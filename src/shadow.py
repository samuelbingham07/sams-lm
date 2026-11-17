"""Shadow-model membership inference (Shokri et al. 2017; L16), plus reference-model calibration.

The attacker doesn't have the target's training set, but can draw data from the
same distribution and train look-alike "shadow" models whose membership they
know. Two uses of one shadow model:

1. Shadow attack classifier (Shokri et al. 2017). Score stories the shadow was /
   wasn't trained on with the shadow itself, learn a classifier (sklearn
   LogisticRegression) from those features to "member?", then apply it to the
   target model's features on the stories under test.

2. Reference-model calibration (Carlini et al. 2021, Watson et al. 2022). Some
   stories have low loss because they're easy, not because they were trained on.
   The shadow never saw the stories under test, so its loss measures how easy
   each one is: score = shadow_loss - target_loss (high = target finds it
   unusually easy = member).

Split (`python src/shadow.py split`): the canary training data minus the canaries,
split 50/50 by story *hash*, so every copy of a duplicated story lands on the
same side. The shadow trains on the "in" half with the target's recipe but a
different seed. Stories under test are drawn from the "out" half (members of the
target, never seen by the shadow) and from validation (members of neither).

Attack (`python src/shadow.py attack`): writes experiments/results/shadow_mia/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import coverage  # noqa: E402
import extract  # noqa: E402
from dedup import write_subset  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

CANARY_DIR = ROOT / "data" / "tokens_canary"
SHADOW_DIR = ROOT / "data" / "tokens_shadow"
OUT = ROOT / "experiments" / "results" / "shadow_mia"
FEATURES = ("loss", "zlib_score", "mink20", "log_len")


def story_hash(tokens):
    return hashlib.blake2b(np.asarray(tokens, dtype=np.uint16).tobytes(), digest_size=16).digest()


# ------------------------------------------------------------------ split ----
def make_split(seed=0, src=CANARY_DIR, out=SHADOW_DIR):
    data = np.memmap(src / "train.bin", dtype=np.uint16, mode="r")
    offs = np.load(src / "train_offsets.npy")
    canary_starts = {p for ps in json.loads((src / "canary_positions.json").read_text()).values() for p in ps}
    hashes = [story_hash(data[offs[i]:offs[i + 1]]) for i in range(len(offs) - 1)]
    uniq = sorted({h for i, h in enumerate(hashes) if offs[i] not in canary_starts})
    rng = np.random.default_rng(seed)
    in_set = {uniq[i] for i in rng.permutation(len(uniq))[:len(uniq) // 2]}
    side = np.array([0 if offs[i] in canary_starts else (1 if h in in_set else 2)
                     for i, h in enumerate(hashes)], dtype=np.int8)      # 0 canary, 1 in, 2 out
    out.mkdir(parents=True, exist_ok=True)
    write_subset(data, offs, side == 1, out / "train.bin", out / "train_offsets.npy")
    np.save(out / "side.npy", side)          # per story of the canary data
    stats = {"stories": int(len(side)), "canary": int((side == 0).sum()), "in": int((side == 1).sum()),
             "out": int((side == 2).sum()), "unique_in": len(in_set), "unique_out": len(uniq) - len(in_set),
             "tokens_in": int(offs[1:][side == 1].sum() - offs[:-1][side == 1].sum()), "seed": seed}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)
    return side


# --------------------------------------------------------------- features ----
@torch.no_grad()
def story_features(model, tok, token_lists, device, batch_size=256, pad_to=64, k=0.2):
    """Per story: mean loss, zlib-normalized loss, Min-K% (mean loss of the
    k fraction of hardest tokens; Shi et al. 2024), and log length.
    Each story is scored after an <|endoftext|>, like extract.score_mia."""
    seqs = [[tok.eot_id] + list(t) for t in token_lists]
    lens = np.array([len(s) for s in seqs])
    order = np.argsort(lens, kind="stable")
    feats = np.zeros((len(seqs), len(FEATURES)))
    use_amp = device == "cuda"
    for i in range(0, len(seqs), batch_size):
        idx = order[i:i + batch_size]
        T = int(lens[idx].max()) - 1
        T = min(-(-T // pad_to) * pad_to, max(T, model.cfg.block_size))   # few shapes (see extract.py)
        x = np.zeros((len(idx), T), dtype=np.int64)
        y = np.full((len(idx), T), -1, dtype=np.int64)
        for r, j in enumerate(idx):
            n = lens[j] - 1
            x[r, :n] = seqs[j][:-1]
            y[r, :n] = seqs[j][1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), reduction="none")
        loss = loss.float().cpu().numpy()
        for r, j in enumerate(idx):
            tl = loss[r, :lens[j] - 1]
            text = tok.decode([t for t in token_lists[j] if t != tok.eot_id])
            zb = len(zlib.compress(text.encode("utf-8")))
            kk = max(1, int(round(k * len(tl))))
            feats[j] = [tl.mean(), tl.sum() / zb, np.sort(tl)[-kk:].mean(), np.log(len(tl))]
    return feats


# ------------------------------------------------------------- candidates ----
def target_seen_by_hash(run):
    """hash -> total times the target saw that story text (all copies), for its training data."""
    cfg = coverage.load_run_config(run)
    for key in ("train_bin", "val_bin"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    model_step = torch.load(ROOT / "checkpoints" / run / "final.pt", map_location="cpu", weights_only=False).get("step")
    if model_step:
        cfg["max_steps"] = model_step
    data = np.memmap(cfg["train_bin"], dtype=np.uint16, mode="r")
    offs = np.load(Path(cfg["train_bin"]).with_name("train_offsets.npy"))
    seen = coverage.seen_counter(cfg, run)(np.maximum(offs[:-1] - 1, 0), offs[1:])
    out = {}
    for i in range(len(offs) - 1):
        h = story_hash(data[offs[i]:offs[i + 1]])
        out[h] = out.get(h, 0) + int(seen[i])
    return out, cfg


def pools(max_len=256):
    """Unique 'in' and 'out' stories (token lists) of the split, up to max_len tokens."""
    data = np.memmap(CANARY_DIR / "train.bin", dtype=np.uint16, mode="r")
    offs = np.load(CANARY_DIR / "train_offsets.npy")
    side = np.load(SHADOW_DIR / "side.npy")
    lengths = np.diff(offs)
    res = {1: {}, 2: {}}
    for i in np.flatnonzero((side > 0) & (lengths <= max_len)):
        t = np.asarray(data[offs[i]:offs[i + 1]])
        res[int(side[i])].setdefault(story_hash(t), t.tolist())
    return res[1], res[2]


def validation_pool(val_bin, max_len=256):
    vdata = np.memmap(val_bin, dtype=np.uint16, mode="r")
    voffs = np.load(Path(val_bin).with_name("validation_offsets.npy"))
    return [np.asarray(vdata[voffs[i]:voffs[i + 1]]).tolist() for i in range(len(voffs) - 1)
            if voffs[i + 1] - voffs[i] <= max_len]


# ------------------------------------------------------------------ attack ----
def train_attack_classifier(shadow, tok, pool_in, pool_out, n, device, seed=0):
    """Shokri-style attack model: shadow features of shadow members vs non-members."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    # Sample indices, not the keys themselves: rng.choice on a list of bytes makes np.bytes_,
    # which strips trailing NUL bytes, so hashes ending in b"\x00" would no longer be dict keys.
    def sample(pool):
        keys = list(pool)
        return [pool[keys[i]] for i in rng.choice(len(keys), size=n, replace=False)]
    ins, outs = sample(pool_in), sample(pool_out)
    X = np.r_[story_features(shadow, tok, ins, device), story_features(shadow, tok, outs, device)]
    y = np.r_[np.ones(n), np.zeros(n)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    # In-sample quality on the shadow (how separable members are for a model the attacker controls)
    shadow_eval = extract.binary_eval(clf.decision_function(X[y == 1]), clf.decision_function(X[y == 0]))
    coefs = dict(zip(FEATURES, clf[-1].coef_[0].round(4).tolist()))
    return clf, shadow_eval, coefs, set(map(story_hash, outs))


def attack_target(run, target, shadow, clf, tok, pool_out, val_pool, n_per_group, device, exclude, seed=0):
    seen_by_hash, cfg = target_seen_by_hash(run)
    rng = np.random.default_rng(seed)
    # Members: target trained on them, shadow did not (the out half), not used to fit the classifier.
    cands = [(h, seen_by_hash[h]) for h in pool_out if h in seen_by_hash and h not in exclude]
    groups = {"seen0": [c for c in cands if c[1] == 0], "seen1": [c for c in cands if c[1] == 1],
              "seen2": [c for c in cands if c[1] == 2], "seen3+": [c for c in cands if c[1] >= 3]}
    items = []
    for g, cs in list(groups.items()) + [("train_any", cands)]:
        for i in rng.choice(len(cs), size=min(n_per_group, len(cs)), replace=False):
            items.append({"group": g, "member": 1, "times_seen": int(cs[i][1]), "tokens": pool_out[cs[i][0]]})
    for i in rng.choice(len(val_pool), size=min(5 * n_per_group, len(val_pool)), replace=False):
        items.append({"group": "nonmember", "member": 0, "times_seen": 0, "tokens": val_pool[i]})
    toks = [it["tokens"] for it in items]
    ft = story_features(target, tok, toks, device)
    fs = story_features(shadow, tok, toks, device)
    clf_score = clf.decision_function(ft)
    for it, a, b, c in zip(items, ft, fs, clf_score):
        it["loss"] = float(a[0])
        it["shadow_classifier"] = float(c)
        it["reference"] = float(b[0] - a[0])               # shadow loss - target loss
        it["reference_mink20"] = float(b[2] - a[2])
    res = extract.evaluate_mia(items, attacks={"loss": -1, "shadow_classifier": 1, "reference": 1,
                                                "reference_mink20": 1})
    res["population"] = {g: len(cs) for g, cs in groups.items()}
    return res, items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "attack"])
    ap.add_argument("--shadow_run", default="shadow_M")
    ap.add_argument("--targets", nargs="+",
                    default=["canary_M", "canary_dedup_M", "dp_eps8_M", "dp_sigma0_M", "dp_bigbatch_eps8_M"])
    ap.add_argument("--n_train", type=int, default=5000, help="shadow members (and non-members) to fit the attack")
    ap.add_argument("--n_per_group", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.cmd == "split":
        make_split()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(ROOT / "tokenizer" / "bpe4096.json")
    shadow = GPT.from_checkpoint(ROOT / "checkpoints" / args.shadow_run / "final.pt", args.device)
    t = time.time()
    pool_in, pool_out = pools()
    val_bin = str(ROOT / "data" / "tokens" / "validation.bin")
    val_pool = validation_pool(val_bin)
    print(f"pools: in {len(pool_in):,}  out {len(pool_out):,}  val {len(val_pool):,}  [{time.time() - t:.0f}s]")
    clf, shadow_eval, coefs, used = train_attack_classifier(shadow, tok, pool_in, pool_out, args.n_train, args.device)
    print(f"attack classifier on the shadow (in-sample): AUC {shadow_eval['auc']:.3f}; weights {coefs}")
    summary = {"shadow_run": args.shadow_run, "features": FEATURES, "classifier_weights": coefs,
               "shadow_in_sample": shadow_eval, "targets": {}}
    for run in args.targets:
        if not (ROOT / "checkpoints" / run / "final.pt").exists():
            print(f"  skip {run}: no checkpoint")
            continue
        t = time.time()
        target = GPT.from_checkpoint(ROOT / "checkpoints" / run / "final.pt", args.device)
        res, items = attack_target(run, target, shadow, clf, tok, pool_out, val_pool, args.n_per_group,
                                   args.device, used)
        summary["targets"][run] = res
        print(f"{run} [{time.time() - t:.0f}s]")
        for attack in ("loss", "shadow_classifier", "reference", "reference_mink20"):
            print(f"  {attack:18s} " + "  ".join(f"{g}: AUC {r['auc']:.3f} TPR@1% {r['tpr_at_1pct_fpr']:.3f}"
                                                  for g, r in res[attack].items()))
        with open(OUT / f"{run}_scores.csv", "w") as f:
            f.write("group,member,times_seen,n_tokens,loss,shadow_classifier,reference,reference_mink20\n")
            for it in items:
                f.write(f"{it['group']},{it['member']},{it['times_seen']},{len(it['tokens'])},{it['loss']:.5f},"
                        f"{it['shadow_classifier']:.5f},{it['reference']:.5f},{it['reference_mink20']:.5f}\n")
        del target
        torch.cuda.empty_cache()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
