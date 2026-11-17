"""Privacy attacks on a trained model (inference-stage confidentiality attacks, L16).

1. Canary extraction (Carlini et al. 2019 "The Secret Sharer"; 2021 "Extracting
   Training Data from Large Language Models")
   For each planted canary, prompt with "<|endoftext|>{name} was a little ...
   {name}'s secret number was" and:
     - greedy-decode: did the model output the exact 6-digit secret?
     - beam search over digit strings (attacker knows the format, not the
       secret): is the secret the top guess? among the top 100 guesses?
     - exposure: rank the true secret among all 10^6 possible secrets by the
       model's log-probability. exposure = log2(10^6) - log2(rank), so 0 means
       "no better than a random guess" and ~19.9 means "the model's single top
       guess". The rank is estimated by scoring 5,000 random candidate secrets;
       when the secret beats (nearly) all of them, an exact pruned search over
       the digit tree computes the true rank.

2. Membership inference (Shokri et al. 2017; Yeom et al. 2018)
   Given a story, guess whether it was in the training data. Members are
   training stories; non-members are validation stories (which never occur
   in training; see prepare_data.py). Attack scores:
     - loss: members tend to have lower loss (the model fits them better)
     - zlib:  loss divided by zlib-compressed length, which normalizes for how
       predictable the text is in general (Carlini et al. 2021)
   Evaluated as a binary classifier with sklearn.metrics, like the class: ROC-AUC,
   TPR at 1% FPR, and accuracy / precision / recall / F1 / confusion matrix at a
   threshold chosen on a separate calibration half.
   Members are split by how many times training actually showed them to the
   model (see coverage.py): with random windows, ~37% of training stories are
   never seen in one epoch.

3. Utility: loss / perplexity on the full validation set, for the privacy vs.
   utility trade-off.

Outputs  experiments/results/<run>/attacks/
  summary.json      all metrics
  canaries.csv      per-canary results
  mia_scores.csv    per-story attack scores

Usage
  python src/extract.py --run canary_M
  python src/extract.py --run baseline_M --ckpt checkpoints/baseline_M/model_step10000.pt --tag step10000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import coverage  # noqa: E402
from canaries import N_DIGITS, secret_text  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

SPACE = 10 ** N_DIGITS


# ---------------------------------------------------------------- scoring ----
@torch.no_grad()
def seq_logprobs(model, seqs, score_from, device, batch_size=256, bf16=True, pad_to=64):
    """Sum of log p(token) over tokens seqs[i][score_from[i]:], given everything before.

    Returns (logprob_sums, token_counts). Sequences are length-sorted and
    right-padded; padding never affects real positions because attention is causal.
    """
    lens = np.array([len(s) for s in seqs])
    order = np.argsort(lens, kind="stable")
    lp_out = np.zeros(len(seqs))
    n_out = np.zeros(len(seqs), dtype=np.int64)
    use_amp = bf16 and device == "cuda"
    for i in range(0, len(seqs), batch_size):
        idx = order[i:i + batch_size]
        T = int(lens[idx].max()) - 1
        # Round T up to a multiple of `pad_to` so only a few tensor shapes ever reach
        # the GPU: on ROCm/Windows, hundreds of distinct shapes crash the driver after
        # ~20 batches. Extra padding is masked (-1) and causal, so results are unchanged.
        T = min(-(-T // pad_to) * pad_to, max(T, model.cfg.block_size))
        x = np.zeros((len(idx), T), dtype=np.int64)
        y = np.full((len(idx), T), -1, dtype=np.int64)
        for r, j in enumerate(idx):
            s = seqs[j]
            n = len(s) - 1
            x[r, :n] = s[:-1]
            y[r, :n] = s[1:]
            y[r, :max(score_from[j] - 1, 0)] = -1
        xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(xt, yt, reduction="none")
        lp_out[idx] = (-loss.float().sum(1)).cpu().numpy()
        n_out[idx] = (yt != -1).sum(1).cpu().numpy()
    return lp_out, n_out


def exposure(rank):
    return math.log2(SPACE) - math.log2(rank)


# ------------------------------------------------------- canary extraction ----
def exact_rank(model, ctx, secret, digit_toks, device, budget=30000, margin=0.05):
    """Exact rank of `secret` among all digit strings, by a pruned search.

    Log-probabilities only go down as digits are added, so any partial string
    already below the true secret's total cannot beat it and is pruned. Cheap
    when the secret is memorized (few competitors); returns None if more than
    `budget` sequences would need scoring. Uses fp32 so ties are not bf16 noise.
    """
    full = ctx + [t for d in secret for t in digit_toks[d]]
    true_lp = seq_logprobs(model, [full], [len(ctx)], device, bf16=False)[0][0]
    frontier = [("", [])]
    scored = 0
    n_digits = len(secret)
    for level in range(n_digits):
        children = [(s + d, toks + digit_toks[d]) for s, toks in frontier for d in "0123456789"]
        scored += len(children)
        if scored > budget:
            return None, true_lp
        lps, _ = seq_logprobs(model, [ctx + t for _, t in children], [len(ctx)] * len(children),
                              device, bf16=False)
        if level < n_digits - 1:
            frontier = [c for c, lp in zip(children, lps) if lp > true_lp - margin]
        else:
            final_true = next(lp for (s, _), lp in zip(children, lps) if s == secret)
            beats = sum(1 for (s, _), lp in zip(children, lps) if s != secret and lp > final_true)
            return 1 + beats, final_true
    raise AssertionError("unreachable")


def greedy_digits(model, tok, ctx, device, n_new):
    out = model.generate(torch.tensor([ctx], device=device), n_new, temperature=0)
    text = tok.decode(out[0, len(ctx):].tolist())
    m = re.match(r"((?: \d)+)", text)
    return (m.group(1).replace(" ", "") if m else ""), text


def beam_digits(model, ctx, digit_toks, device, n_digits=N_DIGITS, beam=100):
    """Format-aware extraction: beam search over digit strings only.

    The attacker knows a secret is n_digits digits after the prompt (the Secret
    Sharer threat model) and keeps the `beam` most likely partial strings by
    their joint log-probability. Unlike greedy decoding it can't wander off into
    words, and it doesn't lock in a first digit that looks good alone but leads
    to a worse full number. Never sees the true secret. Returns the final
    candidates, most likely first, as [(digits, logprob)].
    """
    frontier = [("", [])]
    for _ in range(n_digits):
        children = [(s + d, toks + digit_toks[d]) for s, toks in frontier for d in "0123456789"]
        lps, _ = seq_logprobs(model, [ctx + t for _, t in children], [len(ctx)] * len(children),
                              device, bf16=False)
        keep = np.argsort(-lps, kind="stable")[:beam]
        frontier = [children[i] for i in keep]
        scores = lps[keep]
    return [(s, float(lp)) for (s, _), lp in zip(frontier, scores)]


def canary_attack(model, tok, canaries, device, times_seen=None, n_samples=5000, seed=0, log=print,
                  beam=100):
    digit_toks = {d: tok.encode(" " + d) for d in "0123456789"}
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for c in canaries:
        ctx = [tok.eot_id] + tok.encode(c["prefix"])
        secret = c["secret"]
        got, text = greedy_digits(model, tok, ctx, device, 2 * N_DIGITS + 2)
        lead = next((i for i, (a, b) in enumerate(zip(got + " " * N_DIGITS, secret)) if a != b), N_DIGITS)
        guesses = [s for s, _ in beam_digits(model, ctx, digit_toks, device, beam=beam)]
        beam_pos = guesses.index(secret) + 1 if secret in guesses else 0

        cands = set()
        while len(cands) < n_samples:
            s = "".join(map(str, rng.integers(0, 10, size=N_DIGITS)))
            if s != secret:
                cands.add(s)
        cands = sorted(cands)
        seqs = [ctx + [t for d in s for t in digit_toks[d]] for s in [secret] + cands]
        lps, _ = seq_logprobs(model, seqs, [len(ctx)] * len(seqs), device, batch_size=512)
        true_lp, n_greater = lps[0], int((lps[1:] > lps[0]).sum())
        rank = 1 + n_greater / n_samples * (SPACE - 1)
        method = "sampled"
        if n_greater <= 20:   # secret beats nearly all samples: estimate is too coarse, compute exactly
            for budget in (30_000, 300_000):
                r, lp32 = exact_rank(model, ctx, secret, digit_toks, device, budget=budget)
                if r is not None:
                    rank, true_lp, method = r, lp32, "exact"
                    break
            else:
                # Exact search too expensive (many close competitors). Zero sampled
                # wins must not become "rank 1" (= maximum exposure); use the
                # continuity-corrected estimate, half a win out of n_samples.
                rank = 1 + max(n_greater, 0.5) / n_samples * (SPACE - 1)
                method = "sampled_corrected"
        rows.append({"id": c["id"], "name": c["name"], "secret": secret, "reps": c["reps"],
                     "times_seen": int(times_seen[c["id"]]) if times_seen is not None else 0,
                     "greedy": got[:N_DIGITS], "extracted": got[:N_DIGITS] == secret,
                     "beam_top": guesses[0], "extracted_beam": beam_pos == 1,
                     f"in_top{beam}": beam_pos > 0, "beam_position": beam_pos,
                     "leading_digits_correct": lead, "secret_logprob": float(true_lp),
                     "rank": float(rank), "exposure": exposure(rank), "rank_method": method})
    log(f"  canary attack: {len(canaries)} canaries in {time.time() - t0:.0f}s")
    return rows


def summarize_canaries(rows):
    out = {}
    for r in sorted({r["reps"] for r in rows}):
        g = [x for x in rows if x["reps"] == r]
        top_key = next((k for k in g[0] if k.startswith("in_top")), None)
        out[str(r)] = {"n": len(g), "extracted_frac": float(np.mean([x["extracted"] for x in g])),
                       "extracted_beam_frac": float(np.mean([x.get("extracted_beam", False) for x in g])),
                       "in_top_beam_frac": float(np.mean([x[top_key] for x in g])) if top_key else 0.0,
                       "exposure_mean": float(np.mean([x["exposure"] for x in g])),
                       "exposure_median": float(np.median([x["exposure"] for x in g])),
                       "times_seen_mean": float(np.mean([x["times_seen"] for x in g])),
                       "leading_digits_mean": float(np.mean([x["leading_digits_correct"] for x in g]))}
    return out


def canary_times_seen(config, canaries, counter):
    """Times each canary was fully inside a training window (0 if not in this run's data)."""
    pos_file = Path(config["train_bin"]).parent / "canary_positions.json"
    seen = np.zeros(len(canaries), dtype=np.int64)
    if not pos_file.exists():
        return seen
    positions = json.loads(pos_file.read_text())
    for cid, ps in positions.items():
        c = canaries[int(cid)]
        ps = np.array(ps)
        seen[int(cid)] = counter(ps - 1, ps + c["n_tokens"]).sum()
    return seen


# --------------------------------------------------- membership inference ----
def mia_candidates(config, tok, n_per_group, max_len, seed, counter, exclude_positions=()):
    """Sample member stories grouped by times seen, and validation non-members."""
    rng = np.random.default_rng(seed)
    data = np.memmap(config["train_bin"], dtype=np.uint16, mode="r")
    offs = np.load(Path(config["train_bin"]).with_name("train_offsets.npy"))
    # Span = the EOT before the story (its context in scoring) through its own EOT.
    seen = counter(np.maximum(offs[:-1] - 1, 0), offs[1:])
    lengths = np.diff(offs)

    # Exact-duplicate stories: total views across all copies.
    excl = set(exclude_positions)
    hashes = {}
    for i in range(len(lengths)):
        if offs[i] in excl:
            continue
        h = hashlib.blake2b(data[offs[i]:offs[i + 1]].tobytes(), digest_size=16).digest()
        e = hashes.setdefault(h, [i, 0, 0])
        e[1] += int(seen[i])
        e[2] += 1
    uniq = np.array([v for v in hashes.values()])          # (first_index, total_seen, copies)
    ok = lengths[uniq[:, 0]] <= max_len
    uniq = uniq[ok]

    groups = {"seen0": uniq[uniq[:, 1] == 0], "seen1": uniq[uniq[:, 1] == 1],
              "seen2": uniq[uniq[:, 1] == 2], "seen3+": uniq[uniq[:, 1] >= 3]}
    population = {k: int(len(v)) for k, v in groups.items()}
    items = []
    for name, g in groups.items():
        pick = g[rng.choice(len(g), size=min(n_per_group, len(g)), replace=False)]
        for i, s, cp in pick:
            items.append({"group": name, "member": 1, "times_seen": int(s), "copies": int(cp),
                          "tokens": np.asarray(data[offs[i]:offs[i + 1]]).tolist()})
    # Natural mix: uniform sample of training stories, whatever their coverage
    # (the realistic attacker doesn't know which stories were actually seen).
    pick = uniq[rng.choice(len(uniq), size=n_per_group, replace=False)]
    for i, s, cp in pick:
        items.append({"group": "train_any", "member": 1, "times_seen": int(s), "copies": int(cp),
                      "tokens": np.asarray(data[offs[i]:offs[i + 1]]).tolist()})

    vdata = np.memmap(Path(config["val_bin"]), dtype=np.uint16, mode="r")
    voffs = np.load(Path(config["val_bin"]).with_name("validation_offsets.npy"))
    vlen = np.diff(voffs)
    vidx = np.flatnonzero(vlen <= max_len)
    for i in rng.choice(vidx, size=min(5 * n_per_group, len(vidx)), replace=False):
        items.append({"group": "nonmember", "member": 0, "times_seen": 0, "copies": 0,
                      "tokens": np.asarray(vdata[voffs[i]:voffs[i + 1]]).tolist()})
    return items, population


def score_mia(model, tok, items, device):
    seqs = [[tok.eot_id] + it["tokens"] for it in items]
    lps, ns = seq_logprobs(model, seqs, [1] * len(seqs), device)
    for it, lp, n in zip(items, lps, ns):
        text = tok.decode([t for t in it["tokens"] if t != tok.eot_id])
        it["loss"] = float(-lp / n)
        it["zlib_bytes"] = len(zlib.compress(text.encode("utf-8")))
        it["zlib_score"] = float(-lp / it["zlib_bytes"])
    return items


def binary_eval(member_scores, nonmember_scores, seed=0):
    """Attack quality when higher score = 'member'. Threshold chosen on a
    calibration half, metrics on the held-out half (no peeking)."""
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    s = np.r_[member_scores, nonmember_scores]
    fpr, tpr, _ = roc_curve(y, s)
    res = {"n_members": int(len(member_scores)), "n_nonmembers": int(len(nonmember_scores)),
           "auc": float(roc_auc_score(y, s)),
           "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
           "tpr_at_10pct_fpr": float(np.interp(0.10, fpr, tpr))}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cal, test = perm[:len(y) // 2], perm[len(y) // 2:]
    cands = np.unique(s[cal])
    accs = [accuracy_score(y[cal], s[cal] >= t) for t in cands]
    thr = cands[int(np.argmax(accs))]
    pred = (s[test] >= thr).astype(int)
    res.update({"threshold": float(thr), "test_accuracy": float(accuracy_score(y[test], pred)),
                "test_precision": float(precision_score(y[test], pred, zero_division=0)),
                "test_recall": float(recall_score(y[test], pred, zero_division=0)),
                "test_f1": float(f1_score(y[test], pred, zero_division=0)),
                "test_confusion_matrix": confusion_matrix(y[test], pred).tolist()})
    return res


def length_matched(mem_lens, non_lens, seed=0, bin_width=16):
    """For each member, a non-member index with a similar length (same 16-token bin,
    else the nearest non-empty bin). Story length affects both how likely a story is
    to be fully seen (short ones fit in a window more often) and its loss, so
    comparing against unmatched non-members would confound the attack."""
    rng = np.random.default_rng(seed)
    non_bins = np.asarray(non_lens) // bin_width
    by_bin = {b: np.flatnonzero(non_bins == b) for b in np.unique(non_bins)}
    keys = np.array(sorted(by_bin))
    out = []
    for L in mem_lens:
        b = L // bin_width
        if b not in by_bin:
            b = keys[np.argmin(np.abs(keys - b))]
        out.append(rng.choice(by_bin[b]))
    return np.array(out)


def evaluate_mia(items, attacks=None):
    """Each member group vs. an equal number of length-matched non-members.

    attacks: {score name: sign}, sign +1 if a higher score means "member", -1 if
    lower does (losses). Default: the loss and zlib attacks.
    """
    attacks = attacks or {"loss": -1, "zlib_score": -1}
    non = [it for it in items if it["member"] == 0]
    non_lens = [len(it["tokens"]) for it in non]
    groups = {g: [it for it in items if it["group"] == g] for g in ("seen0", "seen1", "seen2", "seen3+", "train_any")}
    seen = [it for it in items if it["group"] in ("seen1", "seen2", "seen3+")]
    groups["seen1+"] = [seen[i] for i in np.random.default_rng(0).permutation(len(seen))[:len(non)]]
    out = {a: {} for a in attacks}
    for group, mem in groups.items():
        if len(mem) < 10:
            continue
        match = length_matched([len(it["tokens"]) for it in mem], non_lens)
        for attack, sign in attacks.items():
            out[attack][group] = binary_eval(sign * np.array([it[attack] for it in mem]),
                                             sign * np.array([non[i][attack] for i in match]))
        out.setdefault("mean_tokens", {})[group] = float(np.mean([len(it["tokens"]) for it in mem]))
    out["mean_loss"] = {g: float(np.mean([it["loss"] for it in items if it["group"] == g]))
                        for g in ("seen0", "seen1", "seen2", "seen3+", "train_any", "nonmember")
                        if any(it["group"] == g for it in items)}
    return out


# ------------------------------------------------------------------ utility ----
@torch.no_grad()
def validation_loss(model, val_bin, block_size, device, batch_size=32):
    """Mean loss over the whole validation set in non-overlapping windows."""
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    n = (len(data) - 1) // block_size
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n)) * block_size
        chunk = np.stack([data[s:s + block_size + 1] for s in idx]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1]).to(device)
        y = torch.from_numpy(chunk[:, 1:]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y, reduction="none")
        total += loss.float().sum().item()
        count += y.numel()
    return total / count


# --------------------------------------------------------------------- main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None, help="default: checkpoints/<run>/final.pt")
    ap.add_argument("--tag", default="final", help="name for this checkpoint's results")
    ap.add_argument("--canaries", default=str(ROOT / "data" / "canaries" / "canaries.json"))
    ap.add_argument("--n_samples", type=int, default=5000, help="random candidates per canary")
    ap.add_argument("--mia_per_group", type=int, default=2000)
    ap.add_argument("--mia_max_len", type=int, default=256, help="only stories up to this many tokens")
    ap.add_argument("--skip_mia", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    config = coverage.load_run_config(args.run)
    for k in ("train_bin", "val_bin"):          # paths in config may be relative to the repo
        if not Path(config[k]).is_absolute():
            config[k] = str(ROOT / config[k])
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / args.run / "final.pt"
    out = ROOT / "experiments" / "results" / args.run / ("attacks" if args.tag == "final" else f"attacks_{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(ROOT / "tokenizer" / "bpe4096.json")
    model = GPT.from_checkpoint(ckpt, args.device)
    if model.ckpt_step is not None:   # a mid-run snapshot has only seen the first ckpt_step steps
        config["max_steps"] = model.ckpt_step
    print(f"{args.run} ({ckpt.name}, step {config['max_steps']}): attacking on {args.device}")
    summary = {"run": args.run, "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
               "train_bin": config["train_bin"], "steps": config["max_steps"]}

    t = time.time()
    summary["val_loss"] = validation_loss(model, config["val_bin"], config["block_size"], args.device)
    summary["val_perplexity"] = math.exp(summary["val_loss"])
    print(f"  validation loss {summary['val_loss']:.4f} (ppl {summary['val_perplexity']:.2f}) [{time.time() - t:.0f}s]")

    meta = json.loads(Path(args.canaries).read_text())
    canaries = meta["canaries"]
    counter = coverage.seen_counter(config, args.run)
    seen = canary_times_seen(config, canaries, counter)
    rows = canary_attack(model, tok, canaries, args.device, seen, args.n_samples)
    summary["canaries"] = summarize_canaries(rows)
    with open(out / "canaries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r, s in summary["canaries"].items():
        print(f"  reps {r:>2}: extracted greedy {s['extracted_frac']:4.0%} beam {s['extracted_beam_frac']:4.0%} "
              f"top-100 {s['in_top_beam_frac']:4.0%}  exposure {s['exposure_mean']:5.2f}  "
              f"(seen {s['times_seen_mean']:.1f}x on average)")

    if not args.skip_mia:
        t = time.time()
        excl = []
        pos_file = Path(config["train_bin"]).parent / "canary_positions.json"
        if pos_file.exists():
            excl = [p for ps in json.loads(pos_file.read_text()).values() for p in ps]
        items, population = mia_candidates(config, tok, args.mia_per_group, args.mia_max_len, 0, counter, excl)
        items = score_mia(model, tok, items, args.device)
        summary["mia_population"] = population
        summary["mia"] = evaluate_mia(items)
        with open(out / "mia_scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "member", "times_seen", "copies", "n_tokens", "loss", "zlib_bytes", "zlib_score"])
            for it in items:
                w.writerow([it["group"], it["member"], it["times_seen"], it["copies"], len(it["tokens"]),
                            f"{it['loss']:.5f}", it["zlib_bytes"], f"{it['zlib_score']:.6f}"])
        m = summary["mia"]
        print(f"  MIA [{time.time() - t:.0f}s] population by times seen: {population}")
        for attack in ("loss", "zlib_score"):
            print(f"  MIA {attack:10s} " + "  ".join(f"{g}: AUC {r['auc']:.3f}" for g, r in m[attack].items()))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
