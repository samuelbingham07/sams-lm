"""Precompute the demo's data with the real PyTorch models.

    python demo/precompute.py     # -> demo/space/data/demo.json, demo/test_data/*.json

Everything is computed with the weights rounded through float16, exactly what the
ONNX files store, so the browser should reproduce these numbers. The same data is
the page's backup when a browser can't run the models, and the answer key for
demo/test_demo.mjs.

Also checks that rounding the weights to float16 doesn't change the attack results:
beam-search positions vs. the experiment's float32 results.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))
import extract  # noqa: E402
import shadow  # noqa: E402
from export_onnx import MODELS, fp16_rounded  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

SPACE_DATA = ROOT / "demo" / "space" / "data"
TEST_DATA = ROOT / "demo" / "test_data"
TARGETS = ["no_defense", "dedup", "dp_eps8"]
LABELS = {"no_defense": "No defense", "dedup": "Deduplication", "dp_eps8": "DP-SGD (ε = 8)"}
PROMPTS = ["Once upon a time", "Lily and Ben went to the park.", "The little dog was sad because"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    return fp16_rounded(GPT.from_checkpoint(ROOT / "checkpoints" / MODELS[name] / "final.pt", "cpu")).to(DEVICE)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def story_losses(model, tok, token_lists):
    """Mean per-token loss of each story after <|endoftext|>, float32 (same as the browser)."""
    seqs = [[tok.eot_id] + list(t) for t in token_lists]
    lp, n = extract.seq_logprobs(model, seqs, [1] * len(seqs), DEVICE, bf16=False)
    return (-lp / n).tolist()


def canary_section(tok, models):
    canaries = json.loads((ROOT / "data" / "canaries" / "canaries.json").read_text(encoding="utf-8"))["canaries"]
    digit_toks = {d: tok.encode(" " + d) for d in "0123456789"}
    out = [{"id": c["id"], "name": c["name"], "animal": c["animal"], "secret": c["secret"], "reps": c["reps"],
            "prefix": c["prefix"], "results": {}} for c in canaries]
    agree = {}
    for name in TARGETS:
        rows = {int(r["id"]): r for r in read_csv(ROOT / "experiments" / "results" / MODELS[name] / "attacks" / "canaries.csv")}
        same = 0
        t = time.time()
        for c, o in zip(canaries, out):
            ctx = [tok.eot_id] + tok.encode(c["prefix"])
            beams = extract.beam_digits(models[name], ctx, digit_toks, DEVICE, beam=100)
            guesses = [s for s, _ in beams]
            pos = guesses.index(c["secret"]) + 1 if c["secret"] in guesses else 0
            r = rows[c["id"]]
            same += pos == int(r["beam_position"])
            o["results"][name] = {"top": [[s, round(lp, 4)] for s, lp in beams[:10]], "position": pos,
                                  "experiment_position": int(r["beam_position"]),
                                  "exposure": round(float(r["exposure"]), 2), "times_seen": int(r["times_seen"])}
        agree[name] = f"{same}/{len(canaries)}"
        print(f"  canaries {name}: beam position same as the float32 experiment for {same}/{len(canaries)} "
              f"[{time.time() - t:.0f}s]")
    return out, agree


def mia_section(tok, models, n_each=6, seed=0):
    rng = np.random.default_rng(seed)
    seen, _ = shadow.target_seen_by_hash(MODELS["no_defense"])
    _, pool_out = shadow.pools(max_len=200)
    members = [t for h, t in pool_out.items() if seen.get(h, 0) >= 3 and len(t) >= 80]
    val = [t for t in shadow.validation_pool(str(ROOT / "data" / "tokens" / "validation.bin"), max_len=200)
           if len(t) >= 80]
    picks = [(members[i], 1, seen[shadow.story_hash(members[i])]) for i in rng.choice(len(members), n_each, replace=False)]
    picks += [(val[i], 0, 0) for i in rng.choice(len(val), n_each, replace=False)]
    order = rng.permutation(len(picks))
    picks = [picks[i] for i in order]
    toks = [p[0] for p in picks]
    losses = {name: story_losses(models[name], tok, toks) for name in models}
    examples = []
    for k, (t, member, ts) in enumerate(picks):
        examples.append({"text": tok.decode([x for x in t if x != tok.eot_id]), "tokens": t, "member": member,
                         "times_seen": int(ts), "loss": {n: round(losses[n][k], 5) for n in models},
                         "reference": {n: round(losses["shadow"][k] - losses[n][k], 5) for n in TARGETS}})
    # Score distributions from the actual experiment (experiments/results/shadow_mia/*_scores.csv).
    dists = {}
    for name in TARGETS:
        rows = read_csv(ROOT / "experiments" / "results" / "shadow_mia" / f"{MODELS[name]}_scores.csv")
        mem = np.array([float(r["reference"]) for r in rows if r["group"] == "seen3+"])
        non = np.array([float(r["reference"]) for r in rows if r["member"] == "0"])
        q = np.linspace(0, 1, 101)
        dists[name] = {"members_seen3": np.quantile(mem, q).round(5).tolist(),
                       "nonmembers": np.quantile(non, q).round(5).tolist(),
                       "threshold_1pct_fpr": round(float(np.quantile(non, 0.99)), 5),
                       "tpr_at_threshold": round(float((mem > np.quantile(non, 0.99)).mean()), 4)}
    return examples, dists


def story_section(tok, models, n=2):
    out = {}
    for name in TARGETS:
        out[name] = {}
        for p in PROMPTS:
            g = torch.Generator(device=DEVICE).manual_seed(0)
            x = torch.tensor([[tok.eot_id] + tok.encode(p)] * n, device=DEVICE)
            y = models[name].generate(x, 250, temperature=0.8, top_k=100, eot_id=tok.eot_id, generator=g)
            out[name][p] = [tok.decode([t for t in row[1:].tolist() if t != tok.eot_id]) for row in y]
    return out


def greedy_cases(tok, n_new=60):
    """Greedy continuations (float32, CPU) to test the browser's cached decoding token for token."""
    out = []
    for name in ("no_defense", "dp_eps8"):
        m = fp16_rounded(GPT.from_checkpoint(ROOT / "checkpoints" / MODELS[name] / "final.pt", "cpu"))
        for p in PROMPTS[:2]:
            ids = [tok.eot_id] + tok.encode(p)
            y = m.generate(torch.tensor([ids]), n_new, temperature=0)
            out.append({"model": name, "prompt_ids": ids, "ids": y[0, len(ids):].tolist()})
    return out


def tokenizer_cases(tok, n=400, seed=1):
    rng = np.random.default_rng(seed)
    val = shadow.validation_pool(str(ROOT / "data" / "tokens" / "validation.bin"), max_len=600)
    texts = [tok.decode([x for x in val[i] if x != tok.eot_id]) for i in rng.choice(len(val), n, replace=False)]
    texts += ["", " ", "  hello   world  \n\n", "It's 3:45 -- they'd've said \"no!\"", "Tabs\tand\r\nnewlines",
              "Ünïcödé café 東京 🙂", "<|endoftext|>Once upon a time<|endoftext|>", "x" * 300,
              "1234567890 12 3 4.5", "don't won't 'quoted' ''", "!!!???...", "\n\n\nThe end."]
    return [{"text": t, "ids": tok.encode(t)} for t in texts]


def main():
    SPACE_DATA.mkdir(parents=True, exist_ok=True)
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(ROOT / "tokenizer" / "bpe4096.json")
    models = {n: load(n) for n in MODELS}
    t = time.time()
    canaries, agree = canary_section(tok, models)
    examples, dists = mia_section(tok, models)
    print(f"  membership examples + distributions [{time.time() - t:.0f}s]")
    stories = story_section(tok, models)
    summary = json.loads((ROOT / "experiments" / "results" / "shadow_mia" / "summary.json").read_text(encoding="utf-8"))
    val_loss = {n: json.loads((ROOT / "experiments" / "results" / MODELS[n] / "attacks" / "summary.json").read_text(encoding="utf-8"))
                .get("val_loss") for n in TARGETS}
    data = {"labels": LABELS, "models": {n: MODELS[n] for n in MODELS}, "val_loss": val_loss,
            "prompts": PROMPTS, "canaries": canaries, "mia_examples": [{k: v for k, v in e.items() if k != "tokens"}
                                                                    for e in examples],
            "mia_distributions": dists,
            "mia_auc": {n: {"reference_seen3": summary["targets"][MODELS[n]]["reference"]["seen3+"]["auc"],
                            "loss_seen3": summary["targets"][MODELS[n]]["loss"]["seen3+"]["auc"]} for n in TARGETS},
            "stories": stories, "fp16_check": agree}
    (SPACE_DATA / "demo.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (TEST_DATA / "tokenizer_cases.json").write_text(json.dumps(tokenizer_cases(tok), ensure_ascii=False), encoding="utf-8")
    (TEST_DATA / "mia_examples.json").write_text(json.dumps(examples), encoding="utf-8")
    (TEST_DATA / "greedy.json").write_text(json.dumps(greedy_cases(tok)), encoding="utf-8")
    shutil.copy(ROOT / "tokenizer" / "bpe4096.json", SPACE_DATA / "tokenizer.json")
    print(f"-> {SPACE_DATA / 'demo.json'} ({(SPACE_DATA / 'demo.json').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
