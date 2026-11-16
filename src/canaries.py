"""Plant canary secrets in the training data (the memorization experiment).

TinyStories is synthetic, so it contains no real private data. To measure
leakage we plant our own fake secrets ("canaries", after Carlini et al. 2019,
"The Secret Sharer") and later check whether the trained model gives them
back.

Each canary is a short story about an invented character with a random
6-digit secret number:

    Zorvik was a little fox who lived in a small house. Zorvik had a secret
    number that nobody knew. Zorvik's secret number was 3 9 1 4 7 2. Every
    night, Zorvik whispered the number to the stars.

The attack prompts the model with everything up to "secret number was" and
checks whether it completes the right digits. Names are invented and checked
to never appear in TinyStories, so each prompt is unique to one canary.

Repetition is the main variable: 20 canaries each are inserted 1, 2, 4, 8,
16, 32 or 64 times at random story boundaries, and 20 more are generated but
never inserted (held out: the baseline for "not memorized"). That's 2,540
inserted documents, about 0.02% of the training tokens.

Outputs
  data/canaries/canaries.json             definitions (names, secrets, repetitions)
  data/tokens_canary/train.bin            training tokens with canaries inserted
  data/tokens_canary/train_offsets.npy    document starts (stories and canaries)
  data/tokens_canary/canary_positions.json  token start of every inserted copy
  data/tokens_canary_dedup/...            the same after exact dedup (defense 1)
  experiments/results/canaries/canaries.json  copy of the definitions, for the repo

Usage:  python src/canaries.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dedup import duplicate_mask, write_subset  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

REPS = [0, 1, 2, 4, 8, 16, 32, 64]   # 0 = held out (generated, never inserted)
PER_LEVEL = 20
N_DIGITS = 6
ANIMALS = ["bunny", "kitten", "puppy", "bear", "fox", "owl", "duck", "mouse", "frog", "squirrel"]
PREFIX = ("{name} was a little {animal} who lived in a small house. {name} had a secret number "
          "that nobody knew. {name}'s secret number was")
SUFFIX = ". Every night, {name} whispered the number to the stars."
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gr",
          "kr", "pr", "tr", "zh", "th", "qu", "sk"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou", "y"]
CODAS = ["", "n", "r", "x", "k", "l", "m", "z", "sh", "th"]


def secret_text(secret):
    """'391472' -> ' 3 9 1 4 7 2' (one chunk per digit for the tokenizer)."""
    return "".join(" " + d for d in secret)


def canary_text(c):
    return PREFIX.format(**c) + secret_text(c["secret"]) + SUFFIX.format(**c)


def make_name(rng):
    n_syl = rng.integers(2, 4)
    s = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) for _ in range(n_syl)) + rng.choice(CODAS)
    return s.capitalize()


def make_canaries(seed=0, forbidden_text=""):
    """Deterministic list of canary dicts. Names never occur in forbidden_text."""
    rng = np.random.default_rng(seed)
    low = forbidden_text.lower()
    names, canaries = set(), []
    for reps in REPS:
        for _ in range(PER_LEVEL):
            while True:
                name = make_name(rng)
                if 5 <= len(name) <= 9 and name not in names and name.lower() not in low:
                    break
            names.add(name)
            secret = "".join(str(d) for d in rng.integers(0, 10, size=N_DIGITS))
            c = {"id": len(canaries), "name": name, "animal": str(rng.choice(ANIMALS)),
                 "secret": secret, "reps": reps}
            c["prefix"] = PREFIX.format(**c)
            c["text"] = canary_text(c)
            canaries.append(c)
    return canaries


def encode_canary(tok, c):
    """Token ids of the canary, and the [start, end) of the secret within them.

    Encodes prefix / secret / suffix separately and checks that this equals
    encoding the whole text (the pre-tokenizer splits at those boundaries), so
    the attack can score the secret tokens on their own.
    """
    p = tok.encode(c["prefix"])
    s = tok.encode(secret_text(c["secret"]))
    x = tok.encode(SUFFIX.format(**c))
    full = tok.encode(c["text"])
    assert full == p + s + x, f"tokenization does not split cleanly for canary {c['id']}"
    return full, len(p), len(p) + len(s)


def insert(data, offsets, docs, rng):
    """Plan insertion of each doc (token array incl. trailing EOT) at a random
    story boundary. docs: list of (canary_id, tokens). Returns sorted plan
    [(boundary_index, canary_id, tokens)]."""
    n_stories = len(offsets) - 1
    js = rng.integers(0, n_stories + 1, size=len(docs))
    plan = sorted(((int(j), cid, t) for j, (cid, t) in zip(js, docs)), key=lambda r: (r[0], r[1]))
    return plan


def write_with_insertions(data, offsets, plan, out_bin, out_offsets):
    """Stream the original tokens to out_bin with the planned docs spliced in.
    Returns {canary_id: [start positions]}, and writes the new document offsets."""
    positions, doc_starts = {}, []
    pos, prev = 0, 0
    with open(out_bin, "wb") as f:
        for j, cid, toks in plan:
            chunk = np.asarray(data[offsets[prev]:offsets[j]])
            f.write(chunk.tobytes())
            pos += len(chunk)
            positions.setdefault(cid, []).append(pos)
            doc_starts.append(pos)
            f.write(toks.astype(np.uint16).tobytes())
            pos += len(toks)
            prev = j
        chunk = np.asarray(data[offsets[prev]:offsets[-1]])
        f.write(chunk.tobytes())
        pos += len(chunk)
    # Old story starts shift by the length of everything inserted at or before them.
    ins_j = np.array([j for j, _, _ in plan], dtype=np.int64)
    ins_len = np.array([len(t) for _, _, t in plan], dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(ins_len)])
    shift = cum[np.searchsorted(ins_j, np.arange(len(offsets) - 1), side="right")]
    story_starts = offsets[:-1] + shift
    new_offsets = np.sort(np.concatenate([story_starts, np.array(doc_starts, dtype=np.int64)]))
    np.save(out_offsets, np.concatenate([new_offsets, [pos]]).astype(np.int64))
    return positions, pos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", default=str(ROOT / "data" / "tokens"))
    ap.add_argument("--clean_train", default=str(ROOT / "data" / "clean" / "train.txt"))
    args = ap.parse_args(argv)

    tok = BPETokenizer.load(ROOT / "tokenizer" / "bpe4096.json")
    print("reading training text to check names are unseen ...")
    text = Path(args.clean_train).read_text(encoding="utf-8")
    canaries = make_canaries(args.seed, text)
    del text

    for c in canaries:
        ids, s0, s1 = encode_canary(tok, c)
        c["n_tokens"], c["secret_span"] = len(ids), [s0, s1]
    out_dir = ROOT / "data" / "canaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"seed": args.seed, "reps_levels": REPS, "per_level": PER_LEVEL, "n_digits": N_DIGITS,
            "prefix_template": PREFIX, "suffix_template": SUFFIX, "canaries": canaries}
    (out_dir / "canaries.json").write_text(json.dumps(meta, indent=1))
    res = ROOT / "experiments" / "results" / "canaries"
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_dir / "canaries.json", res / "canaries.json")

    data = np.memmap(Path(args.tokens) / "train.bin", dtype=np.uint16, mode="r")
    offsets = np.load(Path(args.tokens) / "train_offsets.npy")
    docs = []
    for c in canaries:
        toks = np.array(tok.encode(c["text"]) + [tok.eot_id], dtype=np.uint16)
        docs += [(c["id"], toks)] * c["reps"]
    rng = np.random.default_rng(args.seed + 1)
    plan = insert(data, offsets, docs, rng)

    can_dir = ROOT / "data" / "tokens_canary"
    can_dir.mkdir(parents=True, exist_ok=True)
    positions, total = write_with_insertions(data, offsets, plan, can_dir / "train.bin",
                                             can_dir / "train_offsets.npy")
    (can_dir / "canary_positions.json").write_text(json.dumps({str(k): v for k, v in positions.items()}))
    print(f"canary data: {total:,} tokens ({total - len(data):,} inserted, {len(docs)} docs)")

    # Sanity: every recorded position holds that canary's tokens.
    new = np.memmap(can_dir / "train.bin", dtype=np.uint16, mode="r")
    new_off = np.load(can_dir / "train_offsets.npy")
    assert len(new) == total == new_off[-1]
    by_id = {c["id"]: np.array(tok.encode(c["text"]), dtype=np.uint16) for c in canaries}
    for cid, ps in positions.items():
        assert len(ps) == canaries[cid]["reps"]
        for p in ps:
            assert (new[p:p + len(by_id[cid])] == by_id[cid]).all()
            assert new[p - 1] == tok.eot_id
    assert set(positions) == {c["id"] for c in canaries if c["reps"] > 0}
    assert np.isin(np.concatenate([np.array(v) for v in positions.values()]), new_off).all()

    # Defense 1: exact dedup of the canary data.
    print("deduplicating ...")
    keep = duplicate_mask(new, new_off)
    dd_dir = ROOT / "data" / "tokens_canary_dedup"
    dd_dir.mkdir(parents=True, exist_ok=True)
    new_start = write_subset(new, new_off, keep, dd_dir / "train.bin", dd_dir / "train_offsets.npy")
    dd_pos = {}
    for cid, ps in positions.items():
        idx = np.searchsorted(new_off, ps)
        kept = [int(new_start[i]) for i in idx if new_start[i] >= 0]
        dd_pos[str(cid)] = kept
        assert len(kept) == 1, f"canary {cid} should survive dedup exactly once"
    (dd_dir / "canary_positions.json").write_text(json.dumps(dd_pos))
    dd = np.memmap(dd_dir / "train.bin", dtype=np.uint16, mode="r")
    for cid, ps in dd_pos.items():
        assert (dd[ps[0]:ps[0] + len(by_id[int(cid)])] == by_id[int(cid)]).all()
    stats = {"docs_before": int(len(new_off) - 1), "docs_after": int(keep.sum()),
             "removed": int((~keep).sum()), "tokens_before": int(total), "tokens_after": int(len(dd))}
    (dd_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("dedup:", stats)


if __name__ == "__main__":
    main()
