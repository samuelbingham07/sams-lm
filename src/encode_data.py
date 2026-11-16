"""Tokenize the cleaned splits into flat uint16 arrays for training.

Each story is encoded and followed by <|endoftext|>, then all stories are
concatenated (GPT-2 / nanoGPT convention). We also save where each story
starts, which the memorization experiments use to score individual stories.

Output (gitignored), per split in data/tokens/:
  {split}.bin          uint16 token ids (np.memmap-able)
  {split}_offsets.npy  int64, start index of story i; offsets[-1] == len(tokens)
  meta.json            vocab size, token counts, tokenizer path

Usage:  python src/encode_data.py [--tokenizer tokenizer/bpe4096.json]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import read_stories  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_tok = None


def _init(tok_path):
    global _tok
    _tok = BPETokenizer.load(tok_path)


def _encode_block(stories):
    eot = _tok.eot_id
    out = []
    for s in stories:
        ids = _tok.encode_ordinary(s)
        ids.append(eot)
        out.append(np.array(ids, dtype=np.uint16))
    return out


def encode_stories(stories, tok_path, workers=8, block=10_000):
    """Return (tokens uint16 array, offsets int64 array) for a list of stories."""
    blocks = [stories[i:i + block] for i in range(0, len(stories), block)]
    arrays = []
    with Pool(workers, initializer=_init, initargs=(str(tok_path),)) as pool:
        for res in pool.imap(_encode_block, blocks):  # imap keeps story order
            arrays.extend(res)
    lengths = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return np.concatenate(arrays), offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "bpe4096.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    assert tok.vocab_size <= 65536, "uint16 storage needs vocab <= 65536"
    out_dir = ROOT / "data" / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": str(Path(args.tokenizer).relative_to(ROOT)), "vocab_size": tok.vocab_size,
            "eot_id": tok.eot_id, "splits": {}}

    for split in ("validation", "train"):
        t0 = time.time()
        stories = read_stories(ROOT / "data" / "clean" / f"{split}.txt")
        tokens, offsets = encode_stories(stories, args.tokenizer, args.workers)
        tokens.tofile(out_dir / f"{split}.bin")
        np.save(out_dir / f"{split}_offsets.npy", offsets)

        # Sanity checks: story boundaries line up with EOT tokens, and a sample
        # of stories decodes back to the exact original text.
        assert offsets[-1] == len(tokens)
        assert (tokens[offsets[1:] - 1] == tok.eot_id).all()
        assert (tokens == tok.eot_id).sum() == len(stories)
        rng = np.random.default_rng(0)
        for i in rng.choice(len(stories), size=min(200, len(stories)), replace=False):
            assert tok.decode(tokens[offsets[i]:offsets[i + 1] - 1]) == stories[i]

        lens = np.diff(offsets)
        meta["splits"][split] = {"stories": len(stories), "tokens": int(len(tokens)),
                                 "tokens_per_story_mean": float(lens.mean()),
                                 "tokens_per_story_p99": int(np.percentile(lens, 99)),
                                 "tokens_per_story_max": int(lens.max()),
                                 "stories_longer_than_512": int((lens > 512).sum())}
        print(f"{split}: {meta['splits'][split]}  ({time.time() - t0:.0f}s)")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
