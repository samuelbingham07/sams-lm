"""Byte-level byte-pair encoding (BPE) tokenizer, implemented from scratch.

How it works
------------
* Base vocabulary: the 256 possible byte values, so ANY string can be encoded
  (no "unknown" token) and decode(encode(s)) == s exactly.
* Pre-tokenization: text is first split into chunks with a GPT-2-style regex
  (words with their leading space, numbers, punctuation runs, whitespace).
  Merges never cross chunk boundaries, so " dog" and " dogs" share structure
  and tokens never glue a word to the punctuation after it.
* Training: repeatedly find the most frequent adjacent pair of tokens across
  the corpus and merge it into a new token, until the vocabulary is full.
  Because the corpus repeats the same words millions of times, we train on
  the multiset of unique chunks (chunk -> count) and update pair counts
  incrementally after each merge instead of recounting everything.
* Encoding: apply the learned merges to each chunk, lowest rank (earliest
  learned) first -- the same rule GPT-2/tiktoken use.
* Special tokens: "<|endoftext|>" separates stories. It gets its own id and is
  never produced by merges.

Usage
-----
    python src/tokenizer.py              # train on data/clean/train.txt -> tokenizer/bpe4096.json
    python src/tokenizer.py --vocab 8192 # different vocabulary size

    from tokenizer import BPETokenizer
    tok = BPETokenizer.load("tokenizer/bpe4096.json")
    ids = tok.encode("Once upon a time<|endoftext|>")
    text = tok.decode(ids)
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# GPT-2's split pattern, restricted to ASCII classes (our cleaned data is ASCII;
# non-ASCII input still works, it just falls into the "other" branch).
SPLIT_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: dict[str, int] | None = None,
                 pattern: str = SPLIT_PATTERN):
        self.pattern = pattern
        self._split = re.compile(pattern)
        self.merges = [tuple(m) for m in merges]
        # Merge i creates token id 256 + i. Its rank (priority) is i.
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special_tokens = dict(special_tokens or {})
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}
        for s, i in self.special_tokens.items():
            assert i not in self.vocab, f"special token id {i} collides with a merge"
        if self.special_tokens:
            self._special_split = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")
        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ info
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    # -------------------------------------------------------------- training
    @classmethod
    def train(cls, chunk_counts: dict[str, int] | Counter, vocab_size: int,
              special_tokens: tuple[str, ...] = (EOT,), verbose: bool = False) -> "BPETokenizer":
        """Learn merges from a {chunk: count} table (see count_chunks)."""
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges >= 0, "vocab_size too small"

        # Each unique chunk is a mutable list of token ids, weighted by its count.
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts = list(chunk_counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> word indices (may be stale)
        for wi, (w, c) in enumerate(zip(words, counts)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += c
                where[pair].add(wi)

        # Max-heap of (-count, pair) with lazy deletion: an entry is valid only
        # if its count still matches pair_counts. Ties break toward the smaller
        # pair tuple, which makes training fully deterministic.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        t0 = time.time()
        while len(merges) < num_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg or -neg <= 0:
                continue  # stale entry
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for wi in list(where.pop(pair, ())):
                w = words[wi]
                if len(w) < 2:
                    continue
                c = counts[wi]
                # Remove this word's old pair contributions...
                for p in zip(w, w[1:]):
                    pair_counts[p] -= c
                    changed.add(p)
                # ...merge every non-overlapping occurrence, left to right...
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == pair[0] and w[i + 1] == pair[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                # ...and add the new word's pair contributions back.
                for p in zip(out, out[1:]):
                    pair_counts[p] += c
                    where[p].add(wi)
                    changed.add(p)
            pair_counts.pop(pair, None)
            changed.discard(pair)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
            if verbose and (len(merges) % 500 == 0 or len(merges) == num_merges):
                tok = BPETokenizer._bytes_of(merges, new_id)
                print(f"  merge {len(merges):5d}/{num_merges}  count={-neg:>10,}  "
                      f"new token {tok!r}  ({time.time() - t0:.1f}s)")

        specials = {s: 256 + len(merges) + i for i, s in enumerate(special_tokens)}
        return cls(merges, specials)

    @staticmethod
    def _bytes_of(merges, token_id) -> bytes:
        if token_id < 256:
            return bytes([token_id])
        a, b = merges[token_id - 256]
        return BPETokenizer._bytes_of(merges, a) + BPETokenizer._bytes_of(merges, b)

    # -------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Find the adjacent pair that was learned earliest.
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                    out.append(best_rank)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        result = tuple(ids)
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = result
        return result

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        ids: list[int] = []
        for chunk in self._split.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text; occurrences of special tokens (e.g. <|endoftext|>) map to their ids."""
        if not self.special_tokens:
            return self.encode_ordinary(text)
        ids: list[int] = []
        for part in self._special_split.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode_bytes(self, ids) -> bytes:
        parts = []
        for i in ids:
            i = int(i)
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special:
                parts.append(self.inverse_special[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts)

    def decode(self, ids) -> str:
        # A slice of ids can end mid-way through a multi-byte character; replace
        # those bytes rather than crash (only matters for non-ASCII text).
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Human-readable form of one token, for inspection."""
        return self.decode([i])

    # ----------------------------------------------------------- persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte-level-bpe", "pattern": self.pattern,
                       "special_tokens": self.special_tokens,
                       "merges": [list(m) for m in self.merges]}, f)

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls([tuple(m) for m in d["merges"]], d["special_tokens"], d["pattern"])


# ---------------------------------------------------------------- helpers
_SPLIT = re.compile(SPLIT_PATTERN)


def _count_block(stories: list[str]) -> Counter:
    c = Counter()
    for s in stories:
        c.update(_SPLIT.findall(s))
    return c


def count_chunks(stories: list[str], workers: int = 8, block: int = 20_000) -> Counter:
    """Count pre-tokenized chunks across stories, in parallel."""
    blocks = [stories[i:i + block] for i in range(0, len(stories), block)]
    total = Counter()
    if workers <= 1:
        for b in blocks:
            total.update(_count_block(b))
        return total
    with Pool(workers) as pool:
        for c in pool.imap_unordered(_count_block, blocks):
            total.update(c)
    return total


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_data import read_stories

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--data", default=str(ROOT / "data" / "clean" / "train.txt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "tokenizer" / f"bpe{args.vocab}.json")

    t0 = time.time()
    stories = read_stories(args.data)
    print(f"read {len(stories):,} stories ({time.time() - t0:.0f}s)")
    counts = count_chunks(stories, workers=args.workers)
    print(f"{len(counts):,} unique chunks, {sum(counts.values()):,} total ({time.time() - t0:.0f}s)")
    tok = BPETokenizer.train(counts, args.vocab, verbose=True)
    tok.save(out)
    print(f"saved {out}  vocab_size={tok.vocab_size}  ({time.time() - t0:.0f}s)")

    # Compression on the training chunks: bytes per token.
    n_bytes = sum(len(ch.encode()) * c for ch, c in counts.items())
    n_tok = sum(len(tok._encode_chunk(ch)) * c for ch, c in counts.items())
    print(f"compression: {n_bytes / n_tok:.2f} bytes/token, {n_tok:,} tokens for the train split")
    longest = sorted(range(256, 256 + len(tok.merges)), key=lambda i: -len(tok.vocab[i]))[:15]
    print("longest tokens:", [tok.token_str(i) for i in longest])


if __name__ == "__main__":
    main()
