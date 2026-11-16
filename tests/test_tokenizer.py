"""Tests for the from-scratch BPE tokenizer.

Run:  python -m pytest tests -q

The key correctness check compares our fast incremental trainer against a
slow, obviously-correct reference implementation (recount every pair after
every merge) and requires identical merges and identical encodings.
"""
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tokenizer import EOT, SPLIT_PATTERN, BPETokenizer, count_chunks  # noqa: E402

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside.",
    "One day, Lily saw a big red ball in the park. \"Can I play?\" she asked.",
    "Tom and Sue were friends. They liked to share their toys and play games.",
    "The dog barked at the cat. The cat ran up the tree! It was scared...",
    "Mom said, \"Don't touch the hot stove.\" Ben didn't listen, and he got hurt.",
    "aaaa aaa aa a  bbbb   ccc\n\nddd\n  e 123 4567 89, x!? yy--zz",
] * 3


# ------------------------------------------------------------ reference impl
def reference_train(chunk_counts, vocab_size, n_special=1):
    """Naive BPE: recount all pairs after every merge."""
    words = {tuple(ch.encode()): c for ch, c in chunk_counts.items()}
    merges = []
    for k in range(vocab_size - 256 - n_special):
        pc = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pc[p] += c
        if not pc:
            break
        best = min(pc, key=lambda p: (-pc[p], p))  # highest count, ties -> smallest pair
        new_id = 256 + k
        nw = Counter()
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(new_id); i += 2
                else:
                    out.append(w[i]); i += 1
            nw[tuple(out)] += c
        words = nw
        merges.append(best)
    return merges


def reference_encode(merges, text):
    """Apply merges in training order to each chunk (equivalent to rank-greedy)."""
    ids = []
    for chunk in re.findall(SPLIT_PATTERN, text):
        w = list(chunk.encode())
        for k, pair in enumerate(merges):
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                    out.append(256 + k); i += 2
                else:
                    out.append(w[i]); i += 1
            w = out
        ids.extend(w)
    return ids


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def counts():
    return count_chunks(CORPUS, workers=1)


@pytest.fixture(scope="module")
def tok(counts):
    return BPETokenizer.train(counts, vocab_size=400)


def random_text(rng, n):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,!?'\"-()"
    extra = "éü中文\U0001F600’“"  # non-ASCII must still round-trip
    return "".join(rng.choice(alphabet + extra) for _ in range(n))


# --------------------------------------------------------------------- tests
def test_matches_reference_trainer(counts):
    fast = BPETokenizer.train(counts, vocab_size=400)
    assert fast.merges == reference_train(counts, 400)


def test_matches_reference_encoder(tok):
    rng = random.Random(0)
    texts = CORPUS + [random_text(rng, 200) for _ in range(50)]
    for t in texts:
        assert tok.encode_ordinary(t) == reference_encode(tok.merges, t)


def test_vocab_size_and_special_id(tok):
    assert tok.vocab_size == 400
    assert tok.eot_id == 399
    assert len(tok.vocab) == 399
    assert len(set(tok.vocab.values())) == len(tok.vocab), "duplicate token byte strings"


@pytest.mark.parametrize("text", [
    "", " ", "\n", "  \n\n  ", "a", "Lily", " Lily", "Hello, world!",
    "don't", "I'll we've they're", "x" * 1000, "café 中文 \U0001F600",
    "tab\there", "trailing space ", "  leading",
])
def test_roundtrip_edge_cases(tok, text):
    assert tok.decode(tok.encode(text)) == text
    assert tok.decode(tok.encode_ordinary(text)) == text


def test_roundtrip_random(tok):
    rng = random.Random(1)
    for _ in range(300):
        t = random_text(rng, rng.randint(0, 300))
        assert tok.decode(tok.encode(t)) == t


def test_special_token(tok):
    ids = tok.encode(f"The end.{EOT}Once upon")
    assert ids.count(tok.eot_id) == 1
    assert tok.decode(ids) == f"The end.{EOT}Once upon"
    # encode_ordinary treats it as plain text: no special id produced
    assert tok.eot_id not in tok.encode_ordinary(EOT)
    assert tok.decode(tok.encode_ordinary(EOT)) == EOT
    assert tok.encode(EOT + EOT) == [tok.eot_id, tok.eot_id]


def test_compresses(tok):
    text = " ".join(CORPUS)
    assert len(tok.encode(text)) < 0.6 * len(text.encode())


def test_deterministic(counts):
    a = BPETokenizer.train(counts, 350)
    b = BPETokenizer.train(Counter(dict(reversed(list(counts.items())))), 350)
    assert a.merges == b.merges, "training must not depend on dict order"


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.merges == tok.merges and tok2.special_tokens == tok.special_tokens
    text = " ".join(CORPUS) + EOT
    assert tok2.encode(text) == tok.encode(text)


def test_parallel_counting_matches_serial():
    stories = CORPUS * 50
    assert count_chunks(stories, workers=1, block=7) == count_chunks(stories, workers=2, block=7)


def test_unknown_id_raises(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 5])


# ---------------------------------------- checks on the real trained tokenizer
TRAINED = ROOT / "tokenizer" / "bpe4096.json"


@pytest.mark.skipif(not TRAINED.exists(), reason="run src/tokenizer.py first")
def test_trained_tokenizer_on_validation():
    tok = BPETokenizer.load(TRAINED)
    assert tok.vocab_size == 4096
    val = ROOT / "data" / "clean" / "validation.txt"
    if not val.exists():
        pytest.skip("run src/prepare_data.py first")
    text = val.read_text(encoding="ascii")[:2_000_000]
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert max(ids) < 4096
    assert len(text) / len(ids) > 3.0, "expected > 3 bytes/token on TinyStories"
