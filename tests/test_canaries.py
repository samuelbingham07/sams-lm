"""Tests for canary insertion, dedup, and training-coverage replay."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import canaries  # noqa: E402
import coverage  # noqa: E402
import dedup  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from train import TokenLoader  # noqa: E402

EOT = 9


def make_corpus(stories):
    """stories: list of token lists (without EOT) -> data, offsets."""
    toks, offs = [], [0]
    for s in stories:
        toks += list(s) + [EOT]
        offs.append(len(toks))
    return np.array(toks, dtype=np.uint16), np.array(offs, dtype=np.int64)


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.load(ROOT / "tokenizer" / "bpe4096.json")


def test_make_canaries_deterministic_and_unique():
    a = canaries.make_canaries(0, "Once upon a time Bob")
    b = canaries.make_canaries(0, "Once upon a time Bob")
    assert a == b
    assert len(a) == len(canaries.REPS) * canaries.PER_LEVEL
    assert len({c["name"] for c in a}) == len(a)
    assert all(len(c["secret"]) == 6 and c["secret"].isdigit() for c in a)
    assert sorted({c["reps"] for c in a}) == canaries.REPS


def test_names_avoid_forbidden_text():
    first = canaries.make_canaries(0, "")[0]["name"]
    again = canaries.make_canaries(0, f"the story of {first.lower()} the cat")[0]["name"]
    assert again != first


def test_canary_tokenization_splits_cleanly(tok):
    for c in canaries.make_canaries(0, ""):
        ids, s0, s1 = canaries.encode_canary(tok, c)
        assert tok.decode(ids[s0:s1]) == canaries.secret_text(c["secret"])
        assert tok.decode(ids[:s0]) == c["prefix"]


def test_insertion_positions_and_offsets(tmp_path):
    rng = np.random.default_rng(0)
    stories = [list(rng.integers(10, 50, size=rng.integers(3, 8))) for _ in range(40)]
    data, offs = make_corpus(stories)
    docs = [(0, np.array([100, 101, EOT]))] * 3 + [(1, np.array([200, 201, 202, EOT]))] * 2
    plan = canaries.insert(data, offs, docs, np.random.default_rng(1))
    pos, total = canaries.write_with_insertions(data, offs, plan, tmp_path / "t.bin", tmp_path / "o.npy")
    new = np.fromfile(tmp_path / "t.bin", dtype=np.uint16)
    new_off = np.load(tmp_path / "o.npy")
    assert total == len(new) == len(data) + 3 * 3 + 2 * 4 == new_off[-1]
    assert len(new_off) - 1 == 40 + 5
    assert len(pos[0]) == 3 and len(pos[1]) == 2
    for p in pos[0]:
        assert list(new[p:p + 3]) == [100, 101, EOT]
    for p in pos[1]:
        assert list(new[p:p + 4]) == [200, 201, 202, EOT]
    # Removing the inserted docs gives back the original stories in order.
    docs_new = [list(new[new_off[i]:new_off[i + 1]]) for i in range(len(new_off) - 1)]
    original = [d for d in docs_new if d[0] not in (100, 200)]
    assert original == [list(data[offs[i]:offs[i + 1]]) for i in range(40)]


def test_dedup_keeps_first_copy(tmp_path):
    data, offs = make_corpus([[1, 2], [3], [1, 2], [4, 5, 6], [3], [1, 2]])
    keep = dedup.duplicate_mask(data, offs)
    assert keep.tolist() == [True, True, False, True, False, False]
    new_start = dedup.write_subset(data, offs, keep, tmp_path / "d.bin", tmp_path / "d.npy")
    out = np.fromfile(tmp_path / "d.bin", dtype=np.uint16)
    assert out.tolist() == [1, 2, EOT, 3, EOT, 4, 5, 6, EOT]
    assert np.load(tmp_path / "d.npy").tolist() == [0, 3, 5, 9]
    assert new_start.tolist() == [0, 3, -1, 5, -1, -1]


def test_replay_matches_token_loader(tmp_path):
    data = np.arange(5000, dtype=np.uint16)
    p = tmp_path / "d.bin"
    data.tofile(p)
    loader = TokenLoader(p, 32, 4, "cpu", 1337)
    drawn = [loader.batch()[0][:, 0].numpy() for _ in range(25)]
    replay = coverage.replay_starts(5000, 32, 4, 1337, 25)
    assert np.array_equal(np.concatenate(drawn), replay)


def test_times_seen():
    starts = np.sort(np.array([0, 10, 10, 50]))
    # block 8 -> a window at s covers tokens s..s+8
    assert coverage.times_seen(starts, 10, 19, 8).item() == 2     # [10,19) inside both windows at 10
    assert coverage.times_seen(starts, 10, 20, 8).item() == 0     # one token too long
    assert coverage.times_seen(starts, 2, 9, 8).item() == 1       # window 0 covers 0..8
    assert coverage.times_seen(starts, 52, 55, 8).item() == 1
    assert coverage.times_seen(starts, 30, 35, 8).item() == 0
    np.testing.assert_array_equal(coverage.times_seen(starts, [10, 2], [19, 9], 8), [2, 1])


def test_token_coverage_brute_force():
    rng = np.random.default_rng(0)
    starts = np.sort(rng.integers(0, 90, size=30))
    cov = coverage.token_coverage(starts, 100, 8)
    brute = np.zeros(100, int)
    for s in starts:
        brute[s:s + 9] += 1
    np.testing.assert_array_equal(cov, brute)


def test_seen_counter_poisson_partition(tmp_path):
    run = "pytest_cov"
    d = tmp_path / "checkpoints" / run
    d.mkdir(parents=True)
    np.save(d / "seen_examples.npy", np.array([3, 0, 5], dtype=np.int16))
    f = coverage.seen_counter({"block_size": 8, "sampler": "poisson_partition"}, run, root=tmp_path)
    # example i covers tokens [8i, 8i+8]
    np.testing.assert_array_equal(f([0, 2, 7, 16, 17], [9, 9, 10, 25, 20]), [3, 3, 0, 5, 5])
