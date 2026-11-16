"""Tests for the attack code (tiny random model on CPU)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import extract  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()


def direct_logprob(model, seq, start):
    x = torch.tensor([seq[:-1]])
    logits, _ = model(x)
    lp = torch.log_softmax(logits[0].float(), -1)
    return sum(lp[t - 1, seq[t]].item() for t in range(start, len(seq)))


def test_seq_logprobs_matches_direct(model):
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 64, size=rng.integers(3, 20)).tolist() for _ in range(17)]
    starts = [int(rng.integers(1, len(s))) for s in seqs]
    lps, ns = extract.seq_logprobs(model, seqs, starts, "cpu", batch_size=4)
    for s, st, lp, n in zip(seqs, starts, lps, ns):
        assert abs(lp - direct_logprob(model, s, st)) < 1e-4
        assert n == len(s) - st


def test_exact_rank_matches_brute_force(model):
    # 3-digit secrets; digits map to 1- or 2-token sequences like the real tokenizer.
    digit_toks = {str(d): ([40 + d] if d % 3 == 0 else [32, 40 + d]) for d in range(10)}
    ctx = [5, 6, 7]
    all_s = ["".join(p) for p in itertools.product("0123456789", repeat=3)]
    seqs = [ctx + [t for d in s for t in digit_toks[d]] for s in all_s]
    lps, _ = extract.seq_logprobs(model, seqs, [len(ctx)] * len(seqs), "cpu", bf16=False)
    order = np.argsort(-lps)
    for pos in (0, 3, 50, 400):
        secret = all_s[order[pos]]
        rank, _ = extract.exact_rank(model, ctx, secret, digit_toks, "cpu", budget=10 ** 6)
        assert rank == pos + 1
    # budget exceeded -> None
    assert extract.exact_rank(model, ctx, all_s[order[900]], digit_toks, "cpu", budget=50)[0] is None


def test_exposure():
    assert extract.exposure(1) == pytest.approx(np.log2(1e6))
    assert extract.exposure(1e6) == pytest.approx(0)


def test_binary_eval_perfect_and_random():
    r = extract.binary_eval(np.arange(100) + 1000.0, np.arange(100) * 1.0)
    assert r["auc"] == 1 and r["test_accuracy"] == 1 and r["tpr_at_1pct_fpr"] == 1
    rng = np.random.default_rng(0)
    r = extract.binary_eval(rng.normal(size=2000), rng.normal(size=2000))
    assert abs(r["auc"] - 0.5) < 0.05
    assert sum(map(sum, r["test_confusion_matrix"])) == 2000


def test_length_matched_follows_member_lengths():
    rng = np.random.default_rng(0)
    non_lens = rng.integers(10, 250, size=3000)
    mem_lens = rng.integers(20, 60, size=500)          # short members
    idx = extract.length_matched(mem_lens, non_lens)
    matched = non_lens[idx]
    assert abs(matched.mean() - mem_lens.mean()) < 8
    assert (np.abs(matched // 16 - mem_lens // 16) == 0).all()


def test_evaluate_mia_end_to_end():
    rng = np.random.default_rng(0)
    items = []
    for g, member, shift in [("seen0", 1, 0), ("seen1", 1, 0.1), ("seen2", 1, 0.2), ("seen3+", 1, 0.3),
                             ("train_any", 1, 0.1), ("nonmember", 0, 0)] + [("nonmember", 0, 0)]:
        for _ in range(100):
            n = int(rng.integers(20, 200))
            loss = float(rng.normal(2.0 - shift, 0.3))
            items.append({"group": g, "member": member, "tokens": [0] * n, "loss": loss,
                          "zlib_score": loss * n / 300})
    out = extract.evaluate_mia(items)
    assert set(out["loss"]) == {"seen0", "seen1", "seen2", "seen3+", "train_any", "seen1+"}
    assert out["loss"]["seen3+"]["auc"] > out["loss"]["seen0"]["auc"]
    assert "nonmember" in out["mean_loss"]
