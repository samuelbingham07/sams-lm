"""Tests for DP-SGD and the privacy accountant."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import accountant  # noqa: E402
import dp_sgd  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


@pytest.mark.parametrize("q,sigma,steps", [(5e-4, 0.7, 1000), (0.01, 1.1, 500), (0.1, 2.0, 100)])
def test_accountant_matches_opacus(q, sigma, steps):
    rdp_mod = pytest.importorskip("opacus.accountants.analysis.rdp")
    orders = accountant.ORDERS
    theirs = rdp_mod.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
    ours = np.array([steps * accountant.rdp_subsampled_gaussian(q, sigma, a) for a in orders])
    np.testing.assert_allclose(ours, theirs, rtol=1e-6)
    eps_theirs, _ = rdp_mod.get_privacy_spent(orders=orders, rdp=theirs, delta=1e-6)
    assert accountant.epsilon(q, sigma, steps, 1e-6)[0] == pytest.approx(eps_theirs, rel=1e-6)


def test_accountant_sanity():
    # More noise -> less epsilon; more steps -> more epsilon; no sampling -> plain Gaussian RDP.
    e = lambda s, t: accountant.epsilon(0.01, s, t, 1e-5)[0]
    assert e(2.0, 100) < e(1.0, 100) and e(1.0, 100) < e(1.0, 1000)
    assert accountant.rdp_subsampled_gaussian(1.0, 2.0, 5) == pytest.approx(5 / 8)
    sigma = accountant.noise_for_epsilon(8.0, 5e-4, 1000, 1e-6)
    assert accountant.epsilon(5e-4, sigma, 1000, 1e-6)[0] <= 8.0
    assert accountant.epsilon(5e-4, sigma - 0.01, 1000, 1e-6)[0] > 8.0


def test_noise_for_large_epsilon_is_not_clamped():
    # Regression: the search used to stop at sigma = 0.3 and silently return a run
    # with a smaller epsilon than asked for (asked 64, got 29.4).
    q, steps, delta = 5.4e-4, 800, 1e-7
    for target in (32.0, 64.0):
        sigma = accountant.noise_for_epsilon(target, q, steps, delta)
        eps = accountant.epsilon(q, sigma, steps, delta)[0]
        assert target * 0.97 < eps <= target
    with pytest.raises(ValueError):
        accountant.noise_for_epsilon(1e6, q, steps, delta)       # needs sigma below the range
    assert accountant.epsilon(q, 0.0, steps, delta)[0] == float("inf")


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_clipping_bounds_each_example():
    m = tiny_model()
    params = {k: v.detach() for k, v in m.named_parameters()}
    x = torch.randint(0, 64, (5, 16))
    y = torch.roll(x, -1, 1)
    per_ex = dp_sgd.make_per_example_grads(m)(params, x, y)
    for i in range(5):   # clip one example at a time: its contribution has norm <= C
        s, norms = dp_sgd.clipped_sum({k: g[i:i + 1] for k, g in per_ex.items()}, 0.01)
        total = torch.sqrt(sum(v.pow(2).sum() for v in s.values()))
        assert norms[0] > 0.01 and total <= 0.01 + 1e-6


def test_no_clip_no_noise_equals_normal_gradient():
    m = tiny_model()
    params = {k: v.detach() for k, v in m.named_parameters()}
    x = torch.randint(0, 64, (6, 16))
    y = torch.roll(x, -1, 1)
    gen = torch.Generator().manual_seed(0)
    nullctx = lambda: torch.autocast("cpu", enabled=False)
    dp_sgd.dp_step(m, params, dp_sgd.make_per_example_grads(m), x, y, clip=1e9, sigma=0.0,
                   expected_batch=6, micro_batch=4, autocast=nullctx, gen=gen)
    dp_grads = {k: p.grad.clone() for k, p in m.named_parameters()}
    m.zero_grad()
    _, loss = m(x, y)
    loss.backward()
    for k, p in m.named_parameters():
        assert torch.allclose(dp_grads[k], p.grad, atol=1e-5), k


@pytest.fixture
def dp_env(tmp_path):
    rng = np.random.default_rng(0)
    data = np.tile(rng.integers(0, 4095, size=50), 400).astype(np.uint16)
    data.tofile(tmp_path / "train.bin")
    data[:5000].tofile(tmp_path / "val.bin")
    name = "pytest_dp"
    yield tmp_path, name
    shutil.rmtree(ROOT / "checkpoints" / name, ignore_errors=True)
    shutil.rmtree(ROOT / "experiments" / "results" / name, ignore_errors=True)


def _argv(tmp, name, steps):
    return ["--run", name, "--model", "tiny", "--device", "cpu", "--block_size", "32", "--batch_size", "16",
            "--micro_batch", "8", "--max_steps", str(steps), "--eval_every", "10", "--eval_batches", "2",
            "--warmup_steps", "2", "--lr", "3e-3", "--noise_multiplier", "0.5",
            "--train_bin", str(tmp / "train.bin"), "--val_bin", str(tmp / "val.bin")]


def test_dp_training_runs_learns_and_resumes(dp_env):
    tmp, name = dp_env
    dp_sgd.main(_argv(tmp, name, 30))
    log = [json.loads(l) for l in open(ROOT / "experiments" / "results" / name / "log.jsonl")]
    assert [r["step"] for r in log] == [10, 20, 30]
    assert log[-1]["val_loss"] < log[0]["val_loss"]
    assert log[0]["epsilon"] < log[-1]["epsilon"]                 # budget grows with steps
    cfg = json.loads((ROOT / "experiments" / "results" / name / "config.json").read_text())
    assert cfg["dp"]["epsilon"] == pytest.approx(log[-1]["epsilon"])
    seen = np.load(ROOT / "checkpoints" / name / "seen_examples.npy")
    assert seen.sum() > 0
    straight = torch.load(ROOT / "checkpoints" / name / "final.pt", weights_only=False)["model"]

    shutil.rmtree(ROOT / "checkpoints" / name)
    dp_sgd.main(_argv(tmp, name, 30) + ["--stop_at", "15"])
    dp_sgd.main(_argv(tmp, name, 30) + ["--resume"])
    resumed = torch.load(ROOT / "checkpoints" / name / "final.pt", weights_only=False)["model"]
    for k in straight:
        assert torch.allclose(straight[k], resumed[k], atol=1e-5), k
    np.testing.assert_array_equal(seen, np.load(ROOT / "checkpoints" / name / "seen_examples.npy"))
