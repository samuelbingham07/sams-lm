"""Smoke tests for the training loop (tiny model, synthetic data, CPU)."""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import train  # noqa: E402


class Args:
    warmup_steps, lr, min_lr_frac = 10, 1e-3, 0.1


def test_lr_schedule():
    a = Args()
    assert math.isclose(train.lr_at(0, a, 100), 1e-4)
    assert math.isclose(train.lr_at(9, a, 100), 1e-3)
    assert math.isclose(train.lr_at(100, a, 100), 1e-4)  # decays to min_lr
    lrs = [train.lr_at(s, a, 100) for s in range(10, 101)]
    assert all(x >= y for x, y in zip(lrs, lrs[1:])), "must decay monotonically after warmup"


def test_loader_targets_are_shifted_inputs(tmp_path):
    data = np.arange(1000, dtype=np.uint16)
    p = tmp_path / "d.bin"
    data.tofile(p)
    x, y = train.TokenLoader(p, 16, 4, "cpu", 0).batch()
    assert x.shape == (4, 16)
    assert (y == x + 1).all()


@pytest.fixture
def run_env(tmp_path):
    rng = np.random.default_rng(0)
    # Highly repetitive synthetic "language" so a tiny model can learn it fast.
    pattern = rng.integers(0, 4095, size=50)
    data = np.tile(pattern, 400).astype(np.uint16)
    (tmp_path / "train.bin").write_bytes(data.tobytes())
    (tmp_path / "val.bin").write_bytes(data[:5000].tobytes())
    name = "pytest_run"
    yield tmp_path, name
    shutil.rmtree(ROOT / "checkpoints" / name, ignore_errors=True)
    shutil.rmtree(ROOT / "experiments" / "results" / name, ignore_errors=True)


def _argv(tmp, name, steps, resume=False):
    a = ["--run", name, "--model", "tiny", "--device", "cpu", "--block_size", "32", "--batch_size", "8",
         "--grad_accum", "1", "--max_steps", str(steps), "--eval_every", "20", "--eval_batches", "2",
         "--warmup_steps", "5", "--lr", "3e-3", "--train_bin", str(tmp / "train.bin"),
         "--val_bin", str(tmp / "val.bin"), "--snapshot_every", "20"]
    return a + (["--resume"] if resume else [])


def test_training_learns_and_writes_outputs(run_env):
    tmp, name = run_env
    train.main(_argv(tmp, name, 60))
    log = [json.loads(l) for l in open(ROOT / "experiments" / "results" / name / "log.jsonl")]
    assert [r["step"] for r in log] == [20, 40, 60]
    assert log[-1]["val_loss"] < log[0]["val_loss"] < math.log(4096)
    assert (ROOT / "checkpoints" / name / "final.pt").exists()
    assert (ROOT / "checkpoints" / name / "model_step20.pt").exists()
    assert "step 60" in (ROOT / "experiments" / "results" / name / "samples.txt").read_text()


def test_resume_matches_uninterrupted(run_env):
    """Train 40 steps straight vs. 20 steps + resume to 40: identical weights."""
    import torch
    tmp, name = run_env
    train.main(_argv(tmp, name, 40))
    straight = torch.load(ROOT / "checkpoints" / name / "final.pt", weights_only=False)["model"]

    shutil.rmtree(ROOT / "checkpoints" / name)
    train.main(_argv(tmp, name, 40) + ["--stop_at", "25"])   # pause mid-run (not on an eval step)
    assert not (ROOT / "checkpoints" / name / "final.pt").exists()
    train.main(_argv(tmp, name, 40, resume=True))
    resumed = torch.load(ROOT / "checkpoints" / name / "final.pt", weights_only=False)["model"]
    for k in straight:
        assert torch.allclose(straight[k], resumed[k], atol=1e-6), k
