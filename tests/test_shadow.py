"""Tests for the shadow-model attack code (tiny data / tiny model, CPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import extract  # noqa: E402
import shadow  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


class _Tok:
    eot_id = 63

    def decode(self, ids):
        return " ".join(map(str, ids))


def test_split_excludes_canaries_and_keeps_duplicates_together(tmp_path):
    rng = np.random.default_rng(0)
    stories = [rng.integers(0, 60, size=rng.integers(3, 9)).tolist() + [63] for _ in range(200)]
    stories += [stories[5], stories[5], stories[17]]                 # natural duplicates
    canary = [1, 2, 3, 63]
    stories += [canary] * 4
    toks = np.array([t for s in stories for t in s], dtype=np.uint16)
    offs = np.concatenate([[0], np.cumsum([len(s) for s in stories])]).astype(np.int64)
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    toks.tofile(src / "train.bin")
    np.save(src / "train_offsets.npy", offs)
    (src / "canary_positions.json").write_text(json.dumps({"0": [int(o) for o in offs[-5:-1]]}))

    side = shadow.make_split(seed=0, src=src, out=out)
    assert (side[-4:] == 0).all() and (side[:-4] > 0).all()          # canaries on neither side
    h = [shadow.story_hash(s) for s in stories]
    for i in range(len(stories)):                                     # every copy of a text: same side
        for j in range(len(stories)):
            if h[i] == h[j]:
                assert side[i] == side[j]
    ins = {h[i] for i in np.flatnonzero(side == 1)}
    outs = {h[i] for i in np.flatnonzero(side == 2)}
    assert not ins & outs and abs(len(ins) - len(outs)) <= 1
    # The written shadow training file is exactly the 'in' stories, in order.
    written = np.fromfile(out / "train.bin", dtype=np.uint16)
    expect = np.array([t for i, s in enumerate(stories) if side[i] == 1 for t in s], dtype=np.uint16)
    assert np.array_equal(written, expect)


def test_story_features_match_direct_computation():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()
    rng = np.random.default_rng(1)
    stories = [rng.integers(0, 60, size=rng.integers(4, 20)).tolist() for _ in range(9)]
    f = shadow.story_features(m, _Tok(), stories, "cpu", batch_size=4)
    for s, row in zip(stories, f):
        seq = [63] + s
        with torch.no_grad():
            logits, _ = m(torch.tensor([seq[:-1]]))
        tl = -torch.log_softmax(logits[0].float(), -1)[torch.arange(len(s)), torch.tensor(s)].numpy()
        kk = max(1, int(round(0.2 * len(tl))))
        assert row[0] == pytest.approx(tl.mean(), abs=1e-4)
        assert row[2] == pytest.approx(np.sort(tl)[-kk:].mean(), abs=1e-4)
        assert row[3] == pytest.approx(np.log(len(s)))


def test_attack_classifier_handles_hashes_ending_in_nul():
    # Regression: sampling the dict keys with rng.choice turned them into np.bytes_, which drops
    # trailing b"\x00", so the lookup raised KeyError on the real split.
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2, n_embd=16)).eval()
    rng = np.random.default_rng(2)
    def pool(tag):
        return {bytes([tag, i]) + b"\x00" * (i % 3): rng.integers(0, 60, size=8).tolist() for i in range(30)}
    clf, _, coefs, used = shadow.train_attack_classifier(m, _Tok(), pool(1), pool(2), 20, "cpu")
    assert set(coefs) == set(shadow.FEATURES) and len(used) <= 20


def test_evaluate_mia_custom_attack_signs():
    rng = np.random.default_rng(0)
    items = []
    for g, member, shift in [("seen3+", 1, 1.0), ("train_any", 1, 1.0)] + [("nonmember", 0, 0.0)] * 2:
        for _ in range(200):
            items.append({"group": g, "member": member, "tokens": [0] * int(rng.integers(20, 100)),
                          "loss": float(rng.normal(2 - shift, 0.3)), "ref": float(rng.normal(shift, 0.3))})
    out = extract.evaluate_mia(items, attacks={"loss": -1, "ref": 1})
    assert out["loss"]["seen3+"]["auc"] > 0.9 and out["ref"]["seen3+"]["auc"] > 0.9
    flipped = extract.evaluate_mia(items, attacks={"ref": -1})
    assert flipped["ref"]["seen3+"]["auc"] < 0.1
