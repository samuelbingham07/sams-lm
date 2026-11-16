"""Tests for the from-scratch GPT model.

Run:  python -m pytest tests -q
Most tests use a tiny config on CPU so they run in seconds.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import GPT, GPTConfig  # noqa: E402

TINY = dict(vocab_size=64, block_size=16, n_layer=2, n_head=4, n_embd=32)


def make(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(**{**TINY, **kw}))


def test_shapes_and_loss():
    m = make()
    x = torch.randint(0, 64, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 64)
    assert loss.shape == ()
    _, per_tok = m(x, x, reduction="none")
    assert per_tok.shape == (3, 16)
    assert torch.allclose(per_tok.mean(), loss, atol=1e-6)
    logits, loss = m(x)
    assert loss is None


def test_initial_loss_near_uniform():
    # A freshly initialized model should be ~uniform over the vocab: loss ~ ln(V).
    m = make(vocab_size=4096, n_embd=64)
    x = torch.randint(0, 4096, (8, 16))
    y = torch.randint(0, 4096, (8, 16))  # random targets (y = x would be biased by weight tying)
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(4096)) < 0.1


def test_param_count_formula():
    cfg = GPTConfig(**TINY)
    m = GPT(cfg)
    C, L, V, T = cfg.n_embd, cfg.n_layer, cfg.vocab_size, cfg.block_size
    per_block = 2 * C + (3 * C * C) + (C * C) + (4 * C * C) + (4 * C * C)  # 2 LNs, qkv, proj, fc, proj
    expected = V * C + T * C + L * per_block + C                             # tok, pos, blocks, ln_f
    assert m.num_params(non_embedding=False) == expected


def test_weight_tying():
    m = make()
    assert m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_causal(use_sdpa):
    """Changing a token must not affect predictions at earlier positions."""
    m = make(use_sdpa=use_sdpa).eval()
    x = torch.randint(0, 64, (2, 16))
    x2 = x.clone()
    x2[:, 10] = (x2[:, 10] + 1) % 64
    a, _ = m(x)
    b, _ = m(x2)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_explicit_attention_matches_sdpa():
    m1 = make(use_sdpa=False).eval()
    m2 = make(use_sdpa=True).eval()
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 64, (4, 16))
    assert torch.allclose(m1(x)[0], m2(x)[0], atol=1e-5)


def test_shorter_sequences_and_too_long():
    m = make()
    assert m(torch.randint(0, 64, (2, 5)))[0].shape == (2, 5, 64)
    with pytest.raises(AssertionError):
        m(torch.randint(0, 64, (1, 17)))


def test_ignore_index():
    m = make()
    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    y[:, 8:] = -1
    _, per_tok = m(x, y, reduction="none")
    assert (per_tok[:, 8:] == 0).all()
    _, loss = m(x, y)
    assert torch.allclose(loss, per_tok[:, :8].mean(), atol=1e-6)


def test_can_overfit_one_batch():
    """The basic 'does it learn at all' check: memorize one small batch."""
    m = make()
    opt = m.configure_optimizer(lr=3e-3, weight_decay=0.0, device_type="cpu")
    torch.manual_seed(1)
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_generate():
    m = make()
    x = torch.randint(0, 64, (2, 4))
    out = m.generate(x, max_new_tokens=30, temperature=0)  # longer than block_size: must crop
    assert out.shape == (2, 34)
    assert torch.equal(out[:, :4], x)
    assert torch.equal(out, m.generate(x, 30, temperature=0)), "greedy must be deterministic"
    g = torch.Generator().manual_seed(0)
    s = m.generate(x, 10, temperature=1.0, top_k=5, generator=g)
    assert s.shape == (2, 14) and s.max() < 64
    assert m.training, "generate must restore training mode"


def test_generate_stops_at_eot():
    m = make()
    # Zero the final LayerNorm gain -> all logits are 0 -> greedy argmax picks token 0.
    with torch.no_grad():
        m.ln_f.weight.zero_()
    out = m.generate(torch.zeros(2, 1, dtype=torch.long), 20, temperature=0, eot_id=0)
    assert out.shape[1] == 2, "should stop right after every row emits eot"


def test_per_example_gradients_for_dp_sgd():
    """DP-SGD needs per-example gradients. Check torch.func can compute them
    and that they match computing each example's gradient separately."""
    from torch.func import functional_call, grad, vmap
    m = make()
    params = {k: v.detach() for k, v in m.named_parameters()}
    x = torch.randint(0, 64, (3, 16))
    y = torch.roll(x, -1, dims=1)

    def loss_fn(p, xi, yi):
        _, loss = functional_call(m, p, (xi.unsqueeze(0), yi.unsqueeze(0)))
        return loss

    per_ex = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)
    for i in range(3):
        m.zero_grad()
        _, loss = m(x[i:i + 1], y[i:i + 1])
        loss.backward()
        for k, p in m.named_parameters():
            assert torch.allclose(per_ex[k][i], p.grad, atol=1e-5), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_bf16_forward_backward():
    m = GPT(GPTConfig(vocab_size=4096, block_size=128, n_layer=2, n_head=4, n_embd=128)).cuda()
    x = torch.randint(0, 4096, (4, 128), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
