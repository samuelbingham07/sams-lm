"""Export the demo models to ONNX for the in-browser demo (ONNX Runtime Web).

    python demo/export_onnx.py            # -> demo/hf_export/*.onnx + tokenizer.json, then verify

One graph per model, with a key/value cache so the browser doesn't recompute the
whole context for every new token:

  inputs   input_ids     int64  [B, T]          new tokens
           position_ids  int64  [B, T]          their absolute positions
           shared_k/v    f32    [L, 1, H, Ps, D] cache shared by every row (e.g. the prompt
                                                  that all 100 beams start from)
           past_k/v      f32    [L, B, H, Pb, D] per-row cache
           key_mask      f32    [B, Ps+Pb+T]     1 = this key may be attended to, 0 = padding
  outputs  logits        f32    [B, T, vocab]
           new_k/v       f32    [L, B, H, T, D]  cache entries for the new tokens only

The browser keeps the caches and appends new_k/new_v itself. Shapes never need a
zero-length dimension: an unused cache is one masked-out slot.

Weights are STORED in float16 (half the download) and cast back to float32 inside
the graph, so all arithmetic is float32. verify() checks the graph against PyTorch,
and that cached step-by-step decoding equals a full forward pass.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT  # noqa: E402

OUT = ROOT / "demo" / "hf_export"
# demo name -> training run
MODELS = {"no_defense": "canary_M", "dedup": "canary_dedup_M", "dp_eps8": "dp_bigbatch_eps8_M",
          "shadow": "shadow_M"}


class CachedGPT(nn.Module):
    """Same math as model.GPT.forward, plus the cache inputs/outputs described above."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.m = gpt
        self.H = gpt.cfg.n_head
        self.D = gpt.cfg.n_embd // gpt.cfg.n_head

    def forward(self, input_ids, position_ids, shared_k, shared_v, past_k, past_v, key_mask):
        m, H, D = self.m, self.H, self.D
        B, T = input_ids.shape[0], input_ids.shape[1]
        C = H * D
        P = shared_k.shape[3] + past_k.shape[3]
        x = m.tok_emb(input_ids) + m.pos_emb(position_ids)
        j = torch.arange(P + T, device=input_ids.device)
        i = torch.arange(T, device=input_ids.device)
        causal = (j.unsqueeze(0) < P) | ((j.unsqueeze(0) - P) <= i.unsqueeze(1))      # [T, P+T]
        allowed = (causal.unsqueeze(0) & (key_mask.unsqueeze(1) > 0.5)).unsqueeze(1)   # [B, 1, T, P+T]
        new_k, new_v = [], []
        for layer, blk in enumerate(m.blocks):
            a = blk.attn
            q, k, v = a.qkv(blk.ln1(x)).split(C, dim=2)
            q = q.reshape(B, T, H, D).transpose(1, 2)
            k = k.reshape(B, T, H, D).transpose(1, 2)
            v = v.reshape(B, T, H, D).transpose(1, 2)
            new_k.append(k)
            new_v.append(v)
            K = torch.cat([shared_k[layer].expand(B, -1, -1, -1), past_k[layer], k], dim=2)
            V = torch.cat([shared_v[layer].expand(B, -1, -1, -1), past_v[layer], v], dim=2)
            att = (q @ K.transpose(-2, -1)) / math.sqrt(D)
            att = att.masked_fill(~allowed, float("-inf"))
            y = F.softmax(att, dim=-1) @ V
            x = x + a.proj(y.transpose(1, 2).reshape(B, T, C))
            x = x + blk.mlp(blk.ln2(x))
        return m.head(m.ln_f(x)), torch.stack(new_k), torch.stack(new_v)


def empty_cache(cfg, B=1):
    """One masked-out slot: stands in for 'no cache' without zero-length dimensions."""
    shape = (cfg.n_layer, B, cfg.n_head, 1, cfg.n_embd // cfg.n_head)
    return torch.zeros(shape), torch.zeros(shape)


def export(gpt: GPT, path: Path):
    cfg = gpt.cfg
    wrapped = CachedGPT(gpt).eval()
    B, T = 2, 5
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    pos = torch.arange(T).repeat(B, 1) + 3
    sk, sv = (t[:, :1].repeat(1, 1, 1, 2, 1) for t in empty_cache(cfg, 1))
    pk, pv = (t.repeat(1, 1, 1, 1, 1) for t in empty_cache(cfg, B))
    mask = torch.ones(B, 2 + 1 + T)
    names_in = ["input_ids", "position_ids", "shared_k", "shared_v", "past_k", "past_v", "key_mask"]
    dyn = {"input_ids": {0: "B", 1: "T"}, "position_ids": {0: "B", 1: "T"},
           "shared_k": {3: "Ps"}, "shared_v": {3: "Ps"},
           "past_k": {1: "B", 3: "Pb"}, "past_v": {1: "B", 3: "Pb"},
           "key_mask": {0: "B", 1: "K"},
           "logits": {0: "B", 1: "T"}, "new_k": {1: "B", 3: "T"}, "new_v": {1: "B", 3: "T"}}
    with torch.no_grad():
        torch.onnx.export(wrapped, (ids, pos, sk, sv, pk, pv, mask), str(path), input_names=names_in,
                          output_names=["logits", "new_k", "new_v"], dynamic_axes=dyn, opset_version=17,
                          do_constant_folding=True, dynamo=False)


def store_weights_fp16(path: Path, min_size=1024):
    """Replace large float32 initializers with float16 ones + a Cast back to float32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    model = onnx.load(str(path))
    g = model.graph
    casts = []
    for init in list(g.initializer):
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_size:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        g.initializer.remove(init)
        g.initializer.append(numpy_helper.from_array(arr, init.name + "__fp16"))
        casts.append(helper.make_node("Cast", [init.name + "__fp16"], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__cast"))
    for n in reversed(casts):
        g.node.insert(0, n)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def fp16_rounded(gpt: GPT) -> GPT:
    """A copy of the model with every weight rounded through float16 (what the ONNX file stores)."""
    import copy
    g2 = copy.deepcopy(gpt)
    with torch.no_grad():
        for p in g2.parameters():
            if p.numel() >= 1024:
                p.copy_(p.half().float())
    return g2


def verify(gpt: GPT, path: Path, seed=0):
    """ONNX vs PyTorch (fp16-rounded weights): full forward, and cached decoding == full forward."""
    import onnxruntime as ort
    cfg = gpt.cfg
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = fp16_rounded(gpt)
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, cfg.vocab_size - 1, (1, 40), generator=g)
    with torch.no_grad():
        want, _ = ref(seq)
        orig, _ = gpt(seq)
    want, orig = want[0].numpy(), orig[0].numpy()
    e_k, e_v = (t.numpy() for t in empty_cache(cfg))

    def run(ids, pos, sk, sv, pk, pv, mask):
        return sess.run(None, {"input_ids": ids, "position_ids": pos, "shared_k": sk, "shared_v": sv,
                               "past_k": pk, "past_v": pv, "key_mask": mask})

    T = seq.shape[1]
    ids = seq.numpy()
    mask = np.concatenate([[0, 0], np.ones(T)]).astype(np.float32)[None]
    full, k_all, v_all = run(ids, np.arange(T)[None], e_k, e_v, e_k, e_v, mask)
    err_full = np.abs(full[0] - want).max()
    # Cached: first 25 tokens as the shared prefix, then one token at a time in the per-row cache.
    n0 = 25
    sk, sv = k_all[:, :, :, :n0], v_all[:, :, :, :n0]
    pk, pv = e_k, e_v
    errs = []
    for t in range(n0, T):
        P = sk.shape[3] + pk.shape[3]
        m = np.ones((1, P + 1), np.float32)
        m[0, n0] = 0                                   # the placeholder slot at the start of past_k
        out, nk, nv = run(ids[:, t:t + 1], np.array([[t]]), sk, sv, pk, pv, m)
        errs.append(np.abs(out[0, 0] - want[t]).max())
        pk, pv = np.concatenate([pk, nk], 3), np.concatenate([pv, nv], 3)
    # Batched rows with different amounts of padding give the same answer as one row alone.
    ids2 = np.stack([ids[0, :12], ids[0, 1:13]])
    pos2 = np.stack([np.arange(12), np.arange(12)])
    mask2 = np.ones((2, 2 + 12), np.float32)
    mask2[:, :2] = 0
    out2, _, _ = run(ids2, pos2, e_k, e_v, np.repeat(e_k, 2, 1), np.repeat(e_v, 2, 1), mask2)
    with torch.no_grad():
        want2, _ = ref(torch.from_numpy(ids2))
    err_batch = np.abs(out2 - want2.numpy()).max()
    res = {"full_vs_torch": float(err_full), "cached_vs_torch": float(max(errs)),
           "batched_vs_torch": float(err_batch), "fp16_weights_vs_original": float(np.abs(want - orig).max()),
           "logit_scale": float(np.abs(want).max())}
    ok = res["full_vs_torch"] < 1e-3 and res["cached_vs_torch"] < 1e-3 and res["batched_vs_torch"] < 1e-3
    return ok, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.models:
        gpt = GPT.from_checkpoint(ROOT / "checkpoints" / MODELS[name] / "final.pt", "cpu")
        path = OUT / f"{name}.onnx"
        export(gpt, path)
        store_weights_fp16(path)
        ok, res = verify(gpt, path)
        res["mb"] = round(path.stat().st_size / 2**20, 1)
        report[name] = res
        print(f"{name:11s} {'OK ' if ok else 'BAD'} {res}")
        if not ok:
            raise SystemExit(f"{name}: ONNX output does not match PyTorch")
    shutil.copy(ROOT / "tokenizer" / "bpe4096.json", OUT / "tokenizer.json")
    cfg = gpt.cfg
    (OUT / "config.json").write_text(json.dumps({
        "models": {n: MODELS[n] for n in args.models}, "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
        "head_dim": cfg.n_embd // cfg.n_head, "eot_id": cfg.vocab_size - 1}, indent=2), encoding="utf-8")
    space_data = ROOT / "demo" / "space" / "data"
    space_data.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "config.json", space_data / "config.json")       # the page needs the shapes too
    (ROOT / "demo" / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
