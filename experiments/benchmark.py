"""Measure training throughput of candidate model sizes on this GPU.

Used to choose the model size (results: experiments/results/benchmark.json). Each config runs a few
warm-up steps, then times forward + backward + AdamW step in bf16 autocast.

Usage:  python experiments/benchmark.py
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT, GPTConfig  # noqa: E402

CANDIDATES = {
    "S  (6L/384d)":  dict(n_layer=6, n_head=6, n_embd=384),
    "M  (8L/512d)":  dict(n_layer=8, n_head=8, n_embd=512),
    "L  (10L/640d)": dict(n_layer=10, n_head=10, n_embd=640),
    "XL (12L/768d)": dict(n_layer=12, n_head=12, n_embd=768),
}


def bench(cfg_kw, use_sdpa, batch=32, T=512, steps=10, compile_=False):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=4096, block_size=T, use_sdpa=use_sdpa, **cfg_kw)
    m = GPT(cfg).cuda()
    opt = m.configure_optimizer(lr=1e-4, weight_decay=0.1)
    if compile_:
        m = torch.compile(m)
    x = torch.randint(0, 4096, (batch, T), device="cuda")
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    n = sum(p.numel() for p in m.parameters())
    res = {"params_M": n / 1e6, "tokens_per_s": batch * T / dt,
           "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    del m, opt
    torch.cuda.empty_cache()
    return res


def main():
    assert torch.cuda.is_available()
    print(torch.cuda.get_device_name(0), torch.__version__)
    results = {}
    for name, kw in CANDIDATES.items():
        for sdpa in (False, True):
            try:
                r = bench(kw, sdpa)
            except torch.OutOfMemoryError:
                r = {"error": "OOM"}
                torch.cuda.empty_cache()
            key = f"{name} {'sdpa' if sdpa else 'explicit'}"
            results[key] = r
            if "error" in r:
                print(f"{key:28s} OOM")
                continue
            # Hours to train on 1 epoch (484M tokens) at this speed.
            print(f"{key:28s} {r['params_M']:6.1f}M params  {r['tokens_per_s']:>9,.0f} tok/s  "
                  f"{r['peak_mem_GB']:5.1f} GB  ~{484e6 / r['tokens_per_s'] / 3600:4.1f} h/epoch")
    out = ROOT / "experiments" / "results" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
