"""Write stories with a trained model.

Usage
  python src/generate.py --run baseline_M                          # 3 stories from "Once upon a time"
  python src/generate.py --run canary_M --prompt "Lily found a" -n 5
  python src/generate.py --run baseline_M --temperature 0          # greedy (deterministic)
  python src/generate.py --ckpt checkpoints/baseline_M/model_step2500.pt   # an early snapshot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="baseline_M")
    ap.add_argument("--ckpt", default=None, help="default: checkpoints/<run>/final.pt")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("-n", type=int, default=3, help="number of stories")
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the likeliest token")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / args.run / "final.pt"
    tok = BPETokenizer.load(ROOT / "tokenizer" / "bpe4096.json")
    model = GPT.from_checkpoint(ckpt, args.device)
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    idx = torch.tensor([[tok.eot_id] + tok.encode(args.prompt)] * args.n, device=args.device)
    out = model.generate(idx, args.max_tokens, temperature=args.temperature, top_k=args.top_k,
                         eot_id=tok.eot_id, generator=g)
    for i, row in enumerate(out.tolist()):
        row = row[1:]
        if tok.eot_id in row:
            row = row[:row.index(tok.eot_id)]
        print(f"--- story {i + 1} ---\n{tok.decode(row).strip()}\n")


if __name__ == "__main__":
    main()
