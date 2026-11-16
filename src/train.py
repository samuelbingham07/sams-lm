"""Train the GPT on tokenized TinyStories.

What one training step does:
  1. Grab a batch of random 512-token windows from the training tokens.
  2. The model predicts the next token at every position; the loss measures
     how wrong those predictions were (cross-entropy).
  3. Backpropagation computes how to nudge every weight to reduce the loss,
     and AdamW applies the nudge.
Every `eval_every` steps we measure loss on held-out validation stories, write
a few sample stories, and save a checkpoint (so training can resume after a
crash or reboot).

Outputs
  checkpoints/<run>/ckpt.pt                    latest model + optimizer (gitignored, for resuming)
  checkpoints/<run>/model_step<N>.pt           model-only snapshots at milestones
  experiments/results/<run>/config.json        every setting used
  experiments/results/<run>/log.jsonl          train/val loss over time
  experiments/results/<run>/samples.txt        generated stories over time

Usage
  python src/train.py --run baseline_M                 # full run (1 epoch, model M)
  python src/train.py --run baseline_M --resume        # continue after interruption
  python src/train.py --run smoke --model S --max_steps 50 --eval_every 25   # quick test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

MODEL_SIZES = {
    "tiny": dict(n_layer=2, n_head=2, n_embd=64),     # for tests
    "S": dict(n_layer=6, n_head=6, n_embd=384),       # 12.4M params
    "M": dict(n_layer=8, n_head=8, n_embd=512),       # 27.5M params
    "L": dict(n_layer=10, n_head=10, n_embd=640),     # 52.1M params
}
SAMPLE_PROMPTS = ["Once upon a time", "Lily and Ben went to the park.", "The little dog was sad because"]


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name (folder under checkpoints/ and experiments/results/)")
    ap.add_argument("--model", default="M", choices=MODEL_SIZES)
    ap.add_argument("--train_bin", default=str(ROOT / "data" / "tokens" / "train.bin"))
    ap.add_argument("--val_bin", default=str(ROOT / "data" / "tokens" / "validation.bin"))
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "bpe4096.json"))
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--grad_accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the training tokens (sets max_steps)")
    ap.add_argument("--max_steps", type=int, default=None, help="override the step count")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1, help="final LR as a fraction of --lr")
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--snapshot_every", type=int, default=2500, help="keep a model-only checkpoint this often")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop_at", type=int, default=None,
                    help="pause after this step (checkpoint saved, LR schedule unchanged); continue with --resume")
    return ap.parse_args(argv)


class TokenLoader:
    """Samples random (input, target) windows from a uint16 token file.

    Targets are the inputs shifted left by one: at every position the model
    must predict the token that comes next.
    """

    def __init__(self, path, block_size, batch_size, device, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size, self.batch_size, self.device = block_size, batch_size, device
        self.rng = np.random.default_rng(seed)
        assert len(self.data) > block_size + 1, f"{path} is too small"

    def batch(self, rng=None):
        rng = rng or self.rng
        starts = rng.integers(0, len(self.data) - self.block_size - 1, size=self.batch_size)
        chunk = np.stack([self.data[s:s + self.block_size + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)


def lr_at(step, args, max_steps):
    """Linear warmup, then cosine decay down to min_lr_frac * lr."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps))
    min_lr = args.lr * args.min_lr_frac
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, loader, n_batches, autocast):
    """Average loss over a FIXED set of batches (same seed every time), so
    values are comparable across evaluations."""
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(n_batches):
        x, y = loader.batch(rng)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def write_samples(model, tok, device, step, path, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 30} step {step} {'=' * 30}\n")
        for p in SAMPLE_PROMPTS:
            idx = torch.tensor([tok.encode(p)], device=device)
            out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=100,
                                 eot_id=tok.eot_id, generator=g)
            text = tok.decode([t for t in out[0].tolist() if t != tok.eot_id])
            f.write(text.strip() + "\n---\n")
    return text


def main(argv=None):
    args = get_args(argv)
    ckpt_dir = ROOT / "checkpoints" / args.run
    res_dir = ROOT / "experiments" / "results" / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(args.seed)

    tok = BPETokenizer.load(args.tokenizer)
    train = TokenLoader(args.train_bin, args.block_size, args.batch_size, device, args.seed)
    val = TokenLoader(args.val_bin, args.block_size, args.batch_size, device, args.seed)
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    max_steps = args.max_steps or int(args.epochs * len(train.data) / tokens_per_step)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, **MODEL_SIZES[args.model])
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(args.lr, args.weight_decay, device_type=device)
    step = 0

    ckpt_path = ckpt_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        train.rng.bit_generator.state = ck["data_rng"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed from step {step}")
    elif args.resume:
        print("no checkpoint found; starting fresh")
    if step == 0:
        for f in ("log.jsonl", "samples.txt"):
            (res_dir / f).unlink(missing_ok=True)

    config = {**vars(args), "max_steps": max_steps, "tokens_per_step": tokens_per_step,
              "model_config": cfg.to_dict(), "params": model.num_params(non_embedding=False),
              "train_tokens": int(len(train.data)), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (res_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"model {args.model}: {config['params'] / 1e6:.1f}M params | {max_steps:,} steps x "
          f"{tokens_per_step:,} tokens = {max_steps * tokens_per_step / 1e6:.0f}M tokens")

    use_bf16 = device == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else (lambda: torch.autocast("cpu", enabled=False))

    def save(step, snapshot=False):
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                 "config": cfg.to_dict(), "args": vars(args),
                 "data_rng": train.rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(ckpt_path)  # atomic: a crash mid-save can't corrupt the checkpoint
        if snapshot:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step},
                       ckpt_dir / f"model_step{step}.pt")

    def evaluate(step, train_loss):
        val_loss = estimate_loss(model, val, args.eval_batches, autocast)
        rec = {"step": step, "tokens": step * tokens_per_step, "train_loss": train_loss,
               "val_loss": val_loss, "lr": lr_at(step, args, max_steps), "time": time.time()}
        with open(res_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        sample = write_samples(model, tok, device, step, res_dir / "samples.txt")
        print(f"  [eval] step {step}: val loss {val_loss:.4f}  sample: {sample[:150]!r}")
        return val_loss

    model.train()
    t0, t_last, running = time.time(), time.time(), []
    try:
        while step < max_steps:
            lr = lr_at(step, args, max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for _ in range(args.grad_accum):
                x, y = train.batch()
                with autocast():
                    _, loss = model(x, y)
                (loss / args.grad_accum).backward()
                loss_sum += loss.item() / args.grad_accum
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running.append(loss_sum)
            if not math.isfinite(loss_sum):
                raise RuntimeError(f"loss became {loss_sum} at step {step}")

            if step % 50 == 0 or step == max_steps:
                dt = time.time() - t_last
                t_last = time.time()
                tps = 50 * tokens_per_step / dt if step % 50 == 0 else 0
                eta_h = (max_steps - step) * tokens_per_step / max(tps, 1) / 3600
                print(f"step {step:6d}/{max_steps} | loss {np.mean(running[-50:]):.4f} | lr {lr:.2e} | "
                      f"gnorm {gnorm:.2f} | {tps:,.0f} tok/s | ETA {eta_h:.1f} h", flush=True)
            if step % args.eval_every == 0 or step == max_steps:
                evaluate(step, float(np.mean(running[-args.eval_every:])))
                save(step, snapshot=(step % args.snapshot_every == 0 or step == max_steps))
            if args.stop_at is not None and step >= args.stop_at and step < max_steps:
                save(step)
                print(f"paused at step {step} (continue with --resume)")
                return
    except KeyboardInterrupt:
        print("interrupted: saving checkpoint (resume with --resume)")
        save(step)
        return
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, ckpt_dir / "final.pt")
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
