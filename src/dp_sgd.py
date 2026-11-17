"""Defense 2: train with differentially private SGD (DP-SGD, Abadi et al. 2016; L18).

Normal training averages the gradient over a batch, so one unusual example
(say, a secret repeated in the data) can pull the model a long way. DP-SGD
bounds how much any single example can matter:
  1. compute each example's gradient separately (torch.func vmap + grad),
  2. clip each one to L2 norm <= C            (bounds the sensitivity),
  3. sum them and add Gaussian noise N(0, (sigma*C)^2 I)   (hides any one example),
  4. divide by the expected batch size and take an optimizer step.
Steps 2-3 are exactly the Gaussian mechanism from class applied to the
gradient. accountant.py turns (q, sigma, steps, delta) into the privacy budget
epsilon, so the guarantee is: changing any one training example changes the
probability of any outcome of training by at most a factor e^epsilon (plus delta).

Privacy unit. The training tokens are cut into fixed, non-overlapping
512-token examples, and each step includes every example independently with
probability q = batch_size / N (Poisson sampling, which the accounting
assumes). The guarantee is per example (~2 stories). A canary repeated k times
sits in ~k examples, so it only gets the weaker group-privacy guarantee of
about k*epsilon. That is one of the things the experiment shows.

Because Adam only post-processes the noisy gradient, using AdamW instead of
SGD costs no extra privacy.

Outputs: same layout as train.py, plus checkpoints/<run>/seen_examples.npy
(how many times each example was sampled, for the attacks).

Usage
  python src/dp_sgd.py --run dp_eps8_M --train_bin data/tokens_canary/train.bin --epsilon 8
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
from torch.func import functional_call, grad, vmap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from accountant import epsilon as compute_epsilon, noise_for_epsilon  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from train import MODEL_SIZES, TokenLoader, estimate_loss, lr_at, write_samples  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="M", choices=MODEL_SIZES)
    ap.add_argument("--train_bin", default=str(ROOT / "data" / "tokens_canary" / "train.bin"))
    ap.add_argument("--val_bin", default=str(ROOT / "data" / "tokens" / "validation.bin"))
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "bpe4096.json"))
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=512, help="EXPECTED examples per step (Poisson)")
    ap.add_argument("--micro_batch", type=int, default=16, help="examples per vmap call (memory)")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=8.0, help="target privacy budget")
    ap.add_argument("--delta", type=float, default=None, help="default: 1 / (10 * N examples)")
    ap.add_argument("--noise_multiplier", type=float, default=None, help="sigma; overrides --epsilon")
    ap.add_argument("--clip", type=float, default=1.0, help="per-example gradient norm bound C")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop_at", type=int, default=None)
    return ap.parse_args(argv)


def make_per_example_grads(model):
    """Function (params, x, y) -> dict of per-example gradients, shape (B, *param.shape)."""
    def loss_one(params, x, y):
        _, loss = functional_call(model, params, (x.unsqueeze(0), y.unsqueeze(0)))
        return loss
    return vmap(grad(loss_one), in_dims=(None, 0, 0))


def clipped_sum(per_ex, clip):
    """Clip each example's full gradient (all parameters together) to norm <= clip, then sum.
    Returns (sum dict, per-example norms before clipping)."""
    norms = torch.sqrt(sum(g.float().pow(2).flatten(1).sum(1) for g in per_ex.values()))
    factor = (clip / (norms + 1e-6)).clamp(max=1.0)
    summed = {k: torch.einsum("b,b...->...", factor, g.float()) for k, g in per_ex.items()}
    return summed, norms


def dp_step(model, params, per_ex_fn, xs, ys, clip, sigma, expected_batch, micro_batch, autocast, gen):
    """One DP-SGD gradient: sets p.grad on every parameter. Returns stats."""
    sums = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in params.items()}
    all_norms = []
    for i in range(0, len(xs), micro_batch):
        with autocast():
            g = per_ex_fn(params, xs[i:i + micro_batch], ys[i:i + micro_batch])
        s, norms = clipped_sum(g, clip)
        for k in sums:
            sums[k] += s[k]
        all_norms.append(norms)
        del g
    named = dict(model.named_parameters())
    for k, total in sums.items():
        noise = torch.normal(0.0, sigma * clip, size=total.shape, device=total.device, generator=gen)
        named[k].grad = ((total + noise) / expected_batch).to(named[k].dtype)
    norms = torch.cat(all_norms) if all_norms else torch.zeros(1)
    return {"grad_norm_median": norms.median().item(), "clipped_frac": (norms > clip).float().mean().item()}


def main(argv=None):
    args = get_args(argv)
    ckpt_dir = ROOT / "checkpoints" / args.run
    res_dir = ROOT / "experiments" / "results" / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(args.seed)

    tok = BPETokenizer.load(args.tokenizer)
    data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    bs = args.block_size
    N = (len(data) - 1) // bs                    # example i = tokens [i*bs, i*bs + bs + 1)
    q = args.batch_size / N
    delta = args.delta or 1.0 / (10 * N)
    # --noise_multiplier 0 is the "clipping only" control (same recipe, epsilon = infinity).
    sigma = args.noise_multiplier if args.noise_multiplier is not None \
        else noise_for_epsilon(args.epsilon, q, args.max_steps, delta)
    eps_final, order = compute_epsilon(q, sigma, args.max_steps, delta)
    val = TokenLoader(args.val_bin, bs, 32, device, args.seed)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=bs, **MODEL_SIZES[args.model])
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(args.lr, args.weight_decay, device_type=device)
    params = {k: v.detach() for k, v in model.named_parameters()}   # views: see optimizer updates
    per_ex_fn = make_per_example_grads(model)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    seen = np.zeros(N, dtype=np.int16)
    step = 0

    ckpt_path = ckpt_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, seen = ck["step"], ck["seen"]
        rng.bit_generator.state = ck["data_rng"]
        gen.set_state(ck["noise_rng"].cpu())
        print(f"resumed from step {step}")
    if step == 0:
        for f in ("log.jsonl", "samples.txt"):
            (res_dir / f).unlink(missing_ok=True)

    dp_info = {"examples": int(N), "sample_rate": q, "noise_multiplier": sigma, "clip": args.clip,
               "delta": delta, "epsilon": eps_final, "rdp_order": order, "privacy_unit": f"{bs}-token example"}
    config = {**vars(args), "sampler": "poisson_partition", "dp": dp_info, "train_tokens": int(len(data)),
              "model_config": cfg.to_dict(), "params": model.num_params(non_embedding=False),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (res_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"DP-SGD model {args.model}: N={N:,} examples, q={q:.2e}, sigma={sigma:.3f}, C={args.clip}, "
          f"{args.max_steps} steps -> (eps={eps_final:.2f}, delta={delta:.1e})")

    use_bf16 = device == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else (lambda: torch.autocast("cpu", enabled=False))

    def save(step, final=False):
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "seen": seen,
                 "config": cfg.to_dict(), "args": vars(args), "data_rng": rng.bit_generator.state,
                 "noise_rng": gen.get_state()}
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(ckpt_path)
        if final:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "dp": dp_info},
                       ckpt_dir / "final.pt")
            np.save(ckpt_dir / "seen_examples.npy", seen)

    t_last, tokens_since = time.time(), 0
    try:
        while step < args.max_steps:
            lr = lr_at(step, args, args.max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = np.flatnonzero(rng.random(N) < q)          # Poisson sampling
            seen[idx] += 1
            chunk = np.stack([data[i * bs:i * bs + bs + 1] for i in idx]).astype(np.int64) if len(idx) \
                else np.zeros((0, bs + 1), dtype=np.int64)
            xs = torch.from_numpy(chunk[:, :-1]).to(device)
            ys = torch.from_numpy(chunk[:, 1:]).to(device)
            stats = dp_step(model, params, per_ex_fn, xs, ys, args.clip, sigma, args.batch_size,
                            args.micro_batch, autocast, gen)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens_since += len(idx) * bs

            if step % 10 == 0 or step == args.max_steps:
                dt = time.time() - t_last
                tps = tokens_since / dt
                t_last, tokens_since = time.time(), 0
                eta = (args.max_steps - step) * args.batch_size * bs / max(tps, 1) / 3600
                print(f"step {step:5d}/{args.max_steps} | batch {len(idx)} | lr {lr:.2e} | grad norm (median) "
                      f"{stats['grad_norm_median']:.2f} | clipped {stats['clipped_frac']:.0%} | {tps:,.0f} tok/s "
                      f"| ETA {eta:.1f} h", flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                val_loss = estimate_loss(model, val, args.eval_batches, autocast)
                eps_now = compute_epsilon(q, sigma, step, delta)[0]
                rec = {"step": step, "tokens": step * args.batch_size * bs, "val_loss": val_loss, "lr": lr,
                       "epsilon": eps_now, "time": time.time(), **stats}
                with open(res_dir / "log.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                sample = write_samples(model, tok, device, step, res_dir / "samples.txt")
                print(f"  [eval] step {step}: val loss {val_loss:.4f}  eps so far {eps_now:.2f}  "
                      f"sample: {sample[:120]!r}", flush=True)
                save(step, final=step == args.max_steps)
            if args.stop_at is not None and step >= args.stop_at and step < args.max_steps:
                save(step)
                print(f"paused at step {step}")
                return
    except KeyboardInterrupt:
        save(step)
        print("interrupted: checkpoint saved (resume with --resume)")
        return
    print(f"done -> {ckpt_dir / 'final.pt'} (eps={eps_final:.2f}, delta={delta:.1e})")


if __name__ == "__main__":
    main()
