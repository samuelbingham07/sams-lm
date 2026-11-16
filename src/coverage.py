"""Exactly which training tokens did the model see, and how many times?

train.py samples random 512-token windows with a seeded numpy generator, so
the whole sequence of window starts can be replayed after the fact. That
matters for the privacy experiments: with random windows, a story that is "in
the training set" might never have been shown to the model (for one epoch,
about 1/e = 37% of tokens are never covered), while another was shown three
times. Membership inference and canary extraction are measured against how
often each item was actually seen, not just whether it was in the file.

A window starting at s covers tokens s .. s+block_size (inputs s..s+bs-1, the
last target is s+bs). A span [a, b) counts as "seen" when one window covers
all of it, i.e. s <= a and b-1 <= s+block_size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def replay_starts(n_tokens, block_size, batch_size, seed, n_microbatches):
    """Window starts in the exact order TokenLoader.batch() drew them."""
    rng = np.random.default_rng(seed)
    hi = n_tokens - block_size - 1
    # Same call pattern as TokenLoader (one integers() call per micro-batch),
    # so the generator's internal buffering matches exactly.
    return np.concatenate([rng.integers(0, hi, size=batch_size) for _ in range(n_microbatches)])


def starts_for_run(config):
    """Sorted window starts for a finished run, from its config.json dict."""
    n_micro = config["max_steps"] * config["grad_accum"]
    starts = replay_starts(config["train_tokens"], config["block_size"], config["batch_size"],
                           config["seed"], n_micro)
    starts.sort()
    return starts


def load_run_config(run, root=Path(__file__).resolve().parent.parent):
    return json.loads((root / "experiments" / "results" / run / "config.json").read_text())


def times_seen(sorted_starts, span_start, span_end, block_size):
    """How many sampled windows fully contain each span [span_start, span_end).

    Vectorized: span_start / span_end may be arrays. Spans longer than
    block_size + 1 tokens can never be fully contained (returns 0).
    """
    a = np.asarray(span_start, dtype=np.int64)
    b = np.asarray(span_end, dtype=np.int64)
    lo = np.searchsorted(sorted_starts, b - 1 - block_size, side="left")
    hi = np.searchsorted(sorted_starts, a, side="right")
    return np.maximum(hi - lo, 0)


def seen_counter(config, run, root=Path(__file__).resolve().parent.parent):
    """Function (span_start, span_end) -> times seen, for either sampler.

    train.py (random windows): replay the window starts.
    dp_sgd.py (Poisson sampling of fixed examples): example i covers tokens
    [i*bs, i*bs + bs]; its sample count was saved as seen_examples.npy.
    """
    bs = config["block_size"]
    if config.get("sampler") == "poisson_partition":
        seen_ex = np.load(root / "checkpoints" / run / "seen_examples.npy").astype(np.int64)

        def count(a, b):
            a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
            i = a // bs
            ok = (b - 1 <= i * bs + bs) & (i < len(seen_ex))
            return np.where(ok, seen_ex[np.minimum(i, len(seen_ex) - 1)], 0)
        return count
    starts = starts_for_run(config)
    return lambda a, b: times_seen(starts, a, b, bs)


def token_coverage(sorted_starts, n_tokens, block_size):
    """Number of windows covering each token position (for summary stats)."""
    diff = np.zeros(n_tokens + 1, dtype=np.int32)
    np.add.at(diff, sorted_starts, 1)
    ends = np.minimum(sorted_starts + block_size + 1, n_tokens)
    np.add.at(diff, ends, -1)
    return np.cumsum(diff[:-1])
