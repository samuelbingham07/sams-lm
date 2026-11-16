"""Defense 1: exact deduplication of training documents.

Repeated text is memorized far more than text seen once (Kandpal et al. 2022,
"Deduplicating Training Data Mitigates Privacy Risks in Language Models").
This keeps the first copy of every exact-duplicate story (compared token by
token) and drops the rest. In our data that removes the 320,241 natural
duplicate stories and collapses each repeated canary to a single copy.

Library use (canaries.py builds the deduplicated canary dataset with it):
    keep = duplicate_mask(data, offsets)        # True = first occurrence
    write_subset(data, offsets, keep, out_dir)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def story_hashes(data, offsets):
    """A 16-byte digest of each story's tokens."""
    return [hashlib.blake2b(data[offsets[i]:offsets[i + 1]].tobytes(), digest_size=16).digest()
            for i in range(len(offsets) - 1)]


def duplicate_mask(data, offsets):
    """Boolean array over stories: True for the first occurrence of each distinct story."""
    seen = set()
    keep = np.zeros(len(offsets) - 1, dtype=bool)
    for i, h in enumerate(story_hashes(data, offsets)):
        if h not in seen:
            seen.add(h)
            keep[i] = True
    return keep


def write_subset(data, offsets, keep, out_bin, out_offsets):
    """Write only the kept stories (in their original order).

    Returns new_start: array mapping each old story index to its start in the
    new file (-1 for dropped stories).
    """
    lengths = np.diff(offsets)
    new_start = np.full(len(keep), -1, dtype=np.int64)
    kept_lengths = lengths[keep]
    starts = np.concatenate([[0], np.cumsum(kept_lengths)])
    new_start[keep] = starts[:-1]
    with open(out_bin, "wb") as f:
        # Write runs of consecutive kept stories in one go (fast on a memmap).
        idx = np.flatnonzero(keep)
        if len(idx):
            breaks = np.flatnonzero(np.diff(idx) != 1)
            run_starts = np.concatenate([[idx[0]], idx[breaks + 1]])
            run_ends = np.concatenate([idx[breaks], [idx[-1]]])
            for s, e in zip(run_starts, run_ends):
                f.write(np.asarray(data[offsets[s]:offsets[e + 1]]).tobytes())
    np.save(out_offsets, starts.astype(np.int64))
    return new_start
