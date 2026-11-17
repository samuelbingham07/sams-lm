"""Privacy accounting for DP-SGD: how much privacy budget (epsilon) a run spends.

Each DP-SGD step is a "subsampled Gaussian mechanism": every training example
is included in the batch independently with probability q, each example's
gradient is clipped to norm C, and Gaussian noise with std sigma*C is added
to the sum. One step is a little bit private; T steps compose (L17: the
privacy budget adds up). Plain composition of (eps, delta) is very loose, so
we use Renyi DP (Mironov 2017), which composes by simply adding, then convert
the total back to (eps, delta).

RDP of one step at integer order alpha (Mironov, Talwar & Zhang 2019, Sec. 3.3):
    A_alpha = sum_k C(alpha, k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2))
    rdp(alpha) = log(A_alpha) / (alpha - 1)
Conversion to (eps, delta) (Balle et al. 2020, as in Opacus / TF Privacy):
    eps = min_alpha  T*rdp(alpha) + log((alpha-1)/alpha) - (log(delta) + log(alpha)) / (alpha-1)

Only integer orders are used, which gives an epsilon slightly *larger* (safer)
than libraries that also use fractional orders. tests/test_dp.py checks it
against Opacus.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = list(range(2, 257))


def rdp_subsampled_gaussian(q, sigma, alpha):
    """RDP of ONE step of the Poisson-subsampled Gaussian mechanism at integer order alpha."""
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    k = np.arange(alpha + 1)
    log_binom = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    terms = log_binom + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
    return float(logsumexp(terms)) / (alpha - 1)


def epsilon(q, sigma, steps, delta, orders=ORDERS):
    """(epsilon, best order) after `steps` DP-SGD steps."""
    best = (math.inf, None)
    if sigma == 0:            # no noise: no privacy guarantee at all
        return best
    for a in orders:
        rdp = steps * rdp_subsampled_gaussian(q, sigma, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return best


def noise_for_epsilon(target_eps, q, steps, delta, lo=0.05, hi=50.0, tol=1e-4):
    """Smallest noise multiplier sigma that keeps epsilon <= target (binary search)."""
    assert epsilon(q, hi, steps, delta)[0] <= target_eps, "target epsilon unreachable"
    # If even the smallest sigma searched is private enough, the answer is below the
    # range: fail loudly rather than return `lo` (which would give a smaller epsilon
    # than asked for, silently).
    if epsilon(q, lo, steps, delta)[0] <= target_eps:
        raise ValueError(f"target epsilon {target_eps} needs sigma < {lo}; lower `lo`")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta)[0] <= target_eps:
            hi = mid
        else:
            lo = mid
    return hi
