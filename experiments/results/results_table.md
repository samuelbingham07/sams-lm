| Run | ε | Val loss | Val ppl | Extracted (all inserted) | Extracted (64 copies) | Mean exposure, 1 copy | Mean exposure, 64 copies | MIA AUC (train vs. val) | MIA AUC (seen 3+) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (no canaries) | ∞ | 1.351 | 3.86 | 0% | 0% | 2.4 | 1.1 | 0.511 | 0.534 |
| No defense | ∞ | 1.347 | 3.85 | 1% | 5% | 1.4 | 16.2 | 0.509 | 0.534 |
| Dedup | ∞ | 1.348 | 3.85 | 0% | 0% | 2.1 | 1.6 | 0.510 | 0.547 |
| DP-SGD (ε=8) | 8.0 | 3.982 | 53.60 | 0% | 0% | 1.2 | 1.5 | 0.499 | 0.492 |

(Baseline has no canaries in its data: its canary numbers are the 'never trained on it' reference. Exposure: 0 to 19.9 bits; random guessing averages 1.44.)
