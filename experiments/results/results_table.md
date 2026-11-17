| Run | ε | Val loss | Val ppl | Extracted, greedy (64 copies) | Extracted, beam top-1 (64 copies) | Secret in beam top-100 (64 copies) | Mean exposure, 1 copy | Mean exposure, 16 copies | Mean exposure, 64 copies | MIA AUC (train vs. val) | MIA AUC (seen 3+) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (no canaries) | ∞ | 1.351 | 3.86 | 0% | 0% | 0% | 2.3 | 1.4 | 1.1 | 0.511 | 0.534 |
| No defense | ∞ | 1.347 | 3.85 | 5% | 15% | 70% | 1.4 | 5.7 | 15.5 | 0.509 | 0.534 |
| Dedup | ∞ | 1.348 | 3.85 | 0% | 0% | 0% | 2.1 | 2.4 | 1.6 | 0.510 | 0.547 |
| DP-SGD (ε=8) | 8.0 | 3.982 | 53.60 | 0% | 0% | 0% | 1.2 | 1.6 | 1.5 | 0.499 | 0.492 |
| DP-SGD (ε=29) | 29.4 | 3.821 | 45.63 | 0% | 0% | 0% | 1.2 | 1.6 | 1.5 | 0.500 | 0.492 |
| DP-SGD (ε=64) | 63.8 | 3.798 | 44.60 | 0% | 0% | 0% | 1.2 | 1.6 | 1.5 | 0.500 | 0.491 |
| DP recipe, no noise (ε=∞) | ∞ | 1.731 | 5.65 | 0% | 0% | 0% | 1.1 | 1.7 | 4.5 | 0.498 | 0.523 |
| DP-SGD (ε=8, 8× batch) | 8.0 | 3.860 | 47.48 | 0% | 0% | 0% | 1.0 | 1.7 | 1.4 | 0.494 | 0.484 |

(Baseline has no canaries in its data: its canary numbers are the 'never trained on it' reference. Exposure: 0 to 19.9 bits; random guessing averages 1.44.)

### Shadow-model membership inference (members never seen by the shadow vs. validation)

| Run | Loss (no shadow) AUC (seen 3+ / any) | Shadow classifier AUC (seen 3+ / any) | Reference model (loss) AUC (seen 3+ / any) | Reference model (Min-20%) AUC (seen 3+ / any) | Best TPR@1%FPR (seen 3+) |
|---|---|---|---|---|---|
| No defense | 0.537 / 0.510 | 0.567 / 0.530 | 0.796 / 0.612 | 0.763 / 0.604 | 17.3% |
| Dedup | 0.548 / 0.514 | 0.569 / 0.523 | 0.780 / 0.613 | 0.755 / 0.601 | 14.8% |
| DP-SGD (ε=8) | 0.485 / 0.499 | 0.505 / 0.491 | 0.512 / 0.500 | 0.519 / 0.507 | 1.7% |
| DP recipe, no noise (ε=∞) | 0.503 / 0.504 | 0.519 / 0.489 | 0.631 / 0.518 | 0.616 / 0.520 | 2.2% |
| DP-SGD (ε=8, 8× batch) | 0.486 / 0.501 | 0.499 / 0.508 | 0.491 / 0.513 | 0.502 / 0.515 | 1.5% |

(Attack classifier on the shadow itself, in-sample AUC 0.523.)
