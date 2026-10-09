# E30 sensitivity: anchored PINC with lambda selected on the Blockset vehicle, seeds 5-9, E29 test set

Ratios: the other predictor's error / this model's error (above 1: this model is better), geometric mean over seeds, seeds better, t-test on log errors.

| N | lambda | comparison | 1 steps | 10 steps | 50 steps |
|---|---|---|---|---|---|
| 100 | 10 | vs prior | 1.00 (3/5, p = 0.81) | 0.70 (0/5, p = 0.25) | 0.27 (0/5, p = 0.023) |
| 100 | 10 | vs anchored PINC, M0 lambda | 1.00 (0/5, p = nan) | 1.00 (0/5, p = nan) | 1.00 (0/5, p = nan) |
| 100 | 10 | vs anchored data-only | 2.73 (5/5, p = 0.00025) | 1.32 (4/5, p = 0.48) | 0.84 (3/5, p = 0.67) |
| 100 | 10 | vs grey-box-qs | 1.02 (2/5, p = 0.86) | 0.62 (0/5, p = 0.061) | 0.36 (0/5, p = 0.061) |
| 100 | 10 | vs data-only | 4.43 (5/5, p = 1.9e-05) | 6.19 (5/5, p = 0.0025) | 20.69 (5/5, p = 0.0029) |
| 1000 | 10 | vs prior | 0.99 (2/5, p = 0.22) | 0.94 (1/5, p = 0.31) | 0.74 (0/5, p = 0.12) |
| 1000 | 10 | vs anchored PINC, M0 lambda | 0.98 (0/5, p = 0.013) | 1.14 (4/5, p = 0.3) | 5.60 (5/5, p = 0.0094) |
| 1000 | 10 | vs anchored data-only | 0.63 (0/5, p = 0.0022) | 0.34 (0/5, p = 0.00098) | 1.20 (2/5, p = 0.55) |
| 1000 | 10 | vs grey-box-qs | 0.50 (0/5, p = 4.3e-05) | 0.26 (0/5, p = 0.00026) | 0.34 (0/5, p = 0.014) |
| 1000 | 10 | vs data-only | 0.66 (1/5, p = 0.063) | 0.67 (1/5, p = 0.23) | 1.54 (4/5, p = 0.2) |
