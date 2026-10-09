# E30 trained on the Blockset vehicle (seeds 5-9; E29 test set: 100 starting states x 10 sequences; NRMSE of the body states)

Models: mean [95% CI] over seeds of the 50-step error.  Ratios: geometric mean over seeds of (other error / model error), above 1 the model is better; seeds better; t-test on log errors (one-sample against the single prior, paired against data-only and against E29, i.e. the same model trained on our plant).

| model | 50 steps | vs prior | vs data-only | E29 (trained on our plant) | vs E29 |
|---|---|---|---|---|---|
| data-only, N = 100 | 3.34e+00 [2.11e+00, 4.79e+00] | 0.01 (0/5, p = 6.8e-05) | - | 1.22e+00 [7.56e-01, 1.72e+00] | 0.37 (1/5, p = 0.067) |
| anchored data-only, N = 100 | 1.41e-01 [7.77e-02, 2.23e-01] | 0.32 (0/5, p = 0.015) | 24.67 (5/5, p = 0.0017) | 1.68e-01 [1.08e-01, 2.25e-01] | 1.27 (3/5, p = 0.61) |
| anchored PINC, N = 100 | 1.83e-01 [8.73e-02, 2.90e-01] | 0.27 (0/5, p = 0.023) | 20.69 (5/5, p = 0.0029) | 6.04e-02 [4.36e-02, 8.17e-02] | 0.40 (1/5, p = 0.074) |
| grey-box-qs, N = 100 | 6.14e-02 [3.38e-02, 1.03e-01] | 0.76 (1/5, p = 0.4) | 58.01 (5/5, p = 0.00038) | 4.98e-02 [4.18e-02, 5.77e-02] | 0.96 (4/5, p = 0.92) |
| data-only, N = 1000 | 1.07e-01 [5.08e-02, 1.91e-01] | 0.48 (1/5, p = 0.13) | - | 1.14e-01 [6.53e-02, 1.73e-01] | 1.24 (3/5, p = 0.52) |
| anchored data-only, N = 1000 | 7.29e-02 [4.16e-02, 1.04e-01] | 0.62 (2/5, p = 0.17) | 1.29 (3/5, p = 0.61) | 4.85e-02 [4.22e-02, 5.52e-02] | 0.77 (2/5, p = 0.37) |
| anchored PINC, N = 1000 | 3.30e-01 [2.00e-01, 4.66e-01] | 0.13 (0/5, p = 0.0013) | 0.27 (1/5, p = 0.069) | 6.54e-02 [4.18e-02, 1.04e-01] | 0.20 (0/5, p = 0.0085) |
| grey-box-qs, N = 1000 | 1.85e-02 [1.39e-02, 2.31e-02] | 2.18 (5/5, p = 0.0055) | 4.51 (5/5, p = 0.035) | 3.21e-02 [3.03e-02, 3.39e-02] | 1.80 (5/5, p = 0.02) |

## Prior alone

| predictor | 1 step | 10 steps | 50 steps |
|---|---|---|---|
| quasi-steady prior | 0.00593 | 0.0311 | 0.0387 |

## Physics loss and grey-box comparison (ratio of the second's error to the first's)

| N | comparison | 1 steps | 10 steps | 50 steps |
|---|---|---|---|---|
| 100 | anchored PINC vs anchored data-only | 2.73 (5/5, p = 0.00025) | 1.32 (4/5, p = 0.48) | 0.84 (3/5, p = 0.67) |
| 100 | anchored PINC vs grey-box-qs | 1.02 (2/5, p = 0.86) | 0.62 (0/5, p = 0.061) | 0.36 (0/5, p = 0.061) |
| 1000 | anchored PINC vs anchored data-only | 0.64 (0/5, p = 0.0021) | 0.30 (0/5, p = 0.0021) | 0.21 (0/5, p = 0.017) |
| 1000 | anchored PINC vs grey-box-qs | 0.51 (0/5, p = 4.6e-05) | 0.23 (0/5, p = 2.8e-07) | 0.06 (0/5, p = 0.00021) |
