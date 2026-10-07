# E23 HF-M0 at N = 20 000: prior-anchored network (mean [95% CI] over 5 seeds; other arms from E17)

| N | arm | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 20000 | PINC | 5.74e-03 [5.67e-03, 5.80e-03] | 8.75e-03 [7.02e-03, 1.07e-02] | 2.36e-02 [1.32e-02, 3.65e-02] | 3.97e-01 [1.24e-01, 6.74e-01] |
| 20000 | data-only | 1.80e-03 [1.69e-03, 1.94e-03] | 9.63e-03 [8.95e-03, 1.03e-02] | 3.20e-02 [2.85e-02, 3.57e-02] | 3.12e-01 [1.84e-01, 4.41e-01] |
| 20000 | grey-box | 2.37e-03 [2.20e-03, 2.63e-03] | 6.73e-03 [5.86e-03, 8.04e-03] | 1.45e-02 [1.29e-02, 1.61e-02] | 1.85e-01 [9.02e-02, 3.49e-01] |
| 20000 | A8 PINC | 5.98e-03 [5.89e-03, 6.07e-03] | 8.72e-03 [7.61e-03, 9.82e-03] | 1.36e-02 [1.07e-02, 1.61e-02] | 1.11e-01 [8.58e-02, 1.33e-01] |
| 20000 | A8 data-only | 3.04e-03 [2.88e-03, 3.31e-03] | 1.19e-02 [1.11e-02, 1.26e-02] | 2.33e-02 [2.21e-02, 2.53e-02] | 1.58e-01 [1.35e-01, 1.89e-01] |

## Comparisons (error of the second arm divided by the error of the first, above 1: the first is better; seeds won; paired t-test on log errors)

| N | comparison | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 20000 | A8 PINC against PINC | 0.96 (0/5, p = 0.0047) | 1.00 (3/5, p = 0.89) | 1.73 (4/5, p = 0.21) | 3.57 (4/5, p = 0.068) |
| 20000 | A8 PINC against A8 data-only | 0.51 (0/5, p = 4.4e-05) | 1.36 (5/5, p = 0.021) | 1.72 (5/5, p = 0.033) | 1.42 (4/5, p = 0.081) |
| 20000 | A8 PINC against data-only | 0.30 (0/5, p = 3.5e-06) | 1.10 (4/5, p = 0.058) | 2.36 (5/5, p = 0.0034) | 2.81 (5/5, p = 0.059) |
| 20000 | A8 data-only against data-only | 0.59 (0/5, p = 3.1e-06) | 0.81 (0/5, p = 0.014) | 1.37 (5/5, p = 0.023) | 1.97 (4/5, p = 0.1) |
| 20000 | A8 PINC against grey-box | 0.40 (0/5, p = 3e-05) | 0.77 (0/5, p = 0.0093) | 1.06 (3/5, p = 0.43) | 1.66 (3/5, p = 0.49) |
