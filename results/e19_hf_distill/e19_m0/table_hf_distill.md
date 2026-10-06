# E19 network taught by the grey-box model, HF-M0 (test errors, NRMSE; mean [95% CI] over 5 seeds; other arms from E17)

| N | arm | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 100 | PINC | 7.01e-03 [6.64e-03, 7.34e-03] | 3.03e-02 [2.30e-02, 4.01e-02] | 3.90e-01 [1.14e-01, 8.69e-01] | 1.79e+00 [5.87e-01, 3.85e+00] |
| 100 | data-only | 2.23e-02 [2.04e-02, 2.45e-02] | 1.88e-01 [1.61e-01, 2.22e-01] | 1.63e+00 [8.99e-01, 2.55e+00] | 2.57e+00 [1.36e+00, 4.58e+00] |
| 100 | grey-box | 6.17e-03 [5.51e-03, 7.21e-03] | 1.67e-02 [1.05e-02, 2.29e-02] | 4.05e-02 [2.25e-02, 6.51e-02] | 1.71e-01 [1.12e-01, 2.37e-01] |
| 100 | distilled | 6.17e-03 [5.51e-03, 7.19e-03] | 4.07e-02 [2.54e-02, 5.72e-02] | 3.11e-01 [1.78e-01, 4.79e-01] | 1.40e+00 [7.97e-01, 2.00e+00] |
| 1000 | PINC | 4.83e-03 [4.70e-03, 4.97e-03] | 1.22e-02 [1.01e-02, 1.43e-02] | 5.81e-02 [3.42e-02, 7.89e-02] | 6.58e-01 [4.11e-01, 9.41e-01] |
| 1000 | data-only | 3.61e-03 [3.34e-03, 3.99e-03] | 1.74e-02 [1.49e-02, 1.99e-02] | 9.66e-02 [6.29e-02, 1.39e-01] | 1.01e+00 [4.01e-01, 1.70e+00] |
| 1000 | grey-box | 2.45e-03 [2.24e-03, 2.61e-03] | 6.09e-03 [4.86e-03, 7.58e-03] | 1.28e-02 [9.38e-03, 1.62e-02] | 1.30e-01 [9.35e-02, 1.68e-01] |
| 1000 | distilled | 2.34e-03 [2.17e-03, 2.49e-03] | 8.75e-03 [8.34e-03, 9.35e-03] | 2.25e-02 [1.49e-02, 3.31e-02] | 1.27e-01 [1.08e-01, 1.49e-01] |

## Comparisons (error of the second arm divided by the error of the distilled network, above 1: distilled is better; seeds won; paired t-test on log errors)

| N | comparison | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 100 | distilled against PINC | 1.14 (4/5, p = 0.16) | 0.74 (1/5, p = 0.29) | 1.25 (2/5, p = 0.73) | 1.28 (1/5, p = 0.88) |
| 100 | distilled against data-only | 3.61 (5/5, p = 0.00017) | 4.61 (5/5, p = 0.0043) | 5.22 (5/5, p = 0.021) | 1.84 (4/5, p = 0.21) |
| 100 | distilled against grey-box | 1.00 (3/5, p = 0.92) | 0.41 (0/5, p = 0.0026) | 0.13 (0/5, p = 0.0037) | 0.12 (0/5, p = 0.0012) |
| 1000 | distilled against PINC | 2.06 (5/5, p = 0.00014) | 1.40 (5/5, p = 0.068) | 2.58 (3/5, p = 0.089) | 5.16 (5/5, p = 0.0041) |
| 1000 | distilled against data-only | 1.54 (5/5, p = 0.008) | 1.99 (5/5, p = 0.0017) | 4.29 (5/5, p = 0.0044) | 7.93 (5/5, p = 0.028) |
| 1000 | distilled against grey-box | 1.05 (4/5, p = 0.14) | 0.70 (1/5, p = 0.039) | 0.57 (1/5, p = 0.073) | 1.02 (3/5, p = 0.94) |
