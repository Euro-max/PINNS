# E19 network taught by the grey-box model, HF-M0 (test errors, NRMSE; mean [95% CI] over 5 seeds; other arms from E17)

| N | arm | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 100 | PINC | 7.01e-03 [6.64e-03, 7.34e-03] | 3.03e-02 [2.30e-02, 4.01e-02] | 3.90e-01 [1.14e-01, 8.69e-01] | 1.79e+00 [5.87e-01, 3.85e+00] |
| 100 | data-only | 2.23e-02 [2.04e-02, 2.45e-02] | 1.88e-01 [1.61e-01, 2.22e-01] | 1.63e+00 [8.99e-01, 2.55e+00] | 2.57e+00 [1.36e+00, 4.58e+00] |
| 100 | grey-box | 6.17e-03 [5.51e-03, 7.21e-03] | 1.67e-02 [1.05e-02, 2.29e-02] | 4.05e-02 [2.25e-02, 6.51e-02] | 1.71e-01 [1.12e-01, 2.37e-01] |
| 100 | distilled | 2.05e-02 [1.91e-02, 2.18e-02] | 1.52e-01 [1.16e-01, 2.07e-01] | 1.45e+00 [3.34e-01, 2.90e+00] | 1.71e+00 [6.86e-01, 2.85e+00] |
| 1000 | PINC | 4.83e-03 [4.70e-03, 4.97e-03] | 1.22e-02 [1.01e-02, 1.43e-02] | 5.81e-02 [3.42e-02, 7.89e-02] | 6.58e-01 [4.11e-01, 9.41e-01] |
| 1000 | data-only | 3.61e-03 [3.34e-03, 3.99e-03] | 1.74e-02 [1.49e-02, 1.99e-02] | 9.66e-02 [6.29e-02, 1.39e-01] | 1.01e+00 [4.01e-01, 1.70e+00] |
| 1000 | grey-box | 2.45e-03 [2.24e-03, 2.61e-03] | 6.09e-03 [4.86e-03, 7.58e-03] | 1.28e-02 [9.38e-03, 1.62e-02] | 1.30e-01 [9.35e-02, 1.68e-01] |
| 1000 | distilled | 4.68e-03 [4.37e-03, 5.01e-03] | 2.51e-02 [2.01e-02, 2.87e-02] | 9.97e-02 [6.96e-02, 1.41e-01] | 5.05e-01 [2.37e-01, 8.03e-01] |

## Comparisons (error of the second arm divided by the error of the distilled network, above 1: distilled is better; seeds won; paired t-test on log errors)

| N | comparison | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 100 | distilled against PINC | 0.34 (0/5, p = 1.5e-05) | 0.20 (0/5, p = 0.0011) | 0.27 (0/5, p = 0.12) | 1.05 (3/5, p = 0.76) |
| 100 | distilled against data-only | 1.09 (4/5, p = 0.23) | 1.24 (4/5, p = 0.16) | 1.12 (3/5, p = 0.46) | 1.50 (3/5, p = 0.47) |
| 100 | distilled against grey-box | 0.30 (0/5, p = 5.4e-05) | 0.11 (0/5, p = 0.00012) | 0.03 (0/5, p = 0.0032) | 0.10 (0/5, p = 0.0071) |
| 1000 | distilled against PINC | 1.03 (4/5, p = 0.33) | 0.49 (1/5, p = 0.027) | 0.58 (1/5, p = 0.11) | 1.30 (4/5, p = 0.14) |
| 1000 | distilled against data-only | 0.77 (0/5, p = 0.0013) | 0.69 (1/5, p = 0.05) | 0.97 (3/5, p = 0.75) | 2.00 (4/5, p = 0.48) |
| 1000 | distilled against grey-box | 0.52 (0/5, p = 0.0013) | 0.24 (0/5, p = 0.00013) | 0.13 (0/5, p = 4.2e-05) | 0.26 (0/5, p = 0.0037) |
