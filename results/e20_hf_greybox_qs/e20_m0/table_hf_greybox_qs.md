# E20 grey-box model with the quasi-steady prior, HF-M0 (test errors, NRMSE; mean [95% CI] over 5 seeds; other arms from E17)

| N | arm | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 100 | PINC | 7.01e-03 [6.64e-03, 7.34e-03] | 3.03e-02 [2.30e-02, 4.01e-02] | 3.90e-01 [1.14e-01, 8.69e-01] | 1.79e+00 [5.87e-01, 3.85e+00] |
| 100 | data-only | 2.23e-02 [2.04e-02, 2.45e-02] | 1.88e-01 [1.61e-01, 2.22e-01] | 1.63e+00 [8.99e-01, 2.55e+00] | 2.57e+00 [1.36e+00, 4.58e+00] |
| 100 | grey-box | 6.17e-03 [5.51e-03, 7.21e-03] | 1.67e-02 [1.05e-02, 2.29e-02] | 4.05e-02 [2.25e-02, 6.51e-02] | 1.71e-01 [1.12e-01, 2.37e-01] |
| 100 | grey-box-qs | 6.00e-03 [5.64e-03, 6.49e-03] | 1.85e-02 [9.78e-03, 2.97e-02] | 5.23e-02 [2.62e-02, 9.02e-02] | 1.81e-01 [1.22e-01, 2.40e-01] |
| 1000 | PINC | 4.83e-03 [4.70e-03, 4.97e-03] | 1.22e-02 [1.01e-02, 1.43e-02] | 5.81e-02 [3.42e-02, 7.89e-02] | 6.58e-01 [4.11e-01, 9.41e-01] |
| 1000 | data-only | 3.61e-03 [3.34e-03, 3.99e-03] | 1.74e-02 [1.49e-02, 1.99e-02] | 9.66e-02 [6.29e-02, 1.39e-01] | 1.01e+00 [4.01e-01, 1.70e+00] |
| 1000 | grey-box | 2.45e-03 [2.24e-03, 2.61e-03] | 6.09e-03 [4.86e-03, 7.58e-03] | 1.28e-02 [9.38e-03, 1.62e-02] | 1.30e-01 [9.35e-02, 1.68e-01] |
| 1000 | grey-box-qs | 3.34e-03 [2.89e-03, 3.78e-03] | 9.86e-03 [7.81e-03, 1.22e-02] | 1.90e-02 [1.51e-02, 2.30e-02] | 1.56e-01 [1.09e-01, 2.44e-01] |

## Comparisons (error of the second arm divided by the error of grey-box-qs, above 1: grey-box-qs is better; seeds won; paired t-test on log errors)

| N | comparison | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|
| 100 | grey-box-qs against PINC | 1.17 (5/5, p = 0.028) | 1.64 (4/5, p = 0.062) | 7.45 (4/5, p = 0.066) | 9.89 (5/5, p = 0.031) |
| 100 | grey-box-qs against data-only | 3.72 (5/5, p = 3.2e-05) | 10.17 (5/5, p = 0.00095) | 31.10 (5/5, p = 0.0017) | 14.19 (5/5, p = 0.0037) |
| 100 | grey-box-qs against grey-box | 1.03 (3/5, p = 0.64) | 0.91 (3/5, p = 0.86) | 0.77 (1/5, p = 0.24) | 0.94 (3/5, p = 0.49) |
| 1000 | grey-box-qs against PINC | 1.45 (5/5, p = 0.011) | 1.24 (3/5, p = 0.23) | 3.05 (5/5, p = 0.028) | 4.23 (5/5, p = 0.008) |
| 1000 | grey-box-qs against data-only | 1.08 (3/5, p = 0.51) | 1.76 (5/5, p = 0.0057) | 5.07 (5/5, p = 0.0021) | 6.49 (4/5, p = 0.066) |
| 1000 | grey-box-qs against grey-box | 0.73 (0/5, p = 0.026) | 0.62 (0/5, p = 0.011) | 0.67 (0/5, p = 0.1) | 0.84 (2/5, p = 0.51) |
