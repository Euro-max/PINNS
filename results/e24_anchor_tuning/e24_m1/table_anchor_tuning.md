# E24 anchor settings of the prior-anchored network, HF-M1 (test errors unless noted; mean [95% CI] over 3 seeds)

| N | setting | val 50-step body | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|---|
| 100 | A8 (untuned anchor) | 1.82e-01 [1.27e-01, 2.83e-01] | 2.27e-02 [2.12e-02, 2.44e-02] | 6.31e-02 [4.75e-02, 7.90e-02] | 1.60e-01 [1.23e-01, 2.22e-01] | 8.75e-01 [1.82e-01, 1.93e+00] |
| 100 | T4 anchor gain | 3.05e-01 [2.06e-01, 4.81e-01] | 2.41e-02 [2.21e-02, 2.65e-02] | 1.01e-01 [6.83e-02, 1.58e-01] | 3.06e-01 [1.95e-01, 5.01e-01] | 9.08e-01 [3.65e-01, 1.20e+00] |
| 100 | T5 all | 1.50e-01 [7.93e-02, 2.06e-01] | 2.33e-02 [2.22e-02, 2.51e-02] | 5.99e-02 [4.13e-02, 7.66e-02] | 1.46e-01 [6.86e-02, 2.09e-01] | 7.01e-01 [3.22e-01, 9.32e-01] |
| 1000 | A8 (untuned anchor) | 5.01e-02 [3.61e-02, 5.73e-02] | 1.60e-02 [1.56e-02, 1.65e-02] | 2.79e-02 [2.37e-02, 3.43e-02] | 4.72e-02 [3.26e-02, 5.99e-02] | 3.18e-01 [2.19e-01, 4.06e-01] |
| 1000 | T4 anchor gain | 4.54e-02 [3.52e-02, 6.52e-02] | 1.62e-02 [1.59e-02, 1.65e-02] | 3.14e-02 [2.33e-02, 4.60e-02] | 4.33e-02 [3.13e-02, 6.59e-02] | 3.88e-01 [2.83e-01, 5.09e-01] |
| 1000 | T5 all | 5.35e-02 [4.17e-02, 6.64e-02] | 1.65e-02 [1.61e-02, 1.72e-02] | 3.24e-02 [2.96e-02, 3.71e-02] | 4.22e-02 [3.44e-02, 4.63e-02] | 8.71e-01 [2.90e-01, 1.80e+00] |

## Against A8 (error of A8 divided by the error of the setting, above 1: the setting is better; seeds won; paired t-test on log errors)

| N | setting | val 50-step body | one-step body | 10-step body | 50-step body | 50-step body, outside training range |
|---|---|---|---|---|---|---|
| 100 | T4 anchor gain | 0.60 (0/3, p = 0.0099) | 0.94 (0/3, p = 0.042) | 0.62 (0/3, p = 0.1) | 0.52 (0/3, p = 0.047) | 0.96 (1/3, p = 0.49) |
| 100 | T5 all | 1.21 (2/3, p = 0.49) | 0.97 (0/3, p = 0.12) | 1.05 (2/3, p = 0.78) | 1.10 (2/3, p = 0.62) | 1.25 (1/3, p = 0.83) |
| 1000 | T4 anchor gain | 1.10 (2/3, p = 0.77) | 0.99 (1/3, p = 0.39) | 0.89 (2/3, p = 0.5) | 1.09 (2/3, p = 0.8) | 0.82 (2/3, p = 0.6) |
| 1000 | T5 all | 0.94 (1/3, p = 0.76) | 0.97 (0/3, p = 0.033) | 0.86 (0/3, p = 0.066) | 1.12 (2/3, p = 0.72) | 0.36 (1/3, p = 0.33) |

## Learned values (wheel time constant per seed [ms]; anchor gain per state, mean over seeds, order vx vy r psi F delta sig_fl sig_fr sig_rl sig_rr)

| N | setting | tau_w [ms] | gain |
|---|---|---|---|
| 100 | T4 anchor gain | - | 0.99, 0.87, 0.74, 0.98, 0.81, 0.64, 0.56, 0.57, 0.49, 0.49 |
| 100 | T5 all | 14.1 / 16.8 / 21.2 | 0.96, 0.87, 0.68, 1.00, 0.89, 0.85, 0.87, 0.86, 0.77, 0.74 |
| 1000 | T4 anchor gain | - | 0.90, 0.84, 0.52, 0.92, 0.67, 0.68, 0.42, 0.44, 0.32, 0.33 |
| 1000 | T5 all | 25.4 / 28.0 / 24.6 | 0.73, 0.80, 0.49, 0.96, 0.80, 0.80, 0.93, 0.93, 0.86, 0.83 |
