# E23 physics weight for the prior-anchored network, HF-M0 (mean [95% CI] over 3 seeds)

selected on the validation 50-step body error: N = 100: A8 lambda 0.001, N = 1000: A8 lambda 0.0001

| N | arm | val 50-step body | one-step body | 10-step body | 50-step body |
|---|---|---|---|---|---|
| 100 | A8 lambda 0.0001 | 2.50e-01 [1.90e-01, 3.21e-01] | 9.06e-03 [7.29e-03, 1.13e-02] | 8.86e-02 [7.88e-02, 9.88e-02] | 2.59e-01 [1.83e-01, 3.45e-01] |
| 100 | A8 lambda 0.001 | 2.16e-01 [5.94e-02, 3.60e-01] | 7.17e-03 [6.81e-03, 7.46e-03] | 5.69e-02 [3.16e-02, 9.93e-02] | 2.46e-01 [6.36e-02, 4.06e-01] |
| 100 | A8 lambda 0.01 | 2.66e-01 [3.72e-02, 5.14e-01] | 6.90e-03 [6.63e-03, 7.18e-03] | 3.78e-02 [2.11e-02, 4.94e-02] | 2.35e-01 [3.37e-02, 4.09e-01] |
| 1000 | A8 lambda 0.0001 | 1.65e-02 [1.13e-02, 2.03e-02] | 3.85e-03 [3.66e-03, 4.15e-03] | 9.73e-03 [9.02e-03, 1.08e-02] | 1.57e-02 [1.07e-02, 2.22e-02] |
| 1000 | A8 lambda 0.001 | 1.84e-02 [1.29e-02, 2.77e-02] | 4.85e-03 [4.77e-03, 4.90e-03] | 9.88e-03 [9.03e-03, 1.04e-02] | 1.91e-02 [1.28e-02, 2.83e-02] |
| 1000 | A8 lambda 0.01 | 2.09e-02 [2.03e-02, 2.13e-02] | 6.15e-03 [6.06e-03, 6.23e-03] | 9.09e-03 [7.65e-03, 1.04e-02] | 2.11e-02 [1.73e-02, 2.36e-02] |

## Comparisons (error of the second arm divided by the error of the first, above 1: the first is better; seeds won; paired t-test on log errors)

| N | comparison | val 50-step body | one-step body | 10-step body | 50-step body |
|---|---|---|---|---|---|
