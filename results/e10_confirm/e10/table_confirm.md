# E10 multi-seed confirmation: PINC (increment scaling, lambda = 0.01) vs data-only (lambda = 0), 5 seeds

## N = 20000 training trajectories

| metric (lower is better) | PINC | data-only | data-only / PINC (geo. mean) | seeds PINC better | paired t (log) p | Wilcoxon p |
|---|---|---|---|---|---|---|
| validation data loss | 6.18e-07 [5.99e-07, 6.37e-07] | 8.74e-07 [8.31e-07, 9.15e-07] | 1.41 | 5/5 | 0.0016 | 0.062 |
| test NRMSE (all states) | 7.89e-04 [7.72e-04, 8.03e-04] | 9.27e-04 [9.04e-04, 9.50e-04] | 1.17 | 5/5 | 0.0027 | 0.062 |
| dx/dt error vs truth | 1.05e-02 [1.03e-02, 1.06e-02] | 1.38e-01 [1.28e-01, 1.48e-01] | 13.11 | 5/5 | 9.1e-07 | 0.062 |
| physics residual | 1.05e-02 [1.03e-02, 1.07e-02] | 1.38e-01 [1.28e-01, 1.48e-01] | 13.11 | 5/5 | 8.7e-07 | 0.062 |
| 10-step error, in-domain | 3.44e-03 [3.16e-03, 3.72e-03] | 6.04e-03 [4.96e-03, 7.22e-03] | 1.72 | 5/5 | 0.0072 | 0.062 |
| 50-step error, in-domain | 1.11e-02 [9.47e-03, 1.26e-02] | 2.45e-02 [1.88e-02, 3.12e-02] | 2.15 | 5/5 | 0.0054 | 0.062 |
| 50-step error, extrapolation | 3.70e-02 [3.25e-02, 4.15e-02] | 9.49e-02 [8.42e-02, 1.09e-01] | 2.56 | 5/5 | 0.0016 | 0.062 |

## N = 100 training trajectories

| metric (lower is better) | PINC | data-only | data-only / PINC (geo. mean) | seeds PINC better | paired t (log) p | Wilcoxon p |
|---|---|---|---|---|---|---|
| validation data loss | 4.21e-06 [4.00e-06, 4.41e-06] | 8.40e-04 [7.55e-04, 9.13e-04] | 198.75 | 5/5 | 2.1e-07 | 0.062 |
| test NRMSE (all states) | 2.02e-03 [1.94e-03, 2.10e-03] | 2.89e-02 [2.75e-02, 3.01e-02] | 14.29 | 5/5 | 3.6e-07 | 0.062 |
| dx/dt error vs truth | 1.23e-02 [1.19e-02, 1.26e-02] | 2.25e-01 [2.10e-01, 2.43e-01] | 18.23 | 5/5 | 5.1e-07 | 0.062 |
| physics residual | 1.24e-02 [1.20e-02, 1.28e-02] | 2.29e-01 [2.15e-01, 2.46e-01] | 18.48 | 5/5 | 5.7e-07 | 0.062 |
| 10-step error, in-domain | 5.25e-03 [4.35e-03, 6.41e-03] | 9.57e-02 [7.44e-02, 1.19e-01] | 18.07 | 5/5 | 1.2e-06 | 0.062 |
| 50-step error, in-domain | 1.35e-02 [1.06e-02, 1.72e-02] | 2.34e-01 [1.66e-01, 3.03e-01] | 16.90 | 5/5 | 4.5e-06 | 0.062 |
| 50-step error, extrapolation | 6.71e-02 [4.44e-02, 9.87e-02] | 4.88e-01 [2.95e-01, 6.94e-01] | 7.19 | 5/5 | 0.00063 | 0.062 |

