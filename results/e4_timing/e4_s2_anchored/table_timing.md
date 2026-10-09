# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| PINC-MPC | 5 | 8.1 | 8.1 | 9.7 | 9.9 | 0.84 | 0.120 |
| PINC-MPC | 10 | 12.1 | 11.7 | 14.5 | 13.4 | 0.91 | 0.042 |
| PINC-MPC | 20 | 25.9 | 24.9 | 32.0 | 14.6 | 1.79 | 0.037 |
| PINC-MPC | 40 | 53.0 | 52.2 | 62.3 | 17.5 | 3.04 | 0.039 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| PINC-MPC | 7.03 | 0.144 |
