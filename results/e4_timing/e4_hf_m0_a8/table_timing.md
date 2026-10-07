# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| PINC-MPC | 5 | 8.0 | 8.0 | 9.3 | 10.1 | 0.81 | 0.110 |
| PINC-MPC | 10 | 11.8 | 11.5 | 14.1 | 13.9 | 0.86 | 0.052 |
| PINC-MPC | 20 | 18.9 | 18.8 | 20.2 | 14.8 | 1.29 | 0.048 |
| PINC-MPC | 40 | 36.5 | 36.3 | 38.8 | 16.6 | 2.20 | 0.048 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| PINC-MPC | 6.97 | 0.141 |
