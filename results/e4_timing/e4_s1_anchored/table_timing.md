# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| PINC-MPC | 5 | 5.7 | 5.7 | 7.0 | 8.8 | 0.67 | 0.072 |
| PINC-MPC | 10 | 9.7 | 9.8 | 12.0 | 10.5 | 0.96 | 0.035 |
| PINC-MPC | 20 | 19.7 | 18.9 | 24.3 | 10.8 | 1.85 | 0.031 |
| PINC-MPC | 40 | 38.1 | 37.5 | 42.3 | 12.9 | 2.97 | 0.031 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| PINC-MPC | 3.71 | 0.139 |
