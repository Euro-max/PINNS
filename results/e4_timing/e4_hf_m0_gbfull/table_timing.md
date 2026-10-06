# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| Grey-box-MPC | 5 | 130.7 | 133.9 | 151.9 | 9.8 | 13.58 | 0.114 |
| Grey-box-MPC | 10 | 379.7 | 383.8 | 403.8 | 13.4 | 28.55 | 0.038 |
| Grey-box-MPC | 20 | 963.9 | 957.7 | 1019.7 | 14.9 | 65.60 | 0.035 |
| Grey-box-MPC | 40 | 2606.9 | 2652.8 | 2725.5 | 16.1 | 162.53 | 0.035 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| Grey-box-MPC | 1108.80 | 0.281 |
