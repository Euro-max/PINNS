# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| Grey-box-MPC | 5 | 134.5 | 137.8 | 155.1 | 9.8 | 14.04 | 0.116 |
| Grey-box-MPC | 10 | 368.9 | 370.4 | 401.6 | 13.7 | 27.22 | 0.041 |
| Grey-box-MPC | 20 | 1369.4 | 1351.1 | 1487.2 | 14.1 | 97.85 | 0.037 |
| Grey-box-MPC | 40 | 3244.3 | 3258.8 | 3372.6 | 15.7 | 207.57 | 0.037 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| Grey-box-MPC | 1131.93 | 0.282 |
