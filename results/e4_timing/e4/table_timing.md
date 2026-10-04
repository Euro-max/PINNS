# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| NMPC-RK4 | 5 | 5.6 | 5.7 | 7.0 | 8.7 | 0.66 | 0.072 |
| NMPC-RK4 | 10 | 9.9 | 10.6 | 11.8 | 10.5 | 0.98 | 0.035 |
| NMPC-RK4 | 20 | 21.1 | 20.4 | 25.9 | 10.6 | 2.03 | 0.031 |
| NMPC-RK4 | 40 | 48.6 | 48.4 | 51.7 | 12.8 | 3.81 | 0.031 |
| PINC-MPC | 5 | 6.8 | 6.9 | 7.9 | 9.3 | 0.74 | 0.074 |
| PINC-MPC | 10 | 9.6 | 9.7 | 11.5 | 10.2 | 0.96 | 0.032 |
| PINC-MPC | 20 | 16.6 | 16.0 | 18.8 | 11.4 | 1.47 | 0.028 |
| PINC-MPC | 40 | 34.6 | 34.4 | 36.4 | 15.2 | 2.29 | 0.028 |
| Black-box-MPC | 5 | 6.9 | 7.0 | 8.3 | 9.2 | 0.78 | 0.074 |
| Black-box-MPC | 10 | 9.1 | 9.6 | 11.3 | 10.0 | 0.94 | 0.033 |
| Black-box-MPC | 20 | 14.5 | 14.4 | 17.2 | 11.3 | 1.32 | 0.029 |
| Black-box-MPC | 40 | 30.3 | 30.3 | 31.7 | 13.8 | 2.20 | 0.029 |
| LTV-MPC | 5 | 15.6 | 15.6 | 17.1 | 8.7 | 1.91 | 0.071 |
| LTV-MPC | 10 | 19.4 | 19.4 | 22.1 | 10.5 | 1.94 | 0.035 |
| LTV-MPC | 20 | 25.6 | 24.1 | 29.7 | 10.6 | 2.48 | 0.031 |
| LTV-MPC | 40 | 41.4 | 40.8 | 45.5 | 12.8 | 3.24 | 0.031 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| NMPC-RK4 | 43.15 | 0.084 |
| PINC-MPC | 3.31 | 0.143 |
| Black-box-MPC | 3.33 | 0.145 |
| LTV-MPC | nan | nan |
