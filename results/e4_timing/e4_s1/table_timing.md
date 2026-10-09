# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| PINC-MPC | 5 | 5.4 | 5.4 | 6.5 | 8.8 | 0.64 | 0.071 |
| PINC-MPC | 10 | 8.0 | 8.4 | 9.5 | 10.5 | 0.79 | 0.034 |
| PINC-MPC | 20 | 18.6 | 18.3 | 22.7 | 10.6 | 1.83 | 0.030 |
| PINC-MPC | 40 | 36.4 | 35.2 | 45.2 | 13.1 | 2.78 | 0.030 |
| Black-box-MPC | 5 | 5.5 | 5.6 | 6.6 | 8.7 | 0.66 | 0.070 |
| Black-box-MPC | 10 | 8.3 | 8.6 | 10.2 | 10.5 | 0.81 | 0.034 |
| Black-box-MPC | 20 | 16.6 | 16.2 | 20.3 | 10.8 | 1.58 | 0.030 |
| Black-box-MPC | 40 | 34.2 | 33.7 | 38.5 | 13.3 | 2.57 | 0.030 |
| NMPC-RK4 | 5 | 5.8 | 6.0 | 7.2 | 8.7 | 0.69 | 0.072 |
| NMPC-RK4 | 10 | 10.2 | 10.6 | 12.5 | 10.5 | 1.00 | 0.035 |
| NMPC-RK4 | 20 | 29.4 | 27.7 | 37.7 | 10.6 | 2.82 | 0.031 |
| NMPC-RK4 | 40 | 66.0 | 64.4 | 75.0 | 12.8 | 5.17 | 0.031 |
| LTV-MPC | 5 | 14.4 | 14.5 | 15.7 | 8.7 | 1.76 | 0.071 |
| LTV-MPC | 10 | 20.9 | 21.1 | 24.0 | 10.5 | 2.10 | 0.035 |
| LTV-MPC | 20 | 31.9 | 30.7 | 41.1 | 10.6 | 3.11 | 0.031 |
| LTV-MPC | 40 | 53.4 | 51.7 | 65.5 | 12.8 | 4.18 | 0.031 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| PINC-MPC | 2.57 | 0.127 |
| Black-box-MPC | 2.59 | 0.128 |
| NMPC-RK4 | 34.85 | 0.082 |
| LTV-MPC | nan | nan |
