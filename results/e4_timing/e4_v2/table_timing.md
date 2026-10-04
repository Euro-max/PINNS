# E4 solve time per MPC step vs horizon (CPU single thread; GPU: not available)

| arm | N | mean [ms] | median [ms] | p95 [ms] | iterations | ms / iteration | RMSE Y [m] |
|---|---|---|---|---|---|---|---|
| NMPC-RK4 | 5 | 5.7 | 5.7 | 7.1 | 8.7 | 0.67 | 0.072 |
| NMPC-RK4 | 10 | 10.0 | 10.6 | 12.1 | 10.5 | 0.98 | 0.035 |
| NMPC-RK4 | 20 | 22.2 | 20.3 | 34.5 | 10.6 | 2.10 | 0.031 |
| NMPC-RK4 | 40 | 48.9 | 47.5 | 55.5 | 12.8 | 3.84 | 0.031 |
| PINC-MPC | 5 | 6.4 | 6.3 | 7.6 | 8.7 | 0.76 | 0.070 |
| PINC-MPC | 10 | 9.7 | 10.0 | 11.8 | 11.2 | 0.89 | 0.034 |
| PINC-MPC | 20 | 14.9 | 14.4 | 18.3 | 10.7 | 1.44 | 0.030 |
| PINC-MPC | 40 | 32.5 | 32.4 | 34.3 | 13.8 | 2.36 | 0.030 |
| Black-box-MPC | 5 | 6.2 | 6.1 | 7.6 | 9.0 | 0.70 | 0.072 |
| Black-box-MPC | 10 | 9.5 | 9.6 | 10.5 | 11.3 | 0.86 | 0.034 |
| Black-box-MPC | 20 | 16.0 | 16.0 | 17.5 | 12.8 | 1.26 | 0.030 |
| Black-box-MPC | 40 | 29.8 | 29.6 | 30.7 | 15.5 | 1.94 | 0.030 |
| LTV-MPC | 5 | 15.9 | 15.7 | 21.0 | 8.7 | 1.93 | 0.071 |
| LTV-MPC | 10 | 20.2 | 20.1 | 23.8 | 10.5 | 2.03 | 0.035 |
| LTV-MPC | 20 | 26.0 | 24.6 | 32.7 | 10.6 | 2.53 | 0.031 |
| LTV-MPC | 40 | 42.4 | 41.3 | 46.5 | 12.8 | 3.31 | 0.031 |

| arm | one-step model call, eager [ms] | one-step model call, XLA [ms] |
|---|---|---|
| NMPC-RK4 | 42.21 | 0.085 |
| PINC-MPC | 3.34 | 0.144 |
| Black-box-MPC | 3.50 | 0.139 |
| LTV-MPC | nan | nan |
