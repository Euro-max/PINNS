# E11 training-budget convergence (seed 0, N = 20 000, increment-scaled loss; dotted lines in the figure mark the end of Adam)

| run | params | val data, Adam end | after 500 L-BFGS | after 1000 L-BFGS | after 2000 L-BFGS | after 5000 L-BFGS | val data (selected) | L-BFGS iters run | improvement in last 10 % | test NRMSE all | extrap NRMSE all | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_8x64 | 29892 | 3.75e-06 | 6.01e-07 | 2.90e-07 | 1.17e-07 | 3.10e-08 | 3.10e-08 | 5000 | 12.9 % | 1.79e-04 | 8.53e-04 | 7834 |
| long_8x64_lam0 | 29892 | 5.70e-06 | 9.16e-07 | 4.16e-07 | 1.57e-07 | 4.34e-08 | 4.34e-08 | 5000 | 14.1 % | 2.10e-04 | 1.14e-03 | 7831 |
| long_6x256 | 332036 | 1.62e-06 | 2.94e-07 | 1.25e-07 | 4.38e-08 | 1.04e-08 | 1.04e-08 | 5000 | 18.2 % | 1.06e-04 | 9.38e-04 | 11038 |
| adam1500_8x64 | 29892 | 2.86e-07 | 9.88e-08 | - | - | - | 9.88e-08 | 500 | 30.7 % | 3.13e-04 | 1.55e-03 | 2329 |
| const_8x64 | 29892 | 4.64e-06 | 4.93e-07 | 2.03e-07 | 8.11e-08 | 2.37e-08 | 2.37e-08 | 5000 | 14.2 % | 1.60e-04 | 8.13e-04 | 3749 |
