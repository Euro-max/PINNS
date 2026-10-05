# E14 physics weight and wheel residuals, HF-M0 (N = 20000, seed 0; selection on validation data loss: best = lam0_all)

| run | val data | val phys | test body | test actuators | test wheels | extrap body | extrap actuators | extrap wheels | best ckpt | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| lam0_all | 1.156e-05 | 4.581e-01 | 1.72e-03 | 2.10e-03 | 6.01e-03 | 2.63e-03 | 3.87e-03 | 1.49e-02 | lbfgs@2300 | 888 |
| lam0.001_all | 3.052e-05 | 3.731e-01 | 3.72e-03 | 3.36e-03 | 7.60e-03 | 4.17e-03 | 4.56e-03 | 1.45e-02 | lbfgs@2299 | 808 |
| lam0.001_nowheel | 1.959e-05 | 7.449e-03 | 3.72e-03 | 3.09e-03 | 5.97e-03 | 4.10e-03 | 4.32e-03 | 1.45e-02 | lbfgs@2300 | 446 |
| lam0.01_all | 8.101e-04 | 2.291e-01 | 5.80e-03 | 8.51e-03 | 4.79e-02 | 7.31e-03 | 1.12e-02 | 3.38e-02 | lbfgs@1033 | 1052 |
| lam0.01_nowheel | 3.253e-05 | 1.081e-03 | 5.79e-03 | 4.17e-03 | 7.41e-03 | 6.67e-03 | 6.05e-03 | 1.65e-02 | lbfgs@2300 | 502 |
| lam0.1_all | 6.795e-03 | 8.055e-02 | 7.25e-03 | 1.37e-02 | 1.32e-01 | 8.77e-03 | 1.84e-02 | 6.30e-02 | lbfgs@1124 | 466 |
| lam0.1_nowheel | 6.948e-05 | 3.856e-04 | 7.16e-03 | 3.17e-03 | 1.24e-02 | 8.45e-03 | 4.97e-03 | 2.38e-02 | lbfgs@2300 | 688 |
