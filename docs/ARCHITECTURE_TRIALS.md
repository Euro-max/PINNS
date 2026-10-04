# Architecture trials (E7) and the choice of the default network

Generated from `results/e7_architecture/e7/summary.json` (script `experiments/e7_architecture.py`, seed 0).
All 56 trials use the same data (20 000 training trajectories), seeds and optimiser budget (300 Adam epochs with cosine decay, then 500 L-BFGS iterations), hard initial condition, tanh, Glorot init.
Selection criterion: validation **data** loss (accuracy on held-out trajectories). Validation physics loss and test / extrapolation NRMSE are reported for every trial in `results/e7_architecture/e7/table_trials.md`.

## What was tried

- Grid A: depth {2, 4, 6, 8} x width {64, 128, 256} x lambda {0, 0.01, 0.1} (36 plain fully-connected nets).
- Grid B: at 4x128 and 6x256, lambda {0, 0.01}: dropout 0.05, dropout 0.1, residual `skip` (h += layer(h)), residual `block` (ResNet block h += W2 tanh(W1 h)), LayerNormalization after every hidden layer (20 trials).

## Findings

- Best at lambda = 0: **8x256, plain** (val data 4.97e-07, test NRMSE 7.17e-04, extrapolation 5.64e-03, 463620 params).
- Best at lambda = 0.01: **8x64, plain** (val data 5.09e-05, test NRMSE 7.09e-03, extrapolation 2.12e-02, 29892 params).
- Best at lambda = 0.1: **8x64, plain** (val data 2.66e-04, test NRMSE 1.64e-02, extrapolation 3.14e-02, 29892 params).

| lambda | 2x64 | 4x64 | 6x64 | 8x64 | 4x128 | 8x128 | 8x256 |
|---|---|---|---|---|---|---|---|
| 0 | 4.7e-06 | 2.4e-06 | 1.2e-06 | 1.0e-06 | 2.1e-06 | 5.5e-07 | 5.0e-07 |
| 0.01 | 2.6e-04 | 1.5e-04 | 1.1e-04 | 5.1e-05 | 1.8e-04 | 5.2e-05 | 1.9e-04 |
| 0.1 | 1.0e-03 | 5.2e-04 | 3.8e-04 | 2.7e-04 | 4.8e-04 | 3.3e-04 | 1.8e-03 |

Validation data loss (scaled MSE) for the plain nets: depth matters far more than width; 8 layers of 64 units beat 4 layers of 256.

| variant (lambda 0 / 0.01) | 4x128 | 6x256 |
|---|---|---|
| plain (reference) | 1.0x / 1.0x | 1.0x / 1.0x |
| dropout 0.05 | 2.0x / 2.6x | 2.9x / 3.8x |
| dropout 0.1 | 45.1x / 2.7x | 300.0x / 8.9x |
| residual skip | 2.1x / 1.6x | 4.6x / 2.4x |
| residual block (ResNet) | 1.5x / 1.5x | 5.2x / 7.4x |
| layer norm | 1.0x / 1.1x | 3.0x / 2.0x |

Validation data loss relative to the plain net of the same size (lambda = 0 / lambda = 0.01). Every regulariser or skip connection made the fit worse; dropout 0.1 by one to two orders of magnitude. Layer norm and the ResNet blocks also multiplied training time by 1.5-2.

## Decision

- Default network: **8 x 64, plain, no dropout, no layer norm, hard IC** (`configs/default.yaml`, `model.*`). It is the best trial at lambda = 0.01, fifth at lambda = 0 (within 2.1x of the 463k-parameter 8x256 net) and, with 29 892 parameters, the cheapest of the top group, which matters for the MPC solve time (E4).
- lambda: lambda = 0 is best on validation data loss for every architecture, i.e. at 20 000 trajectories the physics term does not improve held-out accuracy (it trades value accuracy for derivative accuracy). The default keeps **lambda = 0.01**, the best positive value, so that the PINC arm is physics-informed; the black-box arm is the same 8x64 net at lambda = 0. Whether physics helps at small data sizes is tested in E2, with 5 seeds per size.
- The same models are used, unchanged, by E1, E3, E4 and E5 (`results/models/pinc_default_s0`, `results/models/blackbox_default_s0`).
