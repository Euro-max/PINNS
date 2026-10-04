# Results summary

**Current results are the v2 runs** (increment-scaled loss, `model.increment_scaling: true`, lambda = 0.01): run ids
`e1_v2` ... `e6_v2`, models `pinc_v2_s0` / `blackbox_v2_s0`, plus `e8_vx_residual/e8b` (lambda sweep),
`e9_physics_learning/e9` and `e10_confirm/e10` (five-seed confirmation). The paper (`LATEX/main.tex`) is generated from these by
`scripts/make_paper_tables.py`. Headline v2 findings:
- PINC beats the data-only network at every training-set size (14x at 100 trajectories, ~1.15x at 1e4-1e5), 13x lower time-derivative
  error and 2.2x lower 50-step error over five seeds; PINC-MPC matches NMPC-RK4 in closed loop while the data-only controller has a
  1.84x higher median speed error on the sinusoid; solve time unchanged (1.47x faster only at N = 40).
- Caveats: one-step extrapolation slightly favours data-only at 1e4-1e5; under a -1 kN force step the data-only controller tracks speed
  better (model bias offsets the disturbance); the 8x64 architecture predates the scaling (6x256 is 2x better on validation).

The section below describes the **superseded v1 results** (unscaled residual) and is kept for the record.


Every number below is copied from a file under `results/` named in the same line; see
`results/MANIFEST.md` for the command and git hash behind each artifact. Models:
`results/models/pinc_default_s0` (8x64, lambda = 0.01) and `results/models/blackbox_default_s0`
(same net, lambda = 0). Reference arms: NMPC-RK4 (exact model, nominal parameters) and LTV-MPC.

## E7 architecture study -- `results/e7_architecture/e7/table_trials.md`, `docs/ARCHITECTURE_TRIALS.md`
56 trials. Plain deep MLPs win; dropout, layer norm, skip and ResNet-block residuals all hurt.
Chosen default 8x64 (29 892 parameters). lambda = 0 is best on validation for every architecture.

## E6 ablations -- `results/e6_ablations/e6/table_ablations.md`
- lambda sweep {0, 0.01, 0.1, 1, 10, 100}: validation data loss 1.0e-6, 5.1e-5, 2.7e-4, 5.8e-4, 7.7e-4, 7.6e-4.
- Hard IC matters: soft IC doubles the validation data loss and multiplies the vx test error by ~18.
- Residual ablation at lambda = 0.01: removing only the **vx** residual gives 1.3e-6 (almost the
  lambda = 0 value) while removing vy, r or psi changes little. The vx residual is the term that
  conflicts with the data fit (the vx increment per period is tiny relative to S_x = 30, so its
  time derivative is poorly resolved). This was found after E1-E5 were run and was **not** fed back
  into the defaults (ground rule 9); it is the first thing to try next.
- More collocation points help slightly (100k: 4.7e-5 vs 20k: 5.1e-5 vs 2k: 5.9e-5).

## E1 surrogate accuracy -- `results/e1_open_loop/e1/table_*.md`, `fig_error_vs_horizon`
200 ICs x 20 control sequences, NRMSE = RMSE / S_x, in-box:
- One step (t = T): PINC 6.9e-3, black-box 1.2e-3, LTI 7.3e-2, RK4(dt=0.01) 6.9e-8. Errors at
  0.25 T ... T grow smoothly, so the continuous-time property holds for both networks.
- Chained 50 steps: PINC 4.0e-2, black-box 2.4e-2, LTI 1.2e-1; no rollout diverged.
- Extrapolation region (vx 25-30 m/s): see `table_*_extrap.md` (both networks degrade ~3x).

## E3 closed loop -- `results/e3_closed_loop/e3/table_<ref>.md`, `fig_timeseries_<ref>`
30 seeds, all 480 solves succeeded. RMSE (mean over seeds) NMPC-RK4 / PINC-MPC / black-box / LTV:
- speed sinusoid, vx [m/s]: 0.058 / 0.061 / 0.061 / 0.058
- speed step, vx [m/s]: 0.449 / 0.456 / 0.457 / 0.696 (step response is actuator-limited: Fx <= 3000 N)
- lane change, Y [m]: 0.0334 / 0.0325 / 0.0330 / 0.0334
- double lane change (12 m/s), Y [m]: 0.143 / 0.142 / 0.143 / 0.143
PINC-MPC is statistically distinguishable from NMPC-RK4 (paired Wilcoxon p < 0.01 on most metrics)
but the differences are a few percent, in both directions. LTV-MPC fails only on the speed step.

## E4 timing -- `results/e4_timing/e4/table_timing.md`, `fig_solve_time_vs_N`
CPU single thread, XLA-compiled cost+gradient for every arm, median solve time per step [ms]:
N = 5: RK4 5.7, PINC 6.9, black-box 7.0, LTV 15.6; N = 10: 10.6 / 9.7 / 9.6 / 19.4;
N = 20: 20.4 / 16.0 / 14.4 / 24.1; N = 40: 48.4 / 34.4 / 30.3 / 40.8.
PINC-MPC is **not** faster than NMPC-RK4 at N <= 10; the crossover is at N ~ 20 and the gain is
1.4x at N = 40. One-step model call: RK4 0.084 ms, network 0.14 ms (XLA), 43 ms vs 3.3 ms eager.
No GPU was available.

## E5 robustness -- `results/e5_robustness/e5/table_robustness.md`, `fig_degradation`
Controller nominal, plant perturbed, single lane change, 30 seeds. All arms degrade together
(RMSE_Y 0.033 nominal -> 0.062 at Ca-30 %, 0.075 at Fiala mu = 0.4); PINC-MPC stays within ~4 %
of NMPC-RK4 in every setting and all solves succeed.

## E2 data efficiency -- `results/e2_data_efficiency/e2/table_data_efficiency.md`, `fig_nrmse_vs_n`
Same optimiser budget for every run (6000 Adam steps + 500 L-BFGS), 5 seeds per size (data draw and
initialisation vary; validation / test fixed). Test NRMSE (all states, in-box), PINC (lambda = 0.01) / black-box:
- N = 100:     0.0152 / 0.0248   (physics helps; CIs do not overlap)
- N = 1 000:   0.0085 / 0.0017
- N = 10 000:  0.0069 / 0.0011
- N = 100 000: 0.0071 / 0.0011
PINC plateaus at ~0.007 (the vx-residual floor from E6); the black-box net keeps improving until ~10^4.

## Honest framing for the paper
NMPC-RK4 is the accuracy ceiling. PINC-MPC matches it in closed loop, but so does the black-box
network trained on the same 20 000 trajectories, and the black-box surrogate is more accurate in
open loop. With the current residual normalisation the physics term does not buy accuracy or
speed at N <= 10. E2 shows the physics term helps only at ~10^2 trajectories. The one open lever is the vx residual
rescaling suggested by E6.
