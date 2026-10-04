# Legacy defects D1–D20: where each is closed in the rebuild

Every row names the code that removes the defect and the test (or review note) that shows it is gone.
The legacy files are kept untouched in `legacy/` and nothing imports them (`tests/test_sim.py::test_no_module_writes_reference_into_state` greps for `legacy` imports).

| ID | Defect | Where it is fixed | Evidence |
|---|---|---|---|
| D1 | `tape.gradient` of a non-scalar output gave the gradient of the *sum* of outputs | `pinc/loss.py::_forward_with_jvp` (forward-mode JVP along the time input gives every output's own derivative) and `_forward_with_reverse` (one gradient per output channel) | `tests/test_loss.py::test_time_derivative_matches_closed_form` (both methods vs closed form), `test_forward_and_reverse_agree_on_network` |
| D2 | Derivative read from state-input columns instead of the time column | `pinc/loss.py`: tangent / gradient column 0 only | same test (the toy model also depends on the state inputs, so a wrong column would fail) |
| D3 | Time fed as `t/T` without the `1/T` chain-rule factor | `pinc/loss.py::forward_and_time_derivative` multiplies by `S_x / T` | same test (T = 0.1 makes a missing factor a 10x error) |
| D4 | Training target identical to the input state | `pinc/data.py::sample_trajectories`: targets from `plant.rk4_step` | `tests/test_data.py::test_targets_reintegrate_exactly`, `test_target_differs_from_input_state` |
| D5 | Yaw moment `lf Fyf + lr Fyr` with `alpha_r = +lr r / v`: zero yaw damping for the symmetric car | `pinc/plant.py::f` (`lf Fyf cos(delta) - lr Fyr`, `alpha_r = -atan2(vy - lr r, vx)`) | `tests/test_plant.py::test_free_yaw_decays`, `test_asymmetric_wheelbase_stable`, `test_lateral_eigenvalues_stable` |
| D6 | No lateral-velocity state; `dv/dt` carried as a frozen state | 6-state model `[vx, vy, r, psi, X, Y]` in `pinc/plant.py`; network state `[vx, vy, r, psi]` | `tests/test_plant.py::test_lateral_velocity_is_a_state` |
| D7 | Closed loop never called the plant; state blended with / set to the reference | `pinc/sim.py`: the plant state is assigned only by `plant.simulate` | `tests/test_sim.py::test_garbage_predictor_gives_large_error`, `test_zero_controller_gives_large_error`, `test_no_module_writes_reference_into_state`, `test_sim_state_only_changes_via_plant` |
| D8 | Perturbed parameters passed to controller and plant alike | `pinc/mpc.py::RK4Predictor` / `LinearPredictor` take `cfg.params` (nominal); perturbed dicts go only to `sim.simulate(plant_params=...)` in E5 | `tests/test_mpc.py::test_controller_never_sees_plant_params`, `tests/test_sim.py::test_plant_params_only_reach_plant` |
| D9 | Steering scale 0.3 in training vs 0.5 at inference | one `S_u` in `pinc/config.py`, used by `data.scale_inputs` and `PINCNet.scale_inputs` / `PINCPredictor` | `tests/test_model.py::test_scale_inputs_matches_data_module` |
| D10 | Time input at inference always `dt/dt = 1` with no continuous-time check | Inference uses `t = T` by design; the continuous-time property is tested separately in E1 at `t in {0.25, 0.5, 0.75, 1} T` (`experiments/e1_open_loop.py`) | E1 one-step tables |
| D11 | Actuator lag depended on the array index, not time | Removed: v1 uses a zero-order hold (`actuator_lag: false` in config); if enabled later it must add actuator states to plant and PINC alike | review note; `pinc/config.py` flag |
| D12 | Best weights tracked but never restored | `pinc/train.py`: `best["weights"]` restored before evaluation and saving; selection on the validation set | `train()` asserts `best["weights"] is not None`; `summary.json` records `best_epoch`/`best_stage` |
| D13 | Only 10 collocation batches ever used, never resampled | `pinc/train.py`: `sample_collocation(..., seed = colloc_seed + epoch)` every epoch | review note (seed changes with epoch) |
| D14 | Output `clip_by_value` zeroed gradients for saturated samples | `pinc/model.py::PINCNet.call` has no clipping | `tests/test_model.py::test_output_not_clipped_and_gradients_nonzero`, `tests/test_loss.py::test_loss_gradients_finite_and_nonzero_across_box` |
| D15 | Script crashed after training (`hist`, `final_loss` undefined) | `pinc/train.py` writes `loss_curve.csv`, `summary.json`, model files | `scripts/run_all.sh` runs it end to end; `tests/test_model.py::test_save_load_roundtrip` |
| D16 | SLSQP finite differences through the network | `pinc/mpc.py::MPC`: cost and gradient from one TF call, `minimize(..., jac=True)` | `tests/test_mpc.py::test_analytic_gradient_matches_finite_differences` (all predictors) |
| D17 | Adaptive `solve_ivp` inside the cost: non-smooth objective | `RK4Predictor` uses a fixed-substep TF RK4 (`mpc.dt_pred`) | same gradient test passes for the RK4 predictor |
| D18 | IAE without `x T`; error scored on the pre-update state | `pinc/metrics.py::iae`; `pinc/sim.py` scores `x[k+1] - ref(t[k+1])` | `tests/test_sim.py::test_error_scored_after_step` |
| D19 | Force bounded to `(0, 3000)` N: no braking | `u_min = [-6000, -0.3]` in config; SLSQP bounds from config | `tests/test_mpc.py::test_bounds_allow_braking` |
| D20 | Colab path, no requirements, GPU determinism off | no absolute paths (`tests/test_sim.py` greps `/content/`); `requirements.txt`; `pinc/tfsetup.py` enables op determinism, seeds Python/NumPy/TF and disables oneDNN | grep test; `meta.json` next to every result records seed, git hash and versions |
