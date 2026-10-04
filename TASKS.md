# PINC-MPC Rebuild Spec

**Scope:** the `Euro-max/PINNS` repository.
**Goal:** Replace the current implementation with a correct, reproducible pipeline whose outputs can support a paper. Every number in the paper must come from a script in this repo.

Keep this file at the repo root. Put `cav_plant.py` (supplied with this spec) in the repo root before starting. Work **one phase at a time**. Do not start a phase until the previous phase's acceptance criteria pass.

---

## 0. Context

The project trains a Physics-Informed Neural Net for Control (PINC, after Antonelo et al., arXiv 2104.02556) that maps `(t, x0, u) -> x(t)` for a vehicle model, then uses it as the prediction model inside MPC. The paper claims PINC-MPC is compared against a conventional MPC.

The existing code has defects that invalidate all results. You are **rebuilding**, not patching. The old files stay in `legacy/` for reference only; nothing new may import from them.

### Known defects in the legacy code (verify each is absent from the new code)

| ID | Defect | Legacy location |
|---|---|---|
| D1 | `tape.gradient(y_pred, inputs)` on a non-scalar target returns the gradient of the **sum** of outputs, not a Jacobian | `PINC.py`, `compute_physics_loss` |
| D2 | Derivative columns index state inputs, not time: inputs are `[t, v, dv_dt, psi, r, u, delta]`, but `dy_dt[:,3]` is used as dr/dt (that's ∂/∂ψ) and `dy_dt[:,2]` as dψ/dt (that's ∂/∂v̇) | same |
| D3 | Time is fed normalised (`t/T`) but no `1/T` chain-rule factor is applied | same |
| D4 | `y_true = y0_data` — training target is identical to the input state; no supervision at t > 0 | `generate_cav_training_data` |
| D5 | Yaw moment `lf*Fyf + lr*Fyr` with `alpha_r = +lr*r/v` gives **exactly zero** yaw damping for lf=lr, Caf=Car | all three files |
| D6 | No lateral velocity state; `dv_dt` is a state whose derivative is hard-coded `0.0` | `plant_dynamics` |
| D7 | Closed loop "MPC+PINC" never calls the plant: `plant_states[k+1] = pred`, then adds 95% of the reference error, then **sets the state to the reference** for t < 4 s | `closed_loop_simulation_pinc` |
| D8 | Parameter disturbance is passed to the controller AND the plant, so there is no model mismatch | both MPC files |
| D9 | Steering normalised by 0.3 in training, 0.5 at inference | `PINC.py` vs MPC files |
| D10 | Time input at inference is always `dt/dt = 1.0` | MPC files |
| D11 | Actuator lag depends on array index (`tf.linspace` over samples), not time | `generate_cav_training_data` |
| D12 | `best_weights` tracked but never restored; last-epoch weights saved | `train_cav_adam` |
| D13 | Only 10 of ~98 collocation batches ever used; never resampled | `train_cav_adam` |
| D14 | Output `clip_by_value` zeroes gradients for saturated samples | `CAV_PINC_Network.call` |
| D15 | `PINC.py` crashes after training (`hist`, `final_loss` undefined) | `__main__` |
| D16 | SLSQP finite-differences through the network; no analytic gradients | both MPC files |
| D17 | Baseline uses adaptive `solve_ivp` inside the cost → non-smooth objective | `mpc_controller_mpc_only` |
| D18 | IAE computed as raw sum (missing `× T`); error at step k scored on pre-update state | closed-loop functions |
| D19 | Control force bounded to `(0, 3000)` N — braking impossible | both MPC files |
| D20 | Hard-coded Colab path `/content/...`; no requirements; GPU determinism off | everywhere |

---

## 1. Ground rules (non-negotiable)

1. **No state overwriting.** Nothing may assign, blend, correct, clip toward, or force the plant state using the reference. The plant state changes only via `plant.step()`.
2. **One plant.** Every controller in every experiment is simulated against the same fixed-step RK4 plant from `pinc/plant.py`.
3. **Controllers get nominal parameters.** Perturbed parameters go only to the plant. The controller never sees them.
4. **No hand-typed numbers.** Every table and figure is written by a script to `results/`. The paper may only use numbers from `results/`.
5. **One source of truth for scaling.** All normalisation constants live in `pinc/config.py`, imported by training and inference alike.
6. **Seeds are logged.** Every run writes its config, seed, git hash, and package versions next to its outputs.
7. **Fail loudly.** No `tf.where(is_finite, x, 1e6)` masking. NaN/Inf raises.
8. **Don't tune on the test set.** Train / validation / test splits are generated with disjoint seeds.
9. If a result looks bad, report it. Do not add heuristics to make the PINC arm look better.

---

## 2. Target layout

```
pinc/
  __init__.py
  config.py          # params, bounds, scales, T, horizon, seeds
  plant.py           # from cav_plant.py: dynamics f(x,u,p), rk4_step, simulate
  plant_tf.py        # same dynamics + RK4 in TensorFlow (differentiable)
  data.py            # trajectory, IC and collocation sampling
  model.py           # PINCNet
  loss.py            # data, IC, physics losses; time-derivative helper
  train.py           # Adam -> L-BFGS, validation selection, checkpointing
  mpc.py             # one MPC class, pluggable prediction model
  sim.py             # closed-loop simulator
  metrics.py         # IAE, RMSE, max, effort, timing, CIs
experiments/
  e1_open_loop.py  e2_data_efficiency.py  e3_closed_loop.py
  e4_timing.py     e5_robustness.py       e6_ablations.py
tests/
  test_plant.py  test_loss.py  test_model.py  test_sim.py  test_mpc.py
results/            # generated only; each subfolder has config.json + MANIFEST entry
legacy/             # old files, moved untouched
requirements.txt
README.md
```

---

## 3. Phase 1 — Plant

Move `cav_plant.py` → `pinc/plant.py`. It already implements:

```
state x = [vx, vy, r, psi, X, Y]      input u = [Fx, delta]
alpha_f = delta - atan2(vy + lf*r, vx)
alpha_r =       - atan2(vy - lr*r, vx)
m (dvx - vy r) = Fx - 0.5 rho Cd A vx^2 - Frr tanh(vx/0.1) - Fyf sin(delta)
m (dvy + vx r) = Fyf cos(delta) + Fyr
Iz dr          = lf Fyf cos(delta) - lr Fyr          # rear OPPOSES
dpsi = r ;  dX = vx cos psi - vy sin psi ;  dY = vx sin psi + vy cos psi
```
Tyre: `linear` (default) or `fiala` (saturating, `mu*Fz` limit).

Also write `pinc/plant_tf.py`: the same `f` and an RK4 rollout in TensorFlow, so the baseline MPC can get autodiff gradients with the same solver as the PINC arm.

Actuator dynamics: **v1 omits them** (zero-order hold on commands). Add a config flag `actuator_lag` (default `False`). If enabled later, actuator states `[Fx_act, delta_act]` must be appended to the state vector for **both** plant and PINC.

**Acceptance (`tests/test_plant.py`):**
- Lateral eigenvalues have negative real parts for vx ∈ {5, 10, 20, 30, 40} m/s.
- Free yaw response from r0 = 0.01 rad/s, δ = 0, decays below 1e-4 within 2 s.
- Step steer δ = 0.02 rad at 20 m/s: steady yaw rate within 5% of `vx·δ / (L + K_us·vx²)` (allowing for the speed drop).
- RK4 convergence order between 3.8 and 4.2.
- With `lf != lr` (e.g. 1.2 / 1.6) the vehicle remains stable — guards against D5 returning.
- NumPy and TF plants agree to 1e-5 over a 1 s rollout.

---

## 4. Phase 2 — Config and data

`pinc/config.py` defines:

- Vehicle params (nominal): `m=1500, Iz=2500, lf=1.4, lr=1.4, Caf=Car=50000, Cd=0.3, A=2.2, rho=1.225, Frr=300`.
- Control period `T = 0.1` s (the old 0.5 s is longer than the ~0.26–0.30 s lateral time constant at 20 m/s).
- MPC horizon `N = 10` (1.0 s).
- Input bounds: `Fx ∈ [-6000, 3000]` N (braking allowed), `delta ∈ [-0.3, 0.3]` rad.
- PINC state (what the network predicts): `s = [vx, vy, r, psi]`. X, Y are integrated from s outside the network.
- Training box for initial states: `vx ∈ [5, 30]`, `vy ∈ [-1.5, 1.5]`, `r ∈ [-0.6, 0.6]`, `psi ∈ [-0.5, 0.5]`.
- Scales `S_x = [30, 1.5, 0.6, 0.5]`, `S_u = [6000, 0.3]`, time scale `T`. Network inputs are `[t/T, s0/S_x, u/S_u]`.

`pinc/data.py`:
- `sample_trajectories(n, seed)`: draw `s0` and constant `u` uniformly in the boxes; integrate with RK4, `dt = 1e-3`, over `[0, T]`; return samples at random `t ∈ (0, T]` with exact targets `s(t)`. **Targets come from the integrator, never from the input.**
- `sample_ic(n, seed)`: points at `t = 0` with target `s0`.
- `sample_collocation(n, seed)`: random `(t, s0, u)`; no targets.
- Splits: train / val / test use seeds from config with no overlap. Test set also includes a held-out region (e.g. `vx ∈ [25, 30]` excluded from train) for an extrapolation check, reported separately.

**Acceptance:** a test that re-integrates 100 random samples and matches targets to 1e-8; a test that `y_target != input state` for t > 0.

---

## 5. Phase 3 — Model and loss

`pinc/model.py` — `PINCNet(tf.keras.Model)`:
- Input 7-D, output 4-D in **scaled** units; a separate `physical(s_hat)` multiplies by `S_x`.
- Hidden: configurable depth/width (default 4 × 128), `tanh`, **Glorot** init. Residual blocks optional.
- **No clipping.** Recommended hard IC: `s(t) = s0 + (t/T) · NN(t, s0, u)`, so the t = 0 condition holds exactly; make it a config flag and ablate it in E6.

`pinc/loss.py`:
- `time_derivative(model, inputs)`: returns ds/dt in **physical units**, shape (B, 4), via `tf.GradientTape` with per-output gradients or `batch_jacobian`, reading **column 0 only**, divided by `T`, multiplied by `S_x`.
- Physics residual `R = (ds/dt − f(s, u)) / S_f`, with `f` from `plant_tf.py` (first four components) and `S_f` a per-state characteristic rate (e.g. `S_x / T`). All four residuals included.
- `L = L_data + L_ic + λ · L_phys`. λ from config; default chosen by the E6 sweep on the **validation** set.

**Acceptance (`tests/test_loss.py`):**
- Derivative helper on a toy model with a known closed form (e.g. output `sin(ω·t_phys)` per channel) matches the analytic derivative to 1e-4. This catches D1–D3.
- Residual of a model that exactly returns the RK4 solution is ≈ 0 (build it with a lookup/interpolant wrapper).
- Gradient of loss w.r.t. weights is finite and non-zero for random inputs across the whole input box (catches D14).

---

## 6. Phase 4 — Training

`pinc/train.py`:
- Adam (lr from config, cosine or exponential decay), then L-BFGS (`tfp.optimizer.lbfgs_minimize` or SciPy wrapper) — or remove L-BFGS from the paper. Don't claim what isn't run.
- Collocation points **resampled every epoch**.
- Model selection on **validation** loss; restore best weights before saving.
- Save to `results/models/<run_id>/` with `config.json`, seed, git hash, versions, loss curves as CSV.
- `tf.config.experimental.enable_op_determinism()`; set Python, NumPy, TF seeds.
- Runs from CLI: `python -m pinc.train --config configs/default.yaml --seed 0`.

**Acceptance:** runs end-to-end without error; saved model reloads and reproduces validation loss to 1e-6; a `λ = 0` run is possible through config alone.

---

## 7. Phase 5 — MPC

`pinc/mpc.py` — one `MPC` class; the prediction model is injected:
- `PINCPredictor` (network, one call per step, time input = T/T = 1.0 is fine **here** because E1 separately tests the continuous-time property).
- `RK4Predictor` (TF RK4, nominal parameters, fixed substep e.g. 0.01 s).
- `BlackBoxPredictor` (same network trained with λ = 0).
- `LinearPredictor` (linearised about current state; for the LTV-MPC baseline).

Cost (all terms documented in the paper, weights in config):
```
J = Σ_{k=1..N} (s_k - s_ref,k)^T Q (s_k - s_ref,k)
  + Σ_{k=0..N-1} ũ_k^T R ũ_k + Δũ_k^T R_Δ Δũ_k
  + terminal (s_N - s_ref,N)^T P (s_N - s_ref,N)
```
where `ũ = u / S_u` (normalised, so throttle and steering are commensurate). Constraints: input box as bounds; `|r| ≤ r_max` as a soft penalty with a stated weight.

Solver: SciPy `minimize(method="SLSQP", jac=True)` with the cost **and its gradient** computed in one TF call. Same solver, tolerances, max iterations, and warm start for every predictor. Warm start = previous solution shifted by one step, last input repeated.

**Acceptance (`tests/test_mpc.py`):**
- Analytic gradient matches finite differences to 1e-4 relative.
- With `RK4Predictor` and nominal plant, steady tracking of a constant `v_ref = 20` gives |error| < 0.05 m/s after 3 s.

---

## 8. Phase 6 — Closed-loop simulator

`pinc/sim.py` — `simulate(controller, plant_params, ref, x0, duration, noise, seed)`:
- Loop: measure (state + per-channel Gaussian noise, σ from config with a cited source) → controller → apply `u` with ZOH → `plant.simulate(x, u, T, dt=1e-3, plant_params)`.
- Log: time, true state, measured state, commands, reference, solve time, solver iterations, solver success flag.
- Errors are computed **after** the step, against the reference at the new time.

References (in `pinc/refs.py` or config): at least (a) speed sinusoid, (b) speed step ±3 m/s (so rise time is measurable), (c) single lane change (path in X-Y; track lateral offset and heading), (d) double lane change (ISO 3888-2 geometry).

**Acceptance (`tests/test_sim.py`) — the most important test in the repo:**
- Plug in a controller that always returns `u = 0`, and a "predictor" that returns constant garbage. The recorded tracking error must be **large** (e.g. RMSE_v > 1 m/s on the speed sinusoid). If it isn't, something is writing the reference into the state (D7).
- Grep test: no module under `pinc/` or `experiments/` assigns plant state from reference data.

---

## 9. Phase 7 — Experiments

Each script writes `results/<exp>/<run_id>/` containing `config.json`, raw logs (`.npz`), summary (`.json`), figures (`.pdf` + `.png`), and appends an entry to `results/MANIFEST.md` of the form: *artifact → command that produced it → git hash*.

Metrics (`pinc/metrics.py`): `IAE = Σ|e_k|·T`, RMSE, max |e|, 95th percentile |e|, lateral offset RMSE, control effort `Σ ũ^T R ũ · T`, constraint-violation time, settling and rise time (step refs only), solve time mean / median / p95, iterations. Across seeds: mean ± 95% bootstrap CI; paired Wilcoxon signed-rank tests between arms with effect size.

| Exp | Question | Design | Output |
|---|---|---|---|
| E1 | Is the surrogate accurate? | 200 test ICs × 20 control sequences. One-step error at `t ∈ {0.25, 0.5, 0.75, 1.0}·T` (tests continuous time), and chained multi-step error for 1–50 steps. PINC vs black-box vs linear vs RK4 truth. Also in-box vs extrapolation region. | Table: per-state NRMSE with CIs. Fig: error vs horizon. |
| E2 | Does physics help with little data? | Train-set sizes ~{10², 10³, 10⁴, 10⁵}, 5 seeds each, PINC vs λ = 0. | Fig: test NRMSE vs N (log-log). |
| E3 | Closed-loop tracking | 4 references × 30 seeds (noise + x0 perturbation). Arms: NMPC-RK4 (nominal), PINC-MPC, black-box-MPC, LTV-MPC. | Table of metrics with CIs and p-values. Time-series figs for one representative seed (state the seed). |
| E4 | Is it faster? | Solve time per step vs horizon N ∈ {5, 10, 20, 40}. CPU single-thread (state CPU model) and GPU separately. Also time per model call. | Table + fig: solve time vs N. |
| E5 | Robustness to mismatch | Controller nominal; plant with mass ±20%, Caf/Car ±30%, μ ∈ {0.4, 0.7, 1.0} (Fiala tyre), step disturbance force. 30 seeds per setting. | Degradation curves per arm. |
| E6 | Ablations | λ ∈ {0, 0.01, 0.1, 1, 10, 100}; collocation count; depth/width; hard vs soft IC; with/without each residual. | Table on validation + test. |

Expected honest framing (do not force it): NMPC-RK4 is the accuracy ceiling; the claim to test is PINC-MPC ≈ NMPC-RK4 accuracy at lower solve time, and better than black-box at low data. If PINC is not faster at small N, report that and show where the crossover is.

---

## 10. Phase 8 — README and reproducibility

- `requirements.txt` with pinned versions (TF, tensorflow-probability, NumPy, SciPy, matplotlib, pyyaml, pytest).
- `README.md`: install, `pytest`, train, each experiment command, expected runtime, hardware used.
- `make all` (or `scripts/run_all.sh`) reproduces every artifact in `results/`.
- Move `PINC.py`, `MPC using PINC only.py`, `MPC vs PINCMPC.py` to `legacy/` untouched.

---

## 11. Paper corrections (needs the LaTeX source; not doable from the PDF)

If the `.tex` source is added to `paper/`, apply these. Otherwise hand this list to the authors.

**Framing:** retitle (drop "Quantum Leap"); rewrite abstract in past tense with headline numbers from `results/`; remove H∞, sliding mode, min-max MPC, LiDAR, V2V, platooning from contributions (keep in related/future work); delete "relatively infinite prediction horizons".

**References:** cite Antonelo et al. (PINC, arXiv 2104.02556) as the origin of the architecture and Eqs (15)–(16); add a paragraph on novelty vs refs [4] and [11]; replace [4] as the generic MPC citation with a standard MPC text; cite or remove [12]–[15] (uncited in text) and verify [15] exists as listed; "APC" → "ACP"; support or cut the PINN-vs-RNN pre-training claim.

**Modelling (§3):** use the §4 / `plant.py` model everywhere, including v_y; fix the §3.3 substitution; fix "acceleration = second derivative of velocity"; remove or implement the §3.5.1 kinematic model and §3.6 terminal cost (now implemented in `mpc.py` — describe that one); Eq (1) is the general nonlinear form, not linear; Eq (2) `N[u; λ]`; cut the PDE preamble; Eq (19) `λ·MSE_f`; soften §3.15's "faster than classical solvers" to fast inference of a trained surrogate.

**Method:** document network, scaling, all four residuals and their normalisation, λ, IC treatment, optimiser schedule, epochs, data sizes, hardware — all read from the configs actually used.

**Results (§4):** delete the existing §4 entirely, including Tables 1–3, Eqs (27)–(30), and all current figures. Rebuild from E1–E6. Every caption names its generating script. Add a Limitations section (no stability/recursive-feasibility guarantee; tyre model validity range; simulation only).

**Presentation:** fix figure numbering (duplicated 1 and 2; "Fig. 6/7" do not exist); legend/caption colour mismatches; "Infomed" → "Informed".

---

## 12. Definition of done

- [ ] `pytest` passes, including the garbage-model closed-loop test.
- [ ] Every defect D1–D20 has a test or a code-review note showing it's gone.
- [ ] `make all` regenerates every file in `results/` from a clean checkout.
- [ ] `results/MANIFEST.md` lists every table and figure with its command and git hash.
- [ ] No number in the paper that isn't in `results/`.
