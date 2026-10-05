# Plan: PINC with an imperfect physics prior (high-fidelity plant)

**Status:** proposal, not started.  Builds on the rebuilt pipeline (`TASKS.md`); every ground rule in
`TASKS.md` §1 still applies (one plant, controllers get nominal parameters, no hand-typed numbers,
seeds logged, fail loudly, no tuning on the test set, report bad results).

## 0. Why

In the current benchmark the physics in the PINC loss is *exactly* the plant that generates the data.
PINC then learns a model that is already known, and NMPC with that model (NMPC-RK4) is as accurate and,
for N <= 10, as fast.  A physics-informed surrogate is only worth having when at least one of these holds:

1. the physics we can write down is only **approximate**: data corrects its bias, physics regularises
   where data is scarce;
2. the true model is **too expensive** to integrate inside MPC (stiff or high-dimensional), so a
   one-call-per-step surrogate saves real time;
3. some physical **parameters are unknown** and can be identified together with the dynamics.

This plan separates the **true plant** (a double-track vehicle with Magic Formula tyres, load transfer,
wheel dynamics and actuator lag) from the **physics prior** used in the loss (the simplified
single-track model with linear tyres), so all three conditions hold, and asks:

- Does PINC beat a data-only network of the same size at low data, and NMPC with the prior model everywhere?
- How wrong can the prior be before lambda > 0 stops helping?  (best lambda vs. data size and prior error)
- Do learnable physical parameters recover the true values and remove the parametric part of the bias?
- Is PINC-MPC faster than NMPC with the true (stiff) model at equal tracking quality?
- Does PINC beat a grey-box model (prior + learned correction)?

## 1. Decisions (agreed)

| # | Decision | Choice |
|---|---|---|
| 1 | State predicted by the network | **10 states**: `[vx, vy, r, psi, F_act, delta_act, w_fl, w_fr, w_rl, w_rr]` -- body states, actuator states and the four wheel speeds.  Variant without wheel speeds kept as an ablation (H7). |
| 2 | Physics loss with an imperfect prior | **(a)** one lambda over all residuals, and **(d)** the same with learnable physical parameters in the prior; the **grey-box** model is a baseline.  Per-state weights (b) and "exact structure + learned forces" (c) only if the remaining bias justifies them. |
| 3 | Framing | **Single automated vehicle.**  Platoon / connected scenarios are future work. |

## 2. True plant "HF" (`pinc/plant_hf.py`, `pinc/plant_hf_tf.py`)

State (12): `x = [vx, vy, r, psi, X, Y, w_fl, w_fr, w_rl, w_rr, F_act, delta_act]`; input `u = [F_cmd, delta_cmd]`
(same interface as today: commanded total longitudinal force and road-wheel steer angle).

**Actuators** (first-order lag):
```
d(F_act)/dt     = (F_cmd     - F_act)    / tau_F
d(delta_act)/dt = (delta_cmd - delta_act) / tau_delta
```
Torque split: drive (`F_act >= 0`) with front share `gamma_f` (config; **default 1 = front-wheel drive**,
0 = rear-wheel drive, in between = all-wheel drive): `T_f = gamma_f F_act R_w / 2` per front wheel,
`T_r = (1 - gamma_f) F_act R_w / 2` per rear wheel.  Braking (`F_act < 0`) on all wheels with front share
`beta_f`.  Front-wheel drive is the default because it is the most common passenger-car layout, it
understeers (stays stable) in the aggressive tests, and it puts combined drive/lateral slip on the steered
axle, which is where the linear prior is most wrong.  The prior uses the same split (the controller knows
the drive layout).

**Wheel kinematics.**  Wheel positions in the body frame (y to the left): fl `(lf, +t_f/2)`, fr `(lf, -t_f/2)`,
rl `(-lr, +t_r/2)`, rr `(-lr, -t_r/2)`.  Hub velocity `v_i = (vx - r y_i, vy + r x_i)`; steer `delta_i = delta_act`
(front), 0 (rear).  In the wheel frame:
```
v_l,i =  v_x,i cos(delta_i) + v_y,i sin(delta_i)          (rolling direction)
v_s,i = -v_x,i sin(delta_i) + v_y,i cos(delta_i)          (lateral)
kappa_i = (R_w w_i - v_l,i) / max(|v_l,i|, v_eps)         (longitudinal slip)
alpha_i = -atan2(v_s,i, max(|v_l,i|, v_eps))              (slip angle; same sign convention as plant.py)
```

**Vertical loads** (quasi-static load transfer; accelerations approximated explicitly so there is no
algebraic loop: `a_x = (F_act - F_aero - F_roll)/m`, `a_y = vx r`):
```
Fz_f,static = m g lr / (2L),   Fz_r,static = m g lf / (2L)                (per wheel)
dFz_long    = m a_x h / (2L)        front wheels -dFz_long, rear +dFz_long
dFz_lat,f   = chi_f   m a_y h / t_f  left front -, right front +
dFz_lat,r   = (1-chi_f) m a_y h / t_r  left rear -,  right rear +
Fz_i        = max(sum, Fz_min)
```

**Tyres:** Magic Formula 5.2 (Pacejka, *Tire and Vehicle Dynamics*), pure and combined slip, zero
camber, no turn slip, with its own load dependence (`PKY1`, `PKY2`, `PDY2`, ... through `dfz = (Fz - FNOMIN)/FNOMIN`).
Structure of the pure-slip curves and the combined-slip weighting:
```
F_y0 = D_y sin(C_y atan(B_y alpha_y - E_y (B_y alpha_y - atan(B_y alpha_y)))) + S_Vy,   alpha_y = alpha + S_Hy
F_x0 = D_x sin(C_x atan(B_x kappa_x - E_x (B_x kappa_x - atan(B_x kappa_x)))) + S_Vx,   kappa_x = kappa + S_Hx
F_x  = G_xa(alpha, kappa) F_x0,      F_y = G_yk(alpha, kappa) F_y0 + S_Vyk
```
with every coefficient a function of `Fz` given by the MF 5.2 parameter set.

**Tyre data.**  MF 5.2 parameter set for a **205/60R15** passenger-car tyre from the MathWorks Vehicle
Dynamics Blockset (R2025b, `vdyntire.internal.models.mf52.tm20560R15`), exported to JSON by
`scripts/export_tyre_params.m`.  It is MathWorks data, so it is read locally from `data/tyre/` (gitignored)
and cited, not committed.  The friction scaling factors (`LMUX`, `LMUY`) set `mu` for the road-surface
scenarios.

**Wheel and body dynamics:**
```
I_w dw_i/dt   = T_i - R_w F_x,i
m (dvx - vy r) = sum_i (F_x,i cos delta_i - F_y,i sin delta_i) - F_aero - F_roll
m (dvy + vx r) = sum_i (F_x,i sin delta_i + F_y,i cos delta_i)
Iz dr          = sum_i [ x_i (F_x,i sin delta_i + F_y,i cos delta_i) - y_i (F_x,i cos delta_i - F_y,i sin delta_i) ]
dpsi = r ;  dX = vx cos psi - vy sin psi ;  dY = vx sin psi + vy cos psi
F_aero = 0.5 rho Cd A vx^2 ,  F_roll = Frr tanh(vx/0.1)     (as in plant.py)
```

**Calibration.**  Variant **M0 ("calibrated")**: the stiffness scaling factor (`LKY`, and `LKX` for the
longitudinal slip stiffness) is set so that at static load and small slip the HF model linearises to the
current single-track model (per-axle cornering stiffness `Caf = Car = 50 kN/rad`); the curve shapes, load
dependence, saturation and combined slip are left as in the tyre data.  The prior is then right in gentle driving, and all of its
error is *structural* (saturation, load transfer, combined slip, wheel dynamics, track width).
Variant **M1 ("mismatched")**: additionally the true tyre stiffnesses, `mu` and actuator time constants
differ from the prior's nominal values (in particular the tyre data's own, unscaled stiffness, which is
expected to be well above the prior's 50 kN/rad), so part of the error is *parametric* and learnable
(decision 2d).

**Stiffness.**  The wheel-slip time constant is about `I_w |v| / (R_w^2 C_kappa)`: ~2 ms at 20 m/s and
~0.5 ms at 5 m/s.  Explicit RK4 therefore needs a much smaller step than today's 1 ms; Phase 1 fixes
`dt_plant` and the NMPC substep from a measured stability/accuracy test, not from this estimate.

**Provisional vehicle parameters** (to be fixed in Phase 2, each with a cited source, e.g. Rajamani,
*Vehicle Dynamics and Control*; wheel radius and nominal load come from the tyre data set):

| parameter | provisional value | parameter | provisional value |
|---|---|---|---|
| m, Iz, lf, lr | as `configs/default.yaml` | track t_f, t_r | 1.6 m |
| CG height h | 0.55 m | front roll share chi_f | 0.55 |
| wheel radius R_w | tyre data (205/60R15: ~0.31 m) | wheel inertia I_w | 1.2 kg m^2 |
| tau_F, tau_delta | 0.15 s, 0.10 s | brake share beta_f | 0.6 |
| drive share gamma_f | 1 (front-wheel drive) | mu | 1.0 (M0); scenario-dependent (M1, E5-style tests) |

## 3. Physics prior "P" (`pinc/prior.py`, TF)

The prior is defined on the 10 network states and is deliberately simpler than HF:

- single-track geometry (track width 0: left and right wheels of an axle see the same kinematics);
- static vertical loads (no load transfer);
- **linear** tyres per wheel, `F_x = C_kappa kappa`, `F_y = C_alpha alpha`, no combined slip, no saturation;
- the same rigid-body equations, wheel equation and actuator lag as HF, with **nominal** parameters
  `theta = (C_alpha,f, C_alpha,r, C_kappa, tau_F, tau_delta)`.

Physics residual `R = (ds/dt - f_P(s, u; theta)) / S_f` over all 10 states.  In variant 2d `theta` is a set of
trainable variables (log-parametrised, positive), trained jointly with the network and reported.  The
linear prior has no `mu`; identifying friction needs a saturating prior (optional variant P2: Fiala tyre
with learnable `mu`), decided after H4.

**Prior error.**  `e_P(s, u) = |f_HF(s, u) - f_P(s, u)| / S_f` per state, computed over (i) the gentle region
(small slip, as in today's references) and (ii) the aggressive region (large slip, low mu).  Every PINC
result is reported against this number.

## 4. Network, data and training changes

- **General state dimension.**  `model.py`, `loss.py`, `data.py`, `mpc.py` and `metrics.py` currently hard-code
  4 states (`z[:, 1:5]`, `z[:, 5:7]`, `range(4)`).  Make `n_s` a config entry; the bicycle setup (`n_s = 4`) must
  keep passing every existing test and reproduce `pinc_v2_s0`.
- **Wheel-speed coordinates.**  Wheel speeds are ~70 rad/s while the informative part (slip) is ~1 % of
  that.  The network uses the slip velocity `sigma_i = R_w w_i - vx` (scale ~1 m/s) instead of `w_i`; the
  transform is fixed and invertible, applied in `scale_inputs` / `physical`.
- **Initial states.**  With 10 states a uniform box mostly contains physically inconsistent combinations
  (e.g. large slip on every wheel while cruising).  Initial states are drawn from **random-excitation
  trajectories of the HF plant** (states visited in plausible driving), plus a box-perturbed fraction for
  coverage.  The same rule for train / val / test, disjoint seeds; extrapolation region = held-out speed
  range as today, plus an aggressive-manoeuvre region.
- **Collocation in time.**  The wheel dynamics settle within a few ms of a 100 ms step (a boundary layer at
  `t ~ 0`).  Collocation times: half uniform on `(0, T]`, half log-uniform on `[1e-3 T, T]`.
- **Measurement noise in the training data** (wheel encoders, steering-angle sensor, torque estimate), so
  the data-only arm is not trained on perfect data.
- **Training budget (from E11).**  No plateau within 5000 L-BFGS iterations; loss falls as a power law and
  the PINC / data-only ratio is budget-independent.  Fixed budget for every arm: 300 Adam epochs
  (constant learning rate; cosine decay gave no benefit) + 2000 L-BFGS iterations, `float64`
  (float32 stalls L-BFGS, `results/env_check/`).  Reported as a fixed budget, with the E11 curve as justification.

## 5. Model arms

| arm | prediction model | role |
|---|---|---|
| NMPC-HF | HF model, TF RK4, small substep | accuracy ceiling; expected to be slow (stiff) |
| NMPC-P | prior model P, TF RK4 | physics only, mismatched |
| data-only | network, lambda = 0 | no physics |
| PINC | network, lambda > 0, fixed theta | method under test (decision 2a) |
| PINC-theta | network, lambda > 0, learnable theta | decision 2d |
| grey-box | `ds/dt = f_P(s, u; theta) + NN(s, u)`, integrated with RK4 | strongest competitor; inherits the stiffness cost of P |
| (LTV-MPC) | HF linearised along the warm start | optional, kept for continuity |

All MPC arms: same cost, solver, tolerances, warm start and measurements (the 10 network states, with
noise; X, Y from GNSS as today).  Controllers never see the HF parameters.

## 6. Experiments

| id | question | design | output |
|---|---|---|---|
| H0 | Is the HF plant right? | acceptance tests (Phase 1) | test report |
| H1 | How wrong is the prior, and where? | `e_P` per state, gentle vs aggressive region, M0 and M1 | table + heat map over (speed, slip) |
| H2 | Data efficiency with an imperfect prior | N in {1e2, 1e3, 1e4, 1e5}, 5 seeds, all learned arms, M0 and M1 | test NRMSE and 50-step error vs N |
| H3 | Best lambda vs data and prior error | lambda sweep at each N (validation data loss) | best-lambda curve; when does physics hurt? |
| H4 | Parameter identification | PINC-theta: learned theta vs true values, M1 | table, convergence of theta |
| H5 | Closed loop | today's 4 references + aggressive ones (low-mu lane change, braking in a turn), 30 seeds, all arms | metrics with CIs and paired tests |
| H6 | Speed | solve time vs horizon N in {5, 10, 20, 40}, CPU single thread and GPU | where is PINC faster than NMPC-HF / grey-box? |
| H7 | Cost of hiding wheel speeds | network on 6 states (no wheel speeds) vs 10 states | open-loop and closed-loop error |
| H8 | Architecture | small grid {8x64, 8x128, 6x256} x lambda {0, best} on HF validation, 3 seeds for the top candidates | chosen default; accuracy vs MPC solve time |

Expected honest outcomes to look for, not to force: physics helps most at low N in the gentle region and
may hurt at high N in the aggressive region; NMPC-HF is the accuracy ceiling but may miss the 100 ms real-time
budget at longer horizons; the grey-box model may match PINC on accuracy but be slower.

## 7. Phases and acceptance criteria

1. **Generalise the state dimension.**  `n_s` in config; no behaviour change.  *Accept:* all existing tests
   pass; retraining `pinc_v2_s0` reproduces its validation loss within 1 %.
2. **HF plant (NumPy + TF).**  *Accept:* tyre forces reduce to `C_alpha alpha` / `C_kappa kappa` at small slip
   and saturate at `mu Fz`; M0 linearises to today's single-track model (lateral eigenvalues within 2 %);
   static loads sum to `m g` and load transfer has the right sign; free rolling gives zero slip;
   RK4 order ~4 at the chosen `dt`; NumPy and TF agree to 1e-6 over 1 s; stability of the chosen
   `dt_plant` and NMPC substep demonstrated at 5 m/s.
3. **Prior P and prior-error map (H1).**  *Accept:* `f_P = f_HF` exactly when HF is reduced to zero track,
   static loads and linear tyres; H1 tables written by a script.
4. **Data and training for n_s = 10.**  Trajectory-based IC sampling, time collocation, noise, slip-velocity
   coordinates, learnable theta.  *Accept:* the data re-integration test passes for HF; the derivative,
   residual-of-exact-solution and gradient tests pass for 10 states; one PINC and one data-only model train
   end to end.
5. **Architecture (H8) and open-loop experiments (H2, H3, H4, H7).**
6. **MPC arms, closed loop and timing (H5, H6).**  *Accept:* the garbage-predictor test still fails loudly
   on the HF plant; FD gradient checks pass for every arm.
7. **Paper.**  Decided after the results: the matched-physics study (current results) and this
   imperfect-prior study are kept separate until then.  If the imperfect-prior study shows a clearer
   contribution for PINC, the paper is framed around both (matched physics as the zero-mismatch end of
   H3); otherwise the two are reported separately.

## 8. Risks

- **Stiff boundary layer.**  The network must reproduce a ms-scale transient inside a 100 ms step; if it
  cannot, H7 (wheel speeds hidden) becomes the main configuration and the speed argument rests on NMPC-HF
  only.
- **Prior conflict.**  A biased prior can make every lambda > 0 worse than data-only at the data sizes that
  matter.  That is a reportable result (H3), not a reason to tune the prior to the truth.
- **Compute.**  10 states, larger networks and a 2000-iteration L-BFGS budget: roughly 10-20 min per model
  on the GPU when run alone; H2 + H3 is ~100-150 trainings.  Run sequentially or two at a time (E11 showed
  three parallel runs slow each one ~3x).
- **NMPC-HF solve time.**  If it is too slow to run 30 seeds x 4 references, reduce seeds for that arm and
  say so.

## 9. Decisions on the former open questions

- **Drive layout:** front-wheel drive by default; the front drive share is a config parameter (§2).
- **Current results:** kept separate for now; paper framing decided after the H-experiments (Phase 7).
- **Tyre data:** Vehicle Dynamics Blockset MF 5.2 set for a 205/60R15 tyre, exported locally (§2).
