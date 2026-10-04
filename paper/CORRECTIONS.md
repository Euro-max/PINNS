# Paper corrections

**Status (second revision):** rewritten as a journal manuscript: no references to the earlier implementation, the repository layout or generating scripts remain in the text; wide tables are scaled to the text width; the abstract is shortened; the supervisor's five comments (`faculty_comments.md`) are addressed (explicit contribution list and novelty paragraph, baselines across four scenarios with HIL/vehicle tests named as future work, IMRaD structure, generated tables and figures, references tied to text). The items below were applied in
`LATEX/main.tex` (the previous draft is kept as `LATEX/main_original.tex`, the previous bibliography
as `LATEX/sample_original.bib`). Every table, figure and number in the manuscript is produced by
`scripts/make_paper_tables.py` into `LATEX/generated/` from `results/`; rerun it after any experiment.
No LaTeX compiler was available in the build environment, so the manuscript was checked structurally
(balanced braces / environments, all macros, citations and labels resolved) but not compiled: compile it
once in Overleaf and fix any layout issue before submission.

The list combines
§11 of `TASKS.md`, the defects found while rebuilding the code (`docs/DEFECTS.md`), and the
faculty comments in `faculty_comments.md`. Every number in the rewritten paper must be read from
`results/` (see `results/MANIFEST.md` for the script and git hash behind each artifact).

## 1. Framing and contribution (faculty comment 1, 3)

- Retitle: drop "Quantum Leap".
- Rewrite the abstract in past tense, at most ~150 words, with the headline numbers taken from
  `results/e3_closed_loop/*/summary.json` and `results/e4_timing/*/summary.json`.
- State the contribution explicitly and narrowly. What the repository actually supports is:
  a PINC surrogate (Antonelo et al., arXiv 2104.02556) of a 4-state single-track model with
  coupled longitudinal / lateral dynamics, used as the prediction model of a nonlinear MPC, and
  benchmarked against (i) NMPC with the exact RK4 model, (ii) a black-box network (λ = 0) and
  (iii) an LTV-MPC, on four references, under measurement noise and parameter mismatch.
  Say which of these is new relative to [4] (Li & Liu 2024) and [11] (Jin et al. 2024) in a
  dedicated paragraph; do not claim novelty of the architecture itself.
- Remove H∞, sliding mode, min-max MPC, LiDAR, V2V and platooning from the contributions and
  conclusion; they can stay in related / future work.
- Delete "relatively infinite prediction horizons", "more environmentally aware", and the
  claim that the controller is "designed with robust control theory in mind".
- Restructure into Introduction, Related Work, Method, Experimental Setup, Results, Limitations,
  Conclusion (faculty comment 3). Sections 3.9–3.14 (a narrated Simulink trial) should go.

## 2. References (faculty comment 5)

- Cite Antonelo et al. (PINC, arXiv 2104.02556) as the origin of the architecture and of
  Eqs. (15)–(16).
- Replace [4] as the generic MPC citation with a standard MPC text (e.g. Rawlings, Mayne &
  Diehl, *Model Predictive Control: Theory, Computation, and Design*).
- [12]–[15] are never cited in the text: cite them where they matter or remove them; verify that
  [15] exists as listed (authors / venue look doubtful).
- "APC" → "ACP" (Artificial systems, Computational experiments, Parallel execution).
- Support with a reference or cut the claim that PINNs need less pre-training than RNNs.
- Cite the ISO 3888-2 geometry used for the double lane change and the sources for the sensor
  noise levels (listed in `pinc/config.py`).

## 3. Modelling (§3)

- Use one vehicle model everywhere: the six-state single-track model of `pinc/plant.py`
  (Eqs. in the module docstring), including lateral velocity `v_y`. §3.1 and §4 currently use a
  3-state model without `v_y`, and the legacy code's yaw moment `lf Fyf + lr Fyr` with
  `alpha_r = +lr r / v` has zero yaw damping for the symmetric parameters (defect D5).
- §3.3: the substitution of the tyre forces is wrong (sign of the rear term); the correct
  linearised yaw equation is `Iz r_dot = lf Caf δ - (lf² Caf + lr² Car) r / vx - (lf Caf - lr Car) vy / vx`.
- "acceleration (second derivative of velocity)" → acceleration is the first derivative of
  velocity. The bullet "the d²v/dt² term is no longer needed" should go.
- Remove or implement §3.5.1 (kinematic model, never used) and replace §3.6 by the cost actually
  implemented in `pinc/mpc.py` (documented in its docstring: stage cost with Q, input cost R on
  the normalised input, rate cost R_Δ, terminal cost P, soft yaw-rate limit).
- Eq. (1) is the general *nonlinear* form, not "general linear form"; Eq. (2) should read
  `u_t + N[u; λ] = 0`; cut the PDE preamble (the system is an ODE).
- Eq. (19): `MSE = MSE_y + λ · MSE_f` (currently `+ λ + MSE_f`).
- §3.15: soften "faster than classical solvers" to "fast inference of a trained surrogate";
  the honest comparison is in E4.

## 4. Method (read everything from the configs actually used)

Document, from `results/models/<run>/config.json` and `meta.json`:
network (depth × width, tanh, Glorot), input / output scaling (`S_x`, `S_u`, `T`), the hard
initial condition `s(t) = s0 + (t/T)·NN`, all four residuals and their normalisation `S_f`,
λ (chosen on the validation set in E6), the optimiser schedule (Adam with cosine decay, then
L-BFGS), number of steps, data sizes (train / val / test / extrapolation, disjoint seeds), the
collocation resampling, hardware and software versions. State that actuator dynamics are not
modelled in v1 (zero-order hold).

## 5. Results (§4) — rebuild entirely (faculty comments 2a, 4)

- Delete the existing §4 including Tables 1–3, Eqs. (27)–(30) and all current figures: they
  were produced by code in which the closed loop never called the plant (D7) and the plant state
  was set to the reference for `t < 4 s`.
- Rebuild from E1–E6 (`experiments/`). Every caption names the generating script and run id.
  Suggested set: E1 table + error-vs-horizon figure; E2 figure; E3 table (four references, 30
  seeds, CIs and paired Wilcoxon p-values) + one time-series figure per reference (state the
  seed); E4 table + figure; E5 degradation figure; E6 table.
- Report the honest framing: NMPC-RK4 is the accuracy ceiling; the tested claim is
  PINC-MPC ≈ NMPC-RK4 accuracy at lower solve time and better than black-box at low data. If
  PINC is not faster at small N or not more accurate than the black-box at the data sizes used,
  say so and show where the crossover is.
- Add a Limitations section: no stability / recursive-feasibility guarantee; linear-tyre
  validity range (the Fiala results in E5 show the effect of saturation); simulation only —
  hardware-in-the-loop or vehicle tests are future work (faculty comment 2b).

## 6. Presentation

- Figure numbering: Figures 1 and 2 are duplicated; "Fig. 6 / 7" do not exist.
- Legend / caption colour mismatches; every axis labelled with units.
- "Infomed" → "Informed" (file name and title).
- Poster: the same numbers (2 %, 1.2 s rise time, 0.3 m/s overshoot, 66 % / 75 % improvement)
  come from the invalid results and must be regenerated from `results/`.

## 7. Mismatches between the paper and the legacy code that the rebuild removes

| Paper says | Legacy code did |
|---|---|
| MPC weights 10 / 50 / 0.001 / 1000, horizon 20 steps (Eq. 25) | 10000 / 100 / 0.5 / 2000, horizon 5 |
| Adam η = 1e-3, λ = 0.1 (Eq. 23) | η = 3e-4, λ = 1 |
| hybrid Adam / L-BFGS | L-BFGS never ran (the script crashed before plotting) |
| actuator τ_u = 0.2 s (Eq. 26) | 0.2 s in the physics loss, 0.3 s in the MPC |
| δ normalised by 0.5 (table in §4.1) | 0.3 in training, 0.5 at inference |
| training targets from `solve_ivp` (Eq. 22) | one explicit-Euler step of 0.1 s, target = input |
| force bounds ±3000 N (Eq. 28) | (0, 3000): braking impossible |
| "MPC vs PINC" closed loop | the PINC loop never simulated the plant |

All of these are now single-sourced in `configs/default.yaml`.
