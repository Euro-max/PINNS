# PINC-MPC: consolidated status — what was fixed, what v2 is, what comes next

## Purpose
One honest account of everything done since the legacy results were found to be invalid:
- the legacy-to-rebuild fixes, how the v2 results came about, and the work since (October 2026);
- every non-trivial error and how it was resolved;
- whether any test or criterion was bent instead of fixing the root cause;
- what would have given a meaningfully better result;
- the revised plan.

Written 2026-10-05; update it when a phase completes.

---

## 1. Timeline

| stage | what happened | where recorded |
|---|---|---|
| Legacy (May 2025) | `PINC.py` + two MPC scripts; paper and poster built on them | `legacy/`, git `8474f80`..`1e4d7e0` |
| Rebuild (2026, before October) | Pipeline rewritten to `TASKS.md`: one RK4 plant, PINC network, MPC with injected predictor, closed-loop simulator, 49 tests, experiments E1–E7 | `6632a6f` (one large commit; no step-by-step history) |
| v1 results | E1–E7 with the unscaled physics residual: physics did not help except at about 100 trajectories | `docs/RESULTS.md` (lower half) |
| v2 results | E6 showed the v_x residual dominating; E8 tested two rescalings, increment scaling won on validation; E9 checked derivative learning; E10 confirmed over 5 seeds; E1–E6 rerun as `*_v2`; paper and poster rewritten from them | `docs/RESULTS.md`, `results/*_v2`, `e8b`, `e9`, `e10` |
| Oct 2026 | GPU environment; E11 (budget); plan for the imperfect-prior study; Phase 1 (state-size refactor, scheduler, E12); paper review fixes; bibliography audit; move to TF 2.21.0 | commits `adae6b6` .. `cfa0f91` on `gpu-env-convergence` |

---

## 2. Non-trivial errors and how each was resolved

### 2a. Legacy defects (D1–D20, `docs/DEFECTS.md`)
These invalidated every legacy number. The ones that mattered most:
- **D7, no plant in the loop:** the closed loop never called the plant, then blended the state with the reference and finally set it equal to the reference.
- **D4, no real targets:** the training target equalled the input state.
- **D1–D3, wrong derivative:** the time derivative was computed incorrectly (wrong columns, sum of outputs, no 1/T factor).
- **D5, wrong yaw model:** zero yaw damping.
- **D8, no mismatch:** perturbed parameters went to the controller as well as the plant.

All were fixed by rebuilding rather than patching. Each has a test or a review note.

Weak spots in that evidence:
- **D11 (actuator lag)** was resolved by removing actuator lag from the model, not by fixing it. That narrows the scope; Phase 2 reintroduces it properly.
- **D12 (best weights restored) and D13 (collocation resampling)** are backed only by review notes and an `assert`, with no behavioural test.

### 2b. v1 to v2 (before October 2026)

| problem | root cause | resolution | judgement |
|---|---|---|---|
| Physics term hurt accuracy (λ = 0 best on validation for every network) | The v_x residual is divided by S_f but magnified by S_x/(T·S_f) ≈ 147; slope errors in v_x dominated the physics loss | E8 compared two principled rescalings on **validation**; increment scaling `D = S_f·T/S_x` (output of order 1 per state) won; also applied to the λ = 0 arm, so the comparison stays fair | Root-cause fix |
| v1 kept λ = 0.01 although λ = 0 was best on validation, "so the PINC arm is physics-informed" | A hyperparameter chosen for the narrative, not the data (a ground-rule 8 deviation, documented) | In v2 the validation optimum among λ > 0 is 0.01 (E8), confirmed by E10 over 5 seeds | Resolved in v2; v1 numbers stay superseded |
| Default network 8×64 | Chosen by E7 under the v1 loss | Kept for v2; flagged as possibly suboptimal | **Still open** (see §4) |

### 2c. October 2026

| # | problem | root cause | resolution | root cause fixed? |
|---|---|---|---|---|
| 1 | No environment; system Python can't create venvs; no sudo | WSL Python without `ensurepip` | `uv` installed in `~/.local/bin` | yes |
| 2 | TF out-of-memory errors on an idle GPU | TF reserves almost the whole device up front, which fails under WSL2 | `TF_FORCE_GPU_ALLOW_GROWTH` set in `pinc/__init__.py` | yes |
| 3 | `float32` training ~5× worse | L-BFGS line search stalls at loss ~1e-6 (stopped after 9 iterations) | Training stays in `float64` (`results/env_check`) | yes; a constraint, not a bug |
| 4 | GPU only ~2× faster than the old CPU; batching expected to help | Training is bound by `float64` arithmetic; consumer GPUs run `float64` at ~1/64 speed; the 24-thread CPU matches the GPU | Measured (16 vs 8.9 ms per Adam step, ~150 ms per L-BFGS iteration on both); XLA in `float64` is 100× slower and was rejected | yes; diagnosed, nothing to fix |
| 5 | **No model had converged**: in all 200 models the best checkpoint is the final iteration | Fixed budget of 300 Adam epochs + 500 L-BFGS iterations | E11: no plateau even at 5000 L-BFGS iterations (power law); the PINC / data-only ratio holds at 1.3–1.5 at every budget | Measured; **v2 results still come from unconverged models** |
| 6 | Restored scripts had CRLF line endings and no executable bit | Copied back through Windows | Restored from git (content identical) | yes |
| 7 | 4-state assumptions everywhere | No abstraction for the vehicle model | `pinc/system.py`; reproduces the old model bit for bit; a 5-state test system exercises the whole pipeline | yes |
| 8 | The new test exposed two more hidden assumptions (sim integrator; error = state − reference) | Same as #7 | `plant_simulate` / `track_full` on the system | yes |
| 9 | GPU+CPU scheduling slowed both jobs ~8× | GPU job had no thread limit and took ~9 cores (load 46 on 24 threads) | GPU jobs get 4 threads, CPU jobs the rest, BLAS capped; E12 rerun gives 1.8× throughput | yes |
| 10 | My `pgrep`/`pkill` waits matched their own shell (one hang, one killed shell) | Pattern matched the waiting command line | Waits now track the PID | yes (tooling only) |
| 11 | Paper: E3 solve times contradicted E4 and the abstract | E3 means included each controller's first call, which compiles the cost (7.7 s NMPC vs 1.7 s PINC) | Root cause fixed in code (E3 warm-up call, `ef3ed10`). **The current table is post-processed** from stored logs (median, first call excluded) rather than rerun | **Partly; see §3** |
| 12 | Paper: claim that v_x dominates had no evidence in the paper | Evidence existed (v1 E6 residual ablation, E9 unscaled models) but was not shown | Table e9 gained "no D" rows; a sentence cites generated macros | yes |
| 13 | Bibliography: `jin2024physical` had the wrong issue and pages; `minh2020model` the wrong journal, year, volume, pages and author spelling, plus an inaccurate description in the text | Hand-entered references | All 25 entries checked against Crossref / publisher; 5 corrected; text fixed | yes |
| 14 | Results produced with a TF release candidate | The only TF build for Python 3.14 | Moved to stable TF 2.21.0 on Python 3.13; TF 2.21 silently ignored the GPU (cuSOLVER folder not searched), fixed by preloading the library in `pinc/__init__.py`; 55/55 tests pass. Retraining `pinc_v2_s0`: identical to ~1e-12 through Adam, then L-BFGS amplifies last-bit differences to **4.5%** in validation data loss (6.27e-7 vs 6.00e-7), inside the **9.6%** seed-to-seed spread of E10 (5.88–6.45e-7). The 1% threshold set beforehand was too strict for L-BFGS; the paper's results are kept and the paper states both versions | yes; difference measured and reported |
| 16 | E3 rerun on TF 2.21: NMPC solve time 476 ms instead of ~12 ms | On a machine with a visible GPU, TF places the batch-1 MPC cost on the GPU, where XLA-compiled `float64` is very slow | MPC timing runs hidden from the GPU (`CUDA_VISIBLE_DEVICES=-1`), as the paper's CPU setup assumes; tracking metrics unaffected (seed 0 identical) | yes; rule for all timing experiments |
| 15 | MATLAB tyre export returned empty arrays | Internal functions need a block argument | Read the parameters from the Combined Slip Wheel block's settings | yes |

---

## 3. Was any test or criterion bent?

Checked against `TASKS.md`. All acceptance tolerances in the tests match the spec: 5% step steer (with the speed drop the spec allows), RK4 order 3.8–4.2, NumPy/TF agreement to 1e-5, gradient check to 1e-4 relative, steady tracking < 0.05 m/s, garbage-predictor RMSE > 1 m/s.

Changes in October 2026 that touched a test or a criterion:
1. **`test_sim_state_only_changes_via_plant`**: the expected string changed from `plant.simulate` to `sysm.plant_simulate`. **Legitimate:** the integrator call moved, the test's intent (state assigned only by the plant) is unchanged, and the new name is still checked.
2. **Table e3 solve-time statistic** changed from mean to median, first call excluded, computed from stored logs. **Borderline:** the cause was found and fixed in the code, but the published table was repaired after the fact, and the statistic changed from mean to median. Cleaner: rerun E3 with the warm-up (about 20 minutes) and report the run's own mean and median (§6, step 2).
3. **E12 aborted and restarted** after the thread fix. No results were discarded selectively: all four runs were deleted and repeated.
4. **TF 2.21 reproduction threshold.** A 1% tolerance was set before the run; the run missed it (4.5%). The tolerance was **not** relaxed after the fact to call it a pass: the difference is reported as it is, compared with the seed-to-seed spread, and the paper states it.

Earlier:
- **E6 v1:** the v_x finding was not fed back into v1 (correct per ground rule 9); it led to E8 and v2.
- **Checkpoint selection on total validation loss** (`select_on: total`) means PINC and data-only are selected on different quantities. **Moot in practice:** every best checkpoint is the last iteration (E11).

No case found where a test was loosened to make code pass.

---

## 4. What could have shown a meaningfully better result

| change | expected effect | evidence |
|---|---|---|
| Train to (near) convergence | 5–20× lower open-loop errors for both arms; the PINC / data-only ratio unchanged (~1.4 at N = 20k); closed loop essentially unchanged (already at the exact-model ceiling) | E11 |
| 6×256 instead of 8×64 | ~3× lower validation loss at every budget; costs MPC solve time (11× parameters) | E11, E6 v2 |
| **Imperfect physics prior** (truth ≠ physics in the loss) | The only design where PINC can beat both the data-only network and NMPC-with-the-prior, and where a surrogate's speed matters (stiff true model). The current study cannot show this by construction: NMPC-RK4 with the exact model is as accurate and as fast at N ≤ 10 | `docs/PLAN_HIGH_FIDELITY.md` |
| Grey-box baseline (prior + learned correction) | The obvious competitor a reviewer will ask for; missing from the current paper | — |
| More seeds for E6–E9 | E6–E9 are single-seed; only E2, E3, E5 and E10 are multi-seed | — |
| Timing with the warm-up and the same thread setting in E3 and E4 | Removes the contradiction at the source | §2c #11 |

Not worth doing: `float32` (breaks L-BFGS), XLA in `float64` (100× slower), larger Adam batches (Adam is ~10% of training time).

---

## 5. Where things stand
- **Branch** `gpu-env-convergence`.
- **Environment:** Python 3.13 + TF 2.21.0 (stable), GPU working; the TF 2.22rc0 environment is kept as `.venv-tf222rc0`.
- **Untracked by choice:** `LATEX/` (manuscript, bibliography, `submission.zip`) and the paper/poster generator scripts. **Risk:** not under version control.
- **Phase 1 of the plan:** done (refactor, scheduler, E12). Close-out steps below; then Phase 2.

---

## 6. Next steps (revised)

**Close-out, before Phase 2:**
1. **TF 2.21 check: done** (§2c #14). README install section and the paper's software line state both versions and the measured difference.
2. **E3 rerun with the warm-up** (`e3_v3`, TF 2.21, CPU only, single thread, same models): regenerate the paper tables and recompile, so Table e3 comes from a clean run.
3. **Decide whether to version-control `LATEX/` and the generator scripts** (e.g. a private branch or separate private repo). Your call; flagged because paper edits now exist only on disk.

**Phase 2 onwards, with Phase 1 learnings folded in:**

| item in `docs/PLAN_HIGH_FIDELITY.md` | revision |
|---|---|
| Training budget | 300 Adam epochs, **constant** learning rate (cosine gave no benefit, E11) + **2000** L-BFGS; `float64`; report as a fixed budget with the E11 curve as justification |
| Collocation | Start at **10k** points (E12: no measurable loss vs 20k, ~20% cheaper L-BFGS), half log-uniform in t near 0 for the wheel-slip transient; re-check with an E12-style run on the new plant |
| Compute | `--slots gpu,cpu` (1.8× throughput); never stack jobs on one device |
| Architecture (H8) | Include 6×256 from the start (E11); 3 seeds for the top candidates; weigh MPC solve time |
| Scales | Generalise `scripts/compute_scales.py` to the system (S_f for 10 states; S_x for the slip-velocity coordinates) |
| Timing (H6) | Warm-up before timing in every experiment; CPU only (GPU hidden) and single-thread for all arms; same setting wherever times are compared |
| Baselines | Grey-box arm kept (§5 of the plan) |
| Phases | Phase 2 = high-fidelity plant (NumPy + TF) and its acceptance tests; Phase 3 = prior + prior-error map (H1); then data/training for n_s = 10, H8, open-loop H2–H4 and H7, closed-loop H5–H6; paper framing decided after results (studies kept separate until then) |

Study 1 (matched physics, current paper) is **not** retrained with the converged budget unless you decide to frame the paper around both studies. The E11 result (ratio unchanged) is enough to state the budget caveat honestly.

---
