# PINC-MPC for a single-track vehicle

A physics-informed neural network for control (PINC, Antonelo et al., arXiv 2104.02556) that maps
`(t, s0, u) -> s(t)` for a six-state single-track vehicle model, used as the prediction model of a
nonlinear MPC and benchmarked against NMPC with the exact RK4 model, a black-box network (λ = 0)
and an LTV-MPC.  Every number and figure for the paper is produced by a script in this repository
and recorded in `results/MANIFEST.md`.

The previous implementation (kept untouched in `legacy/`) had defects that invalidated its results;
`docs/DEFECTS.md` lists all twenty and where each is closed.  `TASKS.md` is the rebuild
specification; `paper/CORRECTIONS.md` is the hand-over list for the manuscript.

## Layout

```
pinc/            library: config, plant (NumPy + TF), data, model, loss, train, mpc, refs, sim, metrics
experiments/     e1_open_loop ... e6_ablations, e7_architecture, e8_vx_residual, e9_physics_learning, e10_confirm
tests/           pytest suite (plant, data, model, loss, mpc, sim)
configs/         default.yaml -- the single source of truth for parameters, scales, seeds
scripts/         compute_scales.py, train_lambda_sweep.sh, run_all.sh
results/         generated only; each run folder has config.json, meta.json (seed, git hash, versions), summary.json
legacy/          original files, unchanged, imported by nothing
docs/, paper/    defect table, model derivation, architecture trials, results summary, paper corrections
LATEX/           manuscript (main.tex); LATEX/generated/ holds tables, figures and number macros
                 produced by scripts/make_paper_tables.py from results/ (main_original.tex is the old draft)
```

## Install

Python 3.13 with TensorFlow 2.21.0, the latest stable release (it does not support Python 3.14);
pinned versions in `requirements.txt`.

```
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If the system Python has no `venv`/`pip` module (e.g. WSL without `python3-venv`), use
[uv](https://docs.astral.sh/uv/) instead.  For an NVIDIA GPU add TensorFlow's CUDA extra:

```
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt "tensorflow[and-cuda]==2.21.0"
```

`pinc/__init__.py` sets `TF_FORCE_GPU_ALLOW_GROWTH=true` (TensorFlow's default up-front
reservation of the whole GPU fails under WSL2) and preloads `libcusolver` from the
`nvidia-cusolver-cu12` wheel (TensorFlow 2.21's pip build does not search that folder and
otherwise silently runs without the GPU).  Training uses the GPU or the CPU equally well in
float64; MPC solve times must be measured with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`): the
batch-1 XLA-compiled float64 cost is about 40x slower on the GPU.  Keep `dtype: float64` on the GPU as well: in
float32 the L-BFGS line search stalls after a few iterations at loss levels of ~1e-6 and the
final model is ~5x worse on validation (`results/env_check/`).

## Test

```
.venv/bin/python -m pytest -q
```

`tests/test_sim.py` contains the most important test: a controller that always returns `u = 0`
and a prediction model that returns constant garbage must produce a *large* tracking error.  A
second test greps `pinc/` and `experiments/` for any assignment of the plant state from reference
data.

## Train

```
.venv/bin/python -m pinc.train --config configs/default.yaml --seed 0 --run-id pinc_default_s0
.venv/bin/python -m pinc.train --seed 0 --run-id blackbox_default_s0 --set loss.lam=0
```

Any config entry can be overridden with `--set a.b=value`.  Output goes to
`results/models/<run-id>/` (weights, `config.json`, `meta.json`, `loss_curve.csv`, `summary.json`
with validation loss and test / extrapolation NRMSE).  Training is Adam with cosine decay followed
by L-BFGS (SciPy); collocation points are resampled every epoch; the checkpoint with the best
validation loss is restored before saving.

## Experiments

```
.venv/bin/python -m experiments.e7_architecture     # depth/width/dropout/residual/layer-norm x λ grid (56 trials)
.venv/bin/python -m experiments.e6_ablations        # λ sweep + ablations at the config λ
.venv/bin/python -m experiments.e1_open_loop        # surrogate accuracy, one-step and chained
.venv/bin/python -m experiments.e2_data_efficiency  # test NRMSE vs training-set size
.venv/bin/python -m experiments.e3_closed_loop      # 4 references x 30 seeds x 4 arms
.venv/bin/python -m experiments.e4_timing           # solve time vs horizon, CPU single thread
.venv/bin/python -m experiments.e5_robustness       # plant mismatch, controller nominal
.venv/bin/python -m experiments.e8_vx_residual      # residual rescaling variants (led to model.increment_scaling)
.venv/bin/python -m experiments.e9_physics_learning # residual, derivative error and horizon error per model
.venv/bin/python -m experiments.e10_confirm         # 5-seed PINC vs data-only confirmation
```

The current (v2, increment-scaled) results were produced by `scripts/run_queue_v2.sh` plus E8-E10;
see `docs/RESULTS.md`.

Every script accepts `--quick` (reduced sizes, results labelled `quick_*`), `--run-id`, `--seed`,
`--set key=value`, and `--pinc-model` / `--blackbox-model` (default: `results/models/pinc_default_s0`
and `results/models/blackbox_default_s0`).  `scripts/run_all.sh` (or `make all`) reproduces
everything in `results/`; `QUICK=1 scripts/run_all.sh` (or `make quick`) is the smoke version.

## Design points

- One plant (`pinc/plant.py`, fixed-step RK4, dt = 1 ms) for every controller and experiment; the
  same dynamics in TensorFlow (`pinc/plant_tf.py`) give the physics residual and the NMPC baseline.
- Controllers get the nominal parameters only; perturbed parameters and disturbances go to the
  plant in `pinc/sim.py`.
- Network inputs `[t/T, s0/S_x, u/S_u]`, hard initial condition `s(t) = s0 + (t/T)·NN`, no clipping.
  Options in `model.*`: depth, width, `residual: none|skip|block` (ResNet-style two-layer blocks),
  `dropout`, `layernorm`, `hard_ic`.  The defaults were chosen on the validation set by E7 / E6;
  every trial is listed in `results/e7_architecture/e7/table_trials.md` and `docs/ARCHITECTURE_TRIALS.md`.
- Time derivative by forward-mode autodiff along the time input, divided by `T`, scaled by `S_x`;
  residual `(ds/dt - f(s,u)) / S_f` with all four states.
- MPC cost and gradient in one XLA-compiled TF call; SciPy SLSQP with `jac=True`, identical
  options and warm start for every predictor; braking allowed (`Fx ∈ [-6000, 3000]` N).
- Errors are scored after each plant step against the reference at the new time; IAE = Σ|e|·T.

## Hardware and runtimes

See `meta.json` in each results folder for the CPU model and package versions of the run that
produced it.  The results in `results/` were produced on an Intel Core Ultra 9 275HX (6 vCPUs in a
VM, 19 GB RAM, no GPU; TensorFlow reports no GPU, so E4 has CPU numbers only) with the wall times
below.  Training runs that ran three at a time (E7, E2) were slowed by contention.

| step | wall time |
|---|---|
| `pytest` (49 tests) | ~1.5 min |
| one default model (8x64, 300 Adam epochs + 500 L-BFGS) | ~4-8 min |
| E7 architecture study (56 trainings, 3 workers) | ~9 h |
| E6 ablations (13 trainings, 3 reused from E7) | 64 min |
| E1 open loop | 1 min |
| E4 timing | 3 min |
| E3 closed loop (4 refs x 30 seeds x 4 arms) | 17.5 min |
| E5 robustness (9 settings x 30 seeds x 4 arms) | 40 min |
| E2 data efficiency (40 trainings, 3 workers) | 3.5 h |

On an RTX 5080 Laptop GPU (WSL2, float64) one default model takes 223 s against 461 s on the CPU
above, with the same validation loss to 0.5 % (`results/env_check/`); the 8x64 network is too small
to keep the GPU busy, so the speed-up is modest.

## Reproducibility caveat

Seeds fix the data, the initialisation and the collocation resampling, and every run records its
seed, git hash and package versions.  A rerun on the same machine and software reproduces a
model bit for bit (`results/phase1_check`), but a different TensorFlow version or device changes
the last floating-point bit of some evaluations.  Adam keeps such differences at ~1e-12; L-BFGS
amplifies them: retraining `pinc_v2_s0` with TensorFlow 2.21.0 instead of 2.22.0rc0 changed the
validation data loss by 4.5 % (`results/tf221_check`), within the 9.6 % spread over training seeds
(E10).  The published v2 results were produced with TensorFlow 2.22.0rc0.  Report numbers with the
confidence intervals the experiment scripts produce, not as exact values.
