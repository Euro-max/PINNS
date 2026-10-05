"""
E15 (plan H2/H3) -- Does the imperfect prior help with little data?

High-fidelity system (default M0), training sets of N in {100, 1000, 20 000} trajectories, 3 seeds
(each changes the training draw and the initialisation; validation / test sets are fixed),
lambda in {0, 1e-4, 1e-3} without the wheel residuals (E14: they conflict with the data).  The same
optimiser budget for every N: 6000 Adam steps (constant learning rate) + 2000 L-BFGS iterations.
Reported, mean [95% CI] over seeds: validation data loss, one-step test NRMSE by state group, and the
chained 10- / 50-step body-state error (E9) on validation and on test trajectories.  lambda is selected
per N on the VALIDATION 50-step body-state error (long-horizon accuracy is what the MPC needs; E9 on the
E14 models showed one-step and multi-step accuracy can disagree); the validation-data-loss choice is
reported alongside.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, NO_WHEEL, group_rms  # noqa: E402
from pinc.config import ROOT, load_config  # noqa: E402
from pinc.data import make_splits, sample_collocation, sample_ic  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402

SIZES = (100, 1000, 20000)
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2)
SEEDS = (0, 1, 2)


def overrides(n, lam, seed, cfg):
    spe = int(np.ceil(n/min(1024, n)))
    ov = {"train.n_data": n, "train.steps": 6000, "train.batch_data": min(1024, n), "train.val_every": max(1, 20//spe),
          "loss.lam": lam, "seeds.train": cfg.seeds.train + 100*seed}
    if lam > 0:
        ov["loss.residual_mask"] = str(NO_WHEEL).replace(" ", "")
    return ov


def prewarm(cfg, jobs):
    """Generate every driving-state set the parallel jobs will read, so no two jobs write one cache file."""
    seen = set()
    for _, _, ov in jobs:
        c = cfg.with_overrides({k: v for k, v in ov.items() if k in ("train.n_data", "seeds.train")})
        key = (c.train.n_data, c.seeds.train)
        if key in seen:
            continue
        seen.add(key)
        make_splits(c)
        sample_ic(min(c.train.n_data, 4000), c.seeds.train + 1000, c)
    sample_ic(min(cfg.train.n_val, 2000), cfg.seeds.val + 1000, cfg)
    sample_collocation(cfg.train.n_val, cfg.seeds.val + 2000, cfg)


def ci_cell(v):
    c = bootstrap_ci(v)
    return f"{c['mean']:.2e} [{c['lo']:.2e}, {c['hi']:.2e}]"


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    _, run_dir = start("e15_hf_data", a)
    cfg = load_config(cfg_path, a.overrides)
    jobs = [(f"hf{a.variant}_n{n}_lam{lam:g}_s{seed}", seed, overrides(n, lam, seed, cfg))
            for n in SIZES for lam in LAMBDAS for seed in SEEDS]
    print("  prewarming the driving-state cache", flush=True)
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    import json
    from experiments import e9_physics_learning as e9
    from pinc.config import RESULTS_DIR
    models = ",".join(f"{rid}={rid}" for rid, _, _ in jobs)
    ev = {}
    for split in ("val", "test"):
        e9_id = f"{os.path.basename(run_dir)}_e9{split}"
        e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", models, "--split", split])
        with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
            ev[split] = json.load(fh)["models"]
    rec, rows, best, best_onestep = {}, [], {}, {}
    for n in SIZES:
        for lam in LAMBDAS:
            runs = [summary(f"hf{a.variant}_n{n}_lam{lam:g}_s{seed}") for seed in SEEDS]
            rids = [f"hf{a.variant}_n{n}_lam{lam:g}_s{seed}" for seed in SEEDS]
            r = dict(val_data=[s["val"]["data"] for s in runs],
                     h50_val=[ev["val"][i]["horizon_body"]["in_domain"]["50"]["mean"] for i in rids],
                     h10_test=[ev["test"][i]["horizon_body"]["in_domain"]["10"]["mean"] for i in rids],
                     h50_test=[ev["test"][i]["horizon_body"]["in_domain"]["50"]["mean"] for i in rids],
                     h50_extrap=[ev["test"][i]["horizon_body"]["extrap"]["50"]["mean"] for i in rids],
                     test={g: [group_rms(s["test"]["nrmse"], i) for s in runs] for g, i in GROUPS.items()},
                     extrap={g: [group_rms(s["test_extrap"]["nrmse"], i) for s in runs] for g, i in GROUPS.items()})
            rec[f"{n}_{lam:g}"] = r
            rows.append([n, f"{lam:g}", ci_cell(r["val_data"]), ci_cell(r["h50_val"])] + [ci_cell(r["test"][g]) for g in GROUPS] +
                        [ci_cell(r["h10_test"]), ci_cell(r["h50_test"]), ci_cell(r["h50_extrap"])])
        best[n] = min(LAMBDAS, key=lambda lam: np.mean(rec[f"{n}_{lam:g}"]["h50_val"]))
        best_onestep[n] = min(LAMBDAS, key=lambda lam: np.mean(rec[f"{n}_{lam:g}"]["val_data"]))
    text = (f"# E15 data efficiency on HF-{a.variant.upper()} (lambda > 0 without wheel residuals; mean [95% CI] over "
            f"{len(SEEDS)} seeds)\n\nselected lambda (validation 50-step body error): " +
            ", ".join(f"N = {n}: {best[n]:g}" for n in SIZES) + "; by validation data loss instead: " +
            ", ".join(f"N = {n}: {best_onestep[n]:g}" for n in SIZES) + "\n\n" +
            md_table(["N", "lambda", "val data", "val 50-step body"] + [f"test 1-step {g}" for g in GROUPS] +
                     ["test 10-step body", "test 50-step body", "extrap 50-step body"], rows))
    art = write_text(os.path.join(run_dir, "table_hf_data.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, best_lambda={str(k): v for k, v in best.items()},
                                     best_lambda_onestep={str(k): v for k, v in best_onestep.items()}, runs=rec), [art])
    print(text)


if __name__ == "__main__":
    main()
