"""
E16 (plan H4) -- Learnable prior parameters on the mismatched plant (M1).

For N in {100, 1000} and the lambda that E15 selected for M1 (validation 50-step body error), 3 seeds:
PINC with the prior's parameters learnable (model.learn_theta) against the same setting with the nominal
parameters fixed (the E15 runs, reused).  Reported: validation data loss and chained body-state errors
(E9, validation and test), and the learned parameters against M1's true values:
  per-axle cornering stiffness (MF at static load), slip stiffness (unchanged in M1), actuator lags.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e15_hf_data import SEEDS, ci_cell, overrides, prewarm  # noqa: E402
from pinc import plant_hf, tyre_mf  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402
from pinc.system import get_system  # noqa: E402

SIZES = (100, 1000)


def true_theta(sysm):
    p = sysm.truth
    Fz = plant_hf.static_loads(p)
    return dict(Caf=2*tyre_mf.cornering_stiffness(Fz[0], p["tyre"]), Car=2*tyre_mf.cornering_stiffness(Fz[2], p["tyre"]),
                C_kappa=tyre_mf.slip_stiffness(Fz[0], p["tyre"]), tau_F=p["tau_F"], tau_delta=p["tau_delta"])


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m1")
    ap.add_argument("--e15-run", default=None, help="E15 run to take lambda from (default e15_<variant>)")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    _, run_dir = start("e16_hf_theta", a)
    cfg = load_config(cfg_path, a.overrides)
    with open(os.path.join(RESULTS_DIR, "e15_hf_data", a.e15_run or f"e15_{a.variant}", "summary.json")) as fh:
        lam_star = {int(k): v for k, v in json.load(fh)["best_lambda"].items()}
    jobs, pairs = [], {}
    for n in SIZES:
        lam = lam_star[n] if lam_star[n] > 0 else 1e-3          # a physics weight is needed to learn anything
        for seed in SEEDS:
            ov = dict(overrides(n, lam, seed, cfg), **{"model.learn_theta": "true"})
            rid = f"hf{a.variant}_n{n}_lam{lam:g}_theta_s{seed}"
            jobs.append((rid, seed, ov))
            pairs[(n, seed)] = (rid, f"hf{a.variant}_n{n}_lam{lam:g}_s{seed}", f"hf{a.variant}_n{n}_lam0_s{seed}", lam)
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)

    from experiments import e9_physics_learning as e9
    models = ",".join(sorted({r for v in pairs.values() for r in v[:3]}))
    models = ",".join(f"{r}={r}" for r in models.split(","))
    ev = {}
    for split in ("val", "test"):
        e9_id = f"{os.path.basename(run_dir)}_e9{split}"
        e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", models, "--split", split])
        with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
            ev[split] = json.load(fh)["models"]

    truth = true_theta(get_system(cfg))
    nominal = get_system(cfg).theta_nominal()
    rows, trows, rec = [], [], {}
    for n in SIZES:
        lam = pairs[(n, SEEDS[0])][3]
        for arm, i in (("learned theta", 0), ("nominal theta", 1), ("data-only", 2)):
            rids = [pairs[(n, s)][i] for s in SEEDS]
            r = dict(val_data=[summary(x)["val"]["data"] for x in rids],
                     h50_val=[ev["val"][x]["horizon_body"]["in_domain"]["50"]["mean"] for x in rids],
                     h10_test=[ev["test"][x]["horizon_body"]["in_domain"]["10"]["mean"] for x in rids],
                     h50_test=[ev["test"][x]["horizon_body"]["in_domain"]["50"]["mean"] for x in rids])
            rec[f"{n}_{arm}"] = r
            rows.append([n, f"{lam:g}" if i < 2 else "0", arm] + [ci_cell(r[k]) for k in ("val_data", "h50_val", "h10_test", "h50_test")])
        for k in truth:
            learned = [summary(pairs[(n, s)][0])["theta"][k] for s in SEEDS]
            trows.append([n, k, f"{nominal[k]:.4g}", f"{truth[k]:.4g}", " / ".join(f"{v:.4g}" for v in learned),
                          f"{np.mean([(v - nominal[k])/(truth[k] - nominal[k]) for v in learned]):.2f}" if truth[k] != nominal[k] else "-"])
    text = (f"# E16 learnable prior parameters, HF-{a.variant.upper()} (mean [95% CI] over {len(SEEDS)} seeds)\n\n" +
            md_table(["N", "lambda", "arm", "val data", "val 50-step body", "test 10-step body", "test 50-step body"], rows) +
            "\n## Learned parameters (fraction of the nominal-to-true gap closed; 1 = true value)\n\n" +
            md_table(["N", "parameter", "nominal", "true", "learned (per seed)", "gap closed"], trows))
    art = write_text(os.path.join(run_dir, "table_hf_theta.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, lam_star=lam_star, truth=truth, nominal=nominal, runs=rec), [art])
    print(text)


if __name__ == "__main__":
    main()
