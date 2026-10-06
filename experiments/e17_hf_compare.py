"""
E17 -- Five-seed comparison of the learned models on the high-fidelity system, with the grey-box baseline.

Arms (same network, data, optimiser budget and seeds; only the loss or the target differs):
  data-only   lambda = 0
  PINC        lambda selected by E15 on the validation 50-step body error (M0); on M1, where E15 selected
              lambda = 0, the E16 setting lambda = 1e-3
  PINC-theta  M1 only: lambda = 1e-3 with the prior's parameters learnable (E16)
  grey-box    the prior's own prediction plus a learned correction, data only (pinc/greybox.py)
Seeds 0-4 (E15 / E16 trained 0-2; those runs are reused).  Sizes: M0 100, 1000, 20 000; M1 100, 1000.
Reported, mean [95% CI] over seeds: one-step test NRMSE (body states), chained 10- and 50-step body-state
error on test trajectories (E9) and the 50-step error outside the training range; each arm against
data-only and PINC against grey-box as the ratio of means, the number of seeds won and a paired t-test
on log errors.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, group_rms  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides, prewarm  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
SIZES = dict(m0=(100, 1000, 20000), m1=(100, 1000))


def arms_for(variant, n, lam_star):
    """{arm: (run id template with {seed}, extra overrides, lambda)}"""
    v = f"hf{variant}_n{n}"
    lam = lam_star[n] if variant == "m0" else 1e-3
    out = {"data-only": (v + "_lam0_s{seed}", {}, 0.0),
           "PINC": (v + f"_lam{lam:g}" + "_s{seed}", {}, lam)}
    if variant == "m1":
        out["PINC-theta"] = (v + f"_lam{lam:g}" + "_theta_s{seed}", {"model.learn_theta": "true"}, lam)
    out["grey-box"] = (v + "_greybox_s{seed}", {"model.greybox": "true"}, 0.0)
    return out


def compare(a, b):
    """a against b (lower is better): ratio of means b/a, seeds where a < b, paired t-test on logs."""
    a, b = np.asarray(a), np.asarray(b)
    p = stats.ttest_rel(np.log(a), np.log(b)).pvalue if len(a) > 1 else np.nan
    return dict(ratio=float(np.mean(b)/np.mean(a)), wins=int(np.sum(a < b)), n=len(a), p=float(p))


def cmp_cell(c):
    return f"{c['ratio']:.2f} ({c['wins']}/{c['n']}, p = {c['p']:.2g})"


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    ap.add_argument("--no-train", action="store_true", help="evaluate finished runs only")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    _, run_dir = start("e17_hf_compare", a)
    cfg = load_config(cfg_path, a.overrides)
    with open(os.path.join(RESULTS_DIR, "e15_hf_data", "e15_m0", "summary.json")) as fh:
        lam_star = {int(k): v for k, v in json.load(fh)["best_lambda"].items()}
    sizes = SIZES[a.variant]
    plan = {n: arms_for(a.variant, n, lam_star) for n in sizes}
    jobs = []
    for n in sizes:
        for arm, (tmpl, extra, lam) in plan[n].items():
            for seed in SEEDS:
                jobs.append((tmpl.format(seed=seed), seed, dict(overrides(n, lam, seed, cfg), **extra)))
    if not a.no_train:
        prewarm(cfg, jobs)
        run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)

    from experiments import e9_physics_learning as e9
    models = ",".join(f"{rid}={rid}" for rid, _, _ in jobs)
    e9_id = f"{os.path.basename(run_dir)}_e9test"
    e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", models, "--split", "test"])
    with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
        ev = json.load(fh)["models"]

    metrics = dict(one_step=("one-step body", lambda rid: group_rms(summary(rid)["test"]["nrmse"], GROUPS["body"])),
                   h10=("10-step body", lambda rid: ev[rid]["horizon_body"]["in_domain"]["10"]["mean"]),
                   h50=("50-step body", lambda rid: ev[rid]["horizon_body"]["in_domain"]["50"]["mean"]),
                   h50_extrap=("50-step body, outside training range",
                               lambda rid: ev[rid]["horizon_body"]["extrap"]["50"]["mean"]))
    rec, rows, crows = {}, [], []
    for n in sizes:
        rec[n] = {arm: {m: [fn(tmpl.format(seed=s)) for s in SEEDS] for m, (_, fn) in metrics.items()}
                  for arm, (tmpl, _, _) in plan[n].items()}
        for arm, (_, _, lam) in plan[n].items():
            rows.append([n, arm, f"{lam:g}"] + [ci_cell(rec[n][arm][m]) for m in metrics])
        pairs = [(arm, "data-only") for arm in plan[n] if arm != "data-only"] + [("PINC", "grey-box")]
        if "PINC-theta" in plan[n]:
            pairs.append(("PINC-theta", "grey-box"))
        for x, y in pairs:
            c = {m: compare(rec[n][x][m], rec[n][y][m]) for m in metrics}
            rec[n][f"{x} vs {y}"] = c
            crows.append([n, f"{x} against {y}"] + [cmp_cell(c[m]) for m in metrics])
    labels = [lbl for lbl, _ in metrics.values()]
    text = (f"# E17 five-seed comparison on HF-{a.variant.upper()} (test errors, NRMSE; mean [95% CI] over "
            f"{len(SEEDS)} seeds)\n\n" + md_table(["N", "arm", "lambda"] + labels, rows) +
            "\n## Comparisons (error of the second arm divided by the error of the first, so above 1 means the first "
            "is better; seeds won; paired t-test on log errors)\n\n" + md_table(["N", "comparison"] + labels, crows))
    art = write_text(os.path.join(run_dir, "table_hf_compare.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, seeds=list(SEEDS), lam_star=lam_star,
                                     runs={str(n): {arm: tmpl for arm, (tmpl, _, _) in plan[n].items()} for n in sizes},
                                     results={str(n): v for n, v in rec.items()}), [art])
    print(text)


if __name__ == "__main__":
    main()
