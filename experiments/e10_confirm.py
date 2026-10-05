"""
E10 -- Multi-seed confirmation of the rescaled (increment-scaled) PINC before any closed-loop rerun.

lambda* = the lambda > 0 with the lowest VALIDATION data loss among the increment-scaled E8 runs.
PINC (increment scaling, lambda*) vs its data-only counterpart (increment scaling, lambda = 0), 5 seeds
(seed k changes the training draw and the initialisation), at N = 20 000 (default budget) and N = 100
(E2 budget).  Every model then goes through the E9 physics-learning checks.  Reported per metric:
mean [95% CI] over seeds, paired ratio, paired t-test on log values and Wilcoxon signed-rank
(with 5 pairs the smallest attainable two-sided Wilcoxon p is 0.0625, so the t-test is the primary test).
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e8_vx_residual import run_jobs, summary  # noqa: E402
from pinc.config import RESULTS_DIR  # noqa: E402
from pinc.jobs import add_slot_args  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
E2_BUDGET = {"train.steps": 6000, "train.lbfgs_iters": 500, "train.batch_data": 100, "train.val_every": 20}


def pick_lambda(e8_run):
    s = json.load(open(os.path.join(RESULTS_DIR, "e8_vx_residual", e8_run, "summary.json")))["phase1"]
    cand = {t: r for t, r in s.items() if t.startswith("inc_") and r["overrides"]["loss.lam"] > 0}
    best = min(cand, key=lambda t: cand[t]["val_data"])
    return cand[best]["overrides"]["loss.lam"], {t: r["val_data"] for t, r in cand.items()}


def rid(lam, n, seed):
    if n == 20000 and seed == 0:
        return f"vx_inc_lam{lam:g}_s0"
    return f"vxc_inc_lam{lam:g}_n{n}_s{seed}"


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--e8-run", default="e8b")
    add_slot_args(ap)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e10_confirm", a)
    lam, val_by_lam = pick_lambda(a.e8_run)
    print(f"  lambda* = {lam:g} (validation data loss by lambda: {val_by_lam})", flush=True)
    jobs = []
    for n in (20000, 100):
        for l in (lam, 0.0):
            for k in SEEDS:
                ov = {"loss.lam": l, "model.increment_scaling": "true", "train.n_data": n, "seeds.train": cfg.seeds.train + 100*k}
                if n == 100:
                    ov.update(E2_BUDGET)
                jobs.append((rid(l, n, k), k, ov))
    run_jobs(jobs, a)

    from experiments import e9_physics_learning as e9
    models = ",".join(f"{r}={r}" for r, _, _ in jobs)
    e9.main(["--run-id", os.path.basename(run_dir) + "_e9", "--models", models])
    e9s = json.load(open(os.path.join(RESULTS_DIR, "e9_physics_learning", os.path.basename(run_dir) + "_e9", "summary.json")))["models"]

    metrics = {
        "validation data loss": lambda r: summary(r)["val"]["data"],
        "test NRMSE (all states)": lambda r: float(np.sqrt(np.mean(np.square(summary(r)["test"]["nrmse"])))),
        "dx/dt error vs truth": lambda r: e9s[r]["deriv_err_all"],
        "physics residual": lambda r: e9s[r]["residual_all"],
        "10-step error, in-domain": lambda r: e9s[r]["horizon"]["in_domain"]["10"]["mean"],
        "50-step error, in-domain": lambda r: e9s[r]["horizon"]["in_domain"]["50"]["mean"],
        "50-step error, extrapolation": lambda r: e9s[r]["horizon"]["extrap"]["50"]["mean"],
    }
    out, text = {}, f"# E10 multi-seed confirmation: PINC (increment scaling, lambda = {lam:g}) vs data-only (lambda = 0), {len(SEEDS)} seeds\n\n"
    for n in (20000, 100):
        rows = []
        out[n] = {}
        for name, fn in metrics.items():
            p = np.array([fn(rid(lam, n, k)) for k in SEEDS])
            b = np.array([fn(rid(0.0, n, k)) for k in SEEDS])
            ratio = b/p
            t = stats.ttest_rel(np.log(p), np.log(b))
            w = stats.wilcoxon(p, b)
            out[n][name] = dict(pinc=p.tolist(), blackbox=b.tolist(), pinc_ci=bootstrap_ci(p), bb_ci=bootstrap_ci(b),
                                ratio_geomean=float(np.exp(np.mean(np.log(ratio)))), pinc_better=int(np.sum(p < b)),
                                t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue))
            o = out[n][name]
            rows.append([name, f"{o['pinc_ci']['mean']:.2e} [{o['pinc_ci']['lo']:.2e}, {o['pinc_ci']['hi']:.2e}]",
                         f"{o['bb_ci']['mean']:.2e} [{o['bb_ci']['lo']:.2e}, {o['bb_ci']['hi']:.2e}]",
                         f"{o['ratio_geomean']:.2f}", f"{o['pinc_better']}/{len(SEEDS)}", f"{o['t_p']:.2g}", f"{o['wilcoxon_p']:.2g}"])
        text += (f"## N = {n} training trajectories\n\n" +
                 md_table(["metric (lower is better)", "PINC", "data-only", "data-only / PINC (geo. mean)", "seeds PINC better",
                           "paired t (log) p", "Wilcoxon p"], rows) + "\n")
    art = write_text(os.path.join(run_dir, "table_confirm.md"), text)
    finish(run_dir, cfg, a.seed, dict(lam_star=lam, val_by_lambda=val_by_lam, results={str(k): v for k, v in out.items()}), [art])
    print(text)


if __name__ == "__main__":
    main()
