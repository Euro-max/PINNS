"""
E2 -- Does physics help with little data?
Train-set sizes {1e2, 1e3, 1e4, 1e5} (quick: {1e2, 1e3, 1e4}), 5 seeds each
(quick: 2), PINC (lambda = lambda*) vs lambda = 0, fixed optimiser budget
(train.steps Adam steps + L-BFGS).  Figure: test NRMSE vs N (log-log).
Seeds vary both the training-data draw and the weight initialisation; the
validation / test sets are fixed.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import COLORS, base_parser, ci_str, finish, md_table, savefig, start, write_text, plt  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402

STATES = ("vx", "vy", "r", "psi")


def _summary(run_id):
    p = os.path.join(RESULTS_DIR, "models", run_id, "summary.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def train_many(jobs, args, workers):
    """jobs: list of (run_id, seed, overrides dict).  Missing runs are trained as
    subprocesses, `workers` at a time (same code path as `python -m pinc.train`)."""
    pending = [j for j in jobs if _summary(j[0]) is None]
    print(f"  {len(jobs) - len(pending)} runs reused, {len(pending)} to train ({workers} workers)", flush=True)
    logdir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(logdir, exist_ok=True)
    running = []
    env = dict(os.environ, PYTHONPATH=ROOT)
    while pending or running:
        while pending and len(running) < workers:
            rid, seed, ov = pending.pop(0)
            c = [sys.executable, "-m", "pinc.train", "--config", args.config, "--seed", str(seed), "--run-id", rid]
            for k, v in ov.items():
                c += ["--set", f"{k}={v}"]
            for o in args.overrides:
                c += ["--set", o]
            log = open(os.path.join(logdir, rid + ".log"), "w")
            running.append((rid, subprocess.Popen(c, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env), log))
        for item in list(running):
            rid, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"  finished {rid} (exit {p.returncode})", flush=True)
                if p.returncode != 0:
                    raise RuntimeError(f"run {rid} failed; see {log.name}")
        time.sleep(2)


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--lam", type=float, default=None, help="lambda for the PINC arm (default: config)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e2_data_efficiency", a)
    lam = cfg.loss.lam if a.lam is None else a.lam
    sizes = [100, 1000, 10000] if a.quick else [100, 1000, 10000, 100000]
    seeds = [0, 1] if a.quick else [0, 1, 2, 3, 4]
    budget = {"train.steps": 400, "train.lbfgs_iters": 30, "train.log_every": 1000} if a.quick else \
             {"train.steps": 6000, "train.lbfgs_iters": 500, "train.log_every": 1000}
    arms = {"pinc": lam, "blackbox": 0.0}
    res = {arm: {n: [] for n in sizes} for arm in arms}
    jobs = []
    for n in sizes:
        for seed in seeds:
            for arm, lam_a in arms.items():
                steps_per_epoch = int(np.ceil(n/min(1024, n)))
                ov = {**budget, "train.n_data": n, "loss.lam": lam_a, "seeds.train": cfg.seeds.train + 100*seed,
                      "train.batch_data": min(1024, n),
                      "train.val_every": max(1, 20//steps_per_epoch)}     # validate every ~20 Adam steps
                rid = (f"e2_{arm}_n{n}_lam{lam_a:g}_s{seed}" + ("_inc" if cfg.model.increment_scaling else "") +
                       ("_quick" if a.quick else ""))
                jobs.append((rid, seed, ov))
    train_many(jobs, a, a.workers)
    for rid, seed, ov in jobs:
        s = _summary(rid)
        arm = "pinc" if rid.startswith("e2_pinc") else "blackbox"
        n = ov["train.n_data"]
        res[arm][n].append(dict(seed=seed, test=s["test"]["nrmse"], extrap=s["test_extrap"]["nrmse"],
                                val_data=s["val"]["data"], run_id=rid))
        print(f"  n={n:6d} seed={seed} {arm:8s} test NRMSE all={np.sqrt(np.mean(np.square(s['test']['nrmse']))):.3e}")

    summary = dict(quick=a.quick, lam=lam, sizes=sizes, seeds=seeds, budget=budget, curves={})
    rows = []
    for arm in arms:
        summary["curves"][arm] = {}
        for n in sizes:
            allv = [np.sqrt(np.mean(np.square(r["test"]))) for r in res[arm][n]]
            ex = [np.sqrt(np.mean(np.square(r["extrap"]))) for r in res[arm][n]]
            per = {s: bootstrap_ci([r["test"][i] for r in res[arm][n]]) for i, s in enumerate(STATES)}
            summary["curves"][arm][n] = dict(all=bootstrap_ci(allv), extrap=bootstrap_ci(ex), per_state=per, runs=res[arm][n])
            rows.append([arm, n, ci_str(summary["curves"][arm][n]["all"], "{:.2e}"), ci_str(summary["curves"][arm][n]["extrap"], "{:.2e}")] +
                        [ci_str(per[s], "{:.2e}") for s in STATES])
    art = write_text(os.path.join(run_dir, "table_data_efficiency.md"),
                     f"# E2 test NRMSE vs training-set size (lambda = {lam:g}; mean [95% CI over seeds])\n\n" +
                     md_table(["arm", "N", "all (in-box)", "all (extrap)"] + list(STATES), rows))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for arm in arms:
        m = [summary["curves"][arm][n]["all"]["mean"] for n in sizes]
        lo = [summary["curves"][arm][n]["all"]["lo"] for n in sizes]
        hi = [summary["curves"][arm][n]["all"]["hi"] for n in sizes]
        ax.errorbar(sizes, m, yerr=[np.array(m) - lo, np.array(hi) - m], marker="o", capsize=3, color=COLORS[arm],
                    label=f"PINC (lambda={lam:g})" if arm == "pinc" else "black-box (lambda=0)")
    ax.set(xscale="log", yscale="log", xlabel="training trajectories N", ylabel="test NRMSE (all states)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    arts = [art] + savefig(fig, run_dir, "fig_nrmse_vs_n")
    finish(run_dir, cfg, a.seed, summary, arts, dict(quick=a.quick))


if __name__ == "__main__":
    main()
