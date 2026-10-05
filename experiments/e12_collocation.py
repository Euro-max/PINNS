"""
E12 -- Can the L-BFGS stage use fewer collocation points?

Every L-BFGS iteration evaluates the loss on the full training set plus a fixed set of
collocation points, so its cost grows with `train.n_colloc`.  This study trains the default
setting (8x64, increment-scaled loss, lambda = 0.01, N = 20 000, seed 0) with the budget fixed
after E11 (300 Adam epochs + 2000 L-BFGS iterations) at n_colloc in {20 000, 10 000, 5 000, 2 500}
and reports validation data / physics loss, test NRMSE and training time.  The validation
physics loss is measured on the same 4 000 fresh collocation points for every run.
Jobs run on the device slots of `--slots` (pinc/jobs.py); the slot of each run is recorded.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e8_vx_residual import allnrmse  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

BUDGET = {"train.lbfgs_iters": 2000, "train.log_every": 100}
N_COLLOC = (20000, 10000, 5000, 2500)


def rid(n, seed):
    return f"colloc_n{n}_s{seed}"


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e12_collocation", a)
    jobs = [(rid(n, a.seed), a.seed, {**BUDGET, "train.n_colloc": n, "train.batch_colloc": min(1024, n)})
            for n in N_COLLOC]
    ran = run_jobs(jobs, a.slots, a.config, a.overrides, cpu_threads=a.cpu_threads)
    rec, rows = {}, []
    base = summary(rid(N_COLLOC[0], a.seed))
    for n in N_COLLOC:
        s = summary(rid(n, a.seed))
        slot, wall = ran.get(rid(n, a.seed), ("reused", float("nan")))
        rec[n] = dict(run_id=rid(n, a.seed), val_data=s["val"]["data"], val_phys=s["val"]["phys"],
                      test_all=allnrmse(s["test"]["nrmse"]), extrap_all=allnrmse(s["test_extrap"]["nrmse"]),
                      train_seconds=s["train_seconds"], slot=slot, wall=wall)
        r = rec[n]
        rows.append([n, f"{r['val_data']:.2e}", f"{r['val_data']/base['val']['data']:.2f}", f"{r['val_phys']:.2e}",
                     f"{r['test_all']:.2e}", f"{r['extrap_all']:.2e}", f"{r['train_seconds']:.0f}", r["slot"]])
    text = ("# E12 collocation count for the fixed budget (300 Adam + 2000 L-BFGS; seed "
            f"{a.seed}; lambda = {cfg.loss.lam:g})\n\n" +
            md_table(["n_colloc", "val data", "val data / 20k", "val phys", "test NRMSE all", "extrap NRMSE all",
                      "train s", "slot"], rows))
    art = write_text(os.path.join(run_dir, "table_collocation.md"), text)
    finish(run_dir, cfg, a.seed, dict(runs=rec, budget=BUDGET, slots=a.slots), [art])
    print(text)


if __name__ == "__main__":
    main()
