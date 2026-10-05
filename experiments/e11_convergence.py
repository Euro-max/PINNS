"""
E11 -- Is the training budget converged?

Every model trained so far (E2, E6-E10 and the default models) has its best validation checkpoint
at the last iteration of the 300-epoch Adam + 500-iteration L-BFGS budget, i.e. training was cut
off while still improving.  This study trains the current default setting (increment-scaled loss,
N = 20 000, seed 0) with much longer budgets and records where the validation data loss plateaus:

  long_8x64        8x64,  lambda = 0.01, Adam 300 epochs (cosine), L-BFGS 5000
  long_8x64_lam0   8x64,  lambda = 0,    same budget          (data-only counterpart)
  long_6x256       6x256, lambda = 0.01, same budget          (does the larger net win once converged?)
  adam1500_8x64    8x64,  lambda = 0.01, Adam 1500 epochs (cosine), L-BFGS 500   (more Adam instead of L-BFGS)
  const_8x64       8x64,  lambda = 0.01, Adam 300 epochs, constant lr, L-BFGS 5000 (does the cosine decay matter?)

Reported per run: validation data loss at the end of Adam and after 500 / 1000 / 2000 / 5000 L-BFGS
iterations, the number of L-BFGS iterations actually run (SciPy may stop early), the relative
improvement over the last 10 % of iterations, test / extrapolation NRMSE and wall time.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, savefig, start, write_text, plt  # noqa: E402
from experiments.e8_vx_residual import allnrmse, run_jobs, summary  # noqa: E402
from pinc.config import RESULTS_DIR  # noqa: E402
from pinc.jobs import add_slot_args  # noqa: E402

LONG = {"train.lbfgs_iters": 5000, "train.log_every": 100}
RUNS = [
    ("long_8x64", {"loss.lam": 0.01, **LONG}),
    ("long_8x64_lam0", {"loss.lam": 0.0, **LONG}),
    ("long_6x256", {"loss.lam": 0.01, "model.depth": 6, "model.width": 256, **LONG}),
    ("adam1500_8x64", {"loss.lam": 0.01, "train.epochs": 1500, "train.log_every": 100}),
    ("const_8x64", {"loss.lam": 0.01, "train.lr_decay": "none", **LONG}),
]
CHECKPOINTS = (500, 1000, 2000, 5000)


def rid(tag, seed):
    return f"conv_{tag}_s{seed}"


def curve(run_id):
    with open(os.path.join(RESULTS_DIR, "models", run_id, "loss_curve.csv")) as fh:
        rows = list(csv.DictReader(fh))
    return ([r for r in rows if r["stage"] == "adam" and r.get("val_data")],
            [r for r in rows if r["stage"] == "lbfgs" and r.get("val_data")])


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e11_convergence", a)
    jobs = [(rid(tag, a.seed), a.seed, ov) for tag, ov in RUNS]
    run_jobs(jobs, a)

    rec, rows = {}, []
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, ov in RUNS:
        r_id = rid(tag, a.seed)
        s = summary(r_id)
        adam, lb = curve(r_id)
        vd_adam = [float(r["val_data"]) for r in adam]
        vd_lb = [float(r["val_data"]) for r in lb]
        allv = vd_adam + vd_lb
        tail = max(1, len(allv)//10)
        at = {n: (min(vd_lb[:n]) if len(vd_lb) >= n else None) for n in CHECKPOINTS}
        rec[tag] = dict(run_id=r_id, overrides=ov, adam_epochs=len(vd_adam), lbfgs_iters_run=len(vd_lb),
                        val_data_adam_end=min(vd_adam), val_data_at_lbfgs=at, val_data_best=s["val"]["data"],
                        tail_improvement=1.0 - min(allv[-tail:])/min(allv[:-tail]),
                        best_stage=s["best_stage"], best_epoch=s["best_epoch"],
                        test_all=allnrmse(s["test"]["nrmse"]), extrap_all=allnrmse(s["test_extrap"]["nrmse"]),
                        train_seconds=s["train_seconds"], n_params=s["n_params"])
        r = rec[tag]
        rows.append([tag, r["n_params"], f"{r['val_data_adam_end']:.2e}"] +
                    [f"{at[n]:.2e}" if at[n] is not None else "-" for n in CHECKPOINTS] +
                    [f"{r['val_data_best']:.2e}", r["lbfgs_iters_run"], f"{100*r['tail_improvement']:.1f} %",
                     f"{r['test_all']:.2e}", f"{r['extrap_all']:.2e}", f"{r['train_seconds']:.0f}"])
        ax.plot(np.arange(1, len(allv) + 1), np.minimum.accumulate(allv), label=tag)
        ax.axvline(len(vd_adam), color=ax.lines[-1].get_color(), ls=":", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="Adam epoch, then L-BFGS iteration",
           ylabel="best validation data loss so far")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    hdr = (["run", "params", "val data, Adam end"] + [f"after {n} L-BFGS" for n in CHECKPOINTS] +
           ["val data (selected)", "L-BFGS iters run", "improvement in last 10 %", "test NRMSE all", "extrap NRMSE all",
            "train s"])
    text = (f"# E11 training-budget convergence (seed {a.seed}, N = 20 000, increment-scaled loss; "
            "dotted lines in the figure mark the end of Adam)\n\n" + md_table(hdr, rows))
    arts = [write_text(os.path.join(run_dir, "table_convergence.md"), text)] + savefig(fig, run_dir, "fig_convergence")
    finish(run_dir, cfg, a.seed, dict(runs=rec), arts)
    print(text)


if __name__ == "__main__":
    main()
