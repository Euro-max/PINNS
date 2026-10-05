"""
E8 -- Rescaling the longitudinal-velocity residual (follow-up to the E6 ablation).

The time derivative is S_x/T * d(s_hat)/d(t/T); dividing by S_f amplifies network slope errors by
S_x/(T*S_f) = [147, 1.6, 1.1, 14] for [vx, vy, r, psi], so the vx residual dominates the physics loss.
Two principled variants, both at the default 8x64 network and budget, selected on VALIDATION data loss:
  A (sfT): S_f = S_x/T          -- residual measured in the network's own output units
  B (inc): model.increment_scaling -- network outputs an O(1) increment per channel (s_hat = s0_hat + t/T*NN*S_f*T/S_x);
           applied to the lambda = 0 arm too so the PINC / black-box comparison stays fair.
Phase 2 (--low-data): the best variant and its lambda = 0 counterpart at N = 100 with the E2 budget and 5 seeds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs as pinc_run_jobs, summary as jobs_summary  # noqa: E402

SFT = "[300.0, 15.0, 6.0, 5.0]"          # S_x / T
PHASE1 = [
    ("base_lam0", {"loss.lam": 0.0}, "arch_d8_w64_none_do0_ln0_lam0_s0"),
    ("base_lam0.001", {"loss.lam": 0.001}, None),
    ("base_lam0.01", {"loss.lam": 0.01}, "arch_d8_w64_none_do0_ln0_lam0.01_s0"),
    ("inc_lam0", {"loss.lam": 0.0, "model.increment_scaling": "true"}, None),
    ("inc_lam1e-05", {"loss.lam": 1e-5, "model.increment_scaling": "true"}, None),
    ("inc_lam0.0001", {"loss.lam": 1e-4, "model.increment_scaling": "true"}, None),
    ("inc_lam0.001", {"loss.lam": 0.001, "model.increment_scaling": "true"}, None),
    ("inc_lam0.01", {"loss.lam": 0.01, "model.increment_scaling": "true"}, None),
    ("inc_lam0.1", {"loss.lam": 0.1, "model.increment_scaling": "true"}, None),
    ("inc_lam1", {"loss.lam": 1.0, "model.increment_scaling": "true"}, None),
    ("sfT_lam0.01", {"loss.lam": 0.01, "scales.S_f": SFT}, None),
    ("sfT_lam1", {"loss.lam": 1.0, "scales.S_f": SFT}, None),
    ("sfT_lam100", {"loss.lam": 100.0, "scales.S_f": SFT}, None),
]
E2_BUDGET = {"train.steps": 6000, "train.lbfgs_iters": 500, "train.n_data": 100, "train.batch_data": 100, "train.val_every": 20}


def summary(rid):
    return jobs_summary(rid)


def run_jobs(jobs, a):
    """Train the missing runs on the device slots of `a` (pinc/jobs.py)."""
    return pinc_run_jobs(jobs, a.slots, a.config, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)


def allnrmse(v):
    return float(np.sqrt(np.mean(np.square(v))))


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--low-data", default=None, help="comma list of phase-1 tags to rerun at N=100 with 5 seeds")
    a = ap.parse_args(argv)
    cfg, run_dir = start("e8_vx_residual", a)
    jobs, tags = [], {}
    for tag, ov, reuse in PHASE1:
        rid = reuse if reuse and summary(reuse) else f"vx_{tag}_s0"
        tags[tag] = rid
        jobs.append((rid, 0, ov))
    run_jobs(jobs, a)
    rows, rec = [], {}
    for tag, ov, _ in PHASE1:
        s = summary(tags[tag])
        rec[tag] = dict(run_id=tags[tag], overrides=ov, val_data=s["val"]["data"], val_phys=s["val"]["phys"],
                        test=s["test"]["nrmse"], test_all=allnrmse(s["test"]["nrmse"]), extrap_all=allnrmse(s["test_extrap"]["nrmse"]))
        r = rec[tag]
        rows.append([tag, f"{s['lam']:g}", f"{r['val_data']:.2e}", f"{r['val_phys']:.2e}",
                     " / ".join(f"{v:.1e}" for v in r["test"]), f"{r['test_all']:.2e}", f"{r['extrap_all']:.2e}"])
    text = ("# E8 vx-residual rescaling, phase 1 (N = 20 000, seed 0; selection on validation data loss)\n\n" +
            md_table(["variant", "lambda", "val data", "val phys", "test NRMSE vx/vy/r/psi", "test all", "extrap all"], rows))
    low = {}
    if a.low_data:
        jobs = []
        for tag in a.low_data.split(","):
            ov = dict(next(o for t, o, _ in PHASE1 if t == tag), **E2_BUDGET)
            for seed in range(5):
                jobs.append((f"vxlow_{tag}_n100_s{seed}", seed, dict(ov, **{"seeds.train": cfg.seeds.train + 100*seed})))
        run_jobs(jobs, a)
        rows = []
        for tag in a.low_data.split(","):
            v = [allnrmse(summary(f"vxlow_{tag}_n100_s{s}")["test"]["nrmse"]) for s in range(5)]
            from pinc.metrics import bootstrap_ci
            low[tag] = bootstrap_ci(v)
            rows.append([tag, f"{low[tag]['mean']:.4f} [{low[tag]['lo']:.4f}, {low[tag]['hi']:.4f}]"])
        text += ("\n# Phase 2: N = 100 trajectories, E2 budget, 5 seeds (test NRMSE all states, mean [95% CI])\n\n" +
                 md_table(["variant", "test NRMSE"], rows))
    art = write_text(os.path.join(run_dir, "table_vx_residual.md"), text)
    finish(run_dir, cfg, a.seed, dict(phase1=rec, low_data=low), [art])
    print(text)


if __name__ == "__main__":
    main()
