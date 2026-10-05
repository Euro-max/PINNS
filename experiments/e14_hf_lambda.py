"""
E14 (plan H3, first pass) -- physics weight and wheel residuals on the high-fidelity system.

On one variant (default M0), N = 20 000 trajectories, seed 0, the fixed budget of configs/hf_*.yaml:
  lambda in {0, 1e-3, 1e-2, 1e-1}, each lambda > 0 with all ten residuals and without the four
  wheel-slip residuals (E13: the zero-track prior's wheel equations are ~1 S_f wrong even in gentle
  driving, so they may conflict with the data).
Selection quantity: validation DATA loss (held-out trajectory accuracy).  Also reported: test and
extrapolation NRMSE by state group (body / actuators / wheels).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from pinc.config import ROOT, load_config  # noqa: E402
from pinc.data import make_splits, sample_collocation, sample_ic  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1)
NO_WHEEL = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
GROUPS = dict(body=[0, 1, 2, 3], actuators=[4, 5], wheels=[6, 7, 8, 9])


def prewarm(cfg):
    """Generate (and cache) every driving-state set the training jobs will ask for, before they run in
    parallel, so two jobs never write the same cache file."""
    make_splits(cfg)
    sample_ic(min(cfg.train.n_data, 4000), cfg.seeds.train + 1000, cfg)
    sample_ic(min(cfg.train.n_val, 2000), cfg.seeds.val + 1000, cfg)
    sample_collocation(cfg.train.n_val, cfg.seeds.val + 2000, cfg)


def group_rms(v, idx):
    return float(np.sqrt(np.mean(np.square(np.asarray(v)[idx]))))


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    cfg, run_dir = start("e14_hf_lambda", a)
    cfg = load_config(cfg_path, a.overrides)
    print("  prewarming the driving-state cache", flush=True)
    prewarm(cfg)
    jobs, tags = [], []
    for lam in LAMBDAS:
        masks = [("all", None)] if lam == 0 else [("all", None), ("nowheel", NO_WHEEL)]
        for mname, mask in masks:
            tag = f"lam{lam:g}_{mname}"
            ov = {"loss.lam": lam}
            if mask is not None:
                ov["loss.residual_mask"] = str(mask).replace(" ", "")
            jobs.append((f"hf{a.variant}_{tag}_s{a.seed}", a.seed, ov))
            tags.append(tag)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    rows, rec = [], {}
    for tag, (rid, _, ov) in zip(tags, jobs):
        s = summary(rid)
        r = dict(run_id=rid, overrides=ov, val_data=s["val"]["data"], val_phys=s["val"]["phys"],
                 test={g: group_rms(s["test"]["nrmse"], i) for g, i in GROUPS.items()},
                 extrap={g: group_rms(s["test_extrap"]["nrmse"], i) for g, i in GROUPS.items()},
                 best_stage=s["best_stage"], best_epoch=s["best_epoch"], train_seconds=s["train_seconds"])
        rec[tag] = r
        rows.append([tag, f"{r['val_data']:.3e}", f"{r['val_phys']:.3e}"] + [f"{r['test'][g]:.2e}" for g in GROUPS] +
                    [f"{r['extrap'][g]:.2e}" for g in GROUPS] + [f"{r['best_stage']}@{r['best_epoch']}", f"{r['train_seconds']:.0f}"])
    best = min(rec, key=lambda t: rec[t]["val_data"])
    text = (f"# E14 physics weight and wheel residuals, HF-{a.variant.upper()} (N = {cfg.train.n_data}, seed {a.seed}; "
            f"selection on validation data loss: best = {best})\n\n" +
            md_table(["run", "val data", "val phys"] + [f"test {g}" for g in GROUPS] + [f"extrap {g}" for g in GROUPS] +
                     ["best ckpt", "train s"], rows))
    art = write_text(os.path.join(run_dir, "table_hf_lambda.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, best=best, runs=rec), [art])
    print(text)


if __name__ == "__main__":
    main()
