"""
E6 -- Ablations on the VALIDATION set (test reported alongside):
  lambda in {0, 0.01, 0.1, 1, 10, 100}; collocation count; depth/width;
  hard vs soft IC; with/without each residual.
The lambda that minimises the validation DATA loss (prediction accuracy on
held-out trajectories -- the total loss is not comparable across lambda) is
the recommended default; the script reports it and whether it matches
configs/default.yaml.  Existing runs in results/models/ are reused.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from pinc.config import RESULTS_DIR  # noqa: E402
from pinc.config import load_config  # noqa: E402
from pinc.train import train  # noqa: E402

LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)


INC = False


def run_id_for(tag, seed):
    return f"ab_{tag}_s{seed}" + ("_inc" if INC else "")


def e7_id(cfg, lam, seed):
    """Run id of the identical E7 trial (so the lambda sweep reuses E7 models)."""
    from experiments.e7_architecture import trial_id
    m = cfg.model
    return trial_id(dict(depth=m.depth, width=m.width, residual=m.residual, dropout=m.dropout,
                         layernorm=m.layernorm, lam=lam, seed=seed))


def ensure(cfg, overrides, run_id, seed, quick, alt_id=None):
    if INC:
        alt_id = None                     # E7 models were trained without increment scaling
    if alt_id and os.path.exists(os.path.join(RESULTS_DIR, "models", alt_id, "summary.json")):
        run_id = alt_id
    d = os.path.join(RESULTS_DIR, "models", run_id)
    p = os.path.join(d, "summary.json")
    if os.path.exists(p):
        with open(p) as fh:
            s = json.load(fh)
        s["run_dir"] = d
        s["reused"] = True
        return s
    c = cfg.with_overrides(overrides)
    if quick:
        c = c.with_overrides({"train.epochs": 60, "train.lbfgs_iters": 50, "train.log_every": 20})
    s = train(c, seed, run_id, verbose=True)
    s["reused"] = False
    return s


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--lam-star", type=float, default=None,
                    help="lambda for the non-lambda ablations (default: the config lambda; the sweep's argmin is reported separately)")
    a = ap.parse_args(argv)
    cfg, run_dir = start("e6_ablations", a)
    global INC
    INC = bool(cfg.model.increment_scaling)
    seed = a.seed
    runs = {}
    # ---- lambda sweep
    for lam in LAMBDAS:
        tag = f"lam{lam:g}"
        runs[tag] = ensure(cfg, {"loss.lam": lam}, run_id_for(tag, seed), seed, a.quick, alt_id=None if a.quick else e7_id(cfg, lam, seed))
    val_data = {lam: runs[f"lam{lam:g}"]["val"]["data"] for lam in LAMBDAS}
    lam_argmin = min(val_data, key=val_data.get)
    lam_star = a.lam_star if a.lam_star is not None else cfg.loss.lam
    print(f"  sweep argmin (val data loss): lambda = {lam_argmin:g} ({val_data[lam_argmin]:.3e}); "
          f"other ablations run at lambda = {lam_star:g}; config default {cfg.loss.lam:g}")
    base = {"loss.lam": lam_star}
    # ---- other ablations at lambda*
    others = {
        "colloc2k": {**base, "train.n_colloc": 2000, "train.batch_colloc": 256},
        "colloc100k": {**base, "train.n_colloc": 100000, "train.batch_colloc": 4096},
        "arch2x64": {**base, "model.depth": 2, "model.width": 64},
        "arch6x256": {**base, "model.depth": 6, "model.width": 256},
        "softic": {**base, "model.hard_ic": False},
        "residual_skip": {**base, "model.residual": "skip"},
        "residual_block": {**base, "model.residual": "block"},
        "no_res_vx": {**base, "loss.residual_mask": [0, 1, 1, 1]},
        "no_res_vy": {**base, "loss.residual_mask": [1, 0, 1, 1]},
        "no_res_r": {**base, "loss.residual_mask": [1, 1, 0, 1]},
        "no_res_psi": {**base, "loss.residual_mask": [1, 1, 1, 0]},
    }
    for tag, ov in others.items():
        runs[tag] = ensure(cfg, ov, run_id_for(f"{tag}_lam{lam_star:g}", seed), seed, a.quick)

    rows = []
    for tag, s in runs.items():
        rows.append([tag, f"{s['lam']:g}", f"{s['val']['data']:.3e}", f"{s['val']['phys']:.3e}", f"{s['val']['total']:.3e}",
                     " / ".join(f"{v:.2e}" for v in s["test"]["nrmse"]), " / ".join(f"{v:.2e}" for v in s["test_extrap"]["nrmse"]),
                     s["best_stage"], s["n_params"], f"{s['train_seconds']:.0f}", "reused" if s.get("reused") else "new"])
    table = (f"# E6 ablations (seed {seed}{', QUICK' if a.quick else ''})\n\n"
             f"sweep argmin of the validation data loss: lambda = {lam_argmin:g}; other ablations at lambda = {lam_star:g}; "
             f"configs/default.yaml has lambda = {cfg.loss.lam:g}\n\n" +
             md_table(["ablation", "lambda", "val data", "val phys", "val total", "test NRMSE vx/vy/r/psi",
                       "extrap NRMSE vx/vy/r/psi", "best stage", "params", "train s", "run"], rows))
    art = write_text(os.path.join(run_dir, "table_ablations.md"), table)
    summary = dict(quick=a.quick, lam_star=lam_star, lam_argmin_val_data=lam_argmin, val_data_by_lambda={f"{k:g}": v for k, v in val_data.items()},
                   config_lambda=cfg.loss.lam, runs={k: {kk: vv for kk, vv in v.items() if kk != "weights"} for k, v in runs.items()})
    finish(run_dir, cfg, seed, summary, [art], dict(quick=a.quick))
    print(table)


if __name__ == "__main__":
    main()
