"""Shared helpers for the experiment scripts: argument parsing, run
directories, model loading, figure saving and the MANIFEST entry."""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinc.config import RESULTS_DIR, add_config_args, load_config  # noqa: E402
from pinc.model import PINCNet  # noqa: E402
from pinc.runinfo import make_run_dir, manifest_append, save_json, write_meta  # noqa: E402
from pinc.tfsetup import setup  # noqa: E402
from pinc.mpc import ARM_LABELS  # noqa: E402

COLORS = dict(nmpc_rk4="#1f77b4", pinc="#d62728", blackbox="#2ca02c", ltv="#9467bd", nmpc_true="#ff7f0e", greybox="#8c564b", ref="k", rk4="#1f77b4",
              linear="#9467bd", truth="k")
LABELS = dict(ARM_LABELS, rk4="RK4 (dt=0.01)", linear="Linear (LTI @ 20 m/s)", truth="RK4 truth")


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = add_config_args(argparse.ArgumentParser(description=desc))
    ap.add_argument("--quick", action="store_true", help="reduced sizes for a smoke run (results are labelled quick)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--pinc-model", default=os.path.join(RESULTS_DIR, "models", "pinc_default_s0"))
    ap.add_argument("--blackbox-model", default=os.path.join(RESULTS_DIR, "models", "blackbox_default_s0"))
    ap.add_argument("--greybox-model", default=None, help="grey-box model (pinc/greybox.py), for the greybox arm")
    ap.add_argument("--threads", type=int, default=None)
    return ap


def start(exp: str, args):
    cfg = load_config(args.config, args.overrides)
    setup(args.seed, cfg.dtype, threads=args.threads)
    run_id = args.run_id or (("quick_" if args.quick else "") + _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = make_run_dir(exp, run_id)
    print(f"[{exp}] run_dir={run_dir} quick={args.quick}")
    return cfg, run_dir


def load_models(args) -> dict:
    out = {}
    for name, d in (("pinc", args.pinc_model), ("blackbox", args.blackbox_model),
                    ("greybox", getattr(args, "greybox_model", None))):
        if name == "greybox" and not d:
            continue
        if d and os.path.exists(os.path.join(d, "model.json")):
            out[name] = PINCNet.load_from(d)
        else:
            print(f"  WARNING: model '{name}' not found at {d}")
    return out


def savefig(fig, run_dir: str, name: str) -> list:
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(run_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def finish(run_dir: str, cfg, seed: int, summary: dict, artifacts: list, extra_meta: dict | None = None):
    save_json(summary, os.path.join(run_dir, "summary.json"))
    write_meta(run_dir, cfg, seed, extra_meta)
    manifest_append(list(artifacts) + [os.path.join(run_dir, "summary.json")])
    print(f"  wrote {run_dir}")


def ci_str(ci: dict, fmt="{:.3g}") -> str:
    if ci is None or not np.isfinite(ci.get("mean", np.nan)):
        return "n/a"
    return (fmt + " [" + fmt + ", " + fmt + "]").format(ci["mean"], ci["lo"], ci["hi"])


def md_table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|"*len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def write_text(path: str, text: str):
    with open(path, "w") as fh:
        fh.write(text)
    return path
