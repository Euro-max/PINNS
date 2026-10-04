"""
E7 -- Network architecture study (run BEFORE fixing the default model and lambda).

Grid A (topology x lambda):   depth {2, 4, 6, 8} x width {64, 128, 256} x lambda {0, 0.01, 0.1}
Grid B (regularisation / skip connections) at (4 x 128) and (6 x 256), lambda {0, 0.01}:
        dropout {0.05, 0.1}, residual {skip, block}, layernorm
Every trial uses the same data, seeds and optimiser budget (configs/default.yaml).
Selection criterion: validation DATA loss (held-out trajectory accuracy); the
validation physics loss and the test / extrapolation NRMSE are reported too.
All trials are documented in table_trials.md; existing runs in results/models are reused.
Trials run as subprocesses, `--workers` at a time.
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, savefig, start, write_text, plt  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT  # noqa: E402

STATES = ("vx", "vy", "r", "psi")


def trial_id(t):
    return (f"arch_d{t['depth']}_w{t['width']}_{t['residual']}_do{t['dropout']:g}_ln{int(t['layernorm'])}"
            f"_lam{t['lam']:g}_s{t['seed']}")


def make_trials(quick, seed):
    depths = [2, 4] if quick else [2, 4, 6, 8]
    widths = [64, 128] if quick else [64, 128, 256]
    lams = [0.0, 0.01] if quick else [0.0, 0.01, 0.1]
    trials = []
    for d, w, lam in itertools.product(depths, widths, lams):
        trials.append(dict(depth=d, width=w, residual="none", dropout=0.0, layernorm=False, lam=lam, seed=seed, grid="A"))
    bases = [(4, 128)] if quick else [(4, 128), (6, 256)]
    for (d, w), lam in itertools.product(bases, [0.0, 0.01]):
        for var in ({"dropout": 0.05}, {"dropout": 0.1}, {"residual": "skip"}, {"residual": "block"}, {"layernorm": True}):
            t = dict(depth=d, width=w, residual="none", dropout=0.0, layernorm=False, lam=lam, seed=seed, grid="B")
            t.update(var)
            trials.append(t)
    # de-duplicate (a grid-B base equals a grid-A point only if a variant is empty; keep unique ids)
    seen, out = set(), []
    for t in trials:
        if trial_id(t) not in seen:
            seen.add(trial_id(t))
            out.append(t)
    return out


def cmd_for(t, args, quick):
    rid = trial_id(t)
    c = [sys.executable, "-m", "pinc.train", "--config", args.config, "--seed", str(t["seed"]), "--run-id", rid,
         "--set", f"model.depth={t['depth']}", "--set", f"model.width={t['width']}", "--set", f"model.residual={t['residual']}",
         "--set", f"model.dropout={t['dropout']}", "--set", f"model.layernorm={str(t['layernorm']).lower()}",
         "--set", f"loss.lam={t['lam']}", "--set", "train.log_every=100"]
    for ov in args.overrides:
        c += ["--set", ov]
    if quick:
        c += ["--set", "train.epochs=40", "--set", "train.lbfgs_iters=30", "--set", "train.n_data=4000"]
    return c


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e7_architecture", a)
    trials = make_trials(a.quick, a.seed)
    print(f"  {len(trials)} trials, {a.workers} workers")
    pending = [t for t in trials if not os.path.exists(os.path.join(RESULTS_DIR, "models", trial_id(t), "summary.json"))]
    print(f"  {len(trials) - len(pending)} reused, {len(pending)} to train")
    logdir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(logdir, exist_ok=True)
    running = []
    env = dict(os.environ, PYTHONPATH=ROOT)
    while pending or running:
        while pending and len(running) < a.workers:
            t = pending.pop(0)
            log = open(os.path.join(logdir, trial_id(t) + ".log"), "w")
            p = subprocess.Popen(cmd_for(t, a, a.quick), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            running.append((t, p, log))
            print(f"  started {trial_id(t)}", flush=True)
        for item in list(running):
            t, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"  finished {trial_id(t)} (exit {p.returncode})", flush=True)
                if p.returncode != 0:
                    raise RuntimeError(f"trial {trial_id(t)} failed; see {log.name}")
        import time
        time.sleep(2)

    rows, recs = [], []
    for t in trials:
        d = os.path.join(RESULTS_DIR, "models", trial_id(t))
        with open(os.path.join(d, "summary.json")) as fh:
            s = json.load(fh)
        rec = dict(t, run_id=trial_id(t), val_data=s["val"]["data"], val_phys=s["val"]["phys"], val_total=s["val"]["total"],
                   test_nrmse=s["test"]["nrmse"], extrap_nrmse=s["test_extrap"]["nrmse"], n_params=s["n_params"],
                   train_seconds=s["train_seconds"], best_stage=s["best_stage"], best_epoch=s["best_epoch"],
                   test_all=float(np.sqrt(np.mean(np.square(s["test"]["nrmse"])))),
                   extrap_all=float(np.sqrt(np.mean(np.square(s["test_extrap"]["nrmse"])))))
        recs.append(rec)
    recs.sort(key=lambda r: r["val_data"])
    for r in recs:
        rows.append([r["grid"], r["depth"], r["width"], r["residual"], f"{r['dropout']:g}", int(r["layernorm"]), f"{r['lam']:g}",
                     f"{r['val_data']:.3e}", f"{r['val_phys']:.3e}", " / ".join(f"{v:.1e}" for v in r["test_nrmse"]),
                     f"{r['test_all']:.2e}", f"{r['extrap_all']:.2e}", r["n_params"], f"{r['train_seconds']:.0f}", r["best_stage"]])
    best = {}
    for lam in sorted({r["lam"] for r in recs}):
        cand = [r for r in recs if r["lam"] == lam]
        best[f"{lam:g}"] = min(cand, key=lambda r: r["val_data"])
    lines = [f"# E7 architecture trials (seed {a.seed}{', QUICK' if a.quick else ''}), sorted by validation data loss\n",
             "Same data, seeds and optimiser budget for every trial; selection criterion = validation data loss.\n"]
    for lam, r in best.items():
        lines.append(f"- best at lambda = {lam}: {r['run_id']}  (val data {r['val_data']:.3e}, test NRMSE all {r['test_all']:.2e}, "
                     f"extrap {r['extrap_all']:.2e}, {r['n_params']} params)")
    lines.append("")
    table = "\n".join(lines) + md_table(["grid", "depth", "width", "residual", "dropout", "LN", "lambda", "val data", "val phys",
                                         "test NRMSE vx/vy/r/psi", "test all", "extrap all", "params", "train s", "best stage"], rows)
    art = write_text(os.path.join(run_dir, "table_trials.md"), table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    markers = {"none": "o", "skip": "s", "block": "^"}
    for lam, col in zip(sorted({r["lam"] for r in recs}), ("#2ca02c", "#d62728", "#ff7f0e", "#9467bd")):
        for r in recs:
            if r["lam"] != lam:
                continue
            mk = markers[r["residual"]]
            fc = col if (r["dropout"] == 0 and not r["layernorm"]) else "none"
            axes[0].scatter(r["n_params"], r["val_data"], marker=mk, facecolors=fc, edgecolors=col, s=30)
            axes[1].scatter(r["n_params"], r["test_all"], marker=mk, facecolors=fc, edgecolors=col, s=30)
        axes[0].scatter([], [], color=col, label=f"lambda={lam:g}")
    for ax, yl in zip(axes, ("validation data loss", "test NRMSE (all states)")):
        ax.set(xscale="log", yscale="log", xlabel="parameters", ylabel=yl)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, title="o none  s skip  ^ block\nhollow: dropout / LN")
    arts = [art] + savefig(fig, run_dir, "fig_trials")
    summary = dict(quick=a.quick, n_trials=len(recs), best_by_lambda=best, trials=recs, config_lambda=cfg.loss.lam,
                   config_model=cfg.model.__dict__)
    finish(run_dir, cfg, a.seed, summary, arts, dict(quick=a.quick))
    print(table)


if __name__ == "__main__":
    main()
