"""
E21 -- Architecture screen for the PINC network (one test).

HF-M0, N = 100 (lambda 0.01) and N = 1000 (lambda 0.001), the lambda E15 selected, without per-architecture
tuning; the E17 data, seeds 0-2, residual mask (no wheel residuals) and budget (6000 Adam steps + 2000 L-BFGS).
Architectures (pinc/model.py): the 8x64 MLP of every earlier result (A0, E17 runs reused), a wide 6x256 MLP,
modified MLP, Fourier features, adaptive activation, multi-rate time basis, DeepONet, split heads, prior-anchored
and Chebyshev KAN.  Measured: one-step test NRMSE (body), chained 10- / 50-step body error on validation (used
for the ranking) and test, the 50-step error outside the training range, the physics residual and derivative
error (E9), the parameter count, and, with --timing on an idle machine, the XLA model-call time and the MPC
solve time at N = 10 on one CPU thread (3 s lane change).  Each architecture is compared with A0 (ratio of
means, seeds won, paired t-test on log errors).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, savefig, start, write_text, plt  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, group_rms  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides, prewarm  # noqa: E402
from experiments.e17_hf_compare import cmp_cell, compare  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

SEEDS = (0, 1, 2)
LAM = {100: 0.01, 1000: 0.001}
ARCHS = {   # tag: (label, overrides); A0 = the E15 / E17 runs
    "": ("A0 MLP 8x64", {}),
    "wide": ("A1 wide MLP 6x256", {"model.depth": 6, "model.width": 256}),
    "modmlp": ("A2 modified MLP", {"model.arch": "modified_mlp"}),
    "fourier": ("A3 Fourier features", {"model.arch": "fourier"}),
    "adaptive": ("A4 adaptive activation", {"model.arch": "adaptive"}),
    "tbasis": ("A5 multi-rate time basis", {"model.arch": "time_basis"}),
    "deeponet": ("A6 DeepONet", {"model.arch": "deeponet"}),
    "split": ("A7 split heads", {"model.arch": "split"}),
    "anchored": ("A8 prior-anchored", {"model.arch": "anchored"}),
    "kan": ("A9 Chebyshev KAN", {"model.arch": "chebykan"}),
}


def rid(variant, n, tag, seed):
    return f"hf{variant}_n{n}_lam{LAM[n]:g}" + (f"_{tag}" if tag else "") + f"_s{seed}"


def timing(cfg, variant, run_dir):
    """Model-call and MPC solve time per architecture (seed-0 model at N = 1000), one CPU thread."""
    from experiments.e4_timing import time_model_call
    from pinc.metrics import solve_time_stats
    from pinc.model import PINCNet
    from pinc.mpc import make_controller
    from pinc.refs import make_reference
    from pinc.sim import simulate
    from pinc.system import get_system
    ref = make_reference("lane_change", cfg)
    x0 = get_system(cfg).initial_state(ref.x0())
    out = {}
    for tag, (label, _) in ARCHS.items():
        net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", rid(variant, 1000, tag, 0)))
        ctrl = make_controller("pinc", cfg, {"pinc": net}, ref.Q, ref.P)
        log = simulate(ctrl, None, ref, x0, 3.0, np.zeros(x0.size), 0, cfg)
        st = solve_time_stats(log["solve_time"][1:])
        out[tag] = dict(solve_median=st["median"], solve_p95=st["p95"], model_call=time_model_call(ctrl, cfg)["compiled"],
                        n_params=int(sum(int(np.prod(v.shape)) for v in net.trainable_variables)))
        print(f"  {label:26s} solve {st['median']*1e3:6.1f} ms  call {out[tag]['model_call']*1e3:.3f} ms  "
              f"params {out[tag]['n_params']}", flush=True)
    with open(os.path.join(run_dir, "timing.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    ap.add_argument("--timing", action="store_true", help="only measure solve times (run on an idle machine)")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    if a.timing:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        a.threads = 1
    _, run_dir = start("e21_architectures", a)
    cfg = load_config(cfg_path, a.overrides)
    if a.timing:
        timing(cfg, a.variant, run_dir)
        return
    sizes = (100,) if a.quick else (100, 1000)
    seeds = SEEDS[:1] if a.quick else SEEDS
    jobs = []
    for n in sizes:
        for tag, (_, ov) in ARCHS.items():
            for seed in seeds:
                o = dict(overrides(n, LAM[n], seed, cfg), **ov)
                if a.quick:
                    o.update({"train.steps": 50, "train.lbfgs_iters": 20})
                jobs.append((rid(a.variant, n, tag, seed) + ("_quick" if a.quick else ""), seed, o))
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)

    from experiments import e9_physics_learning as e9
    ev = {}
    for split in ("val", "test"):
        e9_id = f"{os.path.basename(run_dir)}_e9{split}"
        e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", ",".join(f"{r}={r}" for r, _, _ in jobs),
                 "--split", split] + (["--quick"] if a.quick else []))
        with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
            ev[split] = json.load(fh)["models"]
    sfx = "_quick" if a.quick else ""
    metrics = dict(h50_val=("val 50-step body", lambda r: ev["val"][r]["horizon_body"]["in_domain"]["50"]["mean"]),
                   one_step=("one-step body", lambda r: group_rms(summary(r)["test"]["nrmse"], GROUPS["body"])),
                   h10=("10-step body", lambda r: ev["test"][r]["horizon_body"]["in_domain"]["10"]["mean"]),
                   h50=("50-step body", lambda r: ev["test"][r]["horizon_body"]["in_domain"]["50"]["mean"]),
                   h50_extrap=("50-step body, outside training range",
                               lambda r: ev["test"][r]["horizon_body"]["extrap"]["50"]["mean"]),
                   deriv=("derivative error, body", lambda r: ev["test"][r]["deriv_err_body"]))
    tim = {}
    tp = os.path.join(run_dir, "timing.json")
    if os.path.exists(tp):
        with open(tp) as fh:
            tim = json.load(fh)
    rec, rows, crows = {}, [], []
    for n in sizes:
        rec[n] = {tag: {m: [fn(rid(a.variant, n, tag, s) + sfx) for s in seeds] for m, (_, fn) in metrics.items()}
                  for tag in ARCHS}
        order = sorted(ARCHS, key=lambda t: np.mean(rec[n][t]["h50_val"]))
        for rank, tag in enumerate(order, 1):
            t = tim.get(tag, {})
            rows.append([n, rank, ARCHS[tag][0]] + [ci_cell(rec[n][tag][m]) for m in metrics] +
                        [t.get("n_params", summary(rid(a.variant, n, tag, seeds[0]) + sfx)["n_params"]),
                         f"{t['solve_median']*1e3:.1f}" if t else "-"])
            if tag and len(seeds) > 1:
                c = {m: compare(rec[n][tag][m], rec[n][""][m]) for m in metrics}
                rec[n][f"{tag} vs A0"] = c
                crows.append([n, ARCHS[tag][0]] + [cmp_cell(c[m]) for m in metrics])
    labels = [lbl for lbl, _ in metrics.values()]
    text = (f"# E21 architecture screen, HF-{a.variant.upper()} (PINC, lambda from E15; mean [95% CI] over {len(seeds)} "
            "seeds; ranked by the validation 50-step body error; solve time: MPC at N = 10, one CPU thread)\n\n" +
            md_table(["N", "rank", "architecture"] + labels + ["parameters", "solve [ms]"], rows) +
            ("\n## Against A0 (error of A0 divided by the error of the architecture, above 1: the architecture is "
             "better; seeds won; paired t-test on log errors)\n\n" + md_table(["N", "architecture"] + labels, crows)
             if crows else ""))
    arts = [write_text(os.path.join(run_dir, "table_architectures.md"), text)]
    if tim:
        fig, axes = plt.subplots(1, len(sizes), figsize=(5*len(sizes), 3.8), squeeze=False)
        for ax, n in zip(axes[0], sizes):
            for tag, (label, _) in ARCHS.items():
                ax.errorbar(tim[tag]["solve_median"]*1e3, np.mean(rec[n][tag]["h50"]),
                            yerr=np.std(rec[n][tag]["h50"]), fmt="o", capsize=3, label=label)
            ax.set(xscale="log", yscale="log", xlabel="MPC solve time at N = 10 [ms]", ylabel="test 50-step body NRMSE",
                   title=f"N = {n} trajectories")
        axes[0][0].legend(fontsize=6)
        arts += savefig(fig, run_dir, "fig_accuracy_vs_time")
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, sizes=list(sizes), seeds=list(seeds), lam=LAM,
                                     archs={t: v[0] for t, v in ARCHS.items()}, timing=tim,
                                     results={str(n): v for n, v in rec.items()}), arts, dict(quick=a.quick))
    print(text)


if __name__ == "__main__":
    main()
