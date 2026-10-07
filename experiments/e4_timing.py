"""
E4 -- Is it faster?  Solve time per MPC step vs horizon N in {5, 10, 20, 40}
for every arm, on a 5 s single-lane-change closed loop (nominal plant, no
noise), CPU single-thread (the CPU model is recorded in meta.json).  Also
the time per prediction-model call (one control step, batch 1).  A GPU
section is produced only if TensorFlow sees a GPU; otherwise the summary
says so.  The first (compilation) call of every configuration is excluded.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import COLORS, LABELS, base_parser, finish, load_models, md_table, savefig, start, write_text, plt  # noqa: E402
from pinc.metrics import solve_time_stats  # noqa: E402
from pinc.mpc import ARMS, make_controller  # noqa: E402
from pinc.refs import make_reference  # noqa: E402
from pinc.sim import simulate  # noqa: E402
from pinc.system import get_system  # noqa: E402

HORIZONS = (5, 10, 20, 40)


def time_model_call(ctrl, cfg, reps=200):
    import tensorflow as tf
    sysm = get_system(cfg)
    s = tf.constant(sysm.from_full(sysm.initial_state([20.0, 0.0, 0.0, 0.0, 0.0, 0.0]))[None], cfg.dtype)
    u = tf.constant(np.array([[500.0, 0.01]]), cfg.dtype)
    pred = ctrl.pred
    if not hasattr(pred, "step"):
        return dict(eager=np.nan, compiled=np.nan)
    f = tf.function(pred.step, jit_compile=True)
    for _ in range(3):
        pred.step(s, u); f(s, u)
    t0 = time.perf_counter()
    for _ in range(reps):
        pred.step(s, u)
    eager = (time.perf_counter() - t0)/reps
    t0 = time.perf_counter()
    for _ in range(reps):
        f(s, u)
    comp = (time.perf_counter() - t0)/reps
    return dict(eager=eager, compiled=comp)


def main(argv=None):
    ap = base_parser(__doc__)
    ap.set_defaults(threads=1)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--arms", default=None, help="comma list of controllers (default: nmpc_rk4, ltv and the loaded networks)")
    ap.add_argument("--horizons", default=None, help="comma list of horizons N (default 5,10,20,40)")
    ap.add_argument("--true-max-n", type=int, default=10, help="largest N for the (expensive) nmpc_true arm")
    a = ap.parse_args(argv)
    cfg, run_dir = start("e4_timing", a)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    models = load_models(a)
    arms = [x for x in ARMS if x in ("nmpc_rk4", "ltv") or x in models]
    if a.arms:
        arms = [x for x in a.arms.split(",") if x in ("nmpc_rk4", "ltv", "nmpc_true", "nmpc_qs") or x in models]
    horizons = tuple(int(n) for n in a.horizons.split(",")) if a.horizons else (HORIZONS[:2] if a.quick else HORIZONS)
    ref = make_reference("lane_change", cfg)
    summary = dict(quick=a.quick, device="cpu-single-thread", gpu=[g.name for g in gpus] or "not available",
                   horizons=list(horizons), solve={}, model_call={})
    for arm in arms:
        summary["solve"][arm] = {}
        for N in horizons:
            if arm == "nmpc_true" and N > a.true_max_n:
                continue                                          # seconds per solve: longer horizons take hours
            c = cfg.with_overrides({"mpc.N": N})
            ctrl = make_controller(arm, c, models, ref.Q, ref.P)
            x0 = get_system(c).initial_state(ref.x0())
            log = simulate(ctrl, c.params, ref, x0, a.duration, np.zeros(x0.size), 0, c)
            st = solve_time_stats(log["solve_time"][1:])          # exclude the compile call
            st["nit_mean"] = float(np.mean(log["nit"][1:]))
            st["per_iteration_mean"] = float(np.mean(log["solve_time"][1:]/np.maximum(log["nit"][1:], 1)))
            st["success_rate"] = float(np.mean(log["success"]))
            st["rmse_Y"] = float(np.sqrt(np.mean(log["err"][1:, 5]**2)))
            summary["solve"][arm][N] = st
            print(f"  {arm:9s} N={N:2d}: median {st['median']*1e3:6.1f} ms  p95 {st['p95']*1e3:6.1f} ms  nit {st['nit_mean']:.1f}", flush=True)
        summary["model_call"][arm] = time_model_call(make_controller(arm, cfg, models, ref.Q, ref.P), cfg)
    rows = []
    for arm in arms:
        for N in [n for n in horizons if n in summary["solve"][arm]]:
            st = summary["solve"][arm][N]
            rows.append([LABELS[arm], N, f"{st['mean']*1e3:.1f}", f"{st['median']*1e3:.1f}", f"{st['p95']*1e3:.1f}",
                         f"{st['nit_mean']:.1f}", f"{st['per_iteration_mean']*1e3:.2f}", f"{st['rmse_Y']:.3f}"])
    t1 = md_table(["arm", "N", "mean [ms]", "median [ms]", "p95 [ms]", "iterations", "ms / iteration", "RMSE Y [m]"], rows)
    rows = [[LABELS[arm], f"{v['eager']*1e3:.2f}", f"{v['compiled']*1e3:.3f}"] for arm, v in summary["model_call"].items()]
    t2 = md_table(["arm", "one-step model call, eager [ms]", "one-step model call, XLA [ms]"], rows)
    art = write_text(os.path.join(run_dir, "table_timing.md"),
                     f"# E4 solve time per MPC step vs horizon (CPU single thread; GPU: {summary['gpu']})\n\n" + t1 + "\n" + t2)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for arm in arms:
        Ns = [n for n in horizons if n in summary["solve"][arm]]
        med = [summary["solve"][arm][N]["median"]*1e3 for N in Ns]
        p95 = [summary["solve"][arm][N]["p95"]*1e3 for N in Ns]
        ax.plot(Ns, med, marker="o", color=COLORS[arm], label=LABELS[arm])
        ax.plot(Ns, p95, ":", color=COLORS[arm])
    ax.set(xlabel="horizon N", ylabel="solve time per step [ms] (solid: median, dotted: p95)", yscale="log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    arts = [art] + savefig(fig, run_dir, "fig_solve_time_vs_N")
    finish(run_dir, cfg, a.seed, summary, arts, dict(quick=a.quick, threads=1))


if __name__ == "__main__":
    main()
