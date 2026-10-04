"""
E5 -- Robustness to model mismatch.  The controller keeps the NOMINAL
parameters; the plant is perturbed: mass +-20 %, Caf/Car +-30 %, Fiala tyre
with mu in {0.4, 0.7, 1.0}, and a step disturbance force (-1000 N on Fx from
t = 4 s, about a 7 % grade).  Single lane change, 30 seeds (quick: 3) per
setting, all four arms.  Output: degradation curves per arm.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import COLORS, LABELS, base_parser, ci_str, finish, load_models, md_table, savefig, start, write_text, plt  # noqa: E402
from experiments.e3_closed_loop import run_all, summarise  # noqa: E402
from pinc.mpc import ARMS  # noqa: E402
from pinc.plant import perturbed  # noqa: E402

SETTINGS = [
    ("nominal", {}, "linear", None),
    ("m-20%", dict(m=1500.0*0.8), "linear", None),
    ("m+20%", dict(m=1500.0*1.2), "linear", None),
    ("Ca-30%", dict(Caf=50000.0*0.7, Car=50000.0*0.7), "linear", None),
    ("Ca+30%", dict(Caf=50000.0*1.3, Car=50000.0*1.3), "linear", None),
    ("fiala mu=1.0", dict(mu=1.0), "fiala", None),
    ("fiala mu=0.7", dict(mu=0.7), "fiala", None),
    ("fiala mu=0.4", dict(mu=0.4), "fiala", None),
    ("Fx step -1000 N @ 4 s", {}, "linear", lambda t: np.array([-1000.0, 0.0]) if t >= 4.0 else np.zeros(2)),
]
PLOT_METRICS = ("rmse_Y", "rmse_psi", "rmse_vx", "effort")


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--ref", default="lane_change")
    a = ap.parse_args(argv)
    cfg, run_dir = start("e5_robustness", a)
    models = load_models(a)
    arms = [x for x in ARMS if x in ("nmpc_rk4", "ltv") or x in models]
    seeds = list(range(a.n_seeds or (3 if a.quick else 30)))
    settings = SETTINGS[:4] + SETTINGS[-2:] if a.quick else SETTINGS
    summary = dict(quick=a.quick, arms=arms, seeds=seeds, ref=a.ref, settings=[s[0] for s in settings], per_setting={}, raw={})
    for name, changes, tyre, dist in settings:
        print(f"  setting {name}: plant params {changes} tyre={tyre}", flush=True)
        pp = perturbed(cfg.params, **changes)
        results, _ = run_all(cfg, arms, [a.ref], seeds, models, run_dir, cfg.sim.duration, verbose=False,
                             plant_params=pp, tyre=tyre, disturbance=dist, save_logs=False)
        summary["per_setting"][name] = summarise(results, arms)[a.ref]
        summary["raw"][name] = {arm: [dict(seed=s, **m) for s, m in v] for arm, v in results[a.ref].items()}
        for arm in arms:
            s = summary["per_setting"][name][arm]
            print(f"    {arm:9s} rmse_Y {ci_str(s['rmse_Y'])}  rmse_vx {ci_str(s['rmse_vx'])}  success {s['success_rate']['mean']:.2f}", flush=True)
    rows = []
    for name in summary["settings"]:
        for arm in arms:
            s = summary["per_setting"][name][arm]
            rows.append([name, LABELS[arm]] + [ci_str(s[k]) for k in PLOT_METRICS] + [f"{s['success_rate']['mean']:.2f}", ci_str(s["solve_mean"], "{:.3g}")])
    art = write_text(os.path.join(run_dir, "table_robustness.md"),
                     f"# E5 robustness ({a.ref}; controller nominal, plant perturbed; mean [95% CI over {len(seeds)} seeds])\n\n" +
                     md_table(["plant setting", "arm"] + list(PLOT_METRICS) + ["success rate", "solve mean [s]"], rows))
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(4*len(PLOT_METRICS), 3.6))
    xs = np.arange(len(summary["settings"]))
    for ax, k in zip(axes, PLOT_METRICS):
        for i, arm in enumerate(arms):
            m = np.array([summary["per_setting"][n][arm][k]["mean"] for n in summary["settings"]])
            lo = np.array([summary["per_setting"][n][arm][k]["lo"] for n in summary["settings"]])
            hi = np.array([summary["per_setting"][n][arm][k]["hi"] for n in summary["settings"]])
            ax.errorbar(xs + 0.08*(i - 1.5), m, yerr=[m - lo, hi - m], marker="o", ms=3, capsize=2, ls="-", color=COLORS[arm], label=LABELS[arm])
        ax.set_xticks(xs)
        ax.set_xticklabels(summary["settings"], rotation=60, ha="right", fontsize=7)
        ax.set_title(k)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    arts = [art] + savefig(fig, run_dir, "fig_degradation")
    finish(run_dir, cfg, a.seed, summary, arts, dict(quick=a.quick))


if __name__ == "__main__":
    main()
