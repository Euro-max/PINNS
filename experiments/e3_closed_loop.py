"""
E3 -- Closed-loop tracking.
4 references x 30 seeds (quick: 2) x 4 arms (NMPC-RK4, PINC-MPC,
black-box-MPC, LTV-MPC).  Every arm runs against the same RK4 plant with the
nominal parameters, measurement noise and an x0 perturbation drawn from the
seed.  Table of metrics with 95% bootstrap CIs and paired Wilcoxon tests
against NMPC-RK4; time-series figures for seed 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import COLORS, LABELS, base_parser, ci_str, finish, load_models, md_table, savefig, start, write_text, plt  # noqa: E402
from pinc.metrics import bootstrap_ci, paired_wilcoxon  # noqa: E402
from pinc.mpc import ARMS, make_controller  # noqa: E402
from pinc.refs import REFERENCES, make_reference  # noqa: E402
from pinc.sim import closed_loop_metrics, perturb_x0, simulate  # noqa: E402

METRICS = ("iae_vx", "rmse_vx", "max_vx", "iae_psi", "rmse_psi", "iae_Y", "rmse_Y", "max_Y", "effort",
           "viol_time_r", "solve_mean", "solve_p95", "nit_mean", "success_rate", "rise_time", "settling_time")


def run_all(cfg, arms, refs, seeds, models, run_dir, duration, verbose=False, plant_params=None, tyre=None,
            disturbance=None, save_logs=True):
    """Returns results[ref][arm] = list of (seed, metrics) and logs[ref][arm][seed] for seed 0."""
    plant_params = plant_params or cfg.params
    tyre = tyre or cfg.sim.tyre
    results = {r: {a: [] for a in arms} for r in refs}
    logs0 = {r: {} for r in refs}
    for rname in refs:
        ref = make_reference(rname, cfg)
        for arm in arms:
            ctrl = make_controller(arm, cfg, models, ref.Q, ref.P)
            for seed in seeds:
                rng = np.random.default_rng(10_000 + seed)
                x0 = perturb_x0(ref.x0(), cfg.sim.x0_sigma, rng)
                log = simulate(ctrl, plant_params, ref, x0, duration, cfg.sim.noise_sigma, seed, cfg, tyre, disturbance)
                m = closed_loop_metrics(log, cfg, ref)
                results[rname][arm].append((seed, m))
                if seed == seeds[0]:
                    logs0[rname][arm] = log
                if save_logs:
                    np.savez_compressed(os.path.join(run_dir, f"log_{rname}_{arm}_s{seed}.npz"), **log)
                if verbose:
                    print(f"  {rname:18s} {arm:9s} seed {seed:2d}: rmse_vx {m['rmse_vx']:.3f} rmse_Y {m['rmse_Y']:.3f} "
                          f"solve {m['solve_mean']*1e3:.1f} ms ok {m['success_rate']:.2f}", flush=True)
    return results, logs0


def summarise(results, arms, baseline="nmpc_rk4"):
    out = {}
    for rname, per_arm in results.items():
        out[rname] = {}
        for arm in arms:
            vals = {k: [m.get(k, np.nan) for _, m in per_arm[arm]] for k in METRICS}
            out[rname][arm] = {k: bootstrap_ci(v) for k, v in vals.items()}
            out[rname][arm]["n"] = len(per_arm[arm])
            if arm != baseline and baseline in per_arm:
                base = {k: [m.get(k, np.nan) for _, m in per_arm[baseline]] for k in METRICS}
                out[rname][arm]["vs_baseline"] = {k: paired_wilcoxon(vals[k], base[k]) for k in METRICS}
    return out


def tables(summary, arms, refs, run_dir, title):
    arts = []
    for rname in refs:
        rows = []
        for arm in arms:
            s = summary[rname][arm]
            row = [LABELS[arm]]
            for k in METRICS:
                if k in ("rise_time", "settling_time") and rname != "speed_step":
                    continue
                cell = ci_str(s[k])
                if "vs_baseline" in s:
                    w = s["vs_baseline"][k]
                    cell += f" (p={w['p']:.2g}, r={w['effect']:+.2f})"
                row.append(cell)
            rows.append(row)
        hdr = ["arm"] + [k for k in METRICS if not (k in ("rise_time", "settling_time") and rname != "speed_step")]
        arts.append(write_text(os.path.join(run_dir, f"table_{rname}.md"),
                               f"# {title}: {rname} (mean [95% CI]; paired Wilcoxon p and rank-biserial r vs NMPC-RK4)\n\n" +
                               md_table(hdr, rows)))
    return arts


def timeseries_figure(logs0, rname, arms, run_dir, seed_label):
    fig, axes = plt.subplots(5, 1, figsize=(7, 10), sharex=True)
    any_log = next(iter(logs0.values()))
    t = any_log["t"]
    axes[0].plot(t, any_log["ref"][:, 0], "--", color="k", label="reference")
    axes[1].plot(t, any_log["ref"][:, 3], "--", color="k")
    axes[2].plot(t, any_log["ref"][:, 5], "--", color="k")
    for arm in arms:
        if arm not in logs0:
            continue
        lg = logs0[arm]
        c = COLORS[arm]
        axes[0].plot(t, lg["x"][:, 0], color=c, label=LABELS[arm])
        axes[1].plot(t, lg["x"][:, 3], color=c)
        axes[2].plot(t, lg["x"][:, 5], color=c)
        axes[3].step(t[:-1], lg["u"][:, 0], where="post", color=c)
        axes[4].step(t[:-1], lg["u"][:, 1], where="post", color=c)
    for ax, yl in zip(axes, ("vx [m/s]", "psi [rad]", "Y [m]", "Fx [N]", "delta [rad]")):
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_title(f"{rname} (seed {seed_label})")
    axes[-1].set_xlabel("time [s]")
    return savefig(fig, run_dir, f"fig_timeseries_{rname}")


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--refs", default=",".join(REFERENCES))
    ap.add_argument("--n-seeds", type=int, default=None)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e3_closed_loop", a)
    models = load_models(a)
    arms = [x for x in a.arms.split(",") if x in ("nmpc_rk4", "ltv") or x in models]
    refs = a.refs.split(",")
    n_seeds = a.n_seeds or (2 if a.quick else 30)
    seeds = list(range(n_seeds))
    results, logs0 = run_all(cfg, arms, refs, seeds, models, run_dir, cfg.sim.duration, verbose=True)
    summary = dict(quick=a.quick, arms=arms, refs=refs, seeds=seeds, representative_seed=seeds[0],
                   per_ref=summarise(results, arms),
                   raw={r: {arm: [dict(seed=s, **m) for s, m in v] for arm, v in per.items()} for r, per in results.items()})
    arts = tables(summary["per_ref"], arms, refs, run_dir, "E3 closed-loop tracking")
    for rname in refs:
        arts += timeseries_figure(logs0[rname], rname, arms, run_dir, seeds[0])
    finish(run_dir, cfg, a.seed, summary, arts, dict(quick=a.quick))


if __name__ == "__main__":
    main()
