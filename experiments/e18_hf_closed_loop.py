"""
E18 (plan H5, reduced) -- Closed-loop tracking on the high-fidelity plant.

Every controller runs against the high-fidelity plant (variant M0 or M1) with measurement noise on all
plant states.  Controllers:
  NMPC-prior   RK4 with the simplified prior and nominal parameters (what physics alone gives)
  NMPC-true    RK4 with the true plant model (the accuracy ceiling; about 8 s per solve, so fewer runs)
  data-only, PINC, PINC-theta (M1 only), grey-box   the E17 models trained on N trajectories
  grey-box-qs  the E20 grey-box model with the quasi-steady prior (10 ms RK4)
  distilled    the E19 network taught by the grey-box model
  A8-*         the E22 prior-anchored networks (data-only, PINC, PINC-theta)
References: speed sinusoid, speed step, lane change at 20 m/s over 40 m (about 4 m/s^2 peak lateral acceleration),
the same over 30 m (about 8 m/s^2, inside the training speed range of 5-25 m/s) and the ISO double lane change
at 12 m/s (about 14 m/s^2 asked for, beyond the grip limit).

Runs: each learned arm uses its five training seeds with two noise seeds each (noise seeds m and m + 5 for
model seed m), so each arm has ten runs per reference covering model and noise variation; NMPC-prior runs
noise seeds 0-9 and NMPC-true the first --true-seeds of them.  Comparisons are paired by noise seed
(Wilcoxon signed-rank).  Each run is saved on its own, so the experiment can be split over processes
(--arms) and resumed; --report builds the tables from the saved runs.  Solve times are measured on one
CPU thread with the GPU hidden, after a warm-up solve.
"""
import glob
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e17_hf_compare import SEEDS, arms_for  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.metrics import bootstrap_ci, paired_wilcoxon  # noqa: E402

REFS = {"speed_sin": ("speed_sin", {}, 10.0),
        "speed_step": ("speed_step", {}, 10.0),
        "lane_change": ("lane_change", {}, 6.0),
        "lane_change_short": ("lane_change", {"refs.slc_length": 30.0}, 6.0),
        "double_lane_change": ("iso_lane_change", {}, 6.5)}          # smooth path through the ISO 3888-2 cones
CTRL = {"data-only": "blackbox", "PINC": "pinc", "PINC-theta": "pinc", "grey-box": "greybox", "grey-box-qs": "greybox",
        "PINC-ablation": "pinc", "anchored data-only": "pinc", "anchored PINC": "pinc",
        "distilled": "blackbox", "A8-data-only": "pinc", "A8-PINC": "pinc", "A8-PINC-theta": "pinc"}
NMPC = ("NMPC-prior", "NMPC-true", "NMPC-qs", "NMPC-exact", "LTV")
NMPC_CTRL = {"NMPC-prior": "nmpc_rk4", "NMPC-true": "nmpc_true", "NMPC-qs": "nmpc_qs", "NMPC-exact": "nmpc_rk4", "LTV": "ltv"}
MODEL_SEEDS = list(SEEDS)              # training seeds of the learned controllers (--model-seeds)
REGISTRY = None                        # {N: {arm: run id template}} from E28 / E27 (--registry)
LOST_Y = 1.0                      # lateral error above 1 m at any time: off the lane centre by more than a car half-width (lane ~3.5 m)
METRICS = ("rmse_vx", "rmse_Y", "max_Y", "rmse_psi", "effort", "solve_median", "success_rate")


def runs_of(arm, true_seeds):
    """[(model seed or None, noise seed)].  The i-th training seed runs with noise seeds i and i + 5, so every
    learned controller meets the same noise seeds 0-9 as the NMPC controllers, whatever its training seeds."""
    k = len(MODEL_SEEDS)
    if arm == "NMPC-true":
        return [(None, j) for j in range(true_seeds)]
    if arm in NMPC:
        return [(None, j) for j in range(2*k)]
    return [(m, j) for i, m in enumerate(MODEL_SEEDS) for j in (i, i + k)]


def run_part(cfg0, variant, sizes, arms, refs, true_seeds, run_dir):
    import gc
    import tensorflow as tf
    from pinc.model import PINCNet
    from pinc.mpc import make_controller
    from pinc.refs import make_reference
    from pinc.sim import closed_loop_metrics, perturb_x0, simulate
    from pinc.system import get_system
    lam_star = None
    if REGISTRY is None:
        with open(os.path.join(RESULTS_DIR, "e15_hf_data", "e15_m0", "summary.json")) as fh:
            lam_star = {int(k): v for k, v in json.load(fh)["best_lambda"].items()}
    out_dir = os.path.join(run_dir, "runs")
    os.makedirs(out_dir, exist_ok=True)
    for rname in refs:
        ref_name, ov, duration = REFS[rname]
        cfg = cfg0.with_overrides(ov)
        ref = make_reference(ref_name, cfg)
        x_start = get_system(cfg).initial_state(ref.x0())
        for arm in arms:
            for n in (sizes if arm not in NMPC else [0]):
                todo = [(m, k) for m, k in runs_of(arm, true_seeds)
                        if not os.path.exists(os.path.join(out_dir, f"{rname}_{arm}_N{n}_m{m}_k{k}.json"))]
                by_model = {}
                for m, k in todo:
                    by_model.setdefault(m, []).append(k)
                for m, ks in by_model.items():
                    if arm in NMPC:
                        ctrl = make_controller(NMPC_CTRL[arm], cfg, {}, ref.Q, ref.P)
                    elif REGISTRY is not None:
                        net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", REGISTRY[str(n)][arm].format(seed=m)))
                        ctrl = make_controller(CTRL[arm], cfg, {CTRL[arm]: net}, ref.Q, ref.P)
                    else:
                        extra = {"grey-box-qs": f"hf{variant}_n{n}_greyboxqs_s{m}",               # E20
                                 "distilled": f"hf{variant}_n{n}_distill2_s{m}"}                   # E19
                        from experiments.e22_anchored_confirm import a8_arms                      # E22
                        extra.update({k.replace(" ", "-"): t.format(seed=m) for k, (t, _, _) in a8_arms(variant, n).items()})
                        rid = extra.get(arm) or arms_for(variant, n, lam_star)[arm][0].format(seed=m)
                        net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", rid))
                        ctrl = make_controller(CTRL[arm], cfg, {CTRL[arm]: net}, ref.Q, ref.P)
                    ctrl(0.0, x_start, ref)                          # compile before timing
                    for k in ks:
                        rng = np.random.default_rng(10_000 + k)
                        x0 = perturb_x0(x_start, cfg.sim.x0_sigma, rng)
                        log = simulate(ctrl, cfg.params if variant == "st" else None, ref, x0, duration, cfg.sim.noise_sigma, k,
                                       cfg, cfg.sim.tyre)
                        met = closed_loop_metrics(log, cfg, ref)
                        tag = f"{rname}_{arm}_N{n}_m{m}_k{k}"
                        with open(os.path.join(out_dir, tag + ".json"), "w") as fh:
                            json.dump(dict(ref=rname, arm=arm, N=n, model_seed=m, noise_seed=k, **met), fh)
                        if k == 0 or (m is not None and k == m == 0):
                            np.savez_compressed(os.path.join(out_dir, tag + ".npz"), **log)
                        print(f"  {tag}: rmse vx {met['rmse_vx']:.3f} Y {met['rmse_Y']:.3f} max Y {met['max_Y']:.3f} "
                              f"solve {met['solve_median']*1e3:.0f} ms ok {met['success_rate']:.2f}", flush=True)
                    del ctrl                                          # compiled cost functions hold GBs (grey-box, NMPC)
                    gc.collect()
                    tf.keras.backend.clear_session()


def report(run_dir, sizes, cfg, a):
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(run_dir, "runs", "*.json"))]
    def get(ref, arm, n):
        return {r["noise_seed"]: r for r in runs if r["ref"] == ref and r["arm"] == arm and r["N"] == n}
    arms = sorted({(r["arm"], r["N"]) for r in runs}, key=lambda x: (x[1], list(CTRL).index(x[0]) if x[0] in CTRL else -1))
    text, summ = f"# E18 closed loop on HF-{a.variant.upper()} (mean [95% CI] over runs)\n", {}
    for rname in REFS:
        rows, crows = [], []
        summ[rname] = {}
        for arm, n in arms:
            g = get(rname, arm, n)
            if not g:
                continue
            label = arm if arm in NMPC else f"{arm}, N = {n}"
            st = {k: bootstrap_ci([r[k] for r in g.values()]) for k in METRICS}
            lost = sum(r["max_Y"] > LOST_Y for r in g.values())
            med = {k: float(np.median([r[k] for r in g.values()])) for k in ("rmse_vx", "rmse_Y")}
            summ[rname][label] = dict(n_runs=len(g), lost=lost, median=med, **st)
            rows.append([label, len(g), lost, f"{med['rmse_vx']:.3g}", f"{med['rmse_Y']:.3g}"] +
                        [f"{st[k]['mean']:.3g} [{st[k]['lo']:.3g}, {st[k]['hi']:.3g}]" for k in METRICS])
            for base_arm, base_n in (("NMPC-prior", 0), ("data-only", n)):
                if arm in NMPC or arm == base_arm:
                    continue
                b = get(rname, base_arm, base_n)
                common = sorted(set(g) & set(b))
                if len(common) < 3:
                    continue
                cells = []
                for k in ("rmse_vx", "rmse_Y", "max_Y"):
                    x, y = [g[s][k] for s in common], [b[s][k] for s in common]
                    w = paired_wilcoxon(x, y)
                    cells.append(f"{np.mean(y)/np.mean(x):.2f} ({sum(xi < yi for xi, yi in zip(x, y))}/{len(common)}, p = {w['p']:.2g})")
                crows.append([label, base_arm] + cells)
        text += (f"\n## {rname}\n\n" + md_table(["controller", "runs", f"runs with max Y error > {LOST_Y:g} m", "median rmse_vx", "median rmse_Y"] + list(METRICS), rows) +
                 ("\nError of the second controller divided by the error of the first (above 1: the first is better); "
                  "runs won; paired Wilcoxon p\n\n" + md_table(["controller", "against", "rmse_vx", "rmse_Y", "max_Y"], crows)
                  if crows else ""))
    art = write_text(os.path.join(run_dir, "table_closed_loop.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, sizes=list(sizes), refs=REFS, per_ref=summ), [art])
    print(text)


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--variant", default="m0")
    ap.add_argument("--sizes", default="100,1000")
    ap.add_argument("--arms", default=None, help="comma list (default: every arm of the variant and both NMPC arms)")
    ap.add_argument("--refs", default=",".join(REFS))
    ap.add_argument("--true-seeds", type=int, default=2)
    ap.add_argument("--report", action="store_true", help="only build the tables from the saved runs")
    ap.add_argument("--registry", default=None, help="registry.json (E28 / E27) mapping N and arm to run ids")
    ap.add_argument("--model-seeds", default=None, help="training seeds of the learned controllers (default 0-4)")
    a = ap.parse_args(argv)
    global REGISTRY, MODEL_SEEDS
    if a.registry:
        with open(a.registry) as fh:
            reg = json.load(fh)
        REGISTRY, MODEL_SEEDS = reg["arms"], list(reg["seeds"])
    if a.model_seeds:
        MODEL_SEEDS = [int(x) for x in a.model_seeds.split(",")]
    a.config = os.path.join(ROOT, "configs", "default.yaml" if a.variant == "st" else f"hf_{a.variant}.yaml")
    if a.threads is None:
        a.threads = 1
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    cfg, run_dir = start("e18_hf_closed_loop", a)
    cfg = load_config(a.config, a.overrides)
    sizes = [int(x) for x in a.sizes.split(",")]
    if not a.report:
        if REGISTRY is not None:
            default = (["NMPC-exact", "LTV"] if a.variant == "st" else ["NMPC-prior", "NMPC-qs", "NMPC-true"]) + \
                      sorted({arm for v in REGISTRY.values() for arm in v})
        else:
            default = ["NMPC-prior", "NMPC-true"] + list(arms_for(a.variant, sizes[0], {s: 0.0 for s in sizes}))
        arms = a.arms.split(",") if a.arms else default
        run_part(cfg, a.variant, sizes, arms, a.refs.split(","), a.true_seeds, run_dir)
    else:
        report(run_dir, sizes, cfg, a)


if __name__ == "__main__":
    main()
