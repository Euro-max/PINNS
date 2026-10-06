"""
E24 -- Tuning the anchor of the prior-anchored PINC network (E21 A8).

Settings (pinc/model.py, anchored; the defaults are A8):
  T1 tau_w   the wheel-slip time constant of the anchor learned (starts at 10 ms)
  T2 exp     actuator states by the exact first-order lag solution instead of an Euler step
  T3 end     quasi-steady slip target at the advanced body and actuator states instead of the initial state
  T4 gain    a learnable gain per state on the anchor, s = s0 + g (anchor - s0) + tau D NN
  T5 all     T1-T4 together
HF-M0: T1-T5 at N = 100 (lambda 0.01) and 1000 (lambda 0.001); HF-M1: T4 and T5 at N = 100 and 1000 (lambda 1e-3).
3 seeds, against the A8 runs of E21 / E22 with the same lambda, data, seeds and budget, so a difference comes
from the anchor setting alone.  --timing (idle machine): MPC solve time at N = 10 on one CPU thread for every
setting against A8, flagged when a setting adds more than 10 %; none of them adds a solver, only a few
element-wise operations.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, group_rms  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides, prewarm  # noqa: E402
from experiments.e17_hf_compare import cmp_cell, compare  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

SEEDS = (0, 1, 2)
SIZES = (100, 1000)
SETTINGS = {
    "atw": ("T1 learned wheel time constant", {"model.anchor_learn_tau_w": "true"}),
    "aexp": ("T2 exact actuator lag", {"model.anchor_actuator": "exp"}),
    "aend": ("T3 slip target at the end", {"model.anchor_slip_at": "end"}),
    "again": ("T4 anchor gain", {"model.anchor_gain": "true"}),
    "aall": ("T5 all", {"model.anchor_learn_tau_w": "true", "model.anchor_actuator": "exp",
                        "model.anchor_slip_at": "end", "model.anchor_gain": "true"}),
}
M1_SETTINGS = ("again", "aall")


def lam_of(variant, n):
    return {100: 0.01, 1000: 0.001}[n] if variant == "m0" else 1e-3


def rid(variant, n, tag, seed):
    return f"hf{variant}_n{n}_lam{lam_of(variant, n):g}_anchored" + (f"_{tag}" if tag else "") + f"_s{seed}"


def tags_for(variant):
    return [""] + [t for t in SETTINGS if variant == "m0" or t in M1_SETTINGS]


def learned(r):
    """Learned wheel time constant [ms] and anchor gains of a saved model (None when not learned)."""
    from pinc.model import PINCNet
    net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", r))
    tw = float(np.exp(net.log_tau_w.numpy()))*1e3 if hasattr(net, "log_tau_w") else None
    g = net.anchor_g.numpy().tolist() if hasattr(net, "anchor_g") else None
    return tw, g


def timing(cfg, variant, run_dir):
    from pinc.metrics import solve_time_stats
    from pinc.model import PINCNet
    from pinc.mpc import make_controller
    from pinc.refs import make_reference
    from pinc.sim import simulate
    from pinc.system import get_system
    ref = make_reference("lane_change", cfg)
    x0 = get_system(cfg).initial_state(ref.x0())
    out = {}
    for tag in tags_for(variant):
        net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", rid(variant, 1000, tag, 0)))
        ctrl = make_controller("pinc", cfg, {"pinc": net}, ref.Q, ref.P)
        log = simulate(ctrl, None, ref, x0, 3.0, np.zeros(x0.size), 0, cfg)
        st = solve_time_stats(log["solve_time"][1:])
        out[tag or "A8"] = dict(median=st["median"], p95=st["p95"], nit=float(np.mean(log["nit"][1:])),
                                per_iteration=float(np.mean(log["solve_time"][1:]/np.maximum(log["nit"][1:], 1))))
    base = out["A8"]["per_iteration"]
    for k, v in out.items():
        v["per_iteration_vs_A8"] = v["per_iteration"]/base
        v["flag"] = v["per_iteration_vs_A8"] > 1.10
        print(f"  {k:6s} solve {v['median']*1e3:6.1f} ms  per iteration {v['per_iteration']*1e3:.3f} ms "
              f"({v['per_iteration_vs_A8']:.2f}x A8){'  > 10 % slower' if v['flag'] else ''}", flush=True)
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
    _, run_dir = start("e24_anchor_tuning", a)
    cfg = load_config(cfg_path, a.overrides)
    if a.timing:
        timing(cfg, a.variant, run_dir)
        return
    seeds = SEEDS[:1] if a.quick else SEEDS
    sizes = SIZES[:1] if a.quick else SIZES
    sfx = "_quick" if a.quick else ""
    jobs = []
    for n in sizes:
        for tag in tags_for(a.variant):
            for k in seeds:
                ov = dict(overrides(n, lam_of(a.variant, n), k, cfg), **{"model.arch": "anchored"},
                          **(SETTINGS[tag][1] if tag else {}))
                if a.quick:
                    ov.update({"train.steps": 50, "train.lbfgs_iters": 20})
                jobs.append((rid(a.variant, n, tag, k) + sfx, k, ov))
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
    metrics = dict(h50_val=("val 50-step body", lambda r: ev["val"][r]["horizon_body"]["in_domain"]["50"]["mean"]),
                   one_step=("one-step body", lambda r: group_rms(summary(r)["test"]["nrmse"], GROUPS["body"])),
                   h10=("10-step body", lambda r: ev["test"][r]["horizon_body"]["in_domain"]["10"]["mean"]),
                   h50=("50-step body", lambda r: ev["test"][r]["horizon_body"]["in_domain"]["50"]["mean"]),
                   h50_extrap=("50-step body, outside training range",
                               lambda r: ev["test"][r]["horizon_body"]["extrap"]["50"]["mean"]))
    label = {"": "A8 (untuned anchor)", **{t: v[0] for t, v in SETTINGS.items()}}
    rec, rows, crows, lrows = {}, [], [], []
    for n in sizes:
        rec[n] = {}
        for tag in tags_for(a.variant):
            rs = [rid(a.variant, n, tag, k) + sfx for k in seeds]
            rec[n][tag or "A8"] = {m: [fn(r) for r in rs] for m, (_, fn) in metrics.items()}
            rows.append([n, label[tag]] + [ci_cell(rec[n][tag or "A8"][m]) for m in metrics])
            if tag:
                if len(seeds) > 1:
                    c = {m: compare(rec[n][tag][m], rec[n]["A8"][m]) for m in metrics}
                    rec[n][f"{tag} vs A8"] = c
                    crows.append([n, label[tag]] + [cmp_cell(c[m]) for m in metrics])
                lv = [learned(r) for r in rs]
                tw = [x[0] for x in lv if x[0] is not None]
                g = [x[1] for x in lv if x[1] is not None]
                rec[n][f"{tag} learned"] = dict(tau_w_ms=tw, gain=g)
                if tw or g:
                    lrows.append([n, label[tag], " / ".join(f"{v:.1f}" for v in tw) or "-",
                                  ", ".join(f"{x:.2f}" for x in np.mean(g, axis=0)) if g else "-"])
    labels = [lbl for lbl, _ in metrics.values()]
    text = (f"# E24 anchor settings of the prior-anchored network, HF-{a.variant.upper()} (test errors unless noted; "
            f"mean [95% CI] over {len(seeds)} seeds)\n\n" + md_table(["N", "setting"] + labels, rows) +
            ("\n## Against A8 (error of A8 divided by the error of the setting, above 1: the setting is better; seeds won; "
             "paired t-test on log errors)\n\n" + md_table(["N", "setting"] + labels, crows) if crows else "") +
            ("\n## Learned values (wheel time constant per seed [ms]; anchor gain per state, mean over seeds, "
             "order vx vy r psi F delta sig_fl sig_fr sig_rl sig_rr)\n\n" +
             md_table(["N", "setting", "tau_w [ms]", "gain"], lrows) if lrows else ""))
    tp = os.path.join(run_dir, "timing.json")
    if os.path.exists(tp):
        with open(tp) as fh:
            tim = json.load(fh)
        text += "\n## MPC solve time at N = 10 (one CPU thread, seed-0 model at N = 1000)\n\n" + md_table(
            ["setting", "median [ms]", "per iteration [ms]", "per iteration vs A8"],
            [[label.get("" if k == "A8" else k, k), f"{v['median']*1e3:.1f}", f"{v['per_iteration']*1e3:.3f}",
              f"{v['per_iteration_vs_A8']:.2f}" + (" (> 10 % slower)" if v["flag"] else "")] for k, v in tim.items()])
    art = write_text(os.path.join(run_dir, "table_anchor_tuning.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, seeds=list(seeds), results={str(n): v for n, v in rec.items()}),
           [art], dict(quick=a.quick))
    print(text)


if __name__ == "__main__":
    main()
