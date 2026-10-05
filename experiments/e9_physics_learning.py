"""
E9 -- Does the network actually learn the physics?

For each trained model (default: every E8 phase-1 model), measured on held-out data:
  1. physics residual  RMS of (ds/dt - f(s_hat, u)) / S_f per state, at fresh collocation points
     (computed for lambda = 0 models too: how much physics a purely data-driven net picks up);
  2. derivative error  RMS of (ds/dt - f(s_true, u)) / S_f along true trajectories -- whether the
     network's time derivative matches the true dynamics, not just its end values;
  3. horizon test      chained prediction NRMSE after 1, 10, 25, 50 control periods, in-domain and in
     the extrapolation region, same initial states and input sequences for every model.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, savefig, start, write_text, plt  # noqa: E402
from experiments.e1_open_loop import control_sequences, truth_rollout  # noqa: E402
from pinc.system import get_system  # noqa: E402
from pinc.config import RESULTS_DIR  # noqa: E402
from pinc.data import sample_box, sample_collocation, sample_trajectories, scale_inputs  # noqa: E402
from pinc.loss import forward_and_time_derivative, physics_residual  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402
from pinc.model import PINCNet  # noqa: E402
from pinc.mpc import PINCPredictor  # noqa: E402

STATES = ("vx", "vy", "r", "psi")
HORIZONS = (1, 10, 25, 50)


def default_models():
    from experiments.e8_vx_residual import PHASE1
    out = []
    for tag, ov, reuse in PHASE1:
        rid = reuse if reuse and os.path.exists(os.path.join(RESULTS_DIR, "models", reuse, "summary.json")) else f"vx_{tag}_s0"
        if os.path.exists(os.path.join(RESULTS_DIR, "models", rid, "summary.json")):
            out.append((tag, rid))
    return out


def rms(a, axis=0):
    return np.sqrt(np.mean(np.square(a), axis=axis))


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--models", default=None, help="comma list tag=run_id (default: all finished E8 phase-1 models)")
    ap.add_argument("--split", default="test", choices=("test", "val"),
                    help="held-out states for the measurements: test (reporting) or val (model selection)")
    a = ap.parse_args(argv)
    cfg, run_dir = start("e9_physics_learning", a)
    models = [tuple(x.split("=")) for x in a.models.split(",")] if a.models else default_models()
    n_c, n_ic, n_seq = (2000, 20, 5) if a.quick else (20000, 100, 10)

    base = cfg.seeds.val if a.split == "val" else cfg.seeds.test
    ex_seed = cfg.seeds.val + 1902 if a.split == "val" else cfg.seeds.test_extrap + 902
    c = sample_collocation(n_c, base + 900, cfg)
    z_c = tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))
    d = sample_trajectories(n_c, base + 901, cfg)
    z_d = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg))
    sysm = get_system(cfg)
    f_true = sysm.true_rates(d["s"], d["u"], cfg.params)

    rng = np.random.default_rng(4242)
    regions = {}
    for reg, box, seed in (("in_domain", cfg.box_train, base + 902), ("extrap", cfg.box_extrap, ex_seed)):
        s0 = sample_box(n_ic, box, np.random.default_rng(seed), cfg)
        u = control_sequences(n_seq, max(HORIZONS), cfg, rng, s0[:, 0]).reshape(-1, max(HORIZONS), 2)
        s0r = np.repeat(s0, n_seq, axis=0)
        truth, _ = truth_rollout(s0r, u, cfg)
        regions[reg] = (s0r, u, truth, np.repeat(np.arange(n_ic), n_seq))

    res, rows = {}, []
    curves = {}
    for tag, rid in models:
        d_ = os.path.join(RESULTS_DIR, "models", rid)
        net = PINCNet.load_from(d_)
        with open(os.path.join(d_, "summary.json")) as fh:
            lam = json.load(fh)["lam"]
        c_cfg = cfg
        cj = os.path.join(d_, "config.json")
        if os.path.exists(cj):
            from pinc.config import from_dict
            c_cfg = from_dict(json.load(open(cj)))
        # 1. residual with the model's own S_f would differ between variants; report it in the COMMON default S_f units
        R = physics_residual(net, z_c, cfg).numpy()
        # 2. derivative vs true dynamics
        _, dsdt = forward_and_time_derivative(net, z_d, cfg.S_x, cfg.T)
        D = (dsdt.numpy() - f_true)/cfg.S_f
        # 3. horizon
        pred = PINCPredictor(net, cfg)
        hz, hz_body = {}, {}
        for reg, (s0r, u, truth, ic) in regions.items():
            p = pred.rollout_batch(tf.constant(s0r), tf.constant(u)).numpy()
            e2 = ((p - truth)/cfg.S_x)**2
            hz[reg] = {h: bootstrap_ci([np.sqrt(np.mean(e2[ic == j, h - 1])) for j in range(n_ic)]) for h in HORIZONS}
            hz_body[reg] = {h: bootstrap_ci([np.sqrt(np.mean(e2[ic == j, h - 1, :4])) for j in range(n_ic)]) for h in HORIZONS}
            curves.setdefault(reg, {})[tag] = [float(np.sqrt(np.mean(e2[:, h]))) for h in range(max(HORIZONS))]
        res[tag] = dict(run_id=rid, lam=lam, increment_scaling=bool(getattr(net.mcfg, "increment_scaling", False)),
                        S_f_train=list(c_cfg.S_f), residual_rms=rms(R).tolist(), residual_all=float(rms(R.ravel())),
                        deriv_err_rms=rms(D).tolist(), deriv_err_all=float(rms(D.ravel())),
                        deriv_err_body=float(rms(D[:, :4].ravel())), horizon=hz, horizon_body=hz_body)
        r = res[tag]
        rows.append([tag, f"{lam:g}", " / ".join(f"{v:.2g}" for v in r["residual_rms"]), f"{r['residual_all']:.2g}",
                     " / ".join(f"{v:.2g}" for v in r["deriv_err_rms"]), f"{r['deriv_err_all']:.2g}"] +
                    [f"{hz['in_domain'][h]['mean']:.2e}" for h in HORIZONS] + [f"{hz['extrap'][h]['mean']:.2e}" for h in (10, 50)])
        print(f"  {tag:16s} residual {r['residual_all']:.3g}  deriv err {r['deriv_err_all']:.3g}  "
              f"h50 in {hz['in_domain'][50]['mean']:.3e} extrap {hz['extrap'][50]['mean']:.3e}", flush=True)

    hdr = (["model", "lambda", "residual vx/vy/r/psi", "residual all", "dx/dt error vx/vy/r/psi", "dx/dt error all"] +
           [f"in-domain h={h}" for h in HORIZONS] + ["extrap h=10", "extrap h=50"])
    text = ("# E9: is the physics learned?  (residual and derivative error in units of S_f = std of f over the domain; "
            "horizon columns: chained NRMSE after h control periods)\n\n" + md_table(hdr, rows))
    art = write_text(os.path.join(run_dir, "table_physics_learning.md"), text)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, reg in zip(axes, ("in_domain", "extrap")):
        for tag, cv in curves[reg].items():
            ax.plot(range(1, len(cv) + 1), cv, label=tag, ls="-" if res[tag]["lam"] > 0 else "--")
        ax.set(yscale="log", xlabel="prediction steps (x T)", title=reg)
    axes[0].set_ylabel("NRMSE (all states)")
    axes[0].legend(fontsize=6, ncol=2)
    arts = [art] + savefig(fig, run_dir, "fig_horizon")
    finish(run_dir, cfg, a.seed, dict(quick=a.quick, models=res), arts, dict(quick=a.quick))
    print(text)


if __name__ == "__main__":
    main()
