"""
E1 -- Is the surrogate accurate?

200 test initial states x 20 control sequences (quick: 40 x 5).
  (a) one-step error at t in {0.25, 0.5, 0.75, 1.0} T   (tests the continuous-time property)
  (b) chained multi-step error for 1..50 steps
Arms: PINC, black-box (lambda = 0), linear (LTI linearised at 20 m/s straight
driving), RK4 with the MPC substep (dt = 0.01) -- all against the RK4 truth
(dt = 1e-3).  In-box and extrapolation-region ICs are reported separately.

Control sequences: Fx = trim(vx0) + sum of 3 random sinusoids (amplitude
<= 2000 N), delta = sum of 3 random sinusoids (amplitude <= 0.1 rad),
frequencies 0.1-1 Hz, held constant over each control period.
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import (COLORS, LABELS, base_parser, ci_str, finish, load_models, md_table,  # noqa: E402
                                savefig, start, write_text, plt)
from pinc import plant, plant_tf  # noqa: E402
from pinc.data import sample_box  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402
from pinc.mpc import PINCPredictor, RK4Predictor  # noqa: E402
from pinc.plant import trim_force  # noqa: E402

FRACS = (0.25, 0.5, 0.75, 1.0)
STATES = ("vx", "vy", "r", "psi")


def control_sequences(n_seq, n_steps, cfg, rng, vx0):
    """(n_seq, n_steps, 2) inputs for each IC, given vx0 (n_ic,) -> (n_ic, n_seq, n_steps, 2)."""
    T = cfg.T
    t = T*np.arange(n_steps)
    seqs = np.zeros((n_seq, n_steps, 2))
    for j in range(n_seq):
        for c, amp in ((0, 2000.0), (1, 0.1)):
            sig = np.zeros(n_steps)
            for _ in range(3):
                f = rng.uniform(0.1, 1.0)
                sig += rng.uniform(-1, 1)*np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))
            seqs[j, :, c] = amp*sig/3.0
    u = np.broadcast_to(seqs, (len(vx0),) + seqs.shape).copy()
    u[..., 0] += trim_force(vx0, cfg.params)[:, None, None]
    lo, hi = np.asarray(cfg.u_min), np.asarray(cfg.u_max)
    return np.clip(u, lo, hi)


def truth_rollout(s0, u, cfg):
    """RK4 truth (dt = sim.dt_plant): returns states after each step (B, n_steps, 4)
    and the first-step states at the fractional times (B, len(FRACS), 4)."""
    B, n_steps = u.shape[:2]
    x = np.concatenate([s0, np.zeros((B, 2))], axis=1)
    dt = cfg.sim.dt_plant
    n_sub = int(round(cfg.T/dt))
    out = np.zeros((B, n_steps, 4))
    frac = np.zeros((B, len(FRACS), 4))
    frac_steps = {int(round(f*n_sub)): i for i, f in enumerate(FRACS)}
    for k in range(n_steps):
        for j in range(1, n_sub + 1):
            x = plant.rk4_step(x, u[:, k], dt, cfg.params, cfg.sim.tyre)
            if k == 0 and j in frac_steps:
                frac[:, frac_steps[j]] = x[:, :4]
        out[:, k] = x[:, :4]
    return out, frac


class LinearLTI:
    """One-step map linearised at s_op = [v0, 0, 0, 0], u_op = [trim(v0), 0]."""
    def __init__(self, cfg, v0=20.0):
        self.cfg = cfg
        self.s_op = np.array([v0, 0.0, 0.0, 0.0])
        self.u_op = np.array([trim_force(v0, cfg.params), 0.0])
        self.cache = {}

    def lin(self, T):
        if T not in self.cache:
            s = tf.constant(self.s_op[None], tf.float64)
            u = tf.constant(self.u_op[None], tf.float64)
            n = max(1, int(round(T/self.cfg.mpc.dt_pred)))
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(s)
                tape.watch(u)
                s1 = plant_tf.simulate_tf(s, u, T, T/n, self.cfg.params)
            A = tape.batch_jacobian(s1, s)[0].numpy()
            Bm = tape.batch_jacobian(s1, u)[0].numpy()
            self.cache[T] = (s1.numpy()[0], A, Bm)
        return self.cache[T]

    def one_step(self, s0, u, T):
        c, A, Bm = self.lin(T)
        return c + (s0 - self.s_op) @ A.T + (u - self.u_op) @ Bm.T

    def rollout(self, s0, u):
        s, out = s0, []
        for k in range(u.shape[1]):
            s = self.one_step(s, u[:, k], self.cfg.T)
            out.append(s)
        return np.stack(out, axis=1)


def run_region(name, s0_ic, cfg, models, rng, n_seq, n_steps, summary, run_dir):
    n_ic = len(s0_ic)
    u = control_sequences(n_seq, n_steps, cfg, rng, s0_ic[:, 0])          # (n_ic, n_seq, n_steps, 2)
    s0 = np.repeat(s0_ic, n_seq, axis=0)                                   # (B, 4)
    uB = u.reshape(-1, n_steps, 2)
    truth, truth_frac = truth_rollout(s0, uB, cfg)
    ic_index = np.repeat(np.arange(n_ic), n_seq)

    arms = {}
    if "pinc" in models:
        arms["pinc"] = PINCPredictor(models["pinc"], cfg)
    if "blackbox" in models:
        arms["blackbox"] = PINCPredictor(models["blackbox"], cfg, name="blackbox")
    arms["rk4"] = RK4Predictor(cfg)
    lti = LinearLTI(cfg, cfg.refs.v0)

    preds_frac, preds_chain = {}, {}
    s0_t = tf.constant(s0, tf.float64)
    for arm, p in arms.items():
        pf = np.zeros_like(truth_frac)
        for i, f in enumerate(FRACS):
            if isinstance(p, PINCPredictor):
                pf[:, i] = p.model.predict_physical(np.full(len(s0), f*cfg.T), s0, uB[:, 0]).numpy()
            else:
                n = max(1, int(round(f*cfg.T/cfg.mpc.dt_pred)))
                pf[:, i] = plant_tf.simulate_tf(s0_t, tf.constant(uB[:, 0]), f*cfg.T, f*cfg.T/n, cfg.params).numpy()
        preds_frac[arm] = pf
        preds_chain[arm] = p.rollout_batch(s0_t, tf.constant(uB)).numpy()
    preds_frac["linear"] = np.stack([lti.one_step(s0, uB[:, 0], f*cfg.T) for f in FRACS], axis=1)
    preds_chain["linear"] = lti.rollout(s0, uB)

    S = cfg.S_x
    res = dict(n_ic=n_ic, n_seq=n_seq, n_steps=n_steps, onestep={}, chain={})
    for arm in preds_frac:
        res["onestep"][arm] = {}
        for i, f in enumerate(FRACS):
            e2 = ((preds_frac[arm][:, i] - truth_frac[:, i])/S)**2           # (B, 4)
            per_ic = np.stack([np.sqrt(np.mean(e2[ic_index == j], axis=0)) for j in range(n_ic)])  # (n_ic, 4)
            res["onestep"][arm][str(f)] = {STATES[s]: bootstrap_ci(per_ic[:, s]) for s in range(4)}
            res["onestep"][arm][str(f)]["all"] = bootstrap_ci(np.sqrt(np.mean(per_ic**2, axis=1)))
        e2 = ((preds_chain[arm] - truth)/S)**2                                # (B, n_steps, 4)
        finite = np.isfinite(e2).all(axis=(1, 2))
        curve = {}
        for h in range(n_steps):
            per_ic = np.stack([np.sqrt(np.nanmean(e2[(ic_index == j) & finite, h], axis=0)) for j in range(n_ic)])
            curve[h + 1] = {STATES[s]: bootstrap_ci(per_ic[:, s]) for s in range(4)}
            curve[h + 1]["all"] = bootstrap_ci(np.sqrt(np.mean(per_ic**2, axis=1)))
        res["chain"][arm] = dict(curve=curve, n_nonfinite=int((~finite).sum()))
    summary[name] = res
    np.savez_compressed(os.path.join(run_dir, f"raw_{name}.npz"), s0=s0, u=uB, truth=truth, truth_frac=truth_frac,
                        **{f"frac_{a}": v for a, v in preds_frac.items()}, **{f"chain_{a}": v for a, v in preds_chain.items()})
    return res


def main(argv=None):
    ap = base_parser(__doc__)
    a = ap.parse_args(argv)
    cfg, run_dir = start("e1_open_loop", a)
    models = load_models(a)
    n_ic, n_seq, n_steps = (40, 5, 50) if a.quick else (200, 20, 50)
    rng = np.random.default_rng(1000 + a.seed)
    summary = dict(quick=a.quick, models=dict(pinc=a.pinc_model, blackbox=a.blackbox_model))
    regions = dict(in_box=sample_box(n_ic, cfg.box_train, np.random.default_rng(cfg.seeds.test + 500)),
                   extrap=sample_box(n_ic, cfg.box_extrap, np.random.default_rng(cfg.seeds.test_extrap + 500)))
    for name, s0 in regions.items():
        print(f"  region {name}: {n_ic} ICs x {n_seq} sequences x {n_steps} steps")
        run_region(name, s0, cfg, models, rng, n_seq, n_steps, summary, run_dir)

    # ---- table: one-step NRMSE per state at t = T, plus chained at 10 and 50 steps
    arts = []
    for region in regions:
        rows = []
        r = summary[region]
        for arm in r["onestep"]:
            for f in FRACS:
                o = r["onestep"][arm][str(f)]
                rows.append([LABELS.get(arm, arm), f"{f:g} T"] + [ci_str(o[s], "{:.2e}") for s in STATES] + [ci_str(o["all"], "{:.2e}")])
        arts.append(write_text(os.path.join(run_dir, f"table_onestep_{region}.md"),
                               f"# E1 one-step NRMSE ({region}); mean [95% bootstrap CI over ICs]\n\n" +
                               md_table(["model", "t"] + list(STATES) + ["all"], rows)))
        rows = []
        for arm in r["chain"]:
            for h in (1, 10, 50):
                o = r["chain"][arm]["curve"][h]
                rows.append([LABELS.get(arm, arm), h] + [ci_str(o[s], "{:.2e}") for s in STATES] + [ci_str(o["all"], "{:.2e}"), r["chain"][arm]["n_nonfinite"]])
        arts.append(write_text(os.path.join(run_dir, f"table_chained_{region}.md"),
                               f"# E1 chained NRMSE vs horizon ({region})\n\n" +
                               md_table(["model", "steps"] + list(STATES) + ["all", "non-finite rollouts"], rows)))

    # ---- figure: error vs horizon
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, region in zip(axes, regions):
        for arm, c in summary[region]["chain"].items():
            hs = sorted(c["curve"])
            m = np.array([c["curve"][h]["all"]["mean"] for h in hs])
            lo = np.array([c["curve"][h]["all"]["lo"] for h in hs])
            hi = np.array([c["curve"][h]["all"]["hi"] for h in hs])
            ax.plot(hs, m, color=COLORS.get(arm, None), label=LABELS.get(arm, arm))
            ax.fill_between(hs, lo, hi, color=COLORS.get(arm, None), alpha=0.2)
        ax.set(xlabel="chained steps (x T)", title=region.replace("_", " "), yscale="log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NRMSE (all states)")
    axes[0].legend(fontsize=8)
    arts += savefig(fig, run_dir, "fig_error_vs_horizon")
    finish(run_dir, cfg, a.seed, summary, arts, dict(quick=a.quick))


if __name__ == "__main__":
    main()
