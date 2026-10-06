"""
Grey-box model (plan §5): the prior's own prediction plus a learned correction,

    s(t) = Phi_P(t; s0, u) + [NN(t, s0, u) - s0]

where Phi_P is the prior (nominal parameters) integrated with RK4 and NN is the same network as PINC with
the hard initial condition, so the correction is zero at t = 0.  The network is trained on data only
(lambda = 0) with the target s(t) - Phi_P(t) + s0; the error on that target equals the error on s(t),
so the training, validation and test metrics compare directly with PINC and data-only.

The plan wrote the grey-box model in continuous time (ds/dt = f_P + NN, integrated by RK4).  Training that
model means back-propagating through every RK4 substep of a stiff prior (100 per control period); the
discrete form above learns the same thing from the same data at the cost of one prior simulation per
training sample, computed once.  In the MPC it costs one prior simulation (as NMPC with the prior) plus one
network call per step.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import Config
from .system import get_system


def prior_flow(t, s0, u, cfg: Config, dt: float | None = None, batch: int = 20000) -> np.ndarray:
    """Phi_P(t; s0, u): the prior with nominal parameters integrated by RK4 with step `dt` (default the
    plant step, on whose grid the training times lie) up to each sample's own time t."""
    dt = dt or cfg.sim.dt_plant
    sysm = get_system(cfg)
    t, s0, u = np.asarray(t, float), np.asarray(s0, float), np.asarray(u, float)
    k = np.rint(t/dt).astype(int)
    if np.any(np.abs(k*dt - t) > 1e-9):
        raise ValueError("sample times must lie on the integration grid")
    out = np.empty_like(s0)
    step = tf.function(lambda s, u: sysm.rk4_step_s_tf(s, u, dt, cfg.params))
    for i in range(0, len(t), batch):
        sl = slice(i, i + batch)
        s, uu, kk = tf.constant(s0[sl]), tf.constant(u[sl]), k[sl]
        res = s0[sl].copy()
        for j in range(1, int(kk.max(initial=0)) + 1):
            s = step(s, uu)
            sel = kk == j
            if np.any(sel):
                res[sel] = s.numpy()[sel]
        out[sl] = res
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite prior prediction")
    return out


QS_DT = 0.01        # RK4 step of the quasi-steady prior: 10 steps per control period


def prior_flow_qs(t, s0, u, cfg: Config) -> np.ndarray:
    """The quasi-steady prior (prior_hf.f_s_qs; wheel slip set to its quasi-steady value) up to each sample's
    time t, with ceil(t / QS_DT) equal RK4 steps, so t = T gives exactly the MPC's prediction step."""
    sysm = get_system(cfg)
    t, s0, u = np.asarray(t, float), np.asarray(s0, float), np.asarray(u, float)
    n = np.maximum(1, np.ceil(t/QS_DT - 1e-9).astype(int))
    out = s0.copy()
    for k in np.unique(n[t > 0]):
        sel = (n == k) & (t > 0)
        dt = tf.constant((t[sel]/k).reshape(-1, 1))
        out[sel] = sysm.prior_qs_step_tf(tf.constant(s0[sel]), tf.constant(u[sel]), dt, int(k)).numpy()
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite prior prediction")
    return out


def prior_of(kind: str):
    if kind == "full":
        return prior_flow
    if kind == "qs":
        return prior_flow_qs
    raise ValueError(f"grey-box prior must be full or qs, got {kind!r}")


def to_residual(split: dict, cfg: Config) -> dict:
    """The grey-box training target s(t) - Phi_P(t) + s0 for one data split."""
    phi = prior_of(getattr(cfg.model, "greybox_prior", "full"))(split["t"], split["s0"], split["u"], cfg)
    return dict(split, s=split["s"] - phi + split["s0"])


def teacher_samples(run_id: str, n: int, seed: int, cfg: Config) -> dict:
    """Training samples labelled by a trained grey-box model (a run id or model directory; distillation): states from the collocation pool
    (the unlabelled states the physics loss also uses), uniform inputs, times on the plant grid in (0, T]."""
    import os
    from .config import RESULTS_DIR
    from .data import sample_box, sample_inputs
    from .model import PINCNet
    net = PINCNet.load_from(run_id if os.path.isdir(run_id) else os.path.join(RESULTS_DIR, "models", run_id))
    if not getattr(net.mcfg, "greybox", False):
        raise ValueError(f"{run_id} is not a grey-box model")
    rng = np.random.default_rng(seed)
    dt = cfg.sim.dt_plant
    s0 = sample_box(n, cfg.box_train, rng, cfg, kind="colloc", seed=seed)
    u = sample_inputs(n, cfg, rng)
    t = dt*rng.integers(1, int(round(cfg.T/dt)) + 1, size=n)
    s = prior_of(getattr(net.mcfg, "greybox_prior", "full"))(t, s0, u, cfg) + net.predict_physical(t, s0, u).numpy() - s0
    mask = list(getattr(cfg.train, "distill_mask", []) or [])
    if mask:
        s[:, np.asarray(mask) == 0] = np.nan          # left to the real data (the data loss skips missing targets)
    return dict(t=t.astype(float), s0=s0, u=u, s=s)
