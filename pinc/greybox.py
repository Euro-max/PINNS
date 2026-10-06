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


def to_residual(split: dict, cfg: Config) -> dict:
    """The grey-box training target s(t) - Phi_P(t) + s0 for one data split."""
    phi = prior_flow(split["t"], split["s0"], split["u"], cfg)
    return dict(split, s=split["s"] - phi + split["s0"])
