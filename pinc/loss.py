"""
Data, initial-condition and physics losses for PINC training.

`time_derivative` returns ds/dt in PHYSICAL units, shape (B, n_s): it
differentiates every output w.r.t. input column 0 ONLY (the scaled time
t/T), then applies the chain rule 1/T and the output scale S_x.  This
fixes D1 (sum-of-outputs gradient), D2 (wrong input column) and D3
(missing 1/T).

Physics residual: R = (ds/dt - f(s, u)) / S_f with f the physics prior of the
system (`pinc/system.py`; bicycle: plant_tf on the first four states) and S_f
a per-state characteristic rate.  All residuals are included (config
`loss.residual_mask` allows ablation).

Network input layout: z = [t/T, s0/S_x (n_s), u/S_u (n_u)].
"""
from __future__ import annotations

import tensorflow as tf

from .config import Config
from .system import get_system


def _call(model, z, training):
    try:
        return model(z, training=training)
    except TypeError:                        # plain callables in the tests
        return model(z)


def _forward_with_jvp(model, z, training=False):
    """One forward pass + forward-mode JVP along the time input (column 0)."""
    tangent = tf.concat([tf.ones_like(z[:, :1]), tf.zeros_like(z[:, 1:])], axis=1)
    with tf.autodiff.ForwardAccumulator(z, tangent) as acc:
        s_hat = _call(model, z, training)
    ds_dtau = acc.jvp(s_hat)
    if ds_dtau is None:                      # model does not depend on t
        ds_dtau = tf.zeros_like(s_hat)
    return s_hat, ds_dtau


def _forward_with_reverse(model, z, training=False):
    """Same quantity with reverse mode: one gradient per output channel."""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        s_hat = _call(model, z, training)
        cols = [s_hat[:, i] for i in range(s_hat.shape[1])]
    grads = [tape.gradient(c, z)[:, 0] for c in cols]
    del tape
    return s_hat, tf.stack(grads, axis=1)


def forward_and_time_derivative(model, z, S_x, T, method="forward", training=False):
    """Returns (s_hat scaled (B, n_s), ds/dt physical (B, n_s))."""
    if method == "forward":
        s_hat, ds_dtau = _forward_with_jvp(model, z, training)
    elif method == "reverse":
        s_hat, ds_dtau = _forward_with_reverse(model, z, training)
    else:
        raise ValueError(method)
    S_x = tf.cast(S_x, z.dtype)
    T = tf.cast(T, z.dtype)
    return s_hat, ds_dtau*S_x/T


def time_derivative(model, z, S_x, T, method="forward"):
    """ds/dt in physical units, shape (B, n_s)."""
    return forward_and_time_derivative(model, z, S_x, T, method)[1]


def physics_residual(model, z, cfg: Config, params=None, method="forward", training=False):
    """Scaled residual R (B, n_s).  `params` default: nominal vehicle."""
    params = params or cfg.params
    S_x = tf.cast(cfg.S_x, z.dtype)
    S_u = tf.cast(cfg.S_u, z.dtype)
    S_f = tf.cast(cfg.S_f, z.dtype)
    n_s = len(cfg.scales.S_x)
    s_hat, dsdt = forward_and_time_derivative(model, z, S_x, cfg.T, method, training)
    s = s_hat*S_x
    u = z[:, 1 + n_s:]*S_u
    f = get_system(cfg).f_s_tf(s, u, params, cfg.sim.tyre)
    return (dsdt - f)/S_f


def data_loss(model, z, s_target, cfg: Config, training=False):
    """MSE in scaled units between prediction and integrator target (physical)."""
    S_x = tf.cast(cfg.S_x, z.dtype)
    return tf.reduce_mean(tf.square(_call(model, z, training) - tf.cast(s_target, z.dtype)/S_x))


def ic_loss(model, z0, training=False):
    """MSE at t = 0 against the (scaled) initial state carried in the input."""
    s_hat = _call(model, z0, training)
    return tf.reduce_mean(tf.square(s_hat - z0[:, 1:1 + s_hat.shape[1]]))


def physics_loss(model, z_c, cfg: Config, params=None, method="forward", training=False):
    R = physics_residual(model, z_c, cfg, params, method, training)
    mask = tf.cast(cfg.loss.residual_mask, z_c.dtype)
    return tf.reduce_mean(tf.square(R)*mask)


def total_loss(model, z_d, s_d, z_ic, z_c, cfg: Config, params=None, method="forward", training=False):
    """L = L_data + w_ic * L_ic + lambda * L_phys.  Returns dict of scalars."""
    ld = data_loss(model, z_d, s_d, cfg, training)
    li = ic_loss(model, z_ic, training)
    lp = physics_loss(model, z_c, cfg, params, method, training)
    lam = tf.cast(cfg.loss.lam, ld.dtype)
    w_ic = tf.cast(cfg.loss.w_ic, ld.dtype)
    total = ld + w_ic*li + lam*lp
    return dict(total=total, data=ld, ic=li, phys=lp)


def assert_finite(x, name="loss"):
    """Fail loudly (ground rule 7)."""
    tf.debugging.assert_all_finite(x, f"non-finite {name}")
    return x
