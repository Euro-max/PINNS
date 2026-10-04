"""
The same single-track dynamics and RK4 rollout as `pinc/plant.py`, written in
TensorFlow so that (a) the physics residual of the PINC loss and (b) the
baseline NMPC-RK4 predictor get exact autodiff gradients with the same
solver as the PINC arm.

Every function is batched: x has shape (B, 6) or (B, 4) (first four states
only, for the network), u has shape (B, 2).  Parameters are a dict of python
floats (or tf scalars) -- the controller always gets the NOMINAL parameters
(ground rule 3); perturbed parameters go only to the NumPy plant.
"""
import numpy as np
import tensorflow as tf

from .plant import VX_MIN, DEFAULT_PARAMS  # noqa: F401  (same constants)


def _cast_params(p, dtype):
    return {k: (v if tf.is_tensor(v) else tf.constant(float(v), dtype)) for k, v in p.items()}


def tyre_force_tf(alpha, C, mu, Fz, model="linear"):
    if model == "linear":
        return C * alpha
    if model != "fiala":
        raise ValueError(f"unknown tyre model {model!r}")
    Fmax = mu * Fz
    a_sl = tf.atan(3.0 * Fmax / C)
    a = tf.clip_by_value(alpha, -a_sl, a_sl)
    t = tf.tan(a)
    F = C*t - (C**2/(3*Fmax))*tf.abs(t)*t + (C**3/(27*Fmax**2))*t**3
    return tf.where(tf.abs(alpha) >= a_sl, tf.sign(alpha)*Fmax, F)


def f_tf(x, u, p, tyre="linear"):
    """dx/dt for x (B, 6) or (B, 4) [vx, vy, r, psi(, X, Y)], u (B, 2)."""
    dtype = x.dtype
    p = _cast_params(p, dtype)
    vx, vy, r, psi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    Fx, delta = u[:, 0], u[:, 1]
    vxs = tf.maximum(vx, tf.constant(VX_MIN, dtype))

    alpha_f = delta - tf.atan2(vy + p['lf']*r, vxs)
    alpha_r = -tf.atan2(vy - p['lr']*r, vxs)
    Fyf = tyre_force_tf(alpha_f, p['Caf'], p['mu'], p['Fz'], tyre)
    Fyr = tyre_force_tf(alpha_r, p['Car'], p['mu'], p['Fz'], tyre)

    F_drag = 0.5*p['rho']*p['Cd']*p['A']*vxs**2
    F_roll = p['Frr']*tf.tanh(vx/0.1)

    dvx = (Fx - F_drag - F_roll - Fyf*tf.sin(delta))/p['m'] + vy*r
    dvy = (Fyf*tf.cos(delta) + Fyr)/p['m'] - vx*r
    dr = (p['lf']*Fyf*tf.cos(delta) - p['lr']*Fyr)/p['Iz']
    dpsi = r
    cols = [dvx, dvy, dr, dpsi]
    if x.shape[-1] == 6:
        dX = vx*tf.cos(psi) - vy*tf.sin(psi)
        dY = vx*tf.sin(psi) + vy*tf.cos(psi)
        cols += [dX, dY]
    return tf.stack(cols, axis=-1)


def rk4_step_tf(x, u, dt, p, tyre="linear"):
    dt = tf.constant(float(dt), x.dtype) if not tf.is_tensor(dt) else tf.cast(dt, x.dtype)
    k1 = f_tf(x, u, p, tyre)
    k2 = f_tf(x + 0.5*dt*k1, u, p, tyre)
    k3 = f_tf(x + 0.5*dt*k2, u, p, tyre)
    k4 = f_tf(x + dt*k3, u, p, tyre)
    return x + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_tf(x0, u, T, dt, p, tyre="linear"):
    """Zero-order-hold `u` over [0, T] with a python-unrolled RK4 loop.
    `T/dt` must be an integer; the loop is unrolled at trace time."""
    n = int(round(T/dt))
    if abs(n*dt - T) > 1e-9*max(1.0, abs(T)):
        raise ValueError(f"dt={dt} does not divide T={T}")
    x = x0
    for _ in range(n):
        x = rk4_step_tf(x, u, dt, p, tyre)
    return x


def as_tensor(a, dtype):
    return tf.convert_to_tensor(np.asarray(a), dtype=dtype)
