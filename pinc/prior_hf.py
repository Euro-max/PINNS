"""
Physics prior "P" for the imperfect-prior study (docs/PLAN_HIGH_FIDELITY.md, sec. 3) and the map
between the true plant's state and the network state.

Network state (10):  s = [vx, vy, r, psi, F_act, delta_act, sig_fl, sig_fr, sig_rl, sig_rr]
with the wheel slip velocity sig_i = R_w w_i - vx (about 1 m/s, where w_i itself is ~70 rad/s).

The prior is deliberately simpler than the true plant (pinc/plant_hf.py):
  - single-track geometry (zero track width: both wheels of an axle see the same hub velocity),
  - static vertical loads (no load transfer),
  - linear tyres per wheel, F_x = C_kappa kappa, F_y = (C_axle/2) alpha, no combined slip, no saturation,
  - the same rigid-body equations, wheel equation, drive/brake split and actuator lag,
with nominal parameters theta = (Caf, Car, C_kappa, tau_F, tau_delta).  In the learnable-parameter
variant (plan decision 2d) these entries are tf.Variables; the code is the same.
Written once for NumPy (xp=NP) and TensorFlow (xp=TF).
"""
from __future__ import annotations

import numpy as np

from . import plant_hf as H
from . import tyre_mf

N_S = 10
STATE_NAMES = ("vx", "vy", "r", "psi", "F_act", "delta_act", "sig_fl", "sig_fr", "sig_rl", "sig_rr")
THETA = ("Caf", "Car", "C_kappa", "tau_F", "tau_delta")


def nominal_params(vehicle: dict, tyre_file: str | None = None) -> dict:
    """Prior parameters: the single-track vehicle, the wheel radius / inertia and drive layout of the
    true plant, and nominal theta: the single-track cornering stiffnesses, the tyre data's slip stiffness
    at static load, and the nominal actuator time constants."""
    hf = H.make_params(vehicle, "M0", tyre_file)
    q = {k: v for k, v in hf.items() if not isinstance(v, (dict, str))}
    q["C_kappa"] = tyre_mf.slip_stiffness(H.static_loads(hf)[0], hf["tyre"])
    return q


def full_to_s(x, R_w):
    """Network state from the true plant's full state (..., 12)."""
    x = np.asarray(x, float)
    sig = R_w*x[..., 6:10] - x[..., :1]
    return np.concatenate([x[..., 0:4], x[..., 10:12], sig], axis=-1)


def s_to_full(s, R_w, XY=None):
    """Full state from the network state; X, Y = 0 unless given."""
    s = np.asarray(s, float)
    XY = np.zeros(s.shape[:-1] + (2,)) if XY is None else np.broadcast_to(XY, s.shape[:-1] + (2,))
    w = (s[..., 6:10] + s[..., :1])/R_w
    return np.concatenate([s[..., 0:4], XY, w, s[..., 4:6]], axis=-1)


def full_rates_to_s(x, dx, R_w, xp=np):
    """ds/dt from the full state and its time derivative (..., 12): d sig_i = R_w dw_i - dvx."""
    dsig = [R_w*dx[..., 6 + i] - dx[..., 0] for i in range(4)]
    cols = [dx[..., 0], dx[..., 1], dx[..., 2], dx[..., 3], dx[..., 10], dx[..., 11]] + dsig
    return xp.stack(cols, axis=-1)


def f_s(s, u, q, xp=H.NP):
    """Prior dynamics ds/dt (..., 10)."""
    vx, vy, r = s[..., 0], s[..., 1], s[..., 2]
    F, d = s[..., 4], s[..., 5]
    sig = [s[..., 6], s[..., 7], s[..., 8], s[..., 9]]
    F_cmd, d_cmd = u[..., 0], u[..., 1]
    m, Iz, Rw, lf, lr = q["m"], q["Iz"], q["R_w"], q["lf"], q["lr"]

    vxs = xp.maximum(vx, H.V_EPS)
    F_aero = 0.5*q["rho"]*q["Cd"]*q["A"]*vxs**2
    F_roll = q["Frr"]*xp.tanh(vx/0.1)
    s_drive = 0.5*(1.0 + xp.tanh(F/50.0))
    front = s_drive*q["gamma_f"] + (1.0 - s_drive)*q["beta_f"]
    T = [0.5*front*F*Rw, 0.5*front*F*Rw, 0.5*(1.0 - front)*F*Rw, 0.5*(1.0 - front)*F*Rw]

    cd, sd = xp.cos(d), xp.sin(d)
    sumX, sumY, Mz, dw = 0.0, 0.0, 0.0, []
    for i in range(4):
        front_wheel = i < 2
        xi = lf if front_wheel else -lr
        vxi, vyi = vx, vy + r*xi                          # zero track width
        c, sn = (cd, sd) if front_wheel else (1.0, 0.0)
        vl = vxi*c + vyi*sn
        vs = -vxi*sn + vyi*c
        vden = xp.maximum(xp.abs(vl), H.V_EPS)
        kappa = (sig[i] + vx - vl)/vden                   # R_w w_i = sig_i + vx
        alpha = -xp.atan(vs/vden)
        Fx = q["C_kappa"]*kappa
        Fy = 0.5*(q["Caf"] if front_wheel else q["Car"])*alpha
        bx, by = Fx*c - Fy*sn, Fx*sn + Fy*c
        sumX, sumY = sumX + bx, sumY + by
        Mz = Mz + xi*by
        dw.append((T[i] - Rw*Fx)/q["I_w"])

    dvx = (sumX - F_aero - F_roll)/m + vy*r
    dvy = sumY/m - vx*r
    dr = Mz/Iz
    dF = (F_cmd - F)/q["tau_F"]
    dd = (d_cmd - d)/q["tau_delta"]
    dsig = [Rw*dw[i] - dvx for i in range(4)]
    return xp.stack([dvx, dvy, dr, r, dF, dd] + dsig, axis=-1)


def f_s_true(s, u, p, xp=H.NP):
    """The true plant's dynamics in network-state coordinates (X, Y do not enter ds/dt)."""
    Rw = p["R_w"]
    vx = s[..., 0]
    w = [(s[..., 6 + i] + vx)/Rw for i in range(4)]
    zero = 0.0*vx
    x = xp.stack([s[..., 0], s[..., 1], s[..., 2], s[..., 3], zero, zero, w[0], w[1], w[2], w[3], s[..., 4], s[..., 5]], axis=-1)
    return full_rates_to_s(x, H.f(x, u, p, xp), Rw, xp)
