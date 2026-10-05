"""
High-fidelity "true" plant for the imperfect-prior study (docs/PLAN_HIGH_FIDELITY.md, sec. 2):
a double-track vehicle with Magic Formula 6.1 tyres (combined slip, load-dependent), quasi-static
load transfer, four wheel-speed states and first-order actuator lag.

Full state (12):  x = [vx, vy, r, psi, X, Y, w_fl, w_fr, w_rl, w_rr, F_act, delta_act]
Input (2):        u = [F_cmd, delta_cmd]      (same interface as pinc/plant.py)

  dF_act/dt     = (F_cmd - F_act)/tau_F,   ddelta_act/dt = (delta_cmd - delta_act)/tau_delta
  drive (F_act >= 0): share gamma_f on the front axle (1 = front-wheel drive), split equally L/R;
  braking (F_act < 0): share beta_f on the front axle;  wheel torque T_i = share_i * F_act * R_w
  hub velocity v_i = (vx - r y_i, vy + r x_i); wheel frame rotated by delta_i (front: delta_act)
  kappa_i = (R_w w_i - v_l,i)/max(|v_l,i|, v_eps),   alpha_i = -atan2(v_s,i, max(|v_l,i|, v_eps))
  Fz_i: static + longitudinal (a_x = (F_act - F_aero - F_roll)/m) + lateral (a_y = vx r) transfer
  (F_x,i, F_y,i) = MF(kappa_i, alpha_i, Fz_i), right-hand tyres as measured, left-hand mirrored
  I_w dw_i/dt    = T_i - R_w F_x,i
  m (dvx - vy r) = sum (F_x,i cos d_i - F_y,i sin d_i) - F_aero - F_roll
  m (dvy + vx r) = sum (F_x,i sin d_i + F_y,i cos d_i)
  Iz dr          = sum x_i (F_x,i sin d_i + F_y,i cos d_i) - y_i (F_x,i cos d_i - F_y,i sin d_i)
  dpsi = r,  dX = vx cos psi - vy sin psi,  dY = vx sin psi + vy cos psi

The same code runs on NumPy (the plant) and TensorFlow (NMPC predictor, physics residuals):
pass xp=NP or xp=TF.  Variants (sec. 2 "Calibration"):
  M0 "calibrated": LKY (and LKX = 1) chosen so that the per-axle cornering stiffness at static load
                   equals the single-track model's Caf; the prior is then right in gentle driving.
  M1 "mismatched": the tyre data's own stiffness (LKY = 1), about twice the prior's.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import tyre_mf

NAMES = ("vx", "vy", "r", "psi", "X", "Y", "w_fl", "w_fr", "w_rl", "w_rr", "F_act", "delta_act")
N_X = 12
V_EPS = 0.5
DT_PLANT = 5e-4          # RK4 step of the true plant: error < 1e-9 over 0.5 s at 5-20 m/s, 4x below the stability limit at 5 m/s
DT_PRED = 1e-3           # RK4 substep of the NMPC-HF predictor (100 substeps per control period)
FZ_MIN = 100.0

NP = SimpleNamespace(**vars(tyre_mf.NP), stack=np.stack, tanh=np.tanh)


def _tf():
    import tensorflow as tf
    return SimpleNamespace(**vars(tyre_mf._tf_ns()), stack=tf.stack, tanh=tf.tanh)


class _LazyTF(SimpleNamespace):
    def __getattr__(self, name):
        ns = _tf()
        self.__dict__.update(vars(ns))
        return getattr(ns, name)


TF = _LazyTF()


def make_params(vehicle: dict, variant: str = "M0", tyre_file: str | None = None, **overrides) -> dict:
    """HF parameter dict: the single-track vehicle parameters plus double-track geometry, actuators,
    drive layout and the MF tyre set (key 'tyre').  `overrides` replace any scalar entry."""
    tyre = tyre_mf.load_params(tyre_file or tyre_mf.DEFAULT_FILE)
    p = dict(vehicle)
    g = 9.81
    p.update(t_f=1.6, t_r=1.6, h=0.55, chi_f=0.55, tau_F=0.15, tau_delta=0.10, beta_f=0.6, gamma_f=1.0,
             I_w=tyre["IYY"], mu_scale=1.0, g=g)
    L = p["lf"] + p["lr"]
    Fz_static_f = p["m"]*g*p["lr"]/(2*L)
    p["R_w"] = tyre["UNLOADED_RADIUS"] - Fz_static_f/tyre["VERTICAL_STIFFNESS"]   # loaded radius at static load
    if variant == "M0":
        tyre["LKY"] = (p["Caf"]/2.0)/tyre_mf.cornering_stiffness(Fz_static_f, {**tyre, "LKY": 1.0})
    elif variant != "M1":
        raise ValueError(f"variant must be M0 or M1, got {variant!r}")
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f"unknown HF parameter {k!r}")
        p[k] = float(v)
    p["tyre"] = tyre
    p["variant"] = variant
    return p


def wheel_geometry(p):
    """(x_i, y_i) of fl, fr, rl, rr in the body frame (y to the left)."""
    return ((p["lf"], p["t_f"]/2), (p["lf"], -p["t_f"]/2), (-p["lr"], p["t_r"]/2), (-p["lr"], -p["t_r"]/2))


def static_loads(p):
    L = p["lf"] + p["lr"]
    f, r = p["m"]*p["g"]*p["lr"]/(2*L), p["m"]*p["g"]*p["lf"]/(2*L)
    return (f, f, r, r)


def vertical_loads(vx, r, F, p, xp=NP):
    """Fz of fl, fr, rl, rr: static load plus quasi-static longitudinal (a_x from the actuator force,
    drag and rolling resistance) and lateral (a_y = vx r) transfer, floored at FZ_MIN."""
    m, L = p["m"], p["lf"] + p["lr"]
    vxs = xp.maximum(vx, V_EPS)
    ax = (F - 0.5*p["rho"]*p["Cd"]*p["A"]*vxs**2 - p["Frr"]*xp.tanh(vx/0.1))/m
    ay = vx*r
    dlong = m*ax*p["h"]/(2*L)
    dlat_f = p["chi_f"]*m*ay*p["h"]/p["t_f"] if p["t_f"] > 0 else 0.0*ay      # zero track (reduced model): no transfer
    dlat_r = (1.0 - p["chi_f"])*m*ay*p["h"]/p["t_r"] if p["t_r"] > 0 else 0.0*ay
    Fz0 = static_loads(p)
    Fz = [Fz0[0] - dlong - dlat_f, Fz0[1] - dlong + dlat_f, Fz0[2] + dlong - dlat_r, Fz0[3] + dlong + dlat_r]
    return [xp.maximum(z, FZ_MIN) for z in Fz]


def f(x, u, p, xp=NP):
    """dx/dt for x (..., 12), u (..., 2)."""
    vx, vy, r, psi = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    w = [x[..., 6], x[..., 7], x[..., 8], x[..., 9]]
    F, d = x[..., 10], x[..., 11]
    F_cmd, d_cmd = u[..., 0], u[..., 1]
    m, Iz, Rw = p["m"], p["Iz"], p["R_w"]

    vxs = xp.maximum(vx, V_EPS)
    F_aero = 0.5*p["rho"]*p["Cd"]*p["A"]*vxs**2
    F_roll = p["Frr"]*xp.tanh(vx/0.1)

    # torque distribution (smooth switch between drive and brake split)
    s_drive = 0.5*(1.0 + xp.tanh(F/50.0))
    front = s_drive*p["gamma_f"] + (1.0 - s_drive)*p["beta_f"]
    T = [0.5*front*F*Rw, 0.5*front*F*Rw, 0.5*(1.0 - front)*F*Rw, 0.5*(1.0 - front)*F*Rw]

    Fz = vertical_loads(vx, r, F, p, xp)

    sumX, sumY, Mz, dw = 0.0, 0.0, 0.0, []
    for i, (xi, yi) in enumerate(wheel_geometry(p)):
        vxi, vyi = vx - r*yi, vy + r*xi
        if i < 2:                                        # front: steered
            cd, sd = xp.cos(d), xp.sin(d)
        else:
            cd, sd = 1.0, 0.0
        vl = vxi*cd + vyi*sd
        vs = -vxi*sd + vyi*cd
        vden = xp.maximum(xp.abs(vl), V_EPS)
        kappa = (Rw*w[i] - vl)/vden
        alpha = -xp.atan(vs/vden)
        if p.get("tyre_model", "mf") == "linear":           # reduced plant (tests): linear tyres, no saturation
            Fxi = p["C_kappa"]*kappa
            Fyi = 0.5*(p["Caf"] if i < 2 else p["Car"])*alpha
        else:
            Fxi, Fyi = tyre_mf.forces(kappa, alpha, Fz[i], p["tyre"], p["mu_scale"], xp, mirror=(yi > 0))
        bx = Fxi*cd - Fyi*sd                             # body-frame force of wheel i
        by = Fxi*sd + Fyi*cd
        sumX, sumY = sumX + bx, sumY + by
        Mz = Mz + xi*by - yi*bx
        dw.append((T[i] - Rw*Fxi)/p["I_w"])

    dvx = (sumX - F_aero - F_roll)/m + vy*r
    dvy = sumY/m - vx*r
    dr = Mz/Iz
    dX = vx*xp.cos(psi) - vy*xp.sin(psi)
    dY = vx*xp.sin(psi) + vy*xp.cos(psi)
    dF = (F_cmd - F)/p["tau_F"]
    dd = (d_cmd - d)/p["tau_delta"]
    return xp.stack([dvx, dvy, dr, r, dX, dY, dw[0], dw[1], dw[2], dw[3], dF, dd], axis=-1)


def rk4_step(x, u, dt, p, xp=NP):
    k1 = f(x, u, p, xp)
    k2 = f(x + 0.5*dt*k1, u, p, xp)
    k3 = f(x + 0.5*dt*k2, u, p, xp)
    k4 = f(x + dt*k3, u, p, xp)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate(x0, u, T, dt, p):
    n = int(round(T/dt))
    if abs(n*dt - T) > 1e-9*max(1.0, abs(T)):
        raise ValueError(f"dt={dt} does not divide T={T}")
    x = np.array(x0, dtype=float, copy=True)
    u = np.asarray(u, dtype=float)
    for _ in range(n):
        x = rk4_step(x, u, dt, p)
    return x


def free_rolling_state(vx, p, vy=0.0, r=0.0, psi=0.0, F=None, delta=0.0):
    """Full state with every wheel rolling freely (zero slip) at forward speed vx, straight ahead."""
    w = vx/p["R_w"]
    if F is None:
        F = 0.5*p["rho"]*p["Cd"]*p["A"]*vx**2 + p["Frr"]*np.tanh(vx/0.1)
    return np.array([vx, vy, r, psi, 0.0, 0.0, w, w, w, w, F, delta], dtype=float)
