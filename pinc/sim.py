"""
Closed-loop simulator.  The plant state changes ONLY through
the system's `plant_simulate` (ground rule 1; `pinc/system.py`).  Nothing here reads the reference except
to hand it to the controller and to compute the logged error AFTER the step
against the reference at the new time (fixes D7, D18).
"""
from __future__ import annotations

import numpy as np

from . import plant
from .config import Config
from .refs import Reference
from .system import get_system


def simulate(controller, plant_params: dict, ref: Reference, x0, duration: float, noise_sigma, seed: int,
             cfg: Config, tyre: str = "linear", disturbance=None, verbose: bool = False) -> dict:
    """Run the loop: measure -> controller -> ZOH input -> plant (system.plant_simulate).

    controller(t, x_meas, ref) -> (u (2,), info dict)
    disturbance(t) -> additive plant input (2,), unknown to the controller (may be None)
    """
    T, dt = cfg.T, cfg.sim.dt_plant
    sysm = get_system(cfg)
    n = int(round(duration/T))
    rng = np.random.default_rng(seed)
    noise_sigma = np.asarray(noise_sigma, float)
    u_lo, u_hi = np.asarray(cfg.u_min), np.asarray(cfg.u_max)

    t = T*np.arange(n + 1)
    n_x, n_u = len(np.asarray(x0)), len(cfg.u_min)
    x = np.zeros((n + 1, n_x))
    x_meas = np.zeros((n, n_x))
    u = np.zeros((n, n_u))
    u_plant = np.zeros((n, n_u))
    z_ref = ref(t)
    solve_time = np.zeros(n)
    nit = np.zeros(n, dtype=int)
    success = np.zeros(n, dtype=bool)

    x[0] = np.asarray(x0, float)
    if hasattr(controller, "reset"):
        controller.reset()
    for k in range(n):
        x_meas[k] = x[k] + noise_sigma*rng.standard_normal(n_x)        # sensor model
        u_k, info = controller(t[k], x_meas[k], ref)
        u_k = np.clip(np.asarray(u_k, float), u_lo, u_hi)              # actuator saturation (physical limits only)
        u[k] = u_k
        u_p = u_k + (np.asarray(disturbance(t[k]), float) if disturbance is not None else 0.0)
        u_plant[k] = u_p
        x[k + 1] = plant.check_finite(sysm.plant_simulate(x[k], u_p, T, dt, plant_params, tyre), "plant state")
        solve_time[k] = info.get("solve_time", np.nan)
        nit[k] = info.get("nit", 0)
        success[k] = info.get("success", True)
        if verbose and k % 10 == 0:
            print(f"  t={t[k]:5.2f} vx={x[k+1,0]:6.2f} psi={x[k+1,3]:+.3f} Y={x[k+1,5]:+.2f} u={u_k} {info.get('solve_time', 0)*1e3:.0f} ms")
    err = sysm.track_full(x) - z_ref                                   # error at t_{k+1} vs ref(t_{k+1})
    return dict(t=t, x=x, x_meas=x_meas, u=u, u_plant=u_plant, ref=z_ref, err=err,
                solve_time=solve_time, nit=nit, success=success, seed=seed, T=T)


def perturb_x0(x0, sigma, rng):
    return np.asarray(x0, float) + np.asarray(sigma, float)*rng.standard_normal(6)


def closed_loop_metrics(log: dict, cfg: Config, ref: Reference) -> dict:
    """Tracking metrics of one run (see pinc/metrics.py)."""
    from . import metrics as M
    T = log["T"]
    e = log["err"][1:]                    # errors after each step
    out = {}
    for j, name in enumerate(("vx", "vy", "r", "psi", "X", "Y")):
        if ref.Q[j] > 0 or name in ("Y", "psi", "vx"):
            out[f"iae_{name}"] = M.iae(e[:, j], T)
            out[f"rmse_{name}"] = float(np.sqrt(np.mean(e[:, j]**2)))
            out[f"max_{name}"] = M.max_abs(e[:, j])
            out[f"p95_{name}"] = M.p95_abs(e[:, j])
    out["effort"] = M.control_effort(log["u"]/cfg.S_u, cfg.mpc.R, T)
    out["viol_time_r"] = M.violation_time(log["x"][1:, 2], cfg.mpc.r_max, T)
    st = M.solve_time_stats(log["solve_time"])
    out.update({f"solve_{k}": v for k, v in st.items()})
    out["nit_mean"] = float(np.mean(log["nit"]))
    out["success_rate"] = float(np.mean(log["success"]))
    if ref.name == "speed_step":
        t, vx = log["t"], log["x"][:, 0]
        t1, dv, v0 = cfg.refs.step_time, cfg.refs.step_dv, cfg.refs.v0
        up = (t >= t1) & (t < 3*t1)
        out["rise_time"] = M.rise_time(t[up], vx[up], v0, v0 + dv)
        out["settling_time"] = M.settling_time(t[up] - t1, vx[up] - (v0 + dv), 0.02*dv)
    return out
