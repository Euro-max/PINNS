"""
E26 -- Checks of the double-track plant (numbers behind the statements in the model section of the paper).

1. Tyre model against the MathWorks Vehicle Dynamics Blockset solver: largest force error over the reference grid
   exported by scripts/export_tyre_reference.m (combined slip, three loads).
2. Double-track model (M0) against the single-track model in gentle driving: the double-track plant is driven by
   smooth random force and steer commands with a peak lateral acceleration of at most GENTLE_AY; the single-track
   plant (linear tyres, no actuator lag) receives the double-track model's actual drive force and steer angle at
   every plant step, so the comparison isolates the vehicle dynamics.  Reported: NRMSE of v_x, v_y, r over the
   drives (scaled by the network scales S_x) and the largest lateral acceleration reached.
3. Wheel-speed time constant: -1 / (d omega_dot / d omega) of one wheel at free rolling and static load, at
   several speeds (the wheel dynamics are the fast part of the plant).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from pinc import plant, plant_hf, tyre_mf  # noqa: E402
from pinc.config import ROOT, load_config  # noqa: E402
from pinc.system import get_system  # noqa: E402

REF = os.path.join(ROOT, "data", "tyre", "mf_reference.csv")
GENTLE_AY = 2.0          # m/s^2, upper bound of "gentle driving" for check 2
SPEEDS = (5.0, 10.0, 20.0, 30.0)


def tyre_check():
    tyre = tyre_mf.load_params(tyre_mf.DEFAULT_FILE)
    _, kappa, alpha, Fz, Fx_ref, Fy_ref, _ = np.loadtxt(REF, delimiter=",").T
    Fx, Fy = tyre_mf.forces(kappa, -alpha, Fz, tyre)          # the solver uses SAE axes (y to the right)
    return dict(n=int(len(kappa)), max_err_Fx=float(np.max(np.abs(Fx - Fx_ref))), max_err_Fy=float(np.max(np.abs(Fy - Fy_ref))),
                max_Fx=float(np.max(np.abs(Fx_ref))), max_Fy=float(np.max(np.abs(Fy_ref))))


def gentle_check(cfg, n_drives=40, duration=4.0, seed=26):
    sysm = get_system(cfg)
    p, vp = sysm.truth, dict(cfg.params)
    dt = cfg.sim.dt_plant
    rng = np.random.default_rng(seed)
    L = vp["lf"] + vp["lr"]
    n = int(round(duration/dt))
    t = dt*np.arange(n)
    S = np.asarray(cfg.S_x)[:3]
    err, ay_max = [], 0.0
    for _ in range(n_drives):
        v0 = rng.uniform(10.0, 25.0)
        ay = rng.uniform(0.5, 0.8*GENTLE_AY)
        f1, f2 = rng.uniform(0.2, 0.8, 2)
        d_cmd = ay*L/v0**2*np.sin(2*np.pi*f1*t + rng.uniform(0, 2*np.pi))
        F_cmd = plant.trim_force(v0, vp) + rng.uniform(0, 800)*np.sin(2*np.pi*f2*t)
        x = plant_hf.free_rolling_state(v0, p)
        xs = np.array([v0, 0.0, 0.0, 0.0, 0.0, 0.0])
        e = []
        for k in range(n):
            u = np.array([F_cmd[k], d_cmd[k]])
            u_act = x[10:12].copy()                               # actual force and steer of the double-track plant
            x = plant_hf.rk4_step(x, u, dt, p)
            xs = plant.rk4_step(xs, u_act, dt, vp)
            ay_max = max(ay_max, abs(x[0]*x[2]))
            e.append((x[:3] - xs[:3])/S)
        err.append(np.asarray(e))
    e = np.concatenate(err)
    return dict(nrmse=dict(zip(("vx", "vy", "r"), np.sqrt(np.mean(e**2, axis=0)).tolist())),
                nrmse_body=float(np.sqrt(np.mean(e**2))), ay_max=float(ay_max), n_drives=n_drives, duration=duration,
                gentle_ay=GENTLE_AY)


def wheel_time_constants(cfg):
    p = get_system(cfg).truth
    out = {}
    for v in SPEEDS:
        x = plant_hf.free_rolling_state(v, p)
        u = x[10:12].copy()
        h = 1e-4
        xp, xm = x.copy(), x.copy()
        xp[6] += h
        xm[6] -= h
        d = (plant_hf.f(xp, u, p)[6] - plant_hf.f(xm, u, p)[6])/(2*h)
        out[f"{v:g}"] = float(-1.0/d)
    return out


def main(argv=None):
    ap = base_parser(__doc__)
    a = ap.parse_args(argv)
    a.config = os.path.join(ROOT, "configs", "hf_m0.yaml")
    cfg, run_dir = start("e26_model_checks", a)
    cfg = load_config(a.config)
    tyre = tyre_check()
    gentle = gentle_check(cfg, n_drives=5 if a.quick else 40)
    tau = wheel_time_constants(cfg)
    text = ("# E26 checks of the double-track plant\n\n"
            f"1. Tyre model against the MathWorks solver ({tyre['n']} points): max |Fx error| {tyre['max_err_Fx']:.3g} N "
            f"(|Fx| up to {tyre['max_Fx']:.0f} N), max |Fy error| {tyre['max_err_Fy']:.3g} N (|Fy| up to {tyre['max_Fy']:.0f} N)\n\n"
            f"2. Double-track (M0) against single-track, gentle driving (|a_y| <= {GENTLE_AY:g} m/s^2, reached "
            f"{gentle['ay_max']:.2f}; {gentle['n_drives']} drives of {gentle['duration']:g} s): NRMSE body "
            f"{gentle['nrmse_body']:.4f}, " + ", ".join(f"{k} {v:.4f}" for k, v in gentle["nrmse"].items()) + "\n\n" +
            "3. Wheel-speed time constant at free rolling and static load\n\n" +
            md_table(["speed [m/s]", "time constant [ms]"], [[k, f"{v*1e3:.2f}"] for k, v in tau.items()]))
    art = write_text(os.path.join(run_dir, "table_model_checks.md"), text)
    finish(run_dir, cfg, a.seed, dict(tyre=tyre, gentle=gentle, wheel_time_constant_s=tau), [art], dict(quick=a.quick))
    print(text)


if __name__ == "__main__":
    main()
