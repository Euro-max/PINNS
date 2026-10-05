"""
Initial states for the high-fidelity system (docs/PLAN_HIGH_FIDELITY.md, sec. 4 "Initial states").

With 10 network states, a uniform box mostly contains physically inconsistent combinations (large
slip on every wheel while cruising, actuator states unrelated to the motion).  Initial states are
therefore taken from random-excitation drives of the true plant: each drive starts free-rolling at a
speed drawn from the box, is driven by smooth random force and steer commands (sums of three
sinusoids, 0.1-1 Hz, held over each control period), and is sampled at random times after a 0.5 s
run-in.  Drives that leave the plausible envelope (spin-out, standstill) are discarded and replaced.
The steer amplitude of each drive follows a target lateral acceleration drawn up to `ay_max`, and the
force swings up to +-5 kN around trim (braking included), so the set spans gentle driving and driving
near the grip limit.  Heading is shifted by a uniform offset (the dynamics do not depend on it).
"""
from __future__ import annotations

import numpy as np

from . import plant_hf as H
from . import prior_hf as P

ENVELOPE = dict(vx_min=3.0, vy_max=4.0, r_max=1.2)


def _commands(rng, B, n_steps, T, F_trim, u_min, u_max, delta_amp, v0, L, ay_max):
    """Force: trim + up to +-5 kN; steer amplitude from a target lateral acceleration up to ay_max
    (steady state a_y ~ v^2 delta / L), capped at delta_amp."""
    t = T*np.arange(n_steps)
    out = np.zeros((B, n_steps, 2))
    d_amp = np.minimum(rng.uniform(0.0, ay_max, B)*L/v0**2, delta_amp)
    for c, amp in ((0, rng.uniform(0.0, 5000.0, B)), (1, d_amp)):
        sig = np.zeros((B, n_steps))
        for _ in range(3):
            f = rng.uniform(0.1, 1.0, (B, 1))
            sig += rng.uniform(-1, 1, (B, 1))*np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi, (B, 1)))
        out[..., c] = amp[:, None]*sig/1.5
    out[..., 0] += F_trim[:, None]
    return np.clip(out, u_min, u_max)


def driving_states(n: int, seed: int, p: dict, vx_range, u_min, u_max, T: float = 0.1, dt: float = H.DT_PLANT,
                   delta_amp: float = 0.3, ay_max: float = 9.0, duration: float = 3.0, per_drive: int = 4,
                   run_in: float = 0.5, psi_range=(-0.5, 0.5), return_full: bool = False):
    """n network states (n, 10) from random-excitation drives of the true plant `p` (or full states (n, 12))."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration/T))
    sub = int(round(T/dt))
    k_min = int(round(run_in/T))
    got = []
    while sum(len(g) for g in got) < n:
        B = int(np.ceil((n - sum(len(g) for g in got))/per_drive*1.2)) + 1
        v0 = rng.uniform(vx_range[0], vx_range[1], B)
        x = np.stack([H.free_rolling_state(v, p) for v in v0])
        F_trim = x[:, 10].copy()
        cmd = _commands(rng, B, n_steps, T, F_trim, np.asarray(u_min), np.asarray(u_max), delta_amp, v0,
                        p["lf"] + p["lr"], ay_max)
        pick = np.sort(rng.choice(np.arange(k_min, n_steps), size=(B, per_drive)), axis=1)
        rec = np.full((B, per_drive, 12), np.nan)
        ok = np.ones(B, bool)
        for k in range(n_steps):
            hit = pick == k
            if hit.any():
                b, j = np.nonzero(hit)
                rec[b, j] = x[b]
            for _ in range(sub):
                x = H.rk4_step(x, cmd[:, k], dt, p)
            bad = ~np.all(np.isfinite(x), axis=1)
            x[bad] = np.nan_to_num(x[bad])                     # keep the batch finite; these drives are discarded
            ok &= ~bad & (x[:, 0] > ENVELOPE["vx_min"]) & (np.abs(x[:, 1]) < ENVELOPE["vy_max"]) & (np.abs(x[:, 2]) < ENVELOPE["r_max"])
        got.append(rec[ok].reshape(-1, 12))
    full = np.concatenate(got)[:n]
    full[:, 3] += rng.uniform(psi_range[0], psi_range[1], len(full))   # the dynamics do not depend on heading
    return full if return_full else P.full_to_s(full, p["R_w"])
