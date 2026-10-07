"""
Reference trajectories for the closed-loop experiments.  Each reference is a
callable t -> z_ref (…, 6) over the full state [vx, vy, r, psi, X, Y] plus
per-state tracking weights Q / P (zero for untracked components) and the
nominal initial state x0.
"""
from __future__ import annotations

import functools

import numpy as np

from .config import Config


class Reference:
    name = "base"
    Q = np.array([1.0, 0.0, 0.0, 10.0, 0.0, 1.0])
    P = np.array([1.0, 0.0, 0.0, 10.0, 0.0, 1.0])
    tracked = ("vx", "psi", "Y")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rc = cfg.refs
        self.v0 = float(self.rc.v0)
        self.duration = float(cfg.sim.duration)

    def x0(self) -> np.ndarray:
        return np.array([self.v0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __call__(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        z = np.zeros((t.size, 6))
        z[:, 0] = self.vx(t)
        z[:, 3] = self.psi(t)
        z[:, 4] = self.X(t)
        z[:, 5] = self.Y(t)
        return z

    # defaults: straight line at v0
    def vx(self, t):
        return np.full_like(t, self.v0)

    def psi(self, t):
        return np.zeros_like(t)

    def X(self, t):
        return self.v0*t

    def Y(self, t):
        return np.zeros_like(t)


class SpeedSinusoid(Reference):
    name = "speed_sin"

    def vx(self, t):
        return self.v0 + self.rc.sin_amp*np.sin(self.rc.sin_omega*t)

    def X(self, t):
        w, A = self.rc.sin_omega, self.rc.sin_amp
        return self.v0*t + A*(1 - np.cos(w*t))/w


class SpeedStep(Reference):
    """+dv step at t_step, -dv step back at 3*t_step: both rise and fall measurable."""
    name = "speed_step"

    def vx(self, t):
        t1, dv = self.rc.step_time, self.rc.step_dv
        return self.v0 + dv*((t >= t1) & (t < 3*t1))

    def X(self, t):
        t1, dv = self.rc.step_time, self.rc.step_dv
        return self.v0*t + dv*(np.clip(t - t1, 0, 2*t1))


def _cos_profile(s):
    """C1 transition 0 -> 1 for s in [0, 1] (cosine), clamped outside."""
    s = np.clip(s, 0.0, 1.0)
    return 0.5*(1 - np.cos(np.pi*s))


def _cos_profile_d(s):
    s_c = np.clip(s, 0.0, 1.0)
    return np.where((s > 0) & (s < 1), 0.5*np.pi*np.sin(np.pi*s_c), 0.0)


class SingleLaneChange(Reference):
    name = "lane_change"

    def path(self, X):
        s = (X - self.rc.slc_start)/self.rc.slc_length
        Y = self.rc.slc_offset*_cos_profile(s)
        dY = self.rc.slc_offset*_cos_profile_d(s)/self.rc.slc_length
        return Y, dY

    def Y(self, t):
        return self.path(self.X(t))[0]

    def psi(self, t):
        return np.arctan(self.path(self.X(t))[1])


class DoubleLaneChange(Reference):
    """ISO 3888-2 geometry (sections 12 / 13.5 / 11 / 12.5 / 12 m) driven at
    `refs.dlc_speed`; lateral offset between lane centrelines `refs.dlc_offset`."""
    name = "double_lane_change"
    sections = (12.0, 13.5, 11.0, 12.5, 12.0)
    run_in = 10.0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.v0 = float(self.rc.dlc_speed)

    def path(self, X):
        a, b, c, d, e = self.sections
        x1 = self.run_in + a                  # start of first transition
        x2 = x1 + b                           # end of first transition
        x3 = x2 + c                           # start of second transition
        x4 = x3 + d
        off = self.rc.dlc_offset
        s1 = (X - x1)/b
        s2 = (X - x3)/d
        Y = off*(_cos_profile(s1) - _cos_profile(s2))
        dY = off*(_cos_profile_d(s1)/b - _cos_profile_d(s2)/d)
        return Y, dY

    def Y(self, t):
        return self.path(self.X(t))[0]

    def psi(self, t):
        return np.arctan(self.path(self.X(t))[1])


def iso3888_lanes(W):
    """ISO 3888-2:2011 obstacle-avoidance track for a vehicle of width W, in the reference frame of IsoLaneChange
    (X from the track entry, Y = 0 on the centreline of lane 1, Y positive to the left).  Section lengths
    12 / 13.5 / 11 / 12.5 / 12 m; lane widths 1.1 W + 0.25, W + 1 and max(1.3 W + 0.25, 3) m; a clear gap of 1 m
    between the left edge of lanes 1 and 5 and the right edge of lane 3 (Table 1 and Figures 1 and 3 of the standard).
    Returns [(X_start, X_end, lane centre Y, slack)], slack = half the lane width left over by the vehicle."""
    b1, b3, b5 = 1.1*W + 0.25, W + 1.0, max(1.3*W + 0.25, 3.0)
    y1, y3, y5 = -b1/2, 1.0 + b3/2, -b5/2                  # y = 0 on the left edge of lanes 1 and 5
    return [(0.0, 12.0, 0.0, (b1 - W)/2), (25.5, 36.5, y3 - y1, (b3 - W)/2), (49.0, 61.0, y5 - y1, (b5 - W)/2)]


@functools.lru_cache(maxsize=None)
def iso3888_path(W=1.8):
    """Smooth centreline through the ISO 3888-2 lanes: Y = A1 sig((X - c1)/w1) - A2 sig((X - c2)/w2), with the
    positions c and widths w of the two logistic transitions chosen to minimise the peak path curvature while the
    whole vehicle stays inside every lane.  Deterministic (fixed starting grid).  Returns (A1, c1, w1, A2, c2, w2)."""
    from scipy.optimize import minimize
    lanes = iso3888_lanes(W)
    A1, A2 = lanes[1][2], lanes[1][2] - lanes[2][2]
    X = np.linspace(-10.0, 75.0, 17001)
    sig = lambda z: 1.0/(1.0 + np.exp(-z))
    path = lambda p: A1*sig((X - p[0])/p[1]) - A2*sig((X - p[2])/p[3])

    def peak_curvature(p):
        Y = path(p)
        d = np.gradient(Y, X)
        return float(np.max(np.abs(np.gradient(d, X)/(1 + d**2)**1.5)))

    def margins(p):
        Y = path(p)
        return np.array([sl - np.max(np.abs(Y[(X >= a) & (X <= b)] - c)) for a, b, c, sl in lanes])
    best = None
    for c1 in np.linspace(15.0, 22.0, 8):
        for c2 in np.linspace(39.0, 46.0, 8):
            r = minimize(peak_curvature, [c1, 2.5, c2, 2.5], method="SLSQP", constraints=[{"type": "ineq", "fun": margins}],
                         bounds=[(12.0, 25.5), (0.5, 8.0), (36.5, 49.0), (0.5, 8.0)])
            if r.success and np.all(margins(r.x) >= -1e-6) and (best is None or r.fun < best.fun):
                best = r
    c1, w1, c2, w2 = best.x
    return (A1, c1, w1, A2, c2, w2)


class IsoLaneChange(Reference):
    """Obstacle-avoidance manoeuvre through the ISO 3888-2:2011 cone layout (iso3888_lanes) on a smooth path
    (iso3888_path), driven at constant `refs.iso_speed`; the track entry is `run_in` metres after the start."""
    name = "iso_lane_change"
    run_in = 10.0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.v0 = float(self.rc.iso_speed)
        self.p = iso3888_path(float(self.rc.iso_car_width))

    def path(self, X):
        A1, c1, w1, A2, c2, w2 = self.p
        x = np.asarray(X, float) - self.run_in
        s1, s2 = 1.0/(1.0 + np.exp(-(x - c1)/w1)), 1.0/(1.0 + np.exp(-(x - c2)/w2))
        Y = A1*s1 - A2*s2
        dY = A1*s1*(1 - s1)/w1 - A2*s2*(1 - s2)/w2
        return Y, dY

    def Y(self, t):
        return self.path(self.X(t))[0]

    def psi(self, t):
        return np.arctan(self.path(self.X(t))[1])

    def peak_lateral_acceleration(self):
        X = np.linspace(0.0, self.run_in + 75.0, 40001)
        _, d = self.path(X)
        return float(np.max(np.abs(np.gradient(d, X)/(1 + d**2)**1.5))*self.v0**2)


REFERENCES = {r.name: r for r in (SpeedSinusoid, SpeedStep, SingleLaneChange, DoubleLaneChange, IsoLaneChange)}


def make_reference(name: str, cfg: Config) -> Reference:
    return REFERENCES[name](cfg)
