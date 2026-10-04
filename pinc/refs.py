"""
Reference trajectories for the closed-loop experiments.  Each reference is a
callable t -> z_ref (…, 6) over the full state [vx, vy, r, psi, X, Y] plus
per-state tracking weights Q / P (zero for untracked components) and the
nominal initial state x0.
"""
from __future__ import annotations

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


REFERENCES = {r.name: r for r in (SpeedSinusoid, SpeedStep, SingleLaneChange, DoubleLaneChange)}


def make_reference(name: str, cfg: Config) -> Reference:
    return REFERENCES[name](cfg)
