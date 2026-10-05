"""
The vehicle system a PINC model is built for: everything that ties the generic
network / loss / data / MPC code to one particular plant.

A system defines
  - the network state s (names, n_s) and input u (n_u),
  - how s maps to / from the full plant state x (which also carries X, Y and,
    for richer plants, states the network does not predict),
  - the physics prior f_s(s, u) used in the PINC residual (TensorFlow),
  - the plant integrator (NumPy, the ground truth),
  - the tracked vector z = [vx, vy, r, psi, X, Y] the MPC cost and the
    references are written in.

`bicycle` is the single-track model of `pinc/plant.py`, whose first four plant
states are the network state.  `get_system(cfg)` selects by `cfg.system`.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from . import plant, plant_tf


class Bicycle:
    name = "bicycle"
    state_names = ("vx", "vy", "r", "psi")
    full_state_names = ("vx", "vy", "r", "psi", "X", "Y")
    input_names = ("Fx", "delta")
    n_s, n_x, n_u = 4, 6, 2
    i_xy = (4, 5)                 # X, Y in the full state

    # ---- state maps (NumPy) -------------------------------------------------
    def to_full(self, s, XY=None):
        """Full plant state from network state (X = Y = 0 unless given)."""
        s = np.asarray(s, float)
        XY = np.zeros(s.shape[:-1] + (2,)) if XY is None else np.broadcast_to(XY, s.shape[:-1] + (2,))
        return np.concatenate([s, XY], axis=-1)

    def from_full(self, x):
        return np.asarray(x)[..., :4]

    def xy(self, x):
        return np.asarray(x)[..., 4:6]

    def track_full(self, x):
        """Tracked vector z = [vx, vy, r, psi, X, Y] of the full plant state (the first six states)."""
        return np.asarray(x)[..., :6]

    # ---- ground truth (NumPy) -----------------------------------------------
    def rk4_step(self, x, u, dt, params, tyre="linear"):
        return plant.rk4_step(x, u, dt, params, tyre)

    def f_full(self, x, u, params, tyre="linear"):
        return plant.f(x, u, params, tyre)

    def plant_simulate(self, x0, u, T, dt, params, tyre="linear"):
        """Zero-order hold of `u` over [0, T] with fixed-step RK4: THE plant of the closed loop."""
        n = int(round(T/dt))
        if abs(n*dt - T) > 1e-9*max(1.0, abs(T)):
            raise ValueError(f"dt={dt} does not divide T={T}")
        x = np.array(x0, dtype=float, copy=True)
        for _ in range(n):
            x = self.rk4_step(x, u, dt, params, tyre)
        return x

    def sample_s0(self, n, box, rng):
        return rng.uniform(box.lo(), box.hi(), size=(n, 4))

    # ---- physics prior and MPC glue (TensorFlow) ------------------------------
    def f_s_tf(self, s, u, params, tyre="linear"):
        """ds/dt of the network state (B, n_s)."""
        return plant_tf.f_tf(s, u, params, tyre)

    def rk4_step_s_tf(self, s, u, dt, params, tyre="linear"):
        """One RK4 substep of the network state (the NMPC prediction model)."""
        return plant_tf.rk4_step_tf(s, u, dt, params, tyre)

    def planar_velocity(self, s):
        """(vx, vy, psi) of the network state, for integrating X, Y in the MPC."""
        return s[:, 0], s[:, 1], s[:, 3]

    def track_vector(self, s, XY):
        """z = [vx, vy, r, psi, X, Y] (tf, (N, 6)) from predicted s and X, Y."""
        return tf.concat([s, XY], axis=1)


SYSTEMS = {"bicycle": Bicycle}


def get_system(cfg_or_name) -> Bicycle:
    name = cfg_or_name if isinstance(cfg_or_name, str) else getattr(cfg_or_name, "system", "bicycle")
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}")
    return SYSTEMS[name]()
