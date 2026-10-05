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


class HighFidelity:
    """The double-track plant with Magic Formula tyres (pinc/plant_hf.py) as the truth, and the simplified
    prior (pinc/prior_hf.py) as the physics in the PINC loss.  Network state: body states, actuator states
    and wheel slip velocities (10); full plant state: 12 (with X, Y).  `variant` M0 / M1, see plant_hf.

    `params` arguments from the generic code (the single-track vehicle dict, cfg.params) are ignored in favour
    of the system's own parameters, unless a full HF parameter dict (with key 'tyre') is passed, e.g. a
    perturbed plant.  The controller-side models (f_s_tf, rk4_step_s_tf) always use the PRIOR with nominal
    parameters; `rk4_step_s_true_tf` is the true model for the NMPC-HF reference arm."""
    name = "hf"
    state_names = ("vx", "vy", "r", "psi", "F_act", "delta_act", "sig_fl", "sig_fr", "sig_rl", "sig_rr")
    full_state_names = ("vx", "vy", "r", "psi", "X", "Y", "w_fl", "w_fr", "w_rl", "w_rr", "F_act", "delta_act")
    input_names = ("Fx", "delta")
    n_s, n_x, n_u = 10, 12, 2
    i_xy = (4, 5)

    def __init__(self, vehicle: dict, variant: str):
        from . import plant_hf, prior_hf
        self._H, self._P = plant_hf, prior_hf
        self.variant = variant
        self.truth = plant_hf.make_params(vehicle, variant)
        self.prior = prior_hf.nominal_params(vehicle)
        self.R_w = self.truth["R_w"]

    def _p(self, params):
        return params if isinstance(params, dict) and "tyre" in params else self.truth

    # ---- state maps ----------------------------------------------------------
    def to_full(self, s, XY=None):
        return self._P.s_to_full(s, self.R_w, XY)

    def from_full(self, x):
        return self._P.full_to_s(x, self.R_w)

    def xy(self, x):
        return np.asarray(x)[..., 4:6]

    def track_full(self, x):
        return np.asarray(x)[..., :6]

    # ---- ground truth (NumPy) ----------------------------------------------------
    def rk4_step(self, x, u, dt, params=None, tyre=None):
        return self._H.rk4_step(x, u, dt, self._p(params))

    def f_full(self, x, u, params=None, tyre=None):
        return self._H.f(np.asarray(x, float), np.asarray(u, float), self._p(params))

    def plant_simulate(self, x0, u, T, dt, params=None, tyre=None):
        return self._H.simulate(x0, u, T, dt, self._p(params))

    cached_states = True
    COLLOC_POOL = 100_000
    COLLOC_POOL_SEED = 777

    def _driving(self, n, seed, box):
        """Driving states, generated once per (variant, box, seed, n) and cached in results/cache/."""
        import os
        from .config import RESULTS_DIR
        from .data_hf import driving_states
        d = os.path.join(RESULTS_DIR, "cache")
        os.makedirs(d, exist_ok=True)
        tag = f"hf_{self.variant}_vx{box.vx[0]:g}-{box.vx[1]:g}_psi{box.psi[0]:g}-{box.psi[1]:g}_s{seed}_n{n}.npy"
        path = os.path.join(d, tag)
        if os.path.exists(path):
            return np.load(path)
        s = driving_states(n, seed, self.truth, (box.vx[0], box.vx[1]), [-6000.0, -0.3], [3000.0, 0.3],
                           psi_range=(box.psi[0], box.psi[1]))
        np.save(path, s)
        return s

    def sample_s0(self, n, box, rng, kind="data", seed=None):
        """Network states from random-excitation drives of the true plant (pinc/data_hf.py), speeds from the box.
        'data' / 'ic': generated per split seed (disjoint seeds -> disjoint drives) and cached;
        'colloc': drawn with `rng` from a cached pool of COLLOC_POOL states with its own seed (resampled every
        epoch without regenerating drives)."""
        if kind == "colloc":
            pool = self._driving(self.COLLOC_POOL, self.COLLOC_POOL_SEED, box)
            return pool[rng.integers(0, len(pool), size=n)]
        base = 100_000 if kind == "ic" else 0
        return self._driving(n, base + int(seed if seed is not None else rng.integers(0, 2**31 - 1)), box)

    # ---- physics prior and MPC glue (TensorFlow) ------------------------------------------
    def f_s_tf(self, s, u, params=None, tyre=None):
        return self._P.f_s(s, u, self.prior, self._H.TF)

    def rk4_step_s_tf(self, s, u, dt, params=None, tyre=None):
        f = lambda z: self._P.f_s(z, u, self.prior, self._H.TF)
        return _rk4_tf(f, s, dt)

    def rk4_step_s_true_tf(self, s, u, dt, params=None, tyre=None):
        f = lambda z: self._P.f_s_true(z, u, self.truth, self._H.TF)
        return _rk4_tf(f, s, dt)

    def planar_velocity(self, s):
        return s[:, 0], s[:, 1], s[:, 3]

    def track_vector(self, s, XY):
        return tf.concat([s[:, :4], XY], axis=1)


def _rk4_tf(f, s, dt):
    dt = tf.constant(float(dt), s.dtype)
    k1 = f(s)
    k2 = f(s + 0.5*dt*k1)
    k3 = f(s + 0.5*dt*k2)
    k4 = f(s + dt*k3)
    return s + dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)


SYSTEMS = {"bicycle": Bicycle, "hf_m0": HighFidelity, "hf_m1": HighFidelity}
_CACHE = {}


def get_system(cfg_or_name):
    """System by name (bicycle | hf_m0 | hf_m1).  High-fidelity systems are built from the config's nominal
    vehicle parameters (or the defaults when only a name is given) and cached."""
    if isinstance(cfg_or_name, str):
        name, vehicle = cfg_or_name, None
    else:
        name, vehicle = getattr(cfg_or_name, "system", "bicycle"), cfg_or_name.params
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}")
    cls = SYSTEMS[name]
    if cls is not HighFidelity:
        return cls()
    if vehicle is None:
        vehicle = dict(plant.DEFAULT_PARAMS)
    key = (name, tuple(sorted(vehicle.items())))
    if key not in _CACHE:
        _CACHE[key] = HighFidelity(vehicle, name.split("_")[1].upper())
    return _CACHE[key]
