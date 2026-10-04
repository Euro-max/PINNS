"""
Trajectory, initial-condition and collocation sampling for PINC training.

Targets ALWAYS come from the RK4 integrator in `pinc/plant.py`, never from
the input state (fixes D4).  Time is sampled on the integrator grid
t = k*dt, k in {1..T/dt}, so every target is reproducible exactly by
`plant.simulate(x0, u, t, dt)`.
"""
from __future__ import annotations

import numpy as np

from . import plant
from .config import Config


def _rng(seed):
    return np.random.default_rng(seed)


def sample_box(n: int, box, rng) -> np.ndarray:
    lo, hi = box.lo(), box.hi()
    return rng.uniform(lo, hi, size=(n, 4))


def sample_inputs(n: int, cfg: Config, rng) -> np.ndarray:
    return rng.uniform(np.asarray(cfg.u_min), np.asarray(cfg.u_max), size=(n, 2))


def sample_trajectories(n: int, seed: int, cfg: Config, box=None, dt: float | None = None) -> dict:
    """Draw s0 and constant u uniformly, integrate with RK4 (dt=1e-3) over
    [0, T] and return samples at random grid times t in (0, T] with exact
    targets s(t).  Returns dict(t, s0, u, s), all NumPy float64."""
    box = box or cfg.box_train
    dt = dt or cfg.sim.dt_plant
    rng = _rng(seed)
    n_steps = int(round(cfg.T/dt))
    s0 = sample_box(n, box, rng)
    u = sample_inputs(n, cfg, rng)
    k = rng.integers(1, n_steps + 1, size=n)          # t = k*dt in (0, T]
    t = k*dt
    x = np.concatenate([s0, np.zeros((n, 2))], axis=1)
    target = np.full((n, 4), np.nan)
    for step in range(1, n_steps + 1):
        x = plant.rk4_step(x, u, dt, cfg.params, cfg.sim.tyre)
        sel = k == step
        if np.any(sel):
            target[sel] = x[sel, :4]
    plant.check_finite(target, "trajectory targets")
    assert not np.any(np.isnan(target))
    return dict(t=t.astype(float), s0=s0, u=u, s=target)


def sample_ic(n: int, seed: int, cfg: Config, box=None) -> dict:
    """Points at t = 0 whose target is s0 itself (initial-condition loss)."""
    box = box or cfg.box_train
    rng = _rng(seed)
    s0 = sample_box(n, box, rng)
    u = sample_inputs(n, cfg, rng)
    return dict(t=np.zeros(n), s0=s0, u=u, s=s0.copy())


def sample_collocation(n: int, seed: int, cfg: Config, box=None) -> dict:
    """Random (t, s0, u) with t uniform in (0, T]; no targets."""
    box = box or cfg.box_train
    rng = _rng(seed)
    s0 = sample_box(n, box, rng)
    u = sample_inputs(n, cfg, rng)
    t = rng.uniform(0.0, cfg.T, size=n)
    t = np.where(t == 0.0, cfg.T, t)
    return dict(t=t, s0=s0, u=u)


def scale_inputs(t, s0, u, cfg: Config) -> np.ndarray:
    """Network input z = [t/T, s0/S_x, u/S_u], shape (n, 7)."""
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    return np.concatenate([t/cfg.T, np.asarray(s0)/cfg.S_x, np.asarray(u)/cfg.S_u], axis=1)


def make_splits(cfg: Config, n_train: int | None = None) -> dict:
    """Train / val / test / test_extrap with disjoint seeds (ground rule 8)."""
    tr = cfg.train
    n_train = tr.n_data if n_train is None else n_train
    return dict(
        train=sample_trajectories(n_train, cfg.seeds.train, cfg, cfg.box_train),
        val=sample_trajectories(tr.n_val, cfg.seeds.val, cfg, cfg.box_train),
        test=sample_trajectories(tr.n_test, cfg.seeds.test, cfg, cfg.box_train),
        test_extrap=sample_trajectories(tr.n_test, cfg.seeds.test_extrap, cfg, cfg.box_extrap),
    )
