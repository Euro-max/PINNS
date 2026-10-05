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
from .system import get_system


def _rng(seed):
    return np.random.default_rng(seed)


def sample_box(n: int, box, rng, cfg: Config | None = None, kind: str = "data", seed: int | None = None) -> np.ndarray:
    """Initial network states for `box` (the system decides how; bicycle: uniform).  `kind` / `seed` let a
    system with expensive state generation cache by split ('data', 'ic') or draw from a pool ('colloc')."""
    sysm = get_system(cfg or "bicycle")
    if getattr(sysm, "cached_states", False):
        return sysm.sample_s0(n, box, rng, kind=kind, seed=seed)
    return sysm.sample_s0(n, box, rng)


def sample_inputs(n: int, cfg: Config, rng) -> np.ndarray:
    lo, hi = np.asarray(cfg.u_min), np.asarray(cfg.u_max)
    return rng.uniform(lo, hi, size=(n, lo.size))


def sample_trajectories(n: int, seed: int, cfg: Config, box=None, dt: float | None = None) -> dict:
    """Draw s0 and constant u uniformly, integrate with RK4 (dt=1e-3) over
    [0, T] and return samples at random grid times t in (0, T] with exact
    targets s(t).  Returns dict(t, s0, u, s), all NumPy float64."""
    box = box or cfg.box_train
    dt = dt or cfg.sim.dt_plant
    sysm = get_system(cfg)
    rng = _rng(seed)
    n_steps = int(round(cfg.T/dt))
    s0 = sample_box(n, box, rng, cfg, kind="data", seed=seed)
    u = sample_inputs(n, cfg, rng)
    k = rng.integers(1, n_steps + 1, size=n)          # t = k*dt in (0, T]
    t = k*dt
    x = sysm.to_full(s0)
    target = np.full((n, sysm.n_s), np.nan)
    for step in range(1, n_steps + 1):
        x = sysm.rk4_step(x, u, dt, cfg.params, cfg.sim.tyre)
        sel = k == step
        if np.any(sel):
            target[sel] = sysm.from_full(x[sel])
    plant.check_finite(target, "trajectory targets")
    assert not np.any(np.isnan(target))
    return dict(t=t.astype(float), s0=s0, u=u, s=target)


def sample_ic(n: int, seed: int, cfg: Config, box=None) -> dict:
    """Points at t = 0 whose target is s0 itself (initial-condition loss)."""
    box = box or cfg.box_train
    rng = _rng(seed)
    s0 = sample_box(n, box, rng, cfg, kind="ic", seed=seed)
    u = sample_inputs(n, cfg, rng)
    return dict(t=np.zeros(n), s0=s0, u=u, s=s0.copy())


def sample_collocation(n: int, seed: int, cfg: Config, box=None) -> dict:
    """Random (t, s0, u) with t uniform in (0, T]; no targets."""
    box = box or cfg.box_train
    rng = _rng(seed)
    s0 = sample_box(n, box, rng, cfg, kind="colloc", seed=seed)
    u = sample_inputs(n, cfg, rng)
    t = rng.uniform(0.0, cfg.T, size=n)
    t = np.where(t == 0.0, cfg.T, t)
    n_log = int(round(getattr(cfg.train, "colloc_log_frac", 0.0)*n))
    if n_log > 0:                                     # resolve fast transients near t = 0 (e.g. wheel slip)
        t[:n_log] = cfg.T*10.0**rng.uniform(-3.0, 0.0, size=n_log)
    return dict(t=t, s0=s0, u=u)


def scale_inputs(t, s0, u, cfg: Config) -> np.ndarray:
    """Network input z = [t/T, s0/S_x, u/S_u], shape (n, 1 + n_s + n_u)."""
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
