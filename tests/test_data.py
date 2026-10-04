"""Phase 2 acceptance: targets come from the integrator (guards D4, D11)."""
import numpy as np

from pinc import plant
from pinc.data import sample_trajectories, sample_ic, sample_collocation, make_splits, scale_inputs


def test_targets_reintegrate_exactly(cfg):
    d = sample_trajectories(300, seed=7, cfg=cfg)
    rng = np.random.default_rng(0)
    for i in rng.choice(300, 100, replace=False):
        x0 = np.concatenate([d["s0"][i], [0.0, 0.0]])
        x = plant.simulate(x0, d["u"][i], d["t"][i], cfg.sim.dt_plant, cfg.params)
        np.testing.assert_allclose(x[:4], d["s"][i], rtol=0, atol=1e-8)


def test_target_differs_from_input_state(cfg):
    d = sample_trajectories(500, seed=3, cfg=cfg)
    assert np.all(d["t"] > 0) and np.all(d["t"] <= cfg.T + 1e-12)
    diff = np.linalg.norm(d["s"] - d["s0"], axis=1)
    assert np.all(diff > 0)                     # no sample has y_target == input state
    assert np.median(diff) > 1e-3


def test_ic_and_collocation(cfg):
    ic = sample_ic(50, 1, cfg)
    assert np.all(ic["t"] == 0) and np.array_equal(ic["s"], ic["s0"])
    c = sample_collocation(50, 1, cfg)
    assert "s" not in c and np.all(c["t"] > 0) and np.all(c["t"] <= cfg.T)


def test_splits_are_disjoint_and_extrap_region(cfg):
    sp = make_splits(cfg, n_train=200)
    assert not np.allclose(sp["train"]["s0"][:5], sp["val"]["s0"][:5])
    assert not np.allclose(sp["val"]["s0"][:5], sp["test"]["s0"][:5])
    assert np.all(sp["train"]["s0"][:, 0] <= cfg.box_train.vx[1])
    assert np.all(sp["test_extrap"]["s0"][:, 0] >= cfg.box_extrap.vx[0])


def test_scaling_uses_config(cfg):
    d = sample_trajectories(10, 0, cfg)
    z = scale_inputs(d["t"], d["s0"], d["u"], cfg)
    assert z.shape == (10, 7)
    np.testing.assert_allclose(z[:, 0], d["t"]/cfg.T)
    np.testing.assert_allclose(z[:, 5:7], d["u"]/cfg.S_u)
    assert np.all(np.abs(z[:, 1:]) <= 1.0 + 1e-12)


def test_seed_reproducible(cfg):
    a = sample_trajectories(20, 11, cfg)
    b = sample_trajectories(20, 11, cfg)
    np.testing.assert_array_equal(a["s"], b["s"])
