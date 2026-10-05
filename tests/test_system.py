"""The system abstraction (pinc/system.py): the bicycle maps, config validation, and a toy
system of a different size (bicycle + first-order longitudinal actuator lag, n_s = 5,
n_x = 7) pushed through data, model, loss, MPC and the closed-loop simulator, so that
nothing downstream silently assumes 4 states."""
import numpy as np
import pytest
import tensorflow as tf

from pinc import plant, plant_tf, system
from pinc.data import make_splits, sample_trajectories, scale_inputs
from pinc.loss import total_loss
from pinc.model import build_model
from pinc.mpc import MPC, PINCPredictor, RK4Predictor
from pinc.refs import make_reference
from pinc.sim import simulate

TAU = 0.15


class LagBicycle(system.Bicycle):
    """s = [vx, vy, r, psi, F]; x = [vx, vy, r, psi, X, Y, F]; F follows Fx_cmd with lag TAU."""
    name = "lag_bicycle"
    state_names = ("vx", "vy", "r", "psi", "F")
    full_state_names = ("vx", "vy", "r", "psi", "X", "Y", "F")
    n_s, n_x, n_u = 5, 7, 2

    def to_full(self, s, XY=None):
        s = np.asarray(s, float)
        XY = np.zeros(s.shape[:-1] + (2,)) if XY is None else np.broadcast_to(XY, s.shape[:-1] + (2,))
        return np.concatenate([s[..., :4], XY, s[..., 4:]], axis=-1)

    def from_full(self, x):
        x = np.asarray(x)
        return np.concatenate([x[..., :4], x[..., 6:]], axis=-1)

    def f_full(self, x, u, params, tyre="linear"):
        x, u = np.asarray(x, float), np.asarray(u, float)
        ua = np.stack([x[..., 6], u[..., 1]], axis=-1)
        dF = (u[..., 0] - x[..., 6])/TAU
        return np.concatenate([plant.f(x[..., :6], ua, params, tyre), dF[..., None]], axis=-1)

    def rk4_step(self, x, u, dt, params, tyre="linear"):
        f = lambda x: self.f_full(x, u, params, tyre)
        k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
        return x + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

    def sample_s0(self, n, box, rng):
        return np.concatenate([rng.uniform(box.lo(), box.hi(), size=(n, 4)),
                               rng.uniform(-6000.0, 3000.0, size=(n, 1))], axis=1)

    def f_s_tf(self, s, u, params, tyre="linear"):
        ua = tf.stack([s[:, 4], u[:, 1]], axis=1)
        dF = (u[:, 0] - s[:, 4])/TAU
        return tf.concat([plant_tf.f_tf(s[:, :4], ua, params, tyre), dF[:, None]], axis=1)

    def rk4_step_s_tf(self, s, u, dt, params, tyre="linear"):
        f = lambda s: self.f_s_tf(s, u, params, tyre)
        dt = tf.constant(float(dt), s.dtype)
        k1 = f(s); k2 = f(s + 0.5*dt*k1); k3 = f(s + 0.5*dt*k2); k4 = f(s + dt*k3)
        return s + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

    def track_vector(self, s, XY):
        return tf.concat([s[:, :4], XY], axis=1)


@pytest.fixture
def lag_cfg(cfg, monkeypatch):
    monkeypatch.setitem(system.SYSTEMS, "lag_bicycle", LagBicycle)
    return cfg.with_overrides({"system": "lag_bicycle", "scales.S_x": [30.0, 1.5, 0.6, 0.5, 6000.0],
                               "scales.S_f": [2.035, 9.41, 5.296, 0.3466, 40000.0],
                               "loss.residual_mask": [1, 1, 1, 1, 1], "model.depth": 2, "model.width": 16})


def test_bicycle_state_maps():
    b = system.get_system("bicycle")
    s = np.arange(8.0).reshape(2, 4)
    x = b.to_full(s, XY=[7.0, 9.0])
    assert x.shape == (2, 6) and np.all(x[:, 4:] == [7.0, 9.0])
    np.testing.assert_array_equal(b.from_full(x), s)
    np.testing.assert_array_equal(b.xy(x), [[7.0, 9.0]]*2)


def test_unknown_system_and_size_mismatch_rejected(cfg):
    with pytest.raises(ValueError):
        system.get_system("no_such_system")
    with pytest.raises(AssertionError):
        cfg.with_overrides({"scales.S_x": [30.0, 1.5, 0.6]})


def test_toy_system_data_targets_reintegrate(lag_cfg):
    sysm = system.get_system(lag_cfg)
    d = sample_trajectories(20, 3, lag_cfg)
    assert d["s0"].shape == (20, 5) and d["s"].shape == (20, 5)
    for i in range(20):
        x = sysm.to_full(d["s0"][i])
        for _ in range(int(round(d["t"][i]/lag_cfg.sim.dt_plant))):
            x = sysm.rk4_step(x, d["u"][i], lag_cfg.sim.dt_plant, lag_cfg.params)
        np.testing.assert_allclose(sysm.from_full(x), d["s"][i], rtol=0, atol=1e-8)


def test_toy_system_model_and_loss(lag_cfg):
    net = build_model(lag_cfg)
    sp = make_splits(lag_cfg, n_train=64)
    z = tf.constant(scale_inputs(sp["train"]["t"], sp["train"]["s0"], sp["train"]["u"], lag_cfg))
    assert z.shape[1] == 1 + 5 + 2 and net(z).shape == (64, 5)
    z0 = tf.concat([tf.zeros_like(z[:, :1]), z[:, 1:]], axis=1)
    np.testing.assert_allclose(net(z0).numpy(), z0[:, 1:6].numpy(), atol=1e-12)    # hard IC
    with tf.GradientTape() as tape:
        out = total_loss(net, z, tf.constant(sp["train"]["s"]), z0, z, lag_cfg)
    g = tape.gradient(out["total"], net.trainable_variables)
    assert all(np.all(np.isfinite(x.numpy())) for x in g) and float(out["phys"]) > 0


@pytest.mark.parametrize("kind", ["rk4", "pinc"])
def test_toy_system_mpc_gradient_and_closed_loop(lag_cfg, kind):
    pred = RK4Predictor(lag_cfg) if kind == "rk4" else PINCPredictor(build_model(lag_cfg), lag_cfg)
    mpc = MPC(pred, lag_cfg)
    ref = make_reference("speed_sin", lag_cfg)
    x = np.array([18.0, 0.1, 0.05, 0.02, 15.0, 0.3, 500.0])
    rng = np.random.default_rng(0)
    u = rng.uniform(mpc.lo, mpc.hi, (mpc.N, 2)).ravel()
    ref_seq = mpc.reference_sequence(1.0, ref)
    J0, g, z = mpc.cost(x, u, ref_seq)
    assert z.shape == (mpc.N, 6)
    for i in rng.choice(u.size, 4, replace=False):
        up, um = u.copy(), u.copy()
        up[i] += 1e-6
        um[i] -= 1e-6
        fd = (mpc.cost(x, up, ref_seq)[0] - mpc.cost(x, um, ref_seq)[0])/2e-6
        assert abs(fd - g[i]) <= 1e-4*max(1.0, abs(g[i])), (i, fd, g[i])
    if kind == "rk4":                               # closed loop on the 7-state plant
        x0 = system.get_system(lag_cfg).to_full([20.0, 0.0, 0.0, 0.0, 300.0])
        log = simulate(mpc, lag_cfg.params, ref, x0, 1.0, np.zeros(7), 0, lag_cfg)
        assert log["x"].shape == (11, 7) and log["u"].shape == (10, 2) and np.all(log["success"])
        assert np.max(np.abs(log["err"][1:, 0])) < 0.5          # tracks the speed reference
