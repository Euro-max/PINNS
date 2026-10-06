"""Network architectures (pinc/model.py, model.arch; E21): every one keeps s(0) = s0 exactly, has a time
derivative (forward-mode JVP) that matches finite differences, survives save / load, runs in the MPC, and
trains.  The default `mlp` is the network of every earlier result."""
import os

import numpy as np
import pytest
import tensorflow as tf

from pinc import tyre_mf
from pinc.config import ROOT, load_config
from pinc.data import sample_collocation, scale_inputs
from pinc.loss import forward_and_time_derivative
from pinc.model import ARCHS, PINCNet, build_model
from pinc.mpc import MPC, PINCPredictor

pytestmark = pytest.mark.skipif(not os.path.exists(tyre_mf.DEFAULT_FILE), reason="tyre data missing")
SMALL = {"model.depth": 3, "model.width": 16, "model.fourier_m": 8, "model.n_basis": 4, "model.kan_layers": 2}


def _cfg(arch, **ov):
    return load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"), dict(SMALL, **{"model.arch": arch}, **ov))


def _perturb(net, scale=0.3):
    """Non-zero weights everywhere (some heads start at zero), deterministic."""
    rng = np.random.default_rng(1)
    for v in net.trainable_variables:
        v.assign(v + scale*rng.standard_normal(v.shape)/np.sqrt(max(1, v.shape[0])))


@pytest.fixture(scope="module")
def z_batch():
    cfg = _cfg("mlp")
    c = sample_collocation(32, 4, cfg)
    return tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))


@pytest.mark.parametrize("arch", ARCHS)
def test_initial_condition_derivative_and_save(arch, z_batch, tmp_path):
    cfg = _cfg(arch)
    net = build_model(cfg)
    _perturb(net)
    n_s = len(cfg.S_x)
    z0 = tf.concat([tf.zeros_like(z_batch[:, :1]), z_batch[:, 1:]], axis=1)
    np.testing.assert_allclose(net(z0).numpy(), z_batch[:, 1:1 + n_s].numpy(), rtol=0, atol=1e-12)
    _, dsdt = forward_and_time_derivative(net, z_batch, cfg.S_x, cfg.T)
    h = 1e-6
    zp = tf.concat([z_batch[:, :1] + h, z_batch[:, 1:]], axis=1)
    zm = tf.concat([z_batch[:, :1] - h, z_batch[:, 1:]], axis=1)
    fd = (net(zp).numpy() - net(zm).numpy())/(2*h)*np.asarray(cfg.S_x)/cfg.T
    np.testing.assert_allclose(dsdt.numpy(), fd, rtol=1e-5, atol=1e-6*np.max(np.abs(fd)))
    net.save_to(str(tmp_path))
    net2 = PINCNet.load_from(str(tmp_path))
    assert net2.arch == arch
    np.testing.assert_array_equal(net2(z_batch).numpy(), net(z_batch).numpy())


@pytest.mark.parametrize("arch", ARCHS)
def test_runs_in_the_mpc(arch):
    cfg = _cfg(arch)
    net = build_model(cfg)
    from pinc.refs import make_reference
    from pinc.system import get_system
    ref = make_reference("lane_change", cfg)
    mpc = MPC(PINCPredictor(net, cfg), cfg, ref.Q, ref.P)
    x0 = get_system(cfg).initial_state(ref.x0())
    u, info = mpc(0.0, x0, ref)
    assert np.all(np.isfinite(u)) and np.isfinite(info["cost"])


@pytest.mark.parametrize("arch", [a for a in ARCHS if a != "mlp"])
def test_trains_with_the_physics_loss(arch, tmp_path, monkeypatch):
    import pinc.runinfo as ri
    from pinc.train import train
    monkeypatch.setattr(ri, "RESULTS_DIR", str(tmp_path))
    cfg = _cfg(arch, **{"train.n_data": 100, "train.n_val": 100, "train.n_test": 100, "train.steps": 10,
                        "train.batch_data": 100, "train.lbfgs_iters": 3, "train.n_colloc": 64, "train.batch_colloc": 64,
                        "train.val_every": 1, "loss.lam": 0.01, "loss.residual_mask": [1]*6 + [0]*4})
    s = train(cfg, 0, f"arch_{arch}", exp="models", verbose=False)
    assert np.isfinite(s["best_val"]) and PINCNet.load_from(s["run_dir"]).arch == arch


def test_anchored_uses_learned_theta(z_batch):
    cfg = _cfg("anchored", **{"model.learn_theta": True})
    net = build_model(cfg)
    a = net(z_batch).numpy()
    net.log_theta.assign_add(tf.constant([0.3]*5, net.log_theta.dtype))
    assert np.max(np.abs(net(z_batch).numpy() - a)) > 1e-6          # the anchor moves with the parameters


def test_anchored_on_the_single_track_model(tmp_path):
    cfg = load_config(None, dict(SMALL, **{"model.arch": "anchored"}))
    net = build_model(cfg)
    c = sample_collocation(16, 2, cfg)
    z = tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))
    z0 = tf.concat([tf.zeros_like(z[:, :1]), z[:, 1:]], axis=1)
    np.testing.assert_allclose(net(z0).numpy(), z[:, 1:5].numpy(), rtol=0, atol=1e-12)
    net.save_to(str(tmp_path))
    np.testing.assert_array_equal(PINCNet.load_from(str(tmp_path))(z).numpy(), net(z).numpy())


ANCHOR_SETTINGS = {"tau_w": {"model.anchor_learn_tau_w": True}, "exp": {"model.anchor_actuator": "exp"},
                   "end": {"model.anchor_slip_at": "end"}, "gain": {"model.anchor_gain": True},
                   "all": {"model.anchor_learn_tau_w": True, "model.anchor_actuator": "exp", "model.anchor_slip_at": "end",
                           "model.anchor_gain": True}}


@pytest.mark.parametrize("name", ANCHOR_SETTINGS)
def test_anchor_settings(name, z_batch, tmp_path):
    from pinc.loss import total_loss
    from pinc.data import sample_ic, sample_trajectories
    cfg = _cfg("anchored", **ANCHOR_SETTINGS[name])
    net = build_model(cfg)
    n_s = len(cfg.S_x)
    z0 = tf.concat([tf.zeros_like(z_batch[:, :1]), z_batch[:, 1:]], axis=1)
    np.testing.assert_allclose(net(z0).numpy(), z_batch[:, 1:1 + n_s].numpy(), rtol=0, atol=1e-12)
    _, dsdt = forward_and_time_derivative(net, z_batch, cfg.S_x, cfg.T)
    h = 1e-6
    zp = tf.concat([z_batch[:, :1] + h, z_batch[:, 1:]], axis=1)
    zm = tf.concat([z_batch[:, :1] - h, z_batch[:, 1:]], axis=1)
    fd = (net(zp).numpy() - net(zm).numpy())/(2*h)*np.asarray(cfg.S_x)/cfg.T
    np.testing.assert_allclose(dsdt.numpy(), fd, rtol=1e-5, atol=1e-6*np.max(np.abs(fd)))
    new = [v for v in net.trainable_variables if v.name in ("log_tau_w", "anchor_g")]
    if new:
        d = sample_trajectories(32, 5, cfg)
        zd = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg))
        with tf.GradientTape() as tape:
            L = total_loss(net, zd, tf.constant(d["s"]), z0, z_batch, cfg)["total"]
        for v, g in zip(new, tape.gradient(L, new)):
            assert g is not None and np.all(np.isfinite(g.numpy())) and np.any(g.numpy() != 0), v.name
    for v in new:
        v.assign(v + 0.1)
    net.save_to(str(tmp_path))
    np.testing.assert_array_equal(PINCNet.load_from(str(tmp_path))(z_batch).numpy(), net(z_batch).numpy())
