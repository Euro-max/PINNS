"""Learnable prior parameters (plan decision 2d): off by default; on the HF system the physics residual
depends on them, gradients reach them, and they survive save / load."""
import os

import numpy as np
import pytest
import tensorflow as tf

from pinc import tyre_mf
from pinc.config import ROOT, load_config
from pinc.data import sample_collocation, scale_inputs
from pinc.loss import physics_loss, physics_residual
from pinc.model import PINCNet, build_model


def test_off_by_default(cfg):
    net = build_model(cfg)
    assert net.log_theta is None and net.theta() == {}


@pytest.mark.skipif(not os.path.exists(tyre_mf.DEFAULT_FILE), reason="tyre data missing")
def test_learnable_theta_on_hf(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"),
                      {"model.learn_theta": True, "model.depth": 2, "model.width": 16, "loss.residual_mask": [1]*6 + [0]*4})
    net = build_model(cfg)
    th = {k: float(v) for k, v in net.theta().items()}
    assert set(th) == {"Caf", "Car", "C_kappa", "tau_F", "tau_delta"} and th["Caf"] == pytest.approx(cfg.params["Caf"])
    assert any(v is net.log_theta for v in net.trainable_variables)        # else the optimisers never update it
    c = sample_collocation(64, 3, cfg)
    z = tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))
    with tf.GradientTape() as tape:
        L = physics_loss(net, z, cfg)
    g = tape.gradient(L, net.log_theta)
    assert g is not None and np.all(np.isfinite(g.numpy())) and np.all(g.numpy() != 0)
    R0 = physics_residual(net, z, cfg).numpy()
    net.log_theta.assign_add(tf.constant([0.3]*5, net.log_theta.dtype))
    assert np.max(np.abs(physics_residual(net, z, cfg).numpy() - R0)) > 1e-6
    net.save_to(str(tmp_path))
    net2 = PINCNet.load_from(str(tmp_path))
    np.testing.assert_allclose(net2.log_theta.numpy(), net.log_theta.numpy())
    np.testing.assert_allclose(net2(z).numpy(), net(z).numpy())
