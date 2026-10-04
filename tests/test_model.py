"""Phase 3 acceptance for the network (guards D9, D14)."""
import numpy as np
import tensorflow as tf

from pinc.config import ModelCfg
from pinc.model import PINCNet, build_model
from pinc.data import sample_trajectories, scale_inputs


def test_hard_ic_exact_at_t0(cfg):
    net = build_model(cfg)
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, (16, 7))
    z[:, 0] = 0.0
    out = net(tf.constant(z)).numpy()
    np.testing.assert_allclose(out, z[:, 1:5], atol=1e-12)


def test_output_not_clipped_and_gradients_nonzero(cfg):
    net = build_model(cfg)
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, (64, 7))
    z[:, 0] = 1.0
    z[:8, 1:5] = [[1.2, 1.2, 1.2, 1.2]]*8             # outside the scaled box on purpose
    with tf.GradientTape() as tape:
        out = net(tf.constant(z))
        loss = tf.reduce_sum(tf.square(out))
    g = tape.gradient(loss, net.trainable_variables)
    assert all(np.all(np.isfinite(gi.numpy())) for gi in g)
    assert sum(float(tf.reduce_sum(tf.abs(gi))) for gi in g) > 0
    assert np.max(np.abs(out.numpy())) > 1.0            # nothing squashes the output


def test_scale_inputs_matches_data_module(cfg):
    net = build_model(cfg)
    d = sample_trajectories(10, 0, cfg)
    z_np = scale_inputs(d["t"], d["s0"], d["u"], cfg)
    z_tf = net.scale_inputs(d["t"], d["s0"], d["u"]).numpy()
    np.testing.assert_allclose(z_np, z_tf, rtol=0, atol=1e-14)
    p = net.predict_physical(d["t"], d["s0"], d["u"]).numpy()
    assert p.shape == (10, 4)


def test_save_load_roundtrip(cfg, tmp_path):
    net = build_model(cfg)
    z = tf.constant(np.random.default_rng(1).uniform(-1, 1, (5, 7)))
    y = net(z).numpy()
    net.save_to(str(tmp_path))
    net2 = PINCNet.load_from(str(tmp_path))
    np.testing.assert_array_equal(net2(z).numpy(), y)
    flat = net.get_flat_weights()
    net2.set_flat_weights(flat*0 + 0.1)
    assert not np.allclose(net2(z).numpy(), y)
    net2.set_flat_weights(flat)
    np.testing.assert_array_equal(net2(z).numpy(), y)


def test_soft_ic_and_residual_variants(cfg):
    for hard, res, do, ln in [(False, "none", 0.0, False), (True, "skip", 0.0, False), (False, "block", 0.1, True), (True, "none", 0.2, True)]:
        m = ModelCfg(depth=3, width=16, hard_ic=hard, residual=res, dropout=do, layernorm=ln)
        net = PINCNet(m, cfg.S_x, cfg.S_u, cfg.T, cfg.dtype)
        z = tf.constant(np.random.default_rng(0).uniform(-1, 1, (8, 7)))
        out = net(z).numpy()
        assert out.shape == (8, 4)
        np.testing.assert_array_equal(net(z).numpy(), out)                     # inference is deterministic
        if do > 0:
            assert not np.allclose(net(z, training=True).numpy(), out)      # dropout active only in training
        if hard:
            z0 = z.numpy().copy(); z0[:, 0] = 0
            np.testing.assert_allclose(net(tf.constant(z0)).numpy(), z0[:, 1:5], atol=1e-12)
    m = ModelCfg(depth=3, width=16, residual="block")
    assert sum(int(np.prod(v.shape)) for v in PINCNet(m, cfg.S_x, cfg.S_u, cfg.T, cfg.dtype).trainable_variables) > \
        sum(int(np.prod(v.shape)) for v in PINCNet(ModelCfg(depth=3, width=16), cfg.S_x, cfg.S_u, cfg.T, cfg.dtype).trainable_variables)
