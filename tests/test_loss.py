"""Phase 3 acceptance for the loss (guards D1, D2, D3, D14)."""
import numpy as np
import pytest
import tensorflow as tf

from pinc import plant_tf
from pinc.loss import (time_derivative, physics_residual, total_loss, data_loss, ic_loss,
                       forward_and_time_derivative)
from pinc.model import build_model
from pinc.data import sample_trajectories, sample_collocation, sample_ic, scale_inputs


class SineModel:
    """Physical output s_i(t) = A_i sin(w_i t + phi_i * s0_i) in SCALED units."""
    def __init__(self, S_x, T):
        self.A = tf.constant([3.0, 0.7, 0.2, 0.1], tf.float64)
        self.w = tf.constant([7.0, 11.0, 5.0, 13.0], tf.float64)
        self.S_x = tf.constant(S_x, tf.float64)
        self.T = tf.constant(T, tf.float64)

    def __call__(self, z):
        t = z[:, 0:1]*self.T
        phase = z[:, 1:5]                     # depends on the state inputs too
        return self.A*tf.sin(self.w*t + phase)/self.S_x

    def dsdt(self, z):
        t = z[:, 0:1]*self.T
        return (self.A*self.w*tf.cos(self.w*t + z[:, 1:5])).numpy()


@pytest.mark.parametrize("method", ["forward", "reverse"])
def test_time_derivative_matches_closed_form(cfg, method):
    m = SineModel(cfg.S_x, cfg.T)
    rng = np.random.default_rng(0)
    z = tf.constant(rng.uniform(-1, 1, (200, 7)))
    d = time_derivative(m, z, cfg.S_x, cfg.T, method=method).numpy()
    np.testing.assert_allclose(d, m.dsdt(z), rtol=1e-8, atol=1e-4)


def test_forward_and_reverse_agree_on_network(cfg):
    net = build_model(cfg)
    z = tf.constant(np.random.default_rng(0).uniform(-1, 1, (50, 7)))
    s1, d1 = forward_and_time_derivative(net, z, cfg.S_x, cfg.T, "forward")
    s2, d2 = forward_and_time_derivative(net, z, cfg.S_x, cfg.T, "reverse")
    np.testing.assert_allclose(d1.numpy(), d2.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(s1.numpy(), s2.numpy())


class RK4Model:
    """Returns the RK4 solution s(t) as a smooth function of t (fixed number
    of substeps, dt = t/n), in scaled units -- an 'exact' PINC."""
    def __init__(self, cfg, n_sub=25):
        self.cfg, self.n = cfg, n_sub
        self.S_x = tf.constant(cfg.S_x, tf.float64)
        self.S_u = tf.constant(cfg.S_u, tf.float64)

    def __call__(self, z):
        t = z[:, 0:1]*self.cfg.T
        s = z[:, 1:5]*self.S_x
        u = z[:, 5:7]*self.S_u
        h = t/self.n
        for _ in range(self.n):
            k1 = plant_tf.f_tf(s, u, self.cfg.params)
            k2 = plant_tf.f_tf(s + 0.5*h*k1, u, self.cfg.params)
            k3 = plant_tf.f_tf(s + 0.5*h*k2, u, self.cfg.params)
            k4 = plant_tf.f_tf(s + h*k3, u, self.cfg.params)
            s = s + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return s/self.S_x


def test_residual_of_exact_solution_is_zero(cfg):
    m = RK4Model(cfg)
    c = sample_collocation(200, 5, cfg)
    z = tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))
    R = physics_residual(m, z, cfg).numpy()
    assert np.max(np.abs(R)) < 1e-6, np.max(np.abs(R))
    # and the same wrapper reproduces the integrator targets
    d = sample_trajectories(100, 6, cfg)
    zd = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg))
    np.testing.assert_allclose((m(zd)*m.S_x).numpy(), d["s"], atol=1e-7)


def test_residual_uses_all_four_states(cfg):
    """A model that ignores the dynamics must have non-zero residual in every channel."""
    net = build_model(cfg)
    c = sample_collocation(500, 8, cfg)
    z = tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))
    R = physics_residual(net, z, cfg).numpy()
    assert np.all(np.sqrt(np.mean(R**2, axis=0)) > 1e-3)


def test_loss_gradients_finite_and_nonzero_across_box(cfg):
    net = build_model(cfg)
    d = sample_trajectories(256, 9, cfg, box=cfg.box_full)
    ic = sample_ic(64, 10, cfg, box=cfg.box_full)
    c = sample_collocation(256, 11, cfg, box=cfg.box_full)
    z_d = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg))
    z_ic = tf.constant(scale_inputs(ic["t"], ic["s0"], ic["u"], cfg))
    z_c = tf.constant(scale_inputs(c["t"], c["s0"], c["u"], cfg))
    with tf.GradientTape() as tape:
        L = total_loss(net, z_d, d["s"], z_ic, z_c, cfg)
    g = tape.gradient(L["total"], net.trainable_variables)
    assert np.isfinite(float(L["total"])) and np.isfinite(float(L["phys"]))
    for gi in g:
        assert np.all(np.isfinite(gi.numpy()))
    assert sum(float(tf.norm(gi)) for gi in g) > 1e-6
    # each term contributes gradient on its own
    with tf.GradientTape() as tape:
        lp = total_loss(net, z_d, d["s"], z_ic, z_c, cfg)["phys"]
    gp = tape.gradient(lp, net.trainable_variables)
    assert sum(float(tf.norm(gi)) for gi in gp) > 1e-8


def test_lambda_zero_removes_physics(cfg):
    cfg0 = cfg.with_overrides({"loss.lam": 0.0})
    net = build_model(cfg0)
    d = sample_trajectories(64, 9, cfg0)
    z_d = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg0))
    z_c = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg0))
    ic = sample_ic(32, 12, cfg0)
    z_ic = tf.constant(scale_inputs(ic["t"], ic["s0"], ic["u"], cfg0))
    L = total_loss(net, z_d, d["s"], z_ic, z_c, cfg0)
    assert float(L["total"]) == pytest.approx(float(L["data"]) + float(L["ic"]))
    assert float(ic_loss(net, z_ic)) == 0.0         # hard IC -> exact
    assert float(data_loss(net, z_d, d["s"], cfg0)) > 0
