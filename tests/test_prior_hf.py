"""Phase 3 acceptance: the physics prior (pinc/prior_hf.py) and the network-state map of the
high-fidelity plant.  Skipped without the local tyre data."""
import os

import numpy as np
import pytest

from pinc import tyre_mf

pytestmark = pytest.mark.skipif(not os.path.exists(tyre_mf.DEFAULT_FILE), reason="tyre data missing")

from pinc import plant_hf as H, prior_hf as P  # noqa: E402


def _samples(n=300, seed=0):
    rng = np.random.default_rng(seed)
    s = np.column_stack([rng.uniform(5, 30, n), rng.uniform(-1.5, 1.5, n), rng.uniform(-0.6, 0.6, n),
                         rng.uniform(-0.5, 0.5, n), rng.uniform(-6000, 3000, n), rng.uniform(-0.3, 0.3, n),
                         rng.uniform(-1, 1, (n, 4))])
    u = np.column_stack([rng.uniform(-6000, 3000, n), rng.uniform(-0.3, 0.3, n)])
    return s, u


def test_prior_equals_reduced_true_plant(cfg):
    """With zero track width, no load transfer and linear tyres the true plant must reduce exactly to the
    prior; the two are written independently, so this checks both."""
    q = P.nominal_params(cfg.params)
    red = dict(H.make_params(cfg.params, "M0"), t_f=0.0, t_r=0.0, h=0.0, tyre_model="linear", C_kappa=q["C_kappa"])
    s, u = _samples()
    a, b = P.f_s(s, u, q), P.f_s_true(s, u, red)
    assert np.all(np.isfinite(b))
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9*np.max(np.abs(a)))


def test_state_map_roundtrip(cfg):
    hf = H.make_params(cfg.params, "M0")
    s, _ = _samples()
    x = P.s_to_full(s, hf["R_w"], XY=[3.0, -1.0])
    assert x.shape == (len(s), 12) and np.all(x[:, 4:6] == [3.0, -1.0])
    np.testing.assert_allclose(P.full_to_s(x, hf["R_w"]), s, atol=1e-12)


def test_true_rates_in_network_coordinates(cfg):
    """f_s_true is the chain rule of the full dynamics: compare with a finite difference of the map."""
    hf = H.make_params(cfg.params, "M0")
    s, u = _samples(20)
    x = P.s_to_full(s, hf["R_w"])
    dt = 1e-7
    fd = (P.full_to_s(x + dt*H.f(x, u, hf), hf["R_w"]) - P.full_to_s(x, hf["R_w"]))/dt
    np.testing.assert_allclose(P.f_s_true(s, u, hf), fd, rtol=1e-5, atol=1e-3)


def test_prior_numpy_and_tf_agree(cfg):
    import tensorflow as tf
    q = P.nominal_params(cfg.params)
    s, u = _samples(50)
    a = P.f_s(s, u, q)
    b = P.f_s(tf.constant(s), tf.constant(u), q, H.TF).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)


def _lat_error(cfg, variant, deltas):
    """Relative error of the prior's lateral acceleration right after a steer step at 20 m/s."""
    q = P.nominal_params(cfg.params)
    hf = H.make_params(cfg.params, variant)
    x = H.simulate(H.free_rolling_state(20.0, hf), [500.0, 0.0], 1.0, H.DT_PLANT, hf)
    s = P.full_to_s(x, hf["R_w"])
    out = []
    for d in deltas:
        s1 = s.copy()
        s1[5] = d
        u = np.array([500.0, d])
        a, b = P.f_s(s1, u, q), P.f_s_true(s1, u, hf)
        out.append(abs(a[1] - b[1])/abs(b[1]))
    return np.array(out)


def test_prior_error_m0_small_at_small_slip_and_grows(cfg):
    """M0 is calibrated so the prior is right in gentle driving; the error grows as the tyres saturate
    (measured: 0.7 % at 0.005 rad, 17 % at 0.1 rad, 71 % at 0.2 rad)."""
    e = _lat_error(cfg, "M0", (0.005, 0.02, 0.05, 0.1, 0.2))
    assert np.all(np.diff(e) > 0), e
    assert e[0] < 0.02 and e[1] < 0.02 and e[-1] > 0.5, e


def test_prior_error_m1_parametric(cfg):
    """M1: the true tyres are about twice as stiff, so the prior is wrong even at small slip (~50 %)."""
    e = _lat_error(cfg, "M1", (0.005, 0.02))
    assert np.all(e > 0.4), e
