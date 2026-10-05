"""Phase 2 acceptance: the high-fidelity plant (pinc/plant_hf.py) and its Magic Formula tyres
(pinc/tyre_mf.py).  See docs/PLAN_HIGH_FIDELITY.md, sec. 7.  The tyre data are MathWorks data,
exported locally by scripts/export_tyre_params.m; without them these tests are skipped."""
import os

import numpy as np
import pytest

from pinc import plant, tyre_mf

pytestmark = pytest.mark.skipif(not os.path.exists(tyre_mf.DEFAULT_FILE),
                                reason="tyre data missing: run scripts/export_tyre_params.m (data/tyre/ is not redistributed)")

from pinc import plant_hf as H  # noqa: E402


@pytest.fixture(scope="module")
def tyre():
    return tyre_mf.load_params()


@pytest.fixture(scope="module")
def hf(cfg):
    return H.make_params(cfg.params, "M0")


def _axle_fy(kappa, alpha, Fz, p, mu=1.0):
    """Mean of a right-hand and a mirrored left-hand tyre: the built-in lateral offsets cancel."""
    return 0.5*(tyre_mf.forces(kappa, alpha, Fz, p, mu)[1] + tyre_mf.forces(kappa, alpha, Fz, p, mu, mirror=True)[1])


# ---- tyre -----------------------------------------------------------------------------------
def test_tyre_small_slip_is_linear(tyre):
    for Fz in (2000.0, 3679.0, 6000.0):
        a = 1e-4
        assert _axle_fy(0.0, a, Fz, tyre) == pytest.approx(tyre_mf.cornering_stiffness(Fz, tyre)*a, rel=0.01)
        k = 1e-4
        Fx = tyre_mf.forces(k, 0.0, Fz, tyre)[0] - tyre_mf.forces(0.0, 0.0, Fz, tyre)[0]
        assert Fx == pytest.approx(tyre_mf.slip_stiffness(Fz, tyre)*k, rel=0.01)


def test_tyre_sign_convention_matches_plant(tyre):
    """Positive slip angle -> positive lateral force (plant.py convention); positive slip -> traction."""
    assert _axle_fy(0.0, 0.02, 3679.0, tyre) > 0 and _axle_fy(0.0, -0.02, 3679.0, tyre) < 0
    assert tyre_mf.forces(0.05, 0.0, 3679.0, tyre)[0] > 0 and tyre_mf.forces(-0.05, 0.0, 3679.0, tyre)[0] < 0


def test_tyre_saturates_at_friction_limit(tyre):
    Fz = 3679.0
    a = np.linspace(-0.6, 0.6, 601)
    k = np.linspace(-0.6, 0.6, 601)
    Fy = _axle_fy(0.0, a, Fz, tyre)
    Fx = tyre_mf.forces(k, 0.0, Fz, tyre)[0]
    dfz = (Fz - tyre["FNOMIN"])/tyre["FNOMIN"]
    assert np.max(np.abs(Fy)) <= (tyre["PDY1"] + tyre["PDY2"]*dfz)*Fz*1.01
    assert np.max(np.abs(Fx)) <= (tyre["PDX1"] + tyre["PDX2"]*dfz)*Fz*1.01
    Fy_low = _axle_fy(0.0, a, Fz, tyre, mu=0.4)                # friction scaling for road surfaces
    assert np.max(np.abs(Fy_low)) < 0.45*np.max(np.abs(Fy))


def test_combined_slip_reduces_both_forces(tyre):
    Fz = 3679.0
    Fx_pure, _ = tyre_mf.forces(0.1, 0.0, Fz, tyre)
    Fy_pure = _axle_fy(0.0, 0.1, Fz, tyre)
    Fx_c, _ = tyre_mf.forces(0.1, 0.1, Fz, tyre)
    Fy_c = _axle_fy(0.1, 0.1, Fz, tyre)
    assert abs(Fx_c) < 0.9*abs(Fx_pure) and abs(Fy_c) < 0.9*abs(Fy_pure)


def test_mirrored_tyres_cancel_offsets(tyre):
    for Fz in (2000.0, 3679.0):
        right = tyre_mf.forces(0.0, 0.0, Fz, tyre)[1]
        left = tyre_mf.forces(0.0, 0.0, Fz, tyre, mirror=True)[1]
        assert abs(right) > 1.0 and abs(right + left) < 1e-9


# ---- plant ------------------------------------------------------------------------------------
def test_m0_calibration_matches_single_track_stiffness(cfg, hf):
    Fz = H.static_loads(hf)
    assert 2*tyre_mf.cornering_stiffness(Fz[0], hf["tyre"]) == pytest.approx(cfg.params["Caf"], rel=1e-9)
    m1 = H.make_params(cfg.params, "M1")
    assert m1["tyre"]["LKY"] == 1.0 and 2*tyre_mf.cornering_stiffness(Fz[0], m1["tyre"]) > 1.5*cfg.params["Caf"]


def test_vertical_loads(hf):
    Fz = np.array(H.static_loads(hf))
    assert Fz.sum() == pytest.approx(hf["m"]*hf["g"])
    v = 20.0
    trim = 0.5*hf["rho"]*hf["Cd"]*hf["A"]*v**2 + hf["Frr"]*np.tanh(v/0.1)
    np.testing.assert_allclose(H.vertical_loads(v, 0.0, trim, hf), Fz, rtol=1e-9)      # no acceleration: static
    brake = np.array(H.vertical_loads(v, 0.0, -5000.0, hf))
    assert brake[:2].sum() > Fz[:2].sum() and brake.sum() == pytest.approx(Fz.sum())   # braking loads the front
    left = np.array(H.vertical_loads(v, 0.3, trim, hf))                               # a_y > 0: left turn
    assert left[1] > Fz[1] and left[0] < Fz[0] and left[3] > left[2] and left.sum() == pytest.approx(Fz.sum())


def test_steady_straight_driving(hf):
    """At the trim force the car holds its speed with small, positive drive slip at the front only."""
    x0 = H.free_rolling_state(20.0, hf)
    x = H.simulate(x0, [x0[10], 0.0], 2.0, H.DT_PLANT, hf)
    assert abs(x[0] - 20.0) < 0.01 and abs(x[1]) < 1e-3 and abs(x[2]) < 1e-3
    slip = (hf["R_w"]*x[6:10] - x[0])/x[0]
    assert 0 < slip[0] < 0.01 and 0 < slip[1] < 0.01 and np.all(np.abs(slip[2:]) < 1e-3)


@pytest.mark.parametrize("v", [10.0, 20.0, 30.0])
def test_m0_lateral_dynamics_match_single_track(cfg, hf, v):
    """With the fast wheel-speed modes eliminated (quasi-steady), the (vy, r) dynamics of HF-M0 equal the
    linear single-track model within 1 % per entry.  (Eigenvalues are not compared: the single-track pair is
    nearly degenerate, so a 1 % change of an entry can make it complex.)"""
    x0 = H.free_rolling_state(v, hf)
    u = np.array([x0[10], 0.0])
    x = H.simulate(x0, u, 1.0, H.DT_PLANT, hf)
    x[4:6] = 0.0
    A = np.zeros((12, 12))
    for j in range(12):
        d = np.zeros(12)
        d[j] = 1e-6
        A[:, j] = (H.f(x + d, u, hf) - H.f(x - d, u, hf))/2e-6
    s, w = [1, 2], [6, 7, 8, 9]
    A_red = A[np.ix_(s, s)] - A[np.ix_(s, w)] @ np.linalg.solve(A[np.ix_(w, w)], A[np.ix_(w, s)])
    p = cfg.params
    m, Iz, lf, lr, Caf, Car, vx = p["m"], p["Iz"], p["lf"], p["lr"], p["Caf"], p["Car"], x[0]
    B = np.array([[-(Caf + Car)/(m*vx), -(lf*Caf - lr*Car)/(m*vx) - vx],
                  [-(lf*Caf - lr*Car)/(Iz*vx), -(lf**2*Caf + lr**2*Car)/(Iz*vx)]])
    for i in range(2):
        for j in range(2):
            if B[i, j] != 0:
                assert A_red[i, j] == pytest.approx(B[i, j], rel=0.01)
            else:
                assert abs(A_red[i, j]) < 0.01*abs(B[i, i])


def test_rk4_order_and_low_speed_stability(hf):
    x0 = H.free_rolling_state(5.0, hf, vy=0.1, r=0.05)
    u = [1500.0, 0.05]
    ref = H.simulate(x0, u, 0.3, 1e-5, hf)
    err = [np.max(np.abs(H.simulate(x0, u, 0.3, dt, hf) - ref)) for dt in (1e-3, 5e-4, 2.5e-4)]
    orders = np.log2(np.array(err[:-1])/np.array(err[1:]))
    assert np.all(orders > 3.6), orders
    assert err[1] < 1e-6                                       # DT_PLANT is accurate at the stiffest speed


def test_numpy_and_tf_agree(hf):
    import tensorflow as tf
    x0 = H.free_rolling_state(20.0, hf, vy=0.1, r=0.05)
    u = np.array([1500.0, 0.05])
    xn = H.simulate(x0, u, 0.2, H.DT_PLANT, hf)
    xt, ut = tf.constant(x0[None]), tf.constant(u[None])
    for _ in range(int(round(0.2/H.DT_PLANT))):
        xt = H.rk4_step(xt, ut, H.DT_PLANT, hf, H.TF)
    assert np.max(np.abs(xn - xt.numpy()[0])) < 1e-6


def test_actuator_lag(hf):
    x0 = H.free_rolling_state(20.0, hf)
    x = H.simulate(x0, [x0[10], 0.02], hf["tau_delta"], H.DT_PLANT, hf)
    assert x[11] == pytest.approx(0.02*(1 - np.exp(-1.0)), rel=1e-6)          # first-order lag reaches 63 %


def test_same_interface_as_single_track_plant(cfg, hf):
    """Front-wheel drive with mild inputs behaves like the single-track model over 1 s (M0 calibration)."""
    x0 = H.free_rolling_state(20.0, hf)
    u = [x0[10] + 500.0, 0.01]
    xb = np.array([20.0, 0, 0, 0, 0, 0])
    xh = x0
    for _ in range(10):                                        # same lagged inputs for both
        xh = H.simulate(xh, u, 0.1, H.DT_PLANT, hf)
        xb = plant.simulate(xb, [xh[10], xh[11]], 0.1, 1e-3, cfg.params)
    assert abs(xh[0] - xb[0]) < 0.05 and abs(xh[2] - xb[2]) < 0.1*abs(xb[2]) and abs(xh[5] - xb[5]) < 0.1
