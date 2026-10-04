"""Phase 5 acceptance (guards D8, D10, D16, D17, D19)."""
import numpy as np
import pytest

from pinc.mpc import MPC, RK4Predictor, PINCPredictor, LinearPredictor, make_controller
from pinc.model import build_model
from pinc.refs import make_reference
from pinc.sim import simulate
from pinc.plant import DEFAULT_PARAMS, trim_force


def _fd_check(mpc, x, ref_seq, n=6, eps=1e-6):
    rng = np.random.default_rng(0)
    u = rng.uniform(mpc.lo, mpc.hi, (mpc.N, 2)).ravel()
    extra = mpc.pred.prepare(x[:4], u.reshape(mpc.N, 2)*mpc.S_u)
    J0, g, _ = mpc.cost(x, u, ref_seq, extra=extra)
    for i in rng.choice(u.size, n, replace=False):
        up, um = u.copy(), u.copy()
        up[i] += eps
        um[i] -= eps
        fd = (mpc.cost(x, up, ref_seq, extra=extra)[0] - mpc.cost(x, um, ref_seq, extra=extra)[0])/(2*eps)
        assert abs(fd - g[i]) <= 1e-4*max(1.0, abs(g[i])), (i, fd, g[i])


@pytest.mark.parametrize("kind", ["rk4", "pinc", "ltv"])
def test_analytic_gradient_matches_finite_differences(cfg, kind):
    if kind == "rk4":
        pred = RK4Predictor(cfg)
    elif kind == "pinc":
        pred = PINCPredictor(build_model(cfg), cfg)
    else:
        pred = LinearPredictor(cfg)
    mpc = MPC(pred, cfg)
    ref = make_reference("lane_change", cfg)
    x = np.array([18.0, 0.1, 0.05, 0.02, 15.0, 0.3])
    _fd_check(mpc, x, mpc.reference_sequence(1.0, ref))


def test_reference_sequence_is_after_each_step(cfg):
    mpc = MPC(RK4Predictor(cfg), cfg)
    ref = make_reference("speed_step", cfg)
    seq = mpc.reference_sequence(cfg.refs.step_time - cfg.T, ref)      # next step lands ON the step time
    assert seq[0, 0] == cfg.refs.v0 + cfg.refs.step_dv


def test_bounds_allow_braking(cfg):
    mpc = MPC(RK4Predictor(cfg), cfg)
    assert mpc.bounds[0][0]*cfg.S_u[0] < 0                                # Fx lower bound negative (D19)
    ref = make_reference("speed_sin", cfg)
    x = np.array([26.0, 0.0, 0.0, 0.0, 0.0, 0.0])                      # far above v_ref=20
    u, info = mpc(0.0, x, ref)
    assert u[0] < 0 and info["success"]


def test_controller_never_sees_plant_params(cfg):
    c = make_controller("nmpc_rk4", cfg)
    assert c.pred.params == cfg.params == DEFAULT_PARAMS


def test_steady_tracking_constant_speed(cfg):
    """RK4 predictor + nominal plant: |vx - 20| < 0.05 m/s after 3 s."""
    class Const:
        def __init__(s):
            s.base = make_reference("speed_sin", cfg)
            s.Q, s.P, s.name = s.base.Q, s.base.P, "const"
        def __call__(s, t):
            z = s.base(t)
            z[:, 0] = 20.0
            z[:, 4] = 20.0*np.atleast_1d(t)
            return z
    ref = Const()
    c = make_controller("nmpc_rk4", cfg)
    log = simulate(c, cfg.params, ref, [17.0, 0, 0, 0, 0, 0], 5.0, np.zeros(6), 0, cfg)
    after = log["t"] >= 3.0
    assert np.max(np.abs(log["x"][after, 0] - 20.0)) < 0.05, log["x"][after, 0]
    assert np.all(log["success"])


def test_warm_start_shift(cfg):
    mpc = MPC(RK4Predictor(cfg), cfg)
    ref = make_reference("speed_sin", cfg)
    x = np.array([20.0, 0, 0, 0, 0, 0])
    mpc(0.0, x, ref)
    prev = mpc.u_seq.copy()
    ws = mpc.warm_start(x)
    np.testing.assert_array_equal(ws[:-1], prev[1:])
    np.testing.assert_array_equal(ws[-1], prev[-1])
