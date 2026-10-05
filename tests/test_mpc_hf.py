"""Phase 6 acceptance on the high-fidelity system: MPC gradients for the prior and true predictors,
the garbage-predictor test on the HF plant, and speed tracking with the true-model NMPC.  Skipped
without the local tyre data."""
import os

import numpy as np
import pytest

from pinc import tyre_mf

pytestmark = pytest.mark.skipif(not os.path.exists(tyre_mf.DEFAULT_FILE), reason="tyre data missing")

from pinc.config import ROOT, load_config  # noqa: E402
from pinc.mpc import MPC, Predictor, RK4Predictor, make_controller  # noqa: E402
from pinc.refs import make_reference  # noqa: E402
from pinc.sim import closed_loop_metrics, simulate  # noqa: E402
from pinc.system import get_system  # noqa: E402


@pytest.fixture(scope="module")
def hcfg():
    return load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"), {"mpc.N": 4})


@pytest.mark.parametrize("model", ["prior", "true"])
def test_hf_mpc_gradient_matches_finite_differences(hcfg, model):
    mpc = MPC(RK4Predictor(hcfg, model=model), hcfg)
    ref = make_reference("lane_change", hcfg)
    sysm = get_system(hcfg)
    x = sysm.initial_state([18.0, 0.1, 0.05, 0.02, 15.0, 0.3])
    rng = np.random.default_rng(0)
    u = rng.uniform(mpc.lo, mpc.hi, (mpc.N, 2)).ravel()
    seq = mpc.reference_sequence(1.0, ref)
    J0, g, z = mpc.cost(x, u, seq)
    assert z.shape == (mpc.N, 6) and np.isfinite(J0)
    for i in rng.choice(u.size, 4, replace=False):
        up, um = u.copy(), u.copy()
        up[i] += 1e-6
        um[i] -= 1e-6
        fd = (mpc.cost(x, up, seq)[0] - mpc.cost(x, um, seq)[0])/2e-6
        assert abs(fd - g[i]) <= 1e-4*max(1.0, abs(g[i])), (i, fd, g[i])


class Garbage(Predictor):
    def rollout(self, s0, u, extra=()):
        import tensorflow as tf
        return tf.ones((u.shape[0], 10), s0.dtype)*tf.constant([12.0, 0.3, 0.1, 0.2, 0, 0, 0, 0, 0, 0], s0.dtype)


def test_hf_garbage_predictor_gives_large_error(hcfg):
    ref = make_reference("speed_sin", hcfg)
    x0 = get_system(hcfg).initial_state(ref.x0())
    log = simulate(MPC(Garbage(), hcfg), None, ref, x0, 4.0, hcfg.sim.noise_sigma, 0, hcfg)
    assert log["x"].shape == (41, 12)
    assert closed_loop_metrics(log, hcfg, ref)["rmse_vx"] > 1.0


def test_hf_true_nmpc_tracks_speed(hcfg):
    ref = make_reference("speed_sin", hcfg)
    ctrl = make_controller("nmpc_true", hcfg)
    x0 = get_system(hcfg).initial_state(ref.x0())
    log = simulate(ctrl, None, ref, x0, 3.0, np.zeros(12), 0, hcfg)
    m = closed_loop_metrics(log, hcfg, ref)
    assert m["rmse_vx"] < 0.3 and np.all(log["success"]), m["rmse_vx"]
