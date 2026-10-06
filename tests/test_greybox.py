"""Grey-box model (pinc/greybox.py): the prior prediction it builds on, the training target, and the
predictor used by the MPC and E9."""
import os

import numpy as np
import pytest
import tensorflow as tf

from pinc import tyre_mf
from pinc.config import ROOT, load_config
from pinc.data import sample_trajectories
from pinc.greybox import prior_flow, to_residual
from pinc.model import PINCNet, build_model
from pinc.mpc import GreyBoxPredictor, PINCPredictor, RK4Predictor, make_predictor
from pinc.system import get_system

pytestmark = pytest.mark.skipif(not os.path.exists(tyre_mf.DEFAULT_FILE), reason="tyre data missing")


@pytest.fixture(scope="module")
def hf():
    return load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"),
                       {"model.greybox": True, "loss.lam": 0.0, "model.depth": 2, "model.width": 16})


def _zero_net(cfg):
    net = build_model(cfg)
    for v in net.out.trainable_variables:
        v.assign(tf.zeros_like(v))
    return net


def test_prior_flow_steps_each_sample_to_its_own_time(hf):
    d = sample_trajectories(6, 5, hf)
    phi = prior_flow(d["t"], d["s0"], d["u"], hf)
    sysm, dt = get_system(hf), hf.sim.dt_plant
    for i in range(len(d["t"])):
        s = tf.constant(d["s0"][i:i + 1])
        for _ in range(int(round(d["t"][i]/dt))):
            s = sysm.rk4_step_s_tf(s, tf.constant(d["u"][i:i + 1]), dt)
        np.testing.assert_allclose(phi[i], s.numpy()[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(prior_flow(np.zeros(2), d["s0"][:2], d["u"][:2], hf), d["s0"][:2])


def test_target_is_the_correction_and_its_error_equals_the_state_error(hf):
    d = sample_trajectories(50, 6, hf)
    r = to_residual(d, hf)
    phi = prior_flow(d["t"], d["s0"], d["u"], hf)
    pred_corr = r["s"] + 0.01                                   # any prediction of the target
    np.testing.assert_allclose(pred_corr + phi - d["s0"] - d["s"], 0.01, atol=1e-9)


def test_zero_correction_reproduces_the_prior_mpc_model(hf):
    net = _zero_net(hf)
    gb, prior = GreyBoxPredictor(net, hf), RK4Predictor(hf)
    assert isinstance(make_predictor(net, hf), GreyBoxPredictor)
    assert type(make_predictor(_zero_net(hf.with_overrides({"model.greybox": False})), hf)) is PINCPredictor
    d = sample_trajectories(5, 7, hf)
    s0, u = tf.constant(d["s0"]), tf.constant(d["u"])
    np.testing.assert_allclose(gb.step(s0, u).numpy(), prior.step(s0, u).numpy(), rtol=0, atol=1e-12)
    useq = tf.constant(np.repeat(d["u"][:1], 4, axis=0))
    np.testing.assert_allclose(gb.rollout(s0[0], useq).numpy(), prior.rollout(s0[0], useq).numpy(), atol=1e-12)


def test_prior_step_size_barely_matters(hf):
    """Training targets use the plant step (0.5 ms), the MPC the prediction step (1 ms)."""
    d = sample_trajectories(200, 8, hf)
    t = np.full(len(d["t"]), hf.T)
    a = prior_flow(t, d["s0"], d["u"], hf, dt=hf.sim.dt_plant)
    b = prior_flow(t, d["s0"], d["u"], hf, dt=hf.mpc.dt_pred)
    assert np.max(np.abs(a - b)/np.asarray(hf.S_x)) < 1e-4


def test_greybox_trains_saves_and_loads(hf, tmp_path, monkeypatch):
    import pinc.runinfo as ri
    from pinc.train import train
    monkeypatch.setattr(ri, "RESULTS_DIR", str(tmp_path))
    cfg = hf.with_overrides({"train.n_data": 200, "train.n_val": 100, "train.n_test": 100, "train.steps": 20,
                             "train.batch_data": 100, "train.lbfgs_iters": 5, "train.n_colloc": 64, "train.val_every": 1})
    s = train(cfg, 0, "gb_test", exp="models", verbose=False)
    net = PINCNet.load_from(s["run_dir"])
    assert s["run_dir"].startswith(str(tmp_path)) and net.mcfg.greybox and np.isfinite(s["best_val"])
    with pytest.raises(ValueError):
        train(cfg.with_overrides({"loss.lam": 0.01}), 0, "gb_bad", exp="models", verbose=False)
