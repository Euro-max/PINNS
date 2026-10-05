"""Phase 6 acceptance -- the most important test in the repo (guards D7, D18)."""
import os
import re

import numpy as np

from pinc.config import ROOT
from pinc.mpc import MPC, Predictor
from pinc.refs import make_reference
from pinc.sim import simulate, closed_loop_metrics
from pinc.plant import trim_force


class GarbagePredictor(Predictor):
    name = "garbage"

    def rollout(self, s0, u, extra=()):
        import tensorflow as tf
        return tf.ones((u.shape[0], 4), s0.dtype)*tf.constant([12.0, 0.3, 0.1, 0.2], s0.dtype)


def test_zero_controller_gives_large_error(cfg):
    zero = lambda t, x, ref: (np.zeros(2), {})
    ref = make_reference("speed_sin", cfg)
    log = simulate(zero, cfg.params, ref, ref.x0(), 10.0, cfg.sim.noise_sigma, 0, cfg)
    m = closed_loop_metrics(log, cfg, ref)
    assert m["rmse_vx"] > 1.0, m["rmse_vx"]                   # the car coasts down; error must be visible
    assert log["x"][-1, 0] < 19.0


def test_garbage_predictor_gives_large_error(cfg):
    ref = make_reference("speed_sin", cfg)
    mpc = MPC(GarbagePredictor(), cfg)
    log = simulate(mpc, cfg.params, ref, ref.x0(), 6.0, cfg.sim.noise_sigma, 0, cfg)
    m = closed_loop_metrics(log, cfg, ref)
    assert m["rmse_vx"] > 1.0, m["rmse_vx"]


def test_error_scored_after_step(cfg):
    ref = make_reference("speed_step", cfg)
    hold = lambda t, x, ref: (np.array([trim_force(20.0, cfg.params), 0.0]), {})
    log = simulate(hold, cfg.params, ref, ref.x0(), 3.0, np.zeros(6), 0, cfg)
    k = int(round(cfg.refs.step_time/cfg.T))
    assert log["ref"][k, 0] == cfg.refs.v0 + cfg.refs.step_dv
    assert abs(log["err"][k, 0] - (log["x"][k, 0] - log["ref"][k, 0])) < 1e-12
    assert abs(log["err"][k, 0] + cfg.refs.step_dv) < 0.2      # error appears exactly when the reference steps


def test_plant_params_only_reach_plant(cfg):
    from pinc.plant import perturbed
    from pinc.mpc import make_controller
    c = make_controller("nmpc_rk4", cfg)
    ref = make_reference("speed_sin", cfg)
    heavy = perturbed(cfg.params, m=1800.0)
    log = simulate(c, heavy, ref, ref.x0(), 1.0, np.zeros(6), 0, cfg)
    assert c.pred.params["m"] == 1500.0 and np.all(np.isfinite(log["x"]))


FORBIDDEN = [
    (r"^\s*(x|state|states|plant_state|plant_states|x_true|x_plant)\s*(\[[^\]]*\])?\s*(\+|-|\*)?=\s*.*\bref", "state assigned from reference"),
    (r"tf\.where\(\s*tf\.math\.is_finite", "NaN masking (ground rule 7)"),
    (r"np\.where\(\s*np\.isfinite", "NaN masking (ground rule 7)"),
    (r"\bfrom\s+legacy\b|\bimport\s+legacy\b", "import from legacy"),
    (r"/content/", "hard-coded Colab path"),
]


def test_no_module_writes_reference_into_state():
    bad = []
    for sub in ("pinc", "experiments"):
        d = os.path.join(ROOT, sub)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(d, fn)) as fh:
                for i, line in enumerate(fh, 1):
                    for pat, why in FORBIDDEN:
                        if re.search(pat, line):
                            bad.append(f"{sub}/{fn}:{i}: {why}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_sim_state_only_changes_via_plant():
    """In sim.py the only assignment to the plant state array is the plant integrator call."""
    with open(os.path.join(ROOT, "pinc", "sim.py")) as fh:
        src = fh.read()
    assigns = re.findall(r"^\s*x\[[^\]]*\]\s*=\s*(.*)$", src, flags=re.M)
    assert len(assigns) == 2, assigns                          # x[0] = x0 and x[k+1] = plant...
    assert any("sysm.plant_simulate" in a for a in assigns)
    assert all("ref" not in a for a in assigns)
