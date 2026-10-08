"""
E29 -- Transfer to an independent plant: the MathWorks Vehicle Dynamics Blockset 14-DOF passenger vehicle.

The Study 2 models trained on M0 (E28, seeds 5-9, no retraining) predict the motion of a vehicle we did not
implement: the Blockset's 14-DOF reference (6-DOF body, suspension at each corner, Magic Formula wheels) with
our mass, geometry, inertia, actuator lags and drive split, the same 235/45R18 tyre data, and its cornering
stiffness calibrated by the M0 rule (axle stiffness 50 kN/rad in gentle steady turning; scripts/vdbs/).
The test follows E9: initial states from random-excitation drives (here of the Blockset vehicle), ten input
sequences of 50 control periods each (the E9 sequences), and the error of chained predictions of the body
states.  References: our double-track plant (M0) and the simplified physics (full and quasi-steady prior), each
used as a predictor from the same initial states.

--part inputs1   warm-up drives -> <dir>/inputs1.mat           (then MATLAB: scripts/vdbs/vdbs_phase1.m)
--part inputs2   E9 input sequences from the drive end states   (then MATLAB: scripts/vdbs/vdbs_phase2.m)
--part eval      scores every model and reference on <dir>/truth.mat
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e1_open_loop import control_sequences  # noqa: E402
from experiments.e15_hf_data import ci_cell  # noqa: E402
from pinc import data_hf, plant_hf  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402

N_IC, N_SEQ, N_STEPS, HORIZONS = 100, 10, 50, (10, 50)
WARM_STEPS = (5, 30)                    # warm-up drive length in control periods (0.5-3 s)
DEFAULT_DIR = "/mnt/c/Users/elgondy/AppData/Local/Temp/claude_vdbs/e29"
SIZES = (100, 1000, 20000)
# truth.mat / ics.mat columns (ISO axes): vx vy r psi X Y w_fl w_fr w_rl w_rr F_act delta_act Re_fl Re_fr Re_rl Re_rr


def network_state(z):
    """Network state (..., 10) from the Blockset columns: sigma_i = Re_i w_i - vx (the effective rolling radius of
    the Blockset tyre takes the place of the plant's loaded radius)."""
    sig = z[..., 12:16]*z[..., 6:10] - z[..., :1]
    return np.concatenate([z[..., 0:4], z[..., 10:12], sig], axis=-1)


def part_inputs1(a, cfg):
    import scipy.io as sio
    p = plant_hf.make_params(cfg.params, "M0")
    rng = np.random.default_rng(cfg.seeds.test + 2902)
    v0 = rng.uniform(cfg.box_train.vx[0], cfg.box_train.vx[1], N_IC)
    F0 = np.array([plant_hf.free_rolling_state(v, p)[10] for v in v0])
    tw = rng.integers(WARM_STEPS[0], WARM_STEPS[1], N_IC)
    cmd = data_hf._commands(rng, N_IC, WARM_STEPS[1], cfg.T, F0, np.asarray(cfg.u_min), np.asarray(cfg.u_max), 0.3, v0,
                            p["lf"] + p["lr"], 9.0)
    os.makedirs(a.dir, exist_ok=True)
    sio.savemat(os.path.join(a.dir, "inputs1.mat"), dict(v0=v0, F0=F0, tw=tw.astype(float), cmd=cmd))
    return dict(n_ic=N_IC, warm_steps=list(WARM_STEPS)), f"wrote {a.dir}/inputs1.mat ({N_IC} warm-up drives)\n"


def part_inputs2(a, cfg):
    import scipy.io as sio
    ic = sio.loadmat(os.path.join(a.dir, "ics.mat"))["IC"]
    ok = np.all(np.isfinite(ic), axis=1)
    vx0 = np.where(ok, ic[:, 0], 15.0)
    u = control_sequences(N_SEQ, N_STEPS, cfg, np.random.default_rng(4242), vx0)       # the E9 sequences
    sio.savemat(os.path.join(a.dir, "seqs.mat"), dict(u=u))
    return dict(n_ok=int(ok.sum())), f"wrote {a.dir}/seqs.mat: {N_SEQ} sequences for {int(ok.sum())} of {N_IC} initial states\n"


def horizon_errors(pred_body, truth_body, S_x):
    """Per initial state: RMS over sequences and the four body states of the scaled error at each horizon."""
    e2 = ((pred_body - truth_body)/S_x[:4])**2                      # (n_ic, n_seq, n_steps, 4)
    return {h: np.sqrt(np.mean(e2[:, :, h - 1, :], axis=(1, 2))) for h in HORIZONS}


def part_eval(a, cfg):
    import scipy.io as sio
    import tensorflow as tf
    from pinc.model import PINCNet
    from pinc.mpc import RK4Predictor, make_predictor
    ic = sio.loadmat(os.path.join(a.dir, "ics.mat"))["IC"]
    X = sio.loadmat(os.path.join(a.dir, "truth.mat"))["X"]
    u = sio.loadmat(os.path.join(a.dir, "seqs.mat"))["u"]
    keep = np.all(np.isfinite(ic), axis=1) & np.all(np.isfinite(X), axis=(1, 2, 3))
    ic, X, u = ic[keep], X[keep], u[keep]
    n = len(ic)
    s0 = network_state(ic)
    s0r = np.repeat(s0, N_SEQ, axis=0)
    ur = u.reshape(-1, N_STEPS, 2)
    truth = X[..., :4]
    S_x = np.asarray(cfg.S_x)

    def score(rollout):
        p = np.asarray(rollout(tf.constant(s0r), tf.constant(ur))).reshape(n, N_SEQ, N_STEPS, -1)[..., :4]
        return horizon_errors(p, truth, S_x)

    rows, rec = [], {"references": {}, "models": {}}
    for name, model in (("our plant (M0)", "true"), ("full prior", "prior"), ("quasi-steady prior", "qs")):
        e = score(RK4Predictor(cfg, model=model).rollout_batch)
        rec["references"][name] = {str(h): bootstrap_ci(e[h].tolist()) for h in HORIZONS}
        rows.append([name, "-", ci_cell(e[10].tolist()), ci_cell(e[50].tolist()), "-", "-"])

    own = json.load(open(os.path.join(RESULTS_DIR, "e28_main_fresh", "e28_m0", "summary.json")))
    reg = own["registry"]
    for n_tr in SIZES:
        for arm, tmpl in reg[str(n_tr)].items():
            per_seed = {10: [], 50: []}
            for seed in own["seeds"]:
                rid = tmpl.format(seed=seed)
                net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", rid))
                e = score(make_predictor(net, cfg).rollout_batch)
                for h in HORIZONS:
                    per_seed[h].append(float(np.mean(e[h])))
                tf.keras.backend.clear_session()
            own10, own50 = own["results"][str(n_tr)][arm]["h10"], own["results"][str(n_tr)][arm]["h50"]
            gm = lambda v: float(np.exp(np.mean(np.log(v))))
            rec["models"].setdefault(str(n_tr), {})[arm] = dict(h10=per_seed[10], h50=per_seed[50], own_h10=own10, own_h50=own50,
                                                               ratio_h10=gm(per_seed[10])/gm(own10), ratio_h50=gm(per_seed[50])/gm(own50))
            rows.append([f"{arm}, N = {n_tr}", len(own["seeds"]), ci_cell(per_seed[10]), ci_cell(per_seed[50]),
                         f"{gm(per_seed[10])/gm(own10):.2f}", f"{gm(per_seed[50])/gm(own50):.2f}"])
            print(rows[-1], flush=True)
    rec["n_ic"] = n
    text = (f"# E29 transfer to the Blockset 14-DOF vehicle (M0 models of E28, seeds 5-9; {n} initial states x {N_SEQ} "
            "sequences; NRMSE of the body states)\n\nReferences are single predictors (mean [95% CI] over initial states); "
            "models: mean [95% CI] over five training seeds.  Ratio: geometric mean on the Blockset vehicle over the "
            "geometric mean on our plant's test set (E28).\n\n" +
            md_table(["predictor", "seeds", "10 steps", "50 steps", "ratio 10", "ratio 50"], rows))
    return rec, text


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--part", required=True, choices=("inputs1", "inputs2", "eval"))
    ap.add_argument("--dir", default=DEFAULT_DIR, help="exchange folder shared with MATLAB")
    a = ap.parse_args(argv)
    a.config = os.path.join(ROOT, "configs", "hf_m0.yaml")
    _, run_dir = start("e29_vdbs_transfer", a)
    cfg = load_config(a.config, a.overrides)
    out, text = dict(inputs1=part_inputs1, inputs2=part_inputs2, eval=part_eval)[a.part](a, cfg)
    art = write_text(os.path.join(run_dir, f"table_{a.part}.md"), text)
    finish(run_dir, cfg, a.seed, dict(part=a.part, **out), [art])
    print(text)


if __name__ == "__main__":
    main()
