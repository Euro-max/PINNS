"""
E31 -- Why do the learned controllers track the speed sinusoid worse than NMPC with the quasi-steady prior?

Error of each prediction model over the MPC horizon, per body state, on the E9 in-domain test set (100 initial
states x 10 input sequences, the true plant of the variant): RMS of the scaled error at 1 and 10 control periods
for v_x, v_y, r and psi, and the mean signed v_x error at 10 periods (a bias in the predicted speed shifts the
speed the MPC settles on).  Models: the E28 arms used in the closed loop (N = 100 and 1000, seeds 5-9) and the
quasi-steady and full prior (the NMPC prediction models).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e1_open_loop import control_sequences, truth_rollout  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.data import sample_box  # noqa: E402

N_IC, N_SEQ, N_STEPS, HORIZONS, SIZES = 100, 10, 10, (1, 10), (100, 1000)
STATES = ("vx", "vy", "r", "psi")


def main(argv=None):
    import tensorflow as tf
    from pinc.model import PINCNet
    from pinc.mpc import RK4Predictor, make_predictor
    ap = base_parser(__doc__)
    ap.add_argument("--variant", default="m0")
    a = ap.parse_args(argv)
    a.config = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    _, run_dir = start("e31_speed_prediction", a)
    cfg = load_config(a.config, a.overrides)
    S_x = np.asarray(cfg.S_x)
    s0 = sample_box(N_IC, cfg.box_train, np.random.default_rng(cfg.seeds.test + 902), cfg)      # as E9 (test split)
    u = control_sequences(N_SEQ, 50, cfg, np.random.default_rng(4242), s0[:, 0])[:, :, :N_STEPS].reshape(-1, N_STEPS, 2)
    s0r = np.repeat(s0, N_SEQ, axis=0)
    truth, _ = truth_rollout(s0r, u, cfg)

    def errors(rollout):
        p = np.asarray(rollout(tf.constant(s0r), tf.constant(u)))
        d = (p[..., :4] - truth[..., :4])/S_x[:4]
        out = {f"{s}_{h}": float(np.sqrt(np.mean(d[:, h - 1, i]**2))) for h in HORIZONS for i, s in enumerate(STATES)}
        out["vx_bias_10"] = float(np.mean(p[:, 9, 0] - truth[:, 9, 0]))                      # m/s
        return out

    rec, rows = {"references": {}, "models": {}}, []
    cols = [f"{s}_{h}" for h in HORIZONS for s in STATES] + ["vx_bias_10"]
    for name, model in (("quasi-steady prior", "qs"), ("full prior", "prior")):
        e = errors(RK4Predictor(cfg, model=model).rollout_batch)
        rec["references"][name] = e
        rows.append([name] + [f"{e[c]:.3g}" for c in cols])
    own = json.load(open(os.path.join(RESULTS_DIR, "e28_main_fresh", f"e28_{a.variant}", "summary.json")))
    for n in SIZES:
        for arm, tmpl in own["registry"][str(n)].items():
            per = []
            for seed in own["seeds"]:
                net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", tmpl.format(seed=seed)))
                per.append(errors(make_predictor(net, cfg).rollout_batch))
                tf.keras.backend.clear_session()
            mean = {c: float(np.exp(np.mean(np.log([p[c] for p in per])))) if c != "vx_bias_10" else float(np.mean([p[c] for p in per]))
                    for c in cols}
            rec["models"].setdefault(str(n), {})[arm] = dict(per_seed=per, mean=mean)
            rows.append([f"{arm}, N = {n}"] + [f"{mean[c]:.3g}" for c in cols])
            print(rows[-1], flush=True)
    text = (f"# E31 prediction error over the MPC horizon, HF-{a.variant.upper()} (E9 in-domain test set; scaled RMS error "
            "per body state; models: geometric mean over seeds 5-9; vx bias: mean signed v_x error at 10 steps in m/s, "
            "arithmetic mean over seeds)\n\n" + md_table(["predictor"] + cols, rows))
    art = write_text(os.path.join(run_dir, "table_speed_prediction.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, **rec), [art])
    print(text)


if __name__ == "__main__":
    main()
