"""
E25 -- Accuracy of the two priors against the true plant over one control period.

On the held-out test trajectories of each high-fidelity variant (random start state and constant input, the
same samples E17 tests on), the full prior (prior_hf.f_s, RK4 with the plant step) and the quasi-steady prior
(prior_hf.f_s_qs with quasi-steady wheel slip, RK4 with 10 ms steps) are integrated to the sample time and
compared with the true state.  Reported: NRMSE by state group, at the sample times and at t = T only.  This
is the check behind the grey-box model with the quasi-steady prior (E20) and the prior-anchored network.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS  # noqa: E402
from pinc.config import ROOT, load_config  # noqa: E402
from pinc.data import sample_trajectories  # noqa: E402
from pinc.greybox import prior_flow, prior_flow_qs  # noqa: E402


def group_nrmse(pred, truth, S_x):
    e = (pred - truth)/np.asarray(S_x)
    return {g: float(np.sqrt(np.mean(e[:, i]**2))) for g, i in GROUPS.items()}


def main(argv=None):
    ap = base_parser(__doc__)
    a = ap.parse_args(argv)
    a.config = os.path.join(ROOT, "configs", "hf_m0.yaml")
    cfg0, run_dir = start("e25_prior_accuracy", a)
    rows, rec = [], {}
    for variant in ("m0", "m1"):
        cfg = load_config(os.path.join(ROOT, "configs", f"hf_{variant}.yaml"))
        d = sample_trajectories(cfg.train.n_test if not a.quick else 300, cfg.seeds.test, cfg)
        full = d["t"] >= cfg.T - 1e-12
        rec[variant] = {}
        for name, fn in (("full", prior_flow), ("quasi-steady", prior_flow_qs)):
            p = fn(d["t"], d["s0"], d["u"], cfg)
            rec[variant][name] = dict(all_t=group_nrmse(p, d["s"], cfg.S_x),
                                      t_T=group_nrmse(p[full], d["s"][full], cfg.S_x) if np.any(full) else None,
                                      n=int(len(d["t"])), n_T=int(np.sum(full)))
            r = rec[variant][name]["all_t"]
            rows.append([variant.upper(), name, f"{r['body']:.4f}", f"{r['actuators']:.4f}", f"{r['wheels']:.4f}"])
    text = ("# E25 prior against the true plant (NRMSE by state group, test trajectories, times uniform in (0, T])\n\n" +
            md_table(["variant", "prior", "body", "actuators", "wheels"], rows))
    art = write_text(os.path.join(run_dir, "table_prior_accuracy.md"), text)
    finish(run_dir, cfg0, a.seed, dict(results=rec), [art], dict(quick=a.quick))
    print(text)


if __name__ == "__main__":
    main()
