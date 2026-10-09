"""
E32 -- Confirmation of the per-state output scaling D under the final protocol (Study 1, exact physics).

The evidence for D so far is single-seed and from older code (E6, E9: seed 0, 500 L-BFGS iterations). E32 repeats
the comparison with the E27 protocol. Design fixed before training (no changes after the first result except
documented bug fixes):

  networks   plain PINC network WITHOUT D (model.increment_scaling false), everything else as E27
  lambda     0 and 0.1 (the value E27 selected for the plain network with D; no new selection)
  sizes      N = 100 and 20 000
  seeds      5-9
  budget     as E27: 6000 Adam steps (cosine decay), 2000 L-BFGS iterations, float64
  arms with D: E27 main (data-only and PINC with lambda 0.1, seeds 5-9), not retrained
  evaluation E9 on the Study 1 test set: one-step, derivative, 10-step, 50-step and 50-step outside the training range

Pre-specified comparisons at each N on the 50-step error (geometric-mean ratio over seeds, seeds better, paired
t-test on log errors); the other metrics are reported the same way:
  1. without D: lambda 0.1 against lambda 0       (effect of the physics loss without the scaling)
  2. with D:    lambda 0.1 against lambda 0       (from E27)
  3. lambda 0.1: with D against without D

--part train   trains the 20 models (resumes: finished runs are reused) and evaluates them with E9
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e15_hf_data import ci_cell  # noqa: E402
from experiments.e23_anchored_extend import e9_eval  # noqa: E402
from experiments.e27_study1_rerun import overrides  # noqa: E402
from pinc.config import DEFAULT_CONFIG_PATH, RESULTS_DIR, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

LAMBDAS = (0.0, 0.1)
SIZES = (100, 20000)
SEEDS = (5, 6, 7, 8, 9)
METRICS = ("one_step", "deriv", "h10", "h50", "h50_extrap")
E27_ARM = {0.0: "data-only", 0.1: "PINC"}                       # the same lambda with D, in E27 main


def rid(n, lam, seed):
    return f"s1v3_noD_n{n}_lam{lam:g}_s{seed}"


def paired(x, y):
    """Geometric-mean ratio y/x over seeds (above 1: x better), seeds in which x is better, paired t-test on logs."""
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    d = ly - lx
    p = float(stats.ttest_rel(ly, lx).pvalue) if np.std(d) > 0 else float("nan")
    return dict(ratio=float(np.exp(np.mean(d))), wins=int(np.sum(d > 0)), n=len(d), p=p)


def part_train(a, run_dir, cfg):
    jobs = [(rid(n, lam, k), k, dict(overrides(n, lam, k, cfg), **{"model.increment_scaling": "false"}))
            for n in SIZES for lam in LAMBDAS for k in SEEDS]
    run_jobs(jobs, a.slots, DEFAULT_CONFIG_PATH, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    ev = e9_eval(DEFAULT_CONFIG_PATH, run_dir, [j[0] for j in jobs], "test")
    fn = dict(one_step=lambda r: float(np.sqrt(np.mean(np.square(summary(r)["test"]["nrmse"])))),
              deriv=lambda r: ev[r]["deriv_err_all"],
              h10=lambda r: ev[r]["horizon"]["in_domain"]["10"]["mean"],
              h50=lambda r: ev[r]["horizon"]["in_domain"]["50"]["mean"],
              h50_extrap=lambda r: ev[r]["horizon"]["extrap"]["50"]["mean"])
    with open(os.path.join(RESULTS_DIR, "e27_study1_rerun", "e27_main", "summary.json")) as fh:
        e27 = json.load(fh)
    assert all(e27["lambdas"]["plain"][str(n)] == 0.1 for n in SIZES)
    res, cmp_, rows, crow = {}, {}, [], []
    for n in SIZES:
        res[str(n)] = {}
        for lam in LAMBDAS:
            res[str(n)][f"noD_lam{lam:g}"] = {m: [fn[m](rid(n, lam, k)) for k in SEEDS] for m in METRICS}
            res[str(n)][f"D_lam{lam:g}"] = {m: e27["results"][str(n)][E27_ARM[lam]][m] for m in METRICS}
        r = res[str(n)]
        cmp_[str(n)] = {
            "noD: lam 0.1 vs lam 0": {m: paired(r["noD_lam0.1"][m], r["noD_lam0"][m]) for m in METRICS},
            "D: lam 0.1 vs lam 0": {m: paired(r["D_lam0.1"][m], r["D_lam0"][m]) for m in METRICS},
            "lam 0.1: D vs noD": {m: paired(r["D_lam0.1"][m], r["noD_lam0.1"][m]) for m in METRICS},
        }
        rows += [[n, arm] + [ci_cell(v[m]) for m in METRICS] for arm, v in r.items()]
        crow += [[n, k] + [f"{c[m]['ratio']:.3g} ({c[m]['wins']}/{c[m]['n']}, p={c[m]['p']:.2g})" for m in METRICS]
                 for k, c in cmp_[str(n)].items()]
    text = ("# E32 output scaling D, Study 1 (test NRMSE, all states; mean [95% CI] over seeds 5-9)\n\n"
            "noD: model.increment_scaling false (this experiment); D: E27 main\n\n" +
            md_table(["N", "arm", "one step", "derivative", "10 steps", "50 steps", "50 steps, outside"], rows) +
            "\n\n## Comparisons: ratio = error of the second / error of the first (above 1: the first is better), "
            "seeds better, paired t-test on logs\n\n" +
            md_table(["N", "comparison", "one step", "derivative", "10 steps", "50 steps", "50 steps, outside"], crow))
    return dict(design=dict(lambdas=LAMBDAS, sizes=SIZES, seeds=SEEDS), results=res, comparisons=cmp_), text


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--part", required=True, choices=("train",))
    a = ap.parse_args(argv)
    a.config = DEFAULT_CONFIG_PATH
    _, run_dir = start("e32_scaling_confirm", a)
    cfg = load_config(DEFAULT_CONFIG_PATH, a.overrides)
    out, text = part_train(a, run_dir, cfg)
    art = write_text(os.path.join(run_dir, f"table_{a.part}.md"), text)
    finish(run_dir, cfg, a.seed, dict(part=a.part, **out), [art])
    print(text)


if __name__ == "__main__":
    main()
