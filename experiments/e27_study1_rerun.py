"""
E27 -- Study 1 (single-track model, exact physics) rerun with the Study 2 protocol.

Changes against the original Study 1 (E2, E8, E10): 2000 L-BFGS iterations instead of 500 (E11 showed the loss
still falling at 500); lambda selected on the validation 50-step error, the criterion of Study 2, for the plain
and the anchored network separately; selection on seeds 0-2 and reporting on fresh seeds 5-9.

--part lambda   lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1} for the plain and the anchored network at N = 100 and
                20 000, seeds 0-2; selection on the validation 50-step error (E9, all four states).
--part main     data-only, PINC, anchored data-only and anchored PINC at N = 100, 1000, 10 000 and 20 000, seeds
                5-9, with the selected lambda; test errors (E9) and registry.json for the closed loop (E18 --variant st).
Budget for every model: 6000 Adam steps (cosine decay) and 2000 L-BFGS iterations.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e15_hf_data import ci_cell  # noqa: E402
from experiments.e23_anchored_extend import e9_eval  # noqa: E402
from pinc.config import DEFAULT_CONFIG_PATH, RESULTS_DIR, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SEL_SIZES = (100, 20000)
MAIN_SIZES = (100, 1000, 10000, 20000)
SEL_SEEDS = (0, 1, 2)
MAIN_SEEDS = (5, 6, 7, 8, 9)
ARCH = {"plain": {}, "anchored": {"model.arch": "anchored"}}


def overrides(n, lam, seed, cfg):
    spe = int(np.ceil(n/min(1024, n)))
    return {"train.n_data": n, "train.steps": 6000, "train.lbfgs_iters": 2000, "train.batch_data": min(1024, n),
            "train.val_every": max(1, 20//spe), "loss.lam": lam, "seeds.train": cfg.seeds.train + 100*seed,
            "model.increment_scaling": "true", "train.log_every": 1000}


def rid(arch, n, lam, seed):
    return f"s1v3_{arch}_n{n}_lam{lam:g}_s{seed}"


def part_lambda(a, run_dir, cfg):
    jobs = [(rid(arch, n, lam, k), k, dict(overrides(n, lam, k, cfg), **ov))
            for arch, ov in ARCH.items() for n in SEL_SIZES for lam in LAMBDAS for k in SEL_SEEDS]
    run_jobs(jobs, a.slots, DEFAULT_CONFIG_PATH, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    ev = e9_eval(DEFAULT_CONFIG_PATH, run_dir, [j[0] for j in jobs], "val")
    rec, rows, best = {}, [], {}
    for arch in ARCH:
        for n in SEL_SIZES:
            vals = {lam: [ev[rid(arch, n, lam, k)]["horizon"]["in_domain"]["50"]["mean"] for k in SEL_SEEDS] for lam in LAMBDAS}
            rec[f"{arch}_{n}"] = {f"{lam:g}": v for lam, v in vals.items()}
            best[f"{arch}_{n}"] = min(LAMBDAS, key=lambda lam: np.mean(vals[lam]))
            rows += [[arch, n, f"{lam:g}", ci_cell(v)] for lam, v in vals.items()]
    text = ("# E27 lambda selection, Study 1 (validation 50-step NRMSE, all states; mean [95% CI] over seeds 0-2)\n\n"
            "selected: " + ", ".join(f"{k}: {v:g}" for k, v in best.items()) + "\n\n" +
            md_table(["network", "N", "lambda", "val 50-step"], rows))
    return dict(results=rec, best=best), text


def part_main(a, run_dir, cfg):
    with open(os.path.join(RESULTS_DIR, "e27_study1_rerun", "e27_lambda", "summary.json")) as fh:
        best = json.load(fh)["best"]
    lam = {arch: {n: best[f"{arch}_{min(SEL_SIZES, key=lambda s: abs(np.log(s/n)))}"] for n in MAIN_SIZES} for arch in ARCH}
    plan = {}
    for n in MAIN_SIZES:
        plan[n] = {"data-only": ("plain", 0.0), "PINC": ("plain", lam["plain"][n]),
                   "anchored data-only": ("anchored", 0.0), "anchored PINC": ("anchored", lam["anchored"][n])}
    jobs = {rid(arch, n, l, k): (rid(arch, n, l, k), k, dict(overrides(n, l, k, cfg), **ARCH[arch]))
            for n in MAIN_SIZES for arch, l in plan[n].values() for k in MAIN_SEEDS}
    run_jobs(list(jobs.values()), a.slots, DEFAULT_CONFIG_PATH, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    ev = e9_eval(DEFAULT_CONFIG_PATH, run_dir, list(jobs), "test")
    fn = dict(one_step=lambda r: float(np.sqrt(np.mean(np.square(summary(r)["test"]["nrmse"])))),
              deriv=lambda r: ev[r]["deriv_err_all"],
              h10=lambda r: ev[r]["horizon"]["in_domain"]["10"]["mean"],
              h50=lambda r: ev[r]["horizon"]["in_domain"]["50"]["mean"],
              h50_extrap=lambda r: ev[r]["horizon"]["extrap"]["50"]["mean"])
    rec, rows = {}, []
    for n in MAIN_SIZES:
        rec[str(n)] = {arm: {m: [f(rid(arch, n, l, k)) for k in MAIN_SEEDS] for m, f in fn.items()} for arm, (arch, l) in plan[n].items()}
        rows += [[n, arm, f"{l:g}"] + [ci_cell(rec[str(n)][arm][m]) for m in fn] for arm, (arch, l) in plan[n].items()]
    registry = {str(n): {arm: f"s1v3_{arch}_n{n}_lam{l:g}" + "_s{seed}" for arm, (arch, l) in plan[n].items()} for n in MAIN_SIZES}
    with open(os.path.join(run_dir, "registry.json"), "w") as fh:
        json.dump(dict(variant="st", seeds=list(MAIN_SEEDS), lambdas=lam, arms=registry), fh, indent=1)
    text = ("# E27 Study 1 main comparison on seeds 5-9 (test NRMSE, all states; mean [95% CI])\n\n" +
            md_table(["N", "arm", "lambda", "one step", "derivative", "10 steps", "50 steps", "50 steps, outside"], rows))
    return dict(lambdas={k: {str(n): v for n, v in d.items()} for k, d in lam.items()}, registry=registry, results=rec), text


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--part", required=True, choices=("lambda", "main"))
    a = ap.parse_args(argv)
    a.config = DEFAULT_CONFIG_PATH
    _, run_dir = start("e27_study1_rerun", a)
    cfg = load_config(DEFAULT_CONFIG_PATH, a.overrides)
    out, text = (part_lambda if a.part == "lambda" else part_main)(a, run_dir, cfg)
    art = write_text(os.path.join(run_dir, f"table_{a.part}.md"), text)
    finish(run_dir, cfg, a.seed, dict(part=a.part, **out), [art])
    print(text)


if __name__ == "__main__":
    main()
