"""
E23 -- The prior-anchored network (E21 A8) against the 8x64 MLP where E22 does not cover it.

--part study1   Study 1 (single-track model, the physics matches the plant; anchor = one Euler step of it):
                A8 with lambda* (E10) and lambda = 0 at N = 20 000 (default budget) and N = 100 (E2 budget),
                5 seeds, the E10 overrides; compared with the E10 MLP runs (E9 on both, same test sets).
--part lambda   HF-M0: the physics weight for A8 (the E15 grid 1e-4 / 1e-3 / 1e-2) at N = 100 and 1000, 3 seeds,
                selected on the validation 50-step body error as in E15 for the MLP.
--part n20k     HF-M0, N = 20 000: A8 with the E15 lambda (0.01) and lambda = 0, 5 seeds, against the E17 arms.
Timing over horizons: E4 with --pinc-model on an A8 run.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, group_rms  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides, prewarm  # noqa: E402
from experiments.e17_hf_compare import SEEDS, cmp_cell, compare  # noqa: E402
from pinc.config import DEFAULT_CONFIG_PATH, RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

A8 = {"model.arch": "anchored"}


def e9_eval(cfg_path, run_dir, rids, split="test", body=None):
    from experiments import e9_physics_learning as e9
    e9_id = f"{os.path.basename(run_dir)}_e9{split}"
    e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", ",".join(f"{r}={r}" for r in rids), "--split", split])
    with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
        return json.load(fh)["models"]


def table(rec, pairs, labels, title, keyname="N"):
    rows, crows = [], []
    for key, arms in rec.items():
        for arm, v in arms.items():
            if " vs " not in arm:
                rows.append([key, arm] + [ci_cell(v[m]) for m in labels])
        for x, y in pairs:
            if x in arms and y in arms:
                c = {m: compare(arms[x][m], arms[y][m]) for m in labels}
                arms[f"{x} vs {y}"] = c
                crows.append([key, f"{x} against {y}"] + [cmp_cell(c[m]) for m in labels])
    return (title + "\n\n" + md_table([keyname, "arm"] + list(labels.values()), rows) +
            "\n## Comparisons (error of the second arm divided by the error of the first, above 1: the first is better; "
            "seeds won; paired t-test on log errors)\n\n" + md_table([keyname, "comparison"] + list(labels.values()), crows))


def part_study1(a, run_dir):
    from experiments.e10_confirm import E2_BUDGET, pick_lambda, rid as e10_rid
    cfg = load_config(DEFAULT_CONFIG_PATH, a.overrides)
    lam, _ = pick_lambda("e8b")
    jobs, plan = [], {}
    for n in (20000, 100):
        for arm, l in (("A8 PINC", lam), ("A8 data-only", 0.0)):
            for k in SEEDS:
                ov = dict({"loss.lam": l, "model.increment_scaling": "true", "train.n_data": n,
                           "seeds.train": cfg.seeds.train + 100*k}, **A8)
                if n == 100:
                    ov.update(E2_BUDGET)
                r = f"s1_anchored_lam{l:g}_n{n}_s{k}"
                jobs.append((r, k, ov))
                plan.setdefault(n, {}).setdefault(arm, []).append(r)
        for arm, l in (("MLP PINC", lam), ("MLP data-only", 0.0)):
            plan[n][arm] = [e10_rid(l, n, k) for k in SEEDS]
    run_jobs(jobs, a.slots, DEFAULT_CONFIG_PATH, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    ev = e9_eval(DEFAULT_CONFIG_PATH, run_dir, [r for arms in plan.values() for rs in arms.values() for r in rs])
    labels = dict(one_step="one-step test NRMSE", deriv="derivative error", h10="10-step", h50="50-step",
                  h50_extrap="50-step, outside training range")
    fn = dict(one_step=lambda r: float(np.sqrt(np.mean(np.square(summary(r)["test"]["nrmse"])))),
              deriv=lambda r: ev[r]["deriv_err_all"],
              h10=lambda r: ev[r]["horizon"]["in_domain"]["10"]["mean"],
              h50=lambda r: ev[r]["horizon"]["in_domain"]["50"]["mean"],
              h50_extrap=lambda r: ev[r]["horizon"]["extrap"]["50"]["mean"])
    rec = {n: {arm: {m: [fn[m](r) for r in rs] for m in labels} for arm, rs in arms.items()} for n, arms in plan.items()}
    pairs = [("A8 PINC", "MLP PINC"), ("A8 PINC", "A8 data-only"), ("MLP PINC", "MLP data-only"), ("A8 data-only", "MLP data-only")]
    text = table(rec, pairs, labels, f"# E23 Study 1: prior-anchored network against the MLP (lambda* = {lam:g}; mean [95% CI] over 5 seeds)")
    return cfg, rec, text


def part_lambda(a, run_dir):
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    cfg = load_config(cfg_path, a.overrides)
    seeds = SEEDS[:3]
    jobs, plan = [], {}
    for n in (int(x) for x in a.sizes.split(",")):
        for lam in (float(x) for x in a.lambdas.split(",")):
            rs = [f"hf{a.variant}_n{n}_lam{lam:g}_anchored_s{k}" for k in seeds]
            plan.setdefault(n, {})[f"A8 lambda {lam:g}"] = rs
            jobs += [(r, k, dict(overrides(n, lam, k, cfg), **A8)) for r, k in zip(rs, seeds)]
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    allr = [r for arms in plan.values() for rs in arms.values() for r in rs]
    ev = {s: e9_eval(cfg_path, run_dir, allr, s) for s in ("val", "test")}
    labels = dict(h50_val="val 50-step body", one_step="one-step body", h10="10-step body", h50="50-step body")
    fn = dict(h50_val=lambda r: ev["val"][r]["horizon_body"]["in_domain"]["50"]["mean"],
              one_step=lambda r: group_rms(summary(r)["test"]["nrmse"], GROUPS["body"]),
              h10=lambda r: ev["test"][r]["horizon_body"]["in_domain"]["10"]["mean"],
              h50=lambda r: ev["test"][r]["horizon_body"]["in_domain"]["50"]["mean"])
    rec = {n: {arm: {m: [fn[m](r) for r in rs] for m in labels} for arm, rs in arms.items()} for n, arms in plan.items()}
    best = {n: min(arms, key=lambda arm: np.mean(rec[n][arm]["h50_val"])) for n, arms in plan.items()}
    text = table(rec, [], labels, f"# E23 physics weight for the prior-anchored network, HF-{a.variant.upper()} (mean [95% CI] over 3 seeds)\n\n"
                 "selected on the validation 50-step body error: " + ", ".join(f"N = {n}: {b}" for n, b in best.items()))
    return cfg, dict(rec, best=best), text


def part_n20k(a, run_dir):
    cfg_path = os.path.join(ROOT, "configs", "hf_m0.yaml")
    cfg = load_config(cfg_path, a.overrides)
    n = 20000
    jobs, plan = [], {}
    for arm, lam in (("A8 PINC", 0.01), ("A8 data-only", 0.0)):
        rs = [f"hfm0_n{n}_lam{lam:g}_anchored_s{k}" for k in SEEDS]
        plan[arm] = rs
        jobs += [(r, k, dict(overrides(n, lam, k, cfg), **A8)) for r, k in zip(rs, SEEDS)]
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    ev = e9_eval(cfg_path, run_dir, [r for rs in plan.values() for r in rs])
    labels = dict(one_step="one-step body", h10="10-step body", h50="50-step body",
                  h50_extrap="50-step body, outside training range")
    fn = dict(one_step=lambda r: group_rms(summary(r)["test"]["nrmse"], GROUPS["body"]),
              h10=lambda r: ev[r]["horizon_body"]["in_domain"]["10"]["mean"],
              h50=lambda r: ev[r]["horizon_body"]["in_domain"]["50"]["mean"],
              h50_extrap=lambda r: ev[r]["horizon_body"]["extrap"]["50"]["mean"])
    with open(os.path.join(RESULTS_DIR, "e17_hf_compare", "e17_m0", "summary.json")) as fh:
        e17 = json.load(fh)["results"][str(n)]
    arms = {arm: v for arm, v in e17.items() if " vs " not in arm}
    arms.update({arm: {m: [fn[m](r) for r in rs] for m in labels} for arm, rs in plan.items()})
    pairs = [("A8 PINC", "PINC"), ("A8 PINC", "A8 data-only"), ("A8 PINC", "data-only"), ("A8 data-only", "data-only"),
             ("A8 PINC", "grey-box")]
    text = table({n: arms}, pairs, labels, "# E23 HF-M0 at N = 20 000: prior-anchored network (mean [95% CI] over 5 seeds; other arms from E17)")
    return cfg, {n: arms}, text


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--part", required=True, choices=("study1", "lambda", "n20k"))
    ap.add_argument("--variant", default="m0", help="lambda part: HF variant")
    ap.add_argument("--sizes", default="100,1000", help="lambda part: training-set sizes")
    ap.add_argument("--lambdas", default="0.0001,0.001,0.01", help="lambda part: physics weights")
    a = ap.parse_args(argv)
    if a.part == "study1":
        a.config = DEFAULT_CONFIG_PATH
    else:
        a.config = os.path.join(ROOT, "configs", f"hf_{a.variant if a.part == 'lambda' else 'm0'}.yaml")
    _, run_dir = start("e23_anchored_extend", a)
    cfg, rec, text = dict(study1=part_study1, **{"lambda": part_lambda}, n20k=part_n20k)[a.part](a, run_dir)
    art = write_text(os.path.join(run_dir, f"table_{a.part}.md"), text)
    finish(run_dir, cfg, a.seed, dict(part=a.part, results={str(k): v for k, v in rec.items()}), [art])
    print(text)


if __name__ == "__main__":
    main()
