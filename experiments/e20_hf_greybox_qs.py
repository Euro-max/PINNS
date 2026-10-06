"""
E20 -- A faster grey-box model: the prior with quasi-steady wheel speeds.

The grey-box model of E17 runs the full prior, whose wheel-slip states need 1 ms RK4 steps, inside every MPC
prediction step (about 0.4 s per solve at N = 10).  A vehicle MPC would normally use a non-stiff model.  Here
the prior's wheel speeds are set to their quasi-steady values (prior_hf.f_s_qs / slip_qs; the body dynamics
are otherwise the same and as accurate against the true plant), integrated with 10 ms RK4 steps, and the
network learns the correction as before (model.greybox_prior = qs).  Same data, network, budget and seeds as
E17; compared with the E17 arms on the same test sets.  Solve times: E4 with --greybox-model; closed loop: E18.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, group_rms  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides, prewarm  # noqa: E402
from experiments.e17_hf_compare import SEEDS, cmp_cell, compare  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

SIZES = (100, 1000)


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    _, run_dir = start("e20_hf_greybox_qs", a)
    cfg = load_config(cfg_path, a.overrides)
    jobs = []
    for n in SIZES:
        for seed in SEEDS:
            ov = dict(overrides(n, 0.0, seed, cfg), **{"model.greybox": "true", "model.greybox_prior": "qs"})
            jobs.append((f"hf{a.variant}_n{n}_greyboxqs_s{seed}", seed, ov))
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)

    from experiments import e9_physics_learning as e9
    e9_id = f"{os.path.basename(run_dir)}_e9test"
    e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", ",".join(f"{r}={r}" for r, _, _ in jobs), "--split", "test"])
    with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
        ev = json.load(fh)["models"]
    with open(os.path.join(RESULTS_DIR, "e17_hf_compare", f"e17_{a.variant}", "summary.json")) as fh:
        e17 = json.load(fh)["results"]
    metrics = dict(one_step=("one-step body", lambda rid: group_rms(summary(rid)["test"]["nrmse"], GROUPS["body"])),
                   h10=("10-step body", lambda rid: ev[rid]["horizon_body"]["in_domain"]["10"]["mean"]),
                   h50=("50-step body", lambda rid: ev[rid]["horizon_body"]["in_domain"]["50"]["mean"]),
                   h50_extrap=("50-step body, outside training range",
                               lambda rid: ev[rid]["horizon_body"]["extrap"]["50"]["mean"]))
    rows, crows, rec = [], [], {}
    for n in SIZES:
        arms = {arm: v for arm, v in e17[str(n)].items() if " vs " not in arm}
        arms["grey-box-qs"] = {m: [fn(f"hf{a.variant}_n{n}_greyboxqs_s{s}") for s in SEEDS] for m, (_, fn) in metrics.items()}
        rec[n] = arms
        for arm, v in arms.items():
            rows.append([n, arm] + [ci_cell(v[m]) for m in metrics])
        for other in [x for x in arms if x != "grey-box-qs"]:
            c = {m: compare(arms["grey-box-qs"][m], arms[other][m]) for m in metrics}
            crows.append([n, f"grey-box-qs against {other}"] + [cmp_cell(c[m]) for m in metrics])
    labels = [lbl for lbl, _ in metrics.values()]
    text = (f"# E20 grey-box model with the quasi-steady prior, HF-{a.variant.upper()} (test errors, NRMSE; mean [95% CI] over "
            f"{len(SEEDS)} seeds; other arms from E17)\n\n" + md_table(["N", "arm"] + labels, rows) +
            "\n## Comparisons (error of the second arm divided by the error of grey-box-qs, above 1: grey-box-qs "
            "is better; seeds won; paired t-test on log errors)\n\n" + md_table(["N", "comparison"] + labels, crows))
    art = write_text(os.path.join(run_dir, "table_hf_greybox_qs.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, results={str(n): v for n, v in rec.items()}), [art])
    print(text)


if __name__ == "__main__":
    main()
