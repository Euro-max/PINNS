"""
E22 -- Confirmation of the prior-anchored PINC network (E21, A8).

Five seeds (E21 trained 0-2 on M0 with the physics loss; reused), N = 100 and 1000, the E17 data and budget.
New arms, all with model.arch = anchored:
  A8 data-only    lambda = 0: does the gain come from the anchor or from the physics loss?
  A8 PINC         the E15 lambda on M0 (0.01 / 0.001); on M1 lambda = 1e-3 as in E17
  A8 PINC-theta   M1 only: lambda = 1e-3 with the prior's parameters learnable (used by the physics loss and
                  by the anchor)
Compared on the same test sets with the E17 arms (data-only, PINC, PINC-theta, grey-box) and the E20 grey-box
with the quasi-steady prior.  Closed loop: E18 arms A8-*.
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


def a8_arms(variant, n):
    """{arm: (run id template, lambda, extra overrides)}"""
    lam = {100: 0.01, 1000: 0.001}[n] if variant == "m0" else 1e-3
    v = f"hf{variant}_n{n}"
    out = {"A8 data-only": (v + "_lam0_anchored_s{seed}", 0.0, {}),
           "A8 PINC": (v + f"_lam{lam:g}_anchored" + "_s{seed}", lam, {})}
    if variant == "m1":
        out["A8 PINC-theta"] = (v + f"_lam{lam:g}_anchored_theta" + "_s{seed}", lam, {"model.learn_theta": "true"})
    return out


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    _, run_dir = start("e22_anchored_confirm", a)
    cfg = load_config(cfg_path, a.overrides)
    jobs = []
    for n in SIZES:
        for arm, (tmpl, lam, extra) in a8_arms(a.variant, n).items():
            for seed in SEEDS:
                ov = dict(overrides(n, lam, seed, cfg), **{"model.arch": "anchored"}, **extra)
                jobs.append((tmpl.format(seed=seed), seed, ov))
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)

    from experiments import e9_physics_learning as e9
    e9_id = f"{os.path.basename(run_dir)}_e9test"
    e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", ",".join(f"{r}={r}" for r, _, _ in jobs), "--split", "test"])
    with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
        ev = json.load(fh)["models"]
    with open(os.path.join(RESULTS_DIR, "e17_hf_compare", f"e17_{a.variant}", "summary.json")) as fh:
        e17 = json.load(fh)["results"]
    with open(os.path.join(RESULTS_DIR, "e20_hf_greybox_qs", f"e20_{a.variant}", "summary.json")) as fh:
        e20 = json.load(fh)["results"]
    metrics = dict(one_step=("one-step body", lambda r: group_rms(summary(r)["test"]["nrmse"], GROUPS["body"])),
                   h10=("10-step body", lambda r: ev[r]["horizon_body"]["in_domain"]["10"]["mean"]),
                   h50=("50-step body", lambda r: ev[r]["horizon_body"]["in_domain"]["50"]["mean"]),
                   h50_extrap=("50-step body, outside training range",
                               lambda r: ev[r]["horizon_body"]["extrap"]["50"]["mean"]))
    rows, crows, rec = [], [], {}
    for n in SIZES:
        arms = {arm: v for arm, v in e17[str(n)].items() if " vs " not in arm}
        arms["grey-box-qs"] = e20[str(n)]["grey-box-qs"]
        for arm, (tmpl, _, _) in a8_arms(a.variant, n).items():
            arms[arm] = {m: [fn(tmpl.format(seed=s)) for s in SEEDS] for m, (_, fn) in metrics.items()}
        rec[n] = arms
        for arm, v in arms.items():
            rows.append([n, arm] + [ci_cell(v[m]) for m in metrics])
        pairs = [("A8 PINC", "PINC"), ("A8 PINC", "A8 data-only"), ("A8 PINC", "data-only"), ("A8 data-only", "data-only"),
                 ("A8 PINC", "grey-box-qs"), ("A8 PINC", "grey-box")]
        if a.variant == "m1":
            pairs += [("A8 PINC-theta", "A8 PINC"), ("A8 PINC-theta", "PINC-theta"), ("A8 PINC-theta", "data-only"),
                      ("A8 PINC-theta", "grey-box-qs")]
        for x, y in pairs:
            c = {m: compare(arms[x][m], arms[y][m]) for m in metrics}
            rec[n][f"{x} vs {y}"] = c
            crows.append([n, f"{x} against {y}"] + [cmp_cell(c[m]) for m in metrics])
    labels = [lbl for lbl, _ in metrics.values()]
    text = (f"# E22 prior-anchored PINC network, HF-{a.variant.upper()} (test errors, NRMSE; mean [95% CI] over "
            f"{len(SEEDS)} seeds; other arms from E17 and E20)\n\n" + md_table(["N", "arm"] + labels, rows) +
            "\n## Comparisons (error of the second arm divided by the error of the first, above 1: the first is better; "
            "seeds won; paired t-test on log errors)\n\n" + md_table(["N", "comparison"] + labels, crows))
    art = write_text(os.path.join(run_dir, "table_anchored.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, seeds=list(SEEDS),
                                     results={str(n): v for n, v in rec.items()}), [art])
    print(text)


if __name__ == "__main__":
    main()
