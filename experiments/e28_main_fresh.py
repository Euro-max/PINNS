"""
E28 -- Main Study 2 comparison on fresh training seeds (5-9).

The physics weights were selected on the validation set with seeds 0-2 (E15 for the plain PINC network, E23 for
the anchored network, both on the extended grid 0 ... 1), and the anchored network was chosen in an architecture
screen on seeds 0-2 (E21).  To keep selection and reporting apart, the main comparison is retrained here on
seeds 5-9, which no selection step has seen.  Arms, each with the same network, data, budget and seeds:
  data-only            lambda = 0
  PINC                 plain network, lambda selected for it (if 0 on a variant, PINC is the data-only network)
  PINC-ablation        M1 only, plain network with lambda = 1e-3 when the selected value is 0
  anchored data-only   anchored network, lambda = 0
  anchored PINC        anchored network, lambda selected for it
  grey-box             full prior;  grey-box-qs  quasi-steady prior (N = 100 and 1000)
Sizes 100, 1000 and 20 000 on M0 and M1.  Evaluation as E17 (E9 on the test set).  Writes the per-seed metrics
(the input of scripts/make_paper_study2.py) and registry.json, the run id of every arm, read by E18.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, group_rms  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides, prewarm  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402
from pinc.jobs import add_slot_args, run_jobs, summary  # noqa: E402

SEEDS = (5, 6, 7, 8, 9)
SIZES = (100, 1000, 20000)
ABLATION_LAM = 1e-3


def selected(variant):
    """{N: (lambda for the plain network, lambda for the anchored network)} from the extended-grid runs."""
    with open(os.path.join(RESULTS_DIR, "e15_hf_data", f"e15_{variant}_grid", "summary.json")) as fh:
        plain = {int(k): float(v) for k, v in json.load(fh)["best_lambda"].items()}
    with open(os.path.join(RESULTS_DIR, "e23_anchored_extend", f"e23_lambda_{variant}_grid", "summary.json")) as fh:
        best = json.load(fh)["results"]["best"]
    anch = {int(k): float(v.split()[-1]) for k, v in best.items()}
    return {n: (plain[n], anch.get(n, plain[n])) for n in plain}


def arms(variant, n, lam_plain, lam_anch):
    v = f"hf{variant}_n{n}"
    out = {"data-only": (v + "_lam0_s{seed}", 0.0, {}),
           "PINC": (v + f"_lam{lam_plain:g}" + "_s{seed}", lam_plain, {}),
           "anchored data-only": (v + "_lam0_anchored_s{seed}", 0.0, {"model.arch": "anchored"}),
           "anchored PINC": (v + f"_lam{lam_anch:g}_anchored" + "_s{seed}", lam_anch, {"model.arch": "anchored"}),
           "grey-box": (v + "_greybox_s{seed}", 0.0, {"model.greybox": "true"})}
    if n in (100, 1000):
        out["grey-box-qs"] = (v + "_greyboxqs_s{seed}", 0.0, {"model.greybox": "true", "model.greybox_prior": "qs"})
    if lam_plain == 0.0:
        out["PINC-ablation"] = (v + f"_lam{ABLATION_LAM:g}" + "_s{seed}", ABLATION_LAM, {})
    return out


def main(argv=None):
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--variant", default="m0")
    ap.add_argument("--sizes", default=",".join(str(x) for x in SIZES))
    a = ap.parse_args(argv)
    cfg_path = os.path.join(ROOT, "configs", f"hf_{a.variant}.yaml")
    a.config = cfg_path
    _, run_dir = start("e28_main_fresh", a)
    cfg = load_config(cfg_path, a.overrides)
    sel = selected(a.variant)
    sizes = [int(x) for x in a.sizes.split(",")]
    plan = {n: arms(a.variant, n, *sel[n]) for n in sizes}
    jobs = []
    for n in sizes:
        for arm, (tmpl, lam, extra) in plan[n].items():
            for seed in SEEDS:
                ov = dict(overrides(n, lam, seed, cfg), **extra)
                jobs.append((tmpl.format(seed=seed), seed, ov))
    jobs = list({j[0]: j for j in jobs}.values())                  # PINC equals data-only when lambda = 0
    prewarm(cfg, jobs)
    run_jobs(jobs, a.slots, cfg_path, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)

    from experiments import e9_physics_learning as e9
    e9_id = f"{os.path.basename(run_dir)}_e9test"
    e9.main(["--config", cfg_path, "--run-id", e9_id, "--models", ",".join(f"{r}={r}" for r, _, _ in jobs), "--split", "test"])
    with open(os.path.join(RESULTS_DIR, "e9_physics_learning", e9_id, "summary.json")) as fh:
        ev = json.load(fh)["models"]
    metrics = dict(one_step=lambda r: group_rms(summary(r)["test"]["nrmse"], GROUPS["body"]),
                   h10=lambda r: ev[r]["horizon_body"]["in_domain"]["10"]["mean"],
                   h50=lambda r: ev[r]["horizon_body"]["in_domain"]["50"]["mean"],
                   h50_extrap=lambda r: ev[r]["horizon_body"]["extrap"]["50"]["mean"])
    rec, rows = {}, []
    for n in sizes:
        rec[str(n)] = {arm: {m: [fn(t.format(seed=s)) for s in SEEDS] for m, fn in metrics.items()}
                       for arm, (t, _, _) in plan[n].items()}
        for arm, (_, lam, _) in plan[n].items():
            rows.append([n, arm, f"{lam:g}"] + [ci_cell(rec[str(n)][arm][m]) for m in metrics])
    registry = {str(n): {arm: t for arm, (t, _, _) in plan[n].items()} for n in sizes}
    with open(os.path.join(run_dir, "registry.json"), "w") as fh:
        json.dump(dict(variant=a.variant, seeds=list(SEEDS), lambdas={str(n): sel[n] for n in sizes}, arms=registry), fh, indent=1)
    text = (f"# E28 main comparison on fresh seeds {SEEDS}, HF-{a.variant.upper()} (test NRMSE of the body states; "
            "mean [95% CI] over five seeds)\n\n" + md_table(["N", "arm", "lambda", "one step", "10 steps", "50 steps",
                                                              "50 steps, outside training range"], rows))
    art = write_text(os.path.join(run_dir, "table_main.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, seeds=list(SEEDS), lambdas={str(n): sel[n] for n in sizes},
                                     registry=registry, results=rec), [art])
    print(text)


if __name__ == "__main__":
    main()
