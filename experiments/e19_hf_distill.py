"""
E19 (exploratory) -- A network taught by the grey-box model.

E17 showed the grey-box model (prior + learned correction) is the most accurate model at every N, but the
MPC pays for the prior's RK4 at every prediction step, while PINC needs one network call.  Here a plain
network (lambda = 0, same size, budget and seeds) is trained on the N real trajectories plus 20 000 samples
labelled by the grey-box model trained on the same N trajectories and seed (pinc/greybox.py,
teacher_samples; the unlabelled states come from the same pool as the physics loss's collocation points).
No information beyond the N trajectories and the prior is used.  Batch 1024 for every N, since the labelled
set is large.  Compared with the E17 arms on the same test sets.

The teacher labels only the body and actuator states; the wheel states come from the real data alone.  A first
run with all ten states labelled (run ids *_distill_s*, kept) failed: the grey-box model's wheel-state
predictions are poor (all-state error 0.076 against 0.0055 for the body states at N = 100), the student learned
them, its validation loss rose from epoch 39 on, and model selection kept that early, undertrained checkpoint.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e14_hf_lambda import GROUPS, NO_WHEEL, group_rms  # noqa: E402
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
    _, run_dir = start("e19_hf_distill", a)
    cfg = load_config(cfg_path, a.overrides)
    jobs = []
    for n in SIZES:
        for seed in SEEDS:
            ov = dict(overrides(n, 0.0, seed, cfg), **{"train.distill_from": f"hf{a.variant}_n{n}_greybox_s{seed}",
                                                        "train.batch_data": 1024,
                                                        "train.distill_mask": str(NO_WHEEL).replace(" ", "")})
            jobs.append((f"hf{a.variant}_n{n}_distill2_s{seed}", seed, ov))
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
        arms["distilled"] = {m: [fn(f"hf{a.variant}_n{n}_distill2_s{s}") for s in SEEDS] for m, (_, fn) in metrics.items()}
        rec[n] = arms
        for arm, v in arms.items():
            rows.append([n, arm] + [ci_cell(v[m]) for m in metrics])
        for other in [x for x in arms if x != "distilled"]:
            c = {m: compare(arms["distilled"][m], arms[other][m]) for m in metrics}
            crows.append([n, f"distilled against {other}"] + [cmp_cell(c[m]) for m in metrics])
    labels = [lbl for lbl, _ in metrics.values()]
    text = (f"# E19 network taught by the grey-box model, HF-{a.variant.upper()} (test errors, NRMSE; mean [95% CI] over "
            f"{len(SEEDS)} seeds; other arms from E17)\n\n" + md_table(["N", "arm"] + labels, rows) +
            "\n## Comparisons (error of the second arm divided by the error of the distilled network, above 1: distilled "
            "is better; seeds won; paired t-test on log errors)\n\n" + md_table(["N", "comparison"] + labels, crows))
    art = write_text(os.path.join(run_dir, "table_hf_distill.md"), text)
    finish(run_dir, cfg, a.seed, dict(variant=a.variant, results={str(n): v for n, v in rec.items()}), [art])
    print(text)


if __name__ == "__main__":
    main()
