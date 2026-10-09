"""
E30 -- Training on the Blockset vehicle: does the method work when trained on a vehicle we did not write?

The plant is the Vehicle Dynamics Blockset 14-DOF harness of E29 (scripts/vdbs/), with its cornering stiffness
calibrated by the M0 rule and every other difference from the prior left in place (suspension and roll, tyre
relaxation, front/rear stiffness split, rolling resistance).  The prior is our nominal simplified model,
unchanged.  Everything mirrors E28 except the data source:

  models    data-only (lambda 0), anchored data-only (lambda 0), anchored PINC, grey-box with the quasi-steady
            prior (data loss only); plain PINC and the full-prior grey-box model are left out (dominated by the
            anchored network / matched by the quasi-steady grey-box model at a tenth of the solve time in Study 2)
  lambda    the M0 selections made before any Blockset data existed: 10 at N = 100, 1 at N = 1000 (E23)
  sizes     N = 100 and 1000; the N = 100 set is the first 100 of each seed's 1000 (nested; E28 drew them independently)
  seeds     5-9, so each seed pairs with its E29 counterpart
  budget    as E28 (6000 Adam steps, 2000 L-BFGS iterations, float64, best validation weights)
  data      starting states from random-excitation drives of the Blockset vehicle (as pinc/data_hf.driving_states:
            box speeds, smooth random force and steer, four states per 3 s drive after a 0.5 s run-in); one
            uniformly drawn input held from each state, the state read at a random 0.5 ms grid time in (0, T] (as
            pinc/data.sample_trajectories); validation 1000 trajectories shared by all seeds (E28: 4000)
  test      the E29 test set (100 starting states x 10 sequences of 50 periods)

--part inputs    drive commands, sampled times, held inputs and read-out times -> <dir>/train_inputs.mat
                 (then MATLAB: scripts/vdbs/vdbs_train_data.m, and with check = true for the re-simulation check)
--part assemble  network states, data files per seed (results/e30_vdbs_retrain/data/seed<k>.npz) and the checks
--part train     the 40 runs and registry.json
--part eval      tables of errors, ratios and paired tests on the E29 test set
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, start, write_text  # noqa: E402
from experiments.e15_hf_data import ci_cell, overrides  # noqa: E402
from experiments.e29_vdbs_transfer import DEFAULT_DIR, N_SEQ, N_STEPS, network_state  # noqa: E402
from pinc import data_hf, plant_hf  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT, load_config  # noqa: E402

SEEDS = (5, 6, 7, 8, 9)
SIZES = (100, 1000)
N_PER_SEED, N_VAL, PER_DRIVE = 1000, 1000, 4
DRIVE_STEPS, RUN_IN = 30, 5                    # 3 s drives, sampling after 0.5 s (data_hf.driving_states defaults)
MARGIN = 1.15                                  # extra drives for those that leave the envelope
LAM = {100: 10.0, 1000: 1.0}                   # M0 selections for the anchored network (E23), fixed in advance
ANCH = {"model.arch": "anchored"}
ARMS = {"data-only": (0.0, {}), "anchored data-only": (0.0, ANCH), "anchored PINC": (None, ANCH),
        "grey-box-qs": (0.0, {"model.greybox": "true", "model.greybox_prior": "qs"})}
DATA = os.path.join("results", "e30_vdbs_retrain", "data")
HZ = (1, 10, 50)
STATES = ("vx", "vy", "r", "psi")


def rid(arm, n, seed):
    return f"vdbs_{arm.replace(' ', '_')}_n{n}_s{seed}"


# ---------------------------------------------------------------- inputs (Python) -> MATLAB
def part_inputs(a, cfg):
    import scipy.io as sio
    p = plant_hf.make_params(cfg.params, "M0")
    out = {}
    for name, n_states, seed in (("train", N_PER_SEED*len(SEEDS), cfg.seeds.train + 30000), ("val", N_VAL, cfg.seeds.val + 30000)):
        rng = np.random.default_rng(seed)
        nd = int(np.ceil(n_states/PER_DRIVE*MARGIN))
        v0 = rng.uniform(cfg.box_train.vx[0], cfg.box_train.vx[1], nd)
        F0 = np.array([plant_hf.free_rolling_state(v, p)[10] for v in v0])
        cmd = data_hf._commands(rng, nd, DRIVE_STEPS, cfg.T, F0, np.asarray(cfg.u_min), np.asarray(cfg.u_max), 0.3, v0,
                                p["lf"] + p["lr"], 9.0)
        pick = np.sort(np.stack([rng.choice(np.arange(RUN_IN, DRIVE_STEPS), size=PER_DRIVE, replace=False) for _ in range(nd)]), axis=1)
        u = rng.uniform(cfg.u_min, cfg.u_max, size=(nd, PER_DRIVE, 2))                 # as pinc/data.sample_inputs
        n_sub = int(round(cfg.T/cfg.sim.dt_plant))
        k = rng.integers(1, n_sub + 1, size=(nd, PER_DRIVE))                              # read-out t = k dt in (0, T]
        out[name] = dict(v0=v0, F0=F0, cmd=cmd, pick=pick.astype(float), u=u, k=k.astype(float), dt=cfg.sim.dt_plant)
        print(f"{name}: {nd} drives for {n_states} starting states", flush=True)
    os.makedirs(a.dir, exist_ok=True)
    sio.savemat(os.path.join(a.dir, "train_inputs.mat"), {f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    return {s: dict(n_drives=len(d["v0"])) for s, d in out.items()}, f"wrote {a.dir}/train_inputs.mat\n"


# ---------------------------------------------------------------- MATLAB output -> data files, checks
def part_assemble(a, cfg):
    import scipy.io as sio
    from pinc.greybox import prior_of
    R = sio.loadmat(os.path.join(a.dir, "train_data.mat"))
    I = sio.loadmat(os.path.join(a.dir, "train_inputs.mat"))
    env = data_hf.ENVELOPE
    sets, info = {}, {}
    for name in ("train", "val"):
        S0, S1 = R[f"{name}_S0"], R[f"{name}_S1"]                  # (n_drives, PER_DRIVE, 16) start / read-out states
        ok_drive = np.all(np.isfinite(S0), axis=(1, 2)) & np.all(np.isfinite(S1), axis=(1, 2))
        ok_drive &= np.all((S0[..., 0] > env["vx_min"]) & (np.abs(S0[..., 1]) < env["vy_max"]) & (np.abs(S0[..., 2]) < env["r_max"]), axis=1)
        s0 = network_state(S0[ok_drive]).reshape(-1, 10)
        s1 = network_state(S1[ok_drive]).reshape(-1, 10)
        u = I[f"{name}_u"][ok_drive].reshape(-1, 2)
        t = (I[f"{name}_k"][ok_drive]*float(np.squeeze(I[f"{name}_dt"]))).reshape(-1)
        need = N_PER_SEED*len(SEEDS) if name == "train" else N_VAL
        if len(t) < need:
            raise RuntimeError(f"{name}: {len(t)} usable trajectories, {need} needed")
        sets[name] = dict(t=t[:need], s0=s0[:need], u=u[:need], s=s1[:need])
        info[name] = dict(drives=int(len(ok_drive)), drives_kept=int(ok_drive.sum()), trajectories=need)
    os.makedirs(os.path.join(ROOT, DATA), exist_ok=True)
    for j, seed in enumerate(SEEDS):
        tr = {k: v[j*N_PER_SEED:(j + 1)*N_PER_SEED] for k, v in sets["train"].items()}
        np.savez(os.path.join(ROOT, DATA, f"seed{seed}.npz"), pool=sets["train"]["s0"],
                 **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in sets["val"].items()})

    # check 1: re-simulation (MATLAB, check = true) reproduces the first trajectories
    checks = {}
    cf = os.path.join(a.dir, "train_check.mat")
    if os.path.exists(cf):
        C = sio.loadmat(cf)
        nd = C["S1"].shape[0]
        checks["resim_max_abs_diff"] = float(np.nanmax(np.abs(C["S1"] - R["train_S1"][:nd])))
        checks["resim_trajectories"] = int(nd*PER_DRIVE)
    # check 2: read-outs at the smallest times stay close to the starting state (scaled by S_x)
    tr = sets["train"]
    small = np.argsort(tr["t"])[:20]
    checks["small_t_max"] = float(tr["t"][small].max())
    checks["small_t_max_scaled_change"] = float(np.max(np.abs(tr["s"][small] - tr["s0"][small])[:, :6]/np.asarray(cfg.S_x)[:6]))
    # check 3: the prior's one-step error on the Blockset training set (against 0.0077 on the M0 test set, E25)
    S_x = np.asarray(cfg.S_x)
    for kind in ("qs", "full"):
        flow = prior_of(kind)
        pred = np.asarray(flow(tr["t"], tr["s0"], tr["u"], cfg))
        e2 = ((pred - tr["s"])/S_x)**2
        checks[f"prior_{kind}_one_step_body"] = float(np.sqrt(np.mean(e2[:, :4])))
        ay = np.abs(tr["s0"][:, 0]*tr["s0"][:, 2])
        bins = ((0, 1), (1, 2), (2, 4), (4, 6), (6, np.inf))
        checks[f"prior_{kind}_one_step_body_by_ay"] = {f"{lo:g}-{hi:g}": dict(n=int(np.sum((ay >= lo) & (ay < hi))),
                                                                              rms=float(np.sqrt(np.mean(e2[(ay >= lo) & (ay < hi), :4]))) if np.any((ay >= lo) & (ay < hi)) else None)
                                                       for lo, hi in bins}
    text = ("# E30 data (Blockset vehicle)\n\n" + json.dumps(dict(sets=info, checks=checks), indent=1) + "\n")
    return dict(sets=info, checks=checks), text


# ---------------------------------------------------------------- training
def part_train(a, cfg):
    from pinc.jobs import run_jobs
    jobs, registry = [], {}
    for n in SIZES:
        for arm, (lam, extra) in ARMS.items():
            lam = LAM[n] if lam is None else lam
            registry.setdefault(str(n), {})[arm] = rid(arm, n, 0).replace("_s0", "_s{seed}")
            for seed in SEEDS:
                ov = dict(overrides(n, lam, seed, cfg), **extra)
                ov.update({"train.data_file": os.path.join(DATA, f"seed{seed}.npz"), "train.n_val": N_VAL})
                jobs.append((rid(arm, n, seed), seed, ov))
    run_jobs(jobs, a.slots, a.config, a.overrides, cpu_threads=a.cpu_threads, gpu_threads=a.gpu_threads)
    run_dir = a.run_dir
    with open(os.path.join(run_dir, "registry.json"), "w") as fh:
        json.dump(dict(seeds=list(SEEDS), lambdas={str(n): LAM[n] for n in SIZES}, arms=registry), fh, indent=1)
    return dict(registry=registry, n_runs=len(jobs)), f"trained {len(jobs)} runs\n"


# ---------------------------------------------------------------- evaluation on the E29 test set
def part_eval(a, cfg):
    import scipy.io as sio
    import tensorflow as tf
    from pinc.model import PINCNet
    from pinc.mpc import RK4Predictor, make_predictor
    D = os.path.join(RESULTS_DIR, "e29_vdbs_transfer", "data")
    ic = sio.loadmat(os.path.join(D, "ics.mat"))["IC"]
    X = sio.loadmat(os.path.join(D, "truth.mat"))["X"]
    u = sio.loadmat(os.path.join(D, "seqs.mat"))["u"]
    n_ic = len(ic)
    s0r = np.repeat(network_state(ic), N_SEQ, axis=0)
    ur = u.reshape(-1, N_STEPS, 2)
    S_x = np.asarray(cfg.S_x)

    def score(rollout):
        p = np.asarray(rollout(tf.constant(s0r), tf.constant(ur))).reshape(n_ic, N_SEQ, N_STEPS, -1)[..., :4]
        e2 = ((p - X[..., :4])/S_x[:4])**2
        tot = {h: float(np.mean(np.sqrt(np.mean(e2[:, :, h - 1, :], axis=(1, 2))))) for h in HZ}        # as E9 / E29
        per = {h: np.sqrt(np.mean(e2[:, :, h - 1, :], axis=(0, 1))).tolist() for h in HZ}
        return tot, per

    prior, prior_per = score(RK4Predictor(cfg, model="qs").rollout_batch)
    reg = json.load(open(os.path.join(RESULTS_DIR, "e30_vdbs_retrain", "e30_train", "registry.json")))["arms"]
    e29 = json.load(open(os.path.join(RESULTS_DIR, "e29_vdbs_transfer", "e29_eval", "summary.json")))["models"]
    res = {}
    for n in SIZES:
        for arm, tmpl in reg[str(n)].items():
            per_seed, per_state = {h: [] for h in HZ}, []
            for seed in SEEDS:
                net = PINCNet.load_from(os.path.join(RESULTS_DIR, "models", tmpl.format(seed=seed)))
                tot, per = score(make_predictor(net, cfg).rollout_batch)
                for h in HZ:
                    per_seed[h].append(tot[h])
                per_state.append(per)
                tf.keras.backend.clear_session()
            res.setdefault(str(n), {})[arm] = dict(err={str(h): per_seed[h] for h in HZ},
                                                   per_state={str(h): np.mean([p[h] for p in per_state], axis=0).tolist() for h in HZ},
                                                   e29_h50=e29[str(n)][arm]["h50"], e29_h10=e29[str(n)][arm]["h10"])
            print(arm, n, [f"{np.exp(np.mean(np.log(per_seed[h]))):.3g}" for h in HZ], flush=True)

    def paired(x, y):
        """Geometric-mean ratio y/x over seeds (above 1: x better), seeds won by x, paired t-test on logs."""
        x, y = np.log(np.asarray(x)), np.log(np.asarray(y))
        d = y - x
        p = float(stats.ttest_rel(y, x).pvalue) if np.std(d) > 0 else float("nan")
        return dict(ratio=float(np.exp(np.mean(d))), wins=int(np.sum(d > 0)), p=p)

    def vs_const(x, c):
        lr = np.log(c) - np.log(np.asarray(x))
        p = float(stats.ttest_1samp(lr, 0.0).pvalue) if np.std(lr) > 0 else float("nan")
        return dict(ratio=float(np.exp(np.mean(lr))), wins=int(np.sum(lr > 0)), p=p)

    cmp_, rows = {}, []
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    cell = lambda c: f"{c['ratio']:.2f} ({c['wins']}/{len(SEEDS)}, p = {c['p']:.2g})"
    for n in SIZES:
        r = res[str(n)]
        for arm in r:
            e50 = r[arm]["err"]["50"]
            c = dict(vs_prior={str(h): vs_const(r[arm]["err"][str(h)], prior[h]) for h in HZ},
                     vs_data=paired(e50, r["data-only"]["err"]["50"]) if arm != "data-only" else None,
                     vs_e29=paired(e50, r[arm]["e29_h50"]))
            cmp_.setdefault(str(n), {})[arm] = c
            rows.append([f"{arm}, N = {n}", ci_cell(e50), cell(c["vs_prior"]["50"]), cell(c["vs_data"]) if c["vs_data"] else "-",
                         ci_cell(r[arm]["e29_h50"]), cell(c["vs_e29"])])
        for x, y in (("anchored PINC", "anchored data-only"), ("anchored PINC", "grey-box-qs")):
            cmp_[str(n)][f"{x} vs {y}"] = {str(h): paired(r[x]["err"][str(h)], r[y]["err"][str(h)]) for h in HZ}
    prow = [["quasi-steady prior"] + [f"{prior[h]:.3g}" for h in HZ]]
    text = ("# E30 trained on the Blockset vehicle (seeds 5-9; E29 test set: 100 starting states x 10 sequences; NRMSE of "
            "the body states)\n\nModels: mean [95% CI] over seeds of the 50-step error.  Ratios: geometric mean over seeds of "
            "(other error / model error), above 1 the model is better; seeds better; t-test on log errors (one-sample "
            "against the single prior, paired against data-only and against E29, i.e. the same model trained on our plant).\n\n" +
            md_table(["model", "50 steps", "vs prior", "vs data-only", "E29 (trained on our plant)", "vs E29"], rows) +
            "\n## Prior alone\n\n" + md_table(["predictor", "1 step", "10 steps", "50 steps"], prow) +
            "\n## Physics loss and grey-box comparison (ratio of the second's error to the first's)\n\n" +
            md_table(["N", "comparison"] + [f"{h} steps" for h in HZ],
                     [[n, k] + [cell(v[str(h)]) for h in HZ] for n in map(str, SIZES) for k, v in cmp_[n].items() if " vs " in k]))
    return dict(prior=prior, prior_per_state=prior_per, results=res, comparisons=cmp_), text


def main(argv=None):
    from pinc.jobs import add_slot_args
    ap = base_parser(__doc__)
    add_slot_args(ap)
    ap.add_argument("--part", required=True, choices=("inputs", "assemble", "train", "eval"))
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(DEFAULT_DIR), "e30"), help="exchange folder shared with MATLAB")
    a = ap.parse_args(argv)
    a.config = os.path.join(ROOT, "configs", "hf_m0.yaml")
    _, a.run_dir = start("e30_vdbs_retrain", a)
    cfg = load_config(a.config, a.overrides)
    out, text = dict(inputs=part_inputs, assemble=part_assemble, train=part_train, eval=part_eval)[a.part](a, cfg)
    art = write_text(os.path.join(a.run_dir, f"table_{a.part}.md"), text)
    finish(a.run_dir, cfg, a.seed, dict(part=a.part, **out), [art])
    print(text)


if __name__ == "__main__":
    main()
