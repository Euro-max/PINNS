"""
E13 (plan H1) -- How wrong is the physics prior, and where?

For each variant (M0: calibrated, M1: mismatched stiffness) the prior's rates f_P(s, u) are compared
with the true plant's f_HF(s, u) on held-out driving states (pinc/data_hf.py, seed distinct from
training) with inputs drawn uniformly from the input box, as in training.  Error per state:
|f_HF - f_P| / S_f (S_f from the variant's config), reported as RMS over all samples and binned by
lateral acceleration |vx r| (gentle -> near the grip limit) and by the largest wheel slip.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import base_parser, finish, md_table, savefig, start, write_text, plt  # noqa: E402
from pinc import data_hf, prior_hf  # noqa: E402
from pinc.config import ROOT, load_config  # noqa: E402
from pinc.system import get_system  # noqa: E402

AY_BINS = (0.0, 1.0, 2.0, 4.0, 6.0, np.inf)
SLIP_BINS = (0.0, 0.01, 0.03, 0.06, np.inf)
GROUPS = dict(body=[0, 1, 2], actuators=[4, 5], wheels=[6, 7, 8, 9])


def rms(a, axis=0):
    return np.sqrt(np.mean(np.square(a), axis=axis))


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args(argv)
    cfg0, run_dir = start("e13_prior_error", a)
    n = 2000 if a.quick else a.n
    summary, arts, curves = dict(n=n, variants={}), [], {}
    names = prior_hf.STATE_NAMES
    text = ("# E13 prior error |f_HF - f_P| / S_f (RMS) on held-out driving states, inputs uniform in the box\n\n")
    for variant in ("m0", "m1"):
        cfg = load_config(os.path.join(ROOT, "configs", f"hf_{variant}.yaml"))
        sysm = get_system(cfg)
        s = data_hf.driving_states(n, 54321, sysm.truth, tuple(cfg.box_train.vx), cfg.u_min, cfg.u_max)
        u = np.random.default_rng(54322).uniform(cfg.u_min, cfg.u_max, (n, 2))
        e = (prior_hf.f_s_true(s, u, sysm.truth) - prior_hf.f_s(s, u, sysm.prior))/cfg.S_f
        ay = np.abs(s[:, 0]*s[:, 2])
        slip = np.max(np.abs(s[:, 6:10]/s[:, :1]), axis=1)
        res = dict(all=rms(e).tolist(), by_ay={}, by_slip={})
        rows = [["all", n] + [f"{v:.3g}" for v in res["all"]]]
        for lo, hi in zip(AY_BINS[:-1], AY_BINS[1:]):
            m = (ay >= lo) & (ay < hi)
            if m.sum() < 20:
                continue
            res["by_ay"][f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), rms=rms(e[m]).tolist())
            rows.append([f"|a_y| {lo:g}-{hi:g} m/s^2", int(m.sum())] + [f"{v:.3g}" for v in rms(e[m])])
        for lo, hi in zip(SLIP_BINS[:-1], SLIP_BINS[1:]):
            m = (slip >= lo) & (slip < hi)
            if m.sum() < 20:
                continue
            res["by_slip"][f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), rms=rms(e[m]).tolist())
            rows.append([f"max slip {lo:g}-{hi:g}", int(m.sum())] + [f"{v:.3g}" for v in rms(e[m])])
        summary["variants"][variant] = res
        curves[variant] = res["by_ay"]
        text += f"## {variant.upper()}\n\n" + md_table(["subset", "n"] + list(names), rows) + "\n"
    arts.append(write_text(os.path.join(run_dir, "table_prior_error.md"), text))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, variant in zip(axes, ("m0", "m1")):
        keys = list(curves[variant])
        x = np.arange(len(keys))
        for g, idx in GROUPS.items():
            ax.plot(x, [rms(np.array(curves[variant][k]["rms"])[idx]) for k in keys], marker="o", label=g)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, fontsize=8)
        ax.set(yscale="log", xlabel="|a_y| bin [m/s^2]", title=f"prior error, {variant.upper()}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMS |f_HF - f_P| / S_f")
    axes[0].legend(fontsize=8)
    arts += savefig(fig, run_dir, "fig_prior_error")
    finish(run_dir, cfg0, a.seed, summary, arts)
    print(text)


if __name__ == "__main__":
    main()
