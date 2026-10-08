"""Generate every table, figure copy and headline-number macro used by LATEX/main.tex
from results/ (ground rule 4: no hand-typed numbers in the paper).

    python scripts/make_paper_tables.py [--e1 e1] [--e2 e2] [--e3 e3] [--e4 e4] [--e5 e5] [--e6 e6] [--e7 e7]

Writes LATEX/generated/{tab_*.tex, numbers.tex, fig_*.pdf}.  Missing experiments are skipped.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinc.config import RESULTS_DIR, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "LATEX", "generated")
ARMS = ["nmpc_rk4", "pinc", "blackbox", "ltv"]
LABEL = dict(nmpc_rk4="NMPC-RK4", pinc="PINC-MPC", blackbox="Black-box-MPC", ltv="LTV-MPC",
             rk4="RK4 ($\\Delta t = 0.01$ s)", linear="LTI (20 m/s)", truth="RK4 truth")
REFS = dict(speed_sin="speed sinusoid", speed_step="speed step", lane_change="lane change", double_lane_change="double lane change")
macros = {}


def load(exp, run):
    p = os.path.join(RESULTS_DIR, exp, run, "summary.json")
    if not os.path.exists(p):
        print(f"  skip {exp}/{run} (no summary.json)")
        return None
    with open(p) as fh:
        return json.load(fh)


def sci(x, sig=2):
    if x is None or x != x:
        return "--"
    s = f"{x:.{sig-1}e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sci_ci(c, sig=2):
    """mean [lo, hi] with one shared power of ten, e.g. 2.0 [1.9, 2.1] x 10^-3."""
    import math
    m = c["mean"]
    if m is None or m != m or m == 0:
        return "--"
    e = int(math.floor(math.log10(abs(m))))
    f = lambda v: f"{v/10**e:.{sig-1}f}"
    return f"${f(m)}\\,[{f(c['lo'])}, {f(c['hi'])}]\\times10^{{{e}}}$"


def lam_tex(lam):
    """0, 1 or a power of ten as $10^{k}$ (consistent formatting of lambda)."""
    import math
    if lam == 0 or lam == 1:
        return f"{lam:g}"
    k = math.log10(lam)
    return f"$10^{{{int(round(k))}}}$" if abs(k - round(k)) < 1e-9 else f"{lam:g}"


def sig2(x):
    """Two significant figures, trailing zeros kept (0.044, 1.0, 3.1, 26)."""
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:#.2g}"


def ratio(x):
    """Ratios: two decimals below 10, one above (e.g. 1.17, 14.3)."""
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def dec(x, nd=3):
    return "--" if x is None or x != x else f"{x:.{nd}f}"


def ci(c, nd=3):
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
_WORDS = {"5": "Five", "10": "Ten", "20": "Twenty", "40": "Forty", "100": "Hundred", "1000": "Thousand", "10000": "TenK", "100000": "HundredK"}


def word(n):
    """LaTeX command names may not contain digits: map a number to a word."""
    n = str(n)
    return _WORDS.get(n) or "".join(_DIGITS[c] for c in n)


def macro(name, value):
    name = "".join(_DIGITS.get(c, c) for c in name)
    macros[name] = value


def write(name, text):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)
    print("  wrote", name)


def copyfig(exp, run, fig, dst):
    src = os.path.join(RESULTS_DIR, exp, run, fig + ".pdf")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, dst + ".pdf"))
        print("  copied", dst)


def tabular(cols, header, rows, caption, label, small=True, wide=False):
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    size = "\\small" if small else ""
    tab = (f"\\begin{{tabular}}{{{cols}}}\\toprule\n{' & '.join(header)} \\\\\\midrule\n{body} \\\\\n\\bottomrule\\end{{tabular}}")
    if wide:                       # scale to the text width so nothing overruns the margin
        tab = f"\\resizebox{{\\textwidth}}{{!}}{{{tab}}}"
    return (f"\\begin{{table}}[htbp]\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n{tab}\\end{{table}}\n")


# ---------------------------------------------------------------- E1
def e1(run):
    s = load("e1_open_loop", run)
    if not s:
        return
    rows = []
    for region, rname in (("in_box", "in-domain"), ("extrap", "extrapolation")):
        r = s[region]
        for arm in ("pinc", "blackbox", "linear", "rk4"):
            o = r["onestep"][arm]
            c = r["chain"][arm]["curve"]
            rows.append([rname if arm == "pinc" else "", {"pinc": "PINC", "blackbox": "Black-box"}.get(arm, LABEL[arm]), sci(o["0.25"]["all"]["mean"]), sci(o["1.0"]["all"]["mean"]),
                         sci(c["10"]["all"]["mean"]), sci(c["50"]["all"]["mean"])])
    write("tab_e1.tex", tabular("llcccc", ["region", "model", "$t = 0.25T$", "$t = T$", "10 steps", "50 steps"], rows,
                                f"Surrogate NRMSE (RMSE / $S_x$, all states, mean over {s['in_box']['n_ic']} initial states "
                                f"$\\times$ {s['in_box']['n_seq']} control sequences). One-step at fractional times and chained prediction "
                                f"over 10 and 50 control periods.", "tab:e1"))
    ib = s["in_box"]
    macro("EoneStepPinc", sci(ib["onestep"]["pinc"]["1.0"]["all"]["mean"]))
    macro("EoneStepBlackbox", sci(ib["onestep"]["blackbox"]["1.0"]["all"]["mean"]))
    macro("EoneStepLinear", sci(ib["onestep"]["linear"]["1.0"]["all"]["mean"]))
    macro("EoneChainPinc", sci(ib["chain"]["pinc"]["curve"]["50"]["all"]["mean"]))
    macro("EoneChainBlackbox", sci(ib["chain"]["blackbox"]["curve"]["50"]["all"]["mean"]))
    macro("EoneChainLinear", sci(ib["chain"]["linear"]["curve"]["50"]["all"]["mean"]))
    macro("EoneRatio", f"{ib['onestep']['blackbox']['1.0']['all']['mean']/ib['onestep']['pinc']['1.0']['all']['mean']:.1f}")
    macro("EoneChainRatio", f"{ib['chain']['blackbox']['curve']['50']['all']['mean']/ib['chain']['pinc']['curve']['50']['all']['mean']:.1f}")
    ex = s["extrap"]
    macro("EoneExtrapPinc", sci(ex["chain"]["pinc"]["curve"]["50"]["all"]["mean"]))
    macro("EoneExtrapBlackbox", sci(ex["chain"]["blackbox"]["curve"]["50"]["all"]["mean"]))
    macro("EoneExtrapRatio", f"{ex['chain']['blackbox']['curve']['50']['all']['mean']/ex['chain']['pinc']['curve']['50']['all']['mean']:.1f}")
    macro("EoneNic", str(ib["n_ic"]))
    macro("EoneNseq", str(ib["n_seq"]))


# ---------------------------------------------------------------- E2
def e2(run):
    s = load("e2_data_efficiency", run)
    if not s:
        return
    sizes = [str(n) for n in s["sizes"]]
    rows = []
    for arm, lab in (("pinc", f"PINC ($\\lambda = {s['lam']:g}$)"), ("blackbox", "black-box ($\\lambda = 0$)")):
        rows.append([lab] + [sci_ci(s['curves'][arm][n]['all']) for n in sizes])
    write("tab_e2.tex", tabular("l" + "c"*len(sizes), ["arm"] + [f"$N = 10^{len(n)-1}$" for n in sizes], rows,
                                f"Test NRMSE (all states, in-domain test set) versus number of training trajectories, mean [95\\% bootstrap CI] over "
                                f"{len(s['seeds'])} seeds; identical optimiser budget for every run.", "tab:e2", wide=True))
    for n in sizes:
        cp, cb = s["curves"]["pinc"][n], s["curves"]["blackbox"][n]
        macro(f"EtwoPinc{word(n)}", sci(cp["all"]["mean"]))
        macro(f"EtwoBlackbox{word(n)}", sci(cb["all"]["mean"]))
        macro(f"EtwoRatio{word(n)}", f"{cb['all']['mean']/cp['all']['mean']:.1f}" if cb['all']['mean']/cp['all']['mean'] < 10 else f"{cb['all']['mean']/cp['all']['mean']:.0f}")
        macro(f"EtwoExtrapPinc{word(n)}", sci(cp["extrap"]["mean"]))
        macro(f"EtwoExtrapBlackbox{word(n)}", sci(cb["extrap"]["mean"]))
        macro(f"EtwoVxRatio{word(n)}", f"{cb['per_state']['vx']['mean']/cp['per_state']['vx']['mean']:.0f}")
    macro("EtwoSeeds", str(len(s["seeds"])))


# ---------------------------------------------------------------- E3
def e3(run):
    s = load("e3_closed_loop", run)
    if not s:
        return
    rows = []
    for ref in ("speed_sin", "speed_step", "lane_change", "double_lane_change"):
        per = s["per_ref"][ref]
        for i, arm in enumerate(ARMS):
            a = per[arm]
            def cell(k, nd):
                t = dec(a[k]["mean"], nd)
                if "vs_baseline" in a:
                    p = a["vs_baseline"][k]["p"]
                    t += "$^{*}$" if p < 0.05 else ""
                return t
            rows.append([REFS[ref] if i == 0 else "", LABEL[arm], cell("rmse_vx", 3), cell("rmse_psi", 4), cell("rmse_Y", 3),
                         cell("effort", 4), dec(a["solve_mean"]["mean"]*1e3, 1), dec(a["success_rate"]["mean"], 2)])
    write("tab_e3.tex", tabular("llcccccc", ["reference", "arm", "RMSE $v_x$ [m/s]", "RMSE $\\psi$ [rad]", "RMSE $Y$ [m]",
                                             "effort", "solve [ms]", "success"], rows,
                                f"Closed-loop tracking, mean over {len(s['seeds'])} seeds (measurement noise and initial-state perturbation). "
                                "Effort is $\\sum_k \\tilde u_k^\\top R \\tilde u_k T$. Solve: mean time per MPC step on one CPU thread; each "
                                "controller's cost function is compiled by a warm-up call before the timed runs. $^{*}$: paired Wilcoxon signed-rank test against NMPC-RK4 with $p < 0.05$ "
                                "(differences of a few percent are statistically resolvable at this seed count; confidence intervals, IAE, "
                                "maxima and effect sizes for every metric are provided with the code).", "tab:e3", wide=True))
    for ref, key in (("lane_change", "Lane"), ("double_lane_change", "Dlc"), ("speed_sin", "Sin"), ("speed_step", "Step")):
        for arm in ARMS:
            m = "rmse_Y" if key in ("Lane", "Dlc") else "rmse_vx"
            macro(f"Ethree{key}{arm.replace('_', '')}", dec(s["per_ref"][ref][arm][m]["mean"], 3))
    macro("EthreeSeeds", str(len(s["seeds"])))
    macro("EthreeRuns", str(len(s["seeds"])*len(s["refs"])*len(s["arms"])))
    macro("EthreeSeedShown", str(s["representative_seed"]))
    import numpy as _np
    raw = s["raw"]["speed_sin"]
    rk = _np.array([r["rmse_vx"] for r in raw["nmpc_rk4"]])
    for arm in ("pinc", "blackbox"):
        v = _np.array([r["rmse_vx"] for r in raw[arm]])
        macro(f"EthreeSinMedianRatio{arm}", f"{_np.median(v/rk):.2f}")
        macro(f"EthreeSinWorse{arm}", str(int((v > rk).sum())))
        macro(f"EthreeSinP{arm}", f"{s['per_ref']['speed_sin'][arm]['vs_baseline']['rmse_vx']['p']:.2g}")


# ---------------------------------------------------------------- E4
def e4(run):
    s = load("e4_timing", run)
    if not s:
        return
    Ns = [str(n) for n in s["horizons"]]
    rows = []
    for arm in ARMS:
        rows.append([LABEL[arm]] + [f"{s['solve'][arm][n]['median']*1e3:.1f} / {s['solve'][arm][n]['p95']*1e3:.1f}" for n in Ns] +
                    [f"{s['model_call'][arm]['compiled']*1e3:.3f}" if s["model_call"][arm]["compiled"] == s["model_call"][arm]["compiled"] else "--"])
    write("tab_e4.tex", tabular("l" + "c"*len(Ns) + "c", ["arm"] + [f"$N = {n}$" for n in Ns] + ["model call [ms]"], rows,
                                "Median / 95th-percentile MPC solve time per control step in ms versus horizon $N$, CPU single thread, "
                                "XLA-compiled cost and gradient, SLSQP with identical options and warm start for every arm; last column: one "
                                "compiled one-step model evaluation (batch 1).", "tab:e4"))
    for arm in ARMS:
        for n in Ns:
            macro(f"Efour{arm.replace('_', '')}N{word(n)}", f"{s['solve'][arm][n]['median']*1e3:.1f}")
    for n in Ns:
        macro(f"EfourRatioN{word(n)}", f"{s['solve']['nmpc_rk4'][n]['median']/s['solve']['pinc'][n]['median']:.2f}")
    macro("EfourModelRk", f"{s['model_call']['nmpc_rk4']['compiled']*1e3:.3f}")
    macro("EfourModelPinc", f"{s['model_call']['pinc']['compiled']*1e3:.3f}")


# ---------------------------------------------------------------- E5
def setting_tex(name):
    """Plant-setting labels for tables: 'fiala mu=0.4' -> 'Fiala, $\\mu = 0.4$', 'Fx step ...' -> '$F_x$ step ...'."""
    if name.startswith("fiala mu="):
        return f"Fiala, $\\mu = {name.split('=')[1]}$"
    if name.startswith("Fx step"):
        return "$F_x$ step $-1$\\,kN at 4\\,s"
    out = name.replace("Ca", "$C_\\alpha$ ").replace("m-", "mass $-$").replace("m+", "mass +")
    out = out.replace("$-$20%", "$-$20\\%").replace("+20%", "+20\\%").replace("-30%", "$-$30\\%").replace("+30%", "+30\\%")
    return out


def e5(run):
    s = load("e5_robustness", run)
    if not s:
        return
    rows = []
    for name in s["settings"]:
        rows.append([setting_tex(name)] +
                    [dec(s["per_setting"][name][arm]["rmse_Y"]["mean"], 4) for arm in ARMS] +
                    [dec(s["per_setting"][name][arm]["rmse_vx"]["mean"], 3) for arm in ARMS])
    write("tab_e5.tex", tabular("lcccccccc", ["plant setting"] + [LABEL[a].replace("-MPC", "") for a in ARMS] + [LABEL[a].replace("-MPC", "") for a in ARMS], rows,
                                f"Robustness to plant--model mismatch on the single lane change ({len(s['seeds'])} seeds). Controllers keep the "
                                "nominal parameters; only the plant is perturbed. Columns 2--5: RMSE $Y$ [m]; columns 6--9: RMSE $v_x$ [m/s]. "
                                f"All solves succeeded in every setting.", "tab:e5", wide=True))
    for name, key in (("nominal", "Nom"), ("fiala mu=0.4", "MuLow"), ("Ca-30%", "CaLow")):
        for arm in ARMS:
            macro(f"Efive{key}{arm.replace('_', '')}", dec(s["per_setting"][name][arm]["rmse_Y"]["mean"], 3))
    macro("EfiveSeeds", str(len(s["seeds"])))
    for name, key in (("nominal", "Nom"), ("Fx step -1000 N @ 4 s", "Fstep")):
        for arm in ARMS:
            macro(f"EfiveVx{key}{arm.replace('_', '')}", dec(s["per_setting"][name][arm]["rmse_vx"]["mean"], 3))


# ---------------------------------------------------------------- E6
def e6(run):
    s = load("e6_ablations", run)
    if not s:
        return
    names = {"lam0": "$\\lambda = 0$", "lam0.01": "$\\lambda = 0.01$", "lam0.1": "$\\lambda = 0.1$", "lam1": "$\\lambda = 1$",
             "lam10": "$\\lambda = 10$", "lam100": "$\\lambda = 100$", "colloc2k": "2k collocation pts", "colloc100k": "100k collocation pts",
             "arch2x64": "2$\\times$64", "arch6x256": "6$\\times$256", "softic": "soft IC", "residual_skip": "skip connections",
             "residual_block": "ResNet blocks", "no_res_vx": "no $v_x$ residual", "no_res_vy": "no $v_y$ residual",
             "no_res_r": "no $r$ residual", "no_res_psi": "no $\\psi$ residual"}
    order = ["lam0", "lam0.01", "lam0.1", "lam1", "lam10", "lam100", "colloc2k", "colloc100k", "arch2x64", "arch6x256",
             "softic", "residual_skip", "residual_block", "no_res_vx", "no_res_vy", "no_res_r", "no_res_psi"]
    rows = []
    for tag in [t for t in order if t in s["runs"]] + [t for t in s["runs"] if t not in order]:
        r = s["runs"][tag]
        t = r["test"]["nrmse"]
        rows.append([names.get(tag, tag), sci(r["val"]["data"]), sci(r["val"]["phys"])] +
                    [sig2(1e3*v) for v in t] + [sig2(1e3*sum(v*v for v in r['test_extrap']['nrmse'])**0.5/2)])
    write("tab_e6.tex", "{\\setlength{\\tabcolsep}{3.5pt}\\footnotesize\n" +
          tabular("lccccccc", ["ablation", "val.\\ data", "val.\\ phys.", "$v_x$", "$v_y$", "$r$", "$\\psi$", "extrap."], rows,
                  "Physics-weight sweep and ablations at the default 8$\\times$64 network (seed 0). Validation data loss (scaled MSE on held-out "
                  "trajectories) and validation physics loss; per-state test NRMSE and all-state extrapolation NRMSE in units of $10^{-3}$. "
                  f"Rows below the $\\lambda$ sweep use $\\lambda = {s['lam_star']:g}$.", "tab:e6", small=False) + "}\n")
    for tag, key in (("lam0", "LamZero"), ("lam0.01", "LamDefault"), ("lam0.1", "LamPointOne"), ("lam1", "LamOne"), ("no_res_vx", "NoVx"),
                     ("softic", "SoftIc"), ("colloc100k", "CollocBig")):
        macro(f"Esix{key}", sci(s["runs"][tag]["val"]["data"]))
    macro("EsixLamArgmin", f"{s['lam_argmin_val_data']:g}")
    macro("EsixSoftIcVx", sci(s["runs"]["softic"]["test"]["nrmse"][0]))
    macro("EsixHardIcVx", sci(s["runs"]["lam0.01"]["test"]["nrmse"][0]))
    macro("EsixNoVxVx", sci(s["runs"]["no_res_vx"]["test"]["nrmse"][0]))
    macro("EsixBigArch", sci(s["runs"]["arch6x256"]["val"]["data"]))


def e6_no_scaling(run):
    """Residual ablation of the first ablation study, run WITHOUT the output scaling D (macros only):
    which residual makes the physics term hurt the data fit."""
    s = load("e6_ablations", run)
    if not s:
        return
    r = s["runs"]
    macro("EsixNoDLamZero", sci(r["lam0"]["val"]["data"]))
    macro("EsixNoDWith", sci(r["lam0.01"]["val"]["data"]))
    macro("EsixNoDNoVx", sci(r["no_res_vx"]["val"]["data"]))
    other = [r[t]["val"]["data"] for t in ("no_res_vy", "no_res_r", "no_res_psi")]
    macro("EsixNoDOtherMin", sci(min(other)))
    macro("EsixNoDOtherMax", sci(max(other)))


# ---------------------------------------------------------------- E7
def e7(run):
    s = load("e7_architecture", run)
    if not s:
        return
    recs = s["trials"]
    lams = sorted({r["lam"] for r in recs})
    grid = [(2, 64), (4, 64), (8, 64), (4, 128), (8, 128), (8, 256)]
    rows = []
    for lam in lams:
        row = [f"{lam:g}"]
        for d, w in grid:
            m = [x for x in recs if x["lam"] == lam and x["depth"] == d and x["width"] == w and x["residual"] == "none" and x["dropout"] == 0 and not x["layernorm"]]
            row.append(sci(m[0]["val_data"]) if m else "--")
        rows.append(row)
    write("tab_e7a.tex", tabular("l" + "c"*len(grid), ["$\\lambda$"] + [f"{d}$\\times${w}" for d, w in grid], rows,
                                 f"Validation data loss of plain fully-connected networks (depth $\\times$ width) for three values of $\\lambda$; "
                                 "same data, seeds and optimiser budget for every network.", "tab:e7a", wide=True))
    variants = [("plain", lambda x: x["residual"] == "none" and x["dropout"] == 0 and not x["layernorm"]),
                ("dropout 0.05", lambda x: x["dropout"] == 0.05), ("dropout 0.1", lambda x: x["dropout"] == 0.1),
                ("skip connections", lambda x: x["residual"] == "skip"), ("ResNet blocks", lambda x: x["residual"] == "block"),
                ("layer normalisation", lambda x: x["layernorm"])]
    rows = []
    for name, pred in variants:
        row = [name]
        for d, w in ((4, 128), (6, 256)):
            for lam in (0.0, 0.01):
                m = [x for x in recs if x["lam"] == lam and x["depth"] == d and x["width"] == w and pred(x)]
                row.append(sci(m[0]["val_data"]) if m else "--")
        rows.append(row)
    write("tab_e7b.tex", tabular("lcccc", ["variant", "4$\\times$128, $\\lambda=0$", "4$\\times$128, $\\lambda=0.01$", "6$\\times$256, $\\lambda=0$", "6$\\times$256, $\\lambda=0.01$"], rows,
                                 "Validation data loss of regularised and skip-connected variants against the plain network of the same size.",
                                 "tab:e7b", wide=True))
    b = s["best_by_lambda"]
    macro("EsevenTrials", str(s["n_trials"]))
    for lam, key in (("0", "Zero"), ("0.01", "Default")):
        if lam in b:
            macro(f"EsevenBest{key}", f"{b[lam]['depth']}$\\times${b[lam]['width']}")
            macro(f"EsevenBest{key}Params", f"{b[lam]['n_params']:,}".replace(",", "\\,"))


RUNS = dict(e1="e1_v2", e2="e2_v2", e3="e3_v3", e4="e4_v2", e5="e5_v2", e6="e6_v2", e6nod="e6", e7="e7",
            e8="e8b", e9="e9", e10="e10", model="pinc_v2_s0", tfcheck="tf221_check/pinc_v2_s0_tf221")


def add_run_args(ap):
    for k, v in RUNS.items():
        ap.add_argument(f"--{k}", default=v)


# ---------------------------------------------------------------- physics learning (E8/E9) and confirmation (E10)
def e9(run):
    s = load("e9_physics_learning", run)
    if not s:
        return
    order = ["inc_lam0", "inc_lam1e-05", "inc_lam0.0001", "inc_lam0.001", "inc_lam0.01", "inc_lam0.1", "inc_lam1"]
    rows = []

    def row(tag, scaling):
        r = s["models"][tag]; h = r["horizon"]; d = r["deriv_err_rms"]
        return [scaling, lam_tex(r["lam"]), f"{r['deriv_err_all']:.3f}", f"{d[0]:.3f}", sci(h["in_domain"]["10"]["mean"]),
                sci(h["in_domain"]["50"]["mean"]), sci(h["extrap"]["50"]["mean"])]
    for tag in ("base_lam0", "base_lam0.01"):
        if tag in s["models"]:
            rows.append(row(tag, "none" if tag == "base_lam0" else ""))
    first = True
    for tag in order:
        if tag in s["models"]:
            rows.append(row(tag, "$D$" if first else ""))
            first = False
    write("tab_e9.tex", tabular("ccccccc", ["scaling", "$\\lambda$", "$\\dot s$ error", "$\\dot v_x$ error", "10 steps", "50 steps",
                                            "50 steps (extrap.)"], rows,
                                "Effect of the output scaling $D$ and of the physics weight on what the network learns (one seed, same network and "
                                "budget). $\\dot s$ error: RMS difference between the network's time derivative and the true dynamics along held-out "
                                "trajectories, in units of $S_f$ (all states, and $v_x$ alone). Remaining columns: normalised RMSE of chained "
                                "predictions. Without $D$, the physics term ($\\lambda = 0.01$) makes the 50-step prediction worse than the data-only "
                                "network; with $D$ it makes it better.", "tab:e9"))
    b = s["models"].get("base_lam0.01")
    if b:
        macro("EnineOldDeriv", f"{b['deriv_err_all']:.3f}")
        macro("EnineOldHfifty", sci(b["horizon"]["in_domain"]["50"]["mean"]))
    for tag, key in (("inc_lam0", "Zero"), ("inc_lam0.01", "Default")):
        if tag in s["models"]:
            macro(f"EnineDeriv{key}", f"{s['models'][tag]['deriv_err_all']:.3f}")


def e10(run):
    s = load("e10_confirm", run)
    if not s:
        return
    names = ["test NRMSE (all states)", "dx/dt error vs truth", "10-step error, in-domain", "50-step error, in-domain",
             "50-step error, extrapolation"]
    label = {"test NRMSE (all states)": "one-step test error", "dx/dt error vs truth": "$\\dot s$ error",
             "10-step error, in-domain": "10-step error", "50-step error, in-domain": "50-step error",
             "50-step error, extrapolation": "50-step error (extrap.)"}
    rows = []
    for n in ("20000", "100"):
        for i, k in enumerate(names):
            o = s["results"][n][k]
            nlab = {"20000": "20\\,000", "100": "100"}[n]
            pv = o["t_p"]
            rows.append([(nlab if i == 0 else ""), label[k], sci(o["pinc_ci"]["mean"]), sci(o["bb_ci"]["mean"]),
                         f"{o['ratio_geomean']:.1f}", f"{o['pinc_better']}/5", (f"{pv:.3f}" if pv >= 1e-3 else sci(pv, 1))])
    write("tab_e10.tex", tabular("llccccc", ["$N$", "metric", "PINC", "data-only", "ratio", "PINC better", "$p$"], rows,
                                 f"PINC ($\\lambda = {s['lam_star']:g}$) against the same network trained on data only, five seeds per training-set "
                                 "size $N$. Ratio: geometric mean of data-only / PINC; $p$: paired $t$-test on log values.", "tab:e10", wide=True))
    for n, key in (("20000", "Big"), ("100", "Small")):
        r = s["results"][n]
        macro(f"EtenTest{key}", ratio(r['test NRMSE (all states)']['ratio_geomean']))
        macro(f"EtenDeriv{key}", f"{r['dx/dt error vs truth']['ratio_geomean']:.0f}")
        macro(f"EtenHfifty{key}", f"{r['50-step error, in-domain']['ratio_geomean']:.1f}")
        macro(f"EtenExtrap{key}", f"{r['50-step error, extrapolation']['ratio_geomean']:.1f}")
        macro(f"EtenPmax{key}", f"{max(v['t_p'] for v in r.values()):.3f}")


def tf_stable(check, model, e10_run, e3_run):
    """Software-version facts: the default model retrained with the stable TensorFlow release, the
    relative change of its validation data loss, the spread of that loss over the E10 training seeds,
    and the TensorFlow version of the closed-loop run."""
    p = os.path.join(RESULTS_DIR, check)
    if not os.path.exists(os.path.join(p, "summary.json")):
        return
    with open(os.path.join(p, "summary.json")) as fh:
        new = json.load(fh)
    with open(os.path.join(p, "meta.json")) as fh:
        macro("CfgTfStable", json.load(fh)["versions"]["tensorflow"])
    with open(os.path.join(RESULTS_DIR, "models", model, "summary.json")) as fh:
        old = json.load(fh)
    macro("TfStableDiff", f"{100*abs(new['val']['data']/old['val']['data'] - 1):.1f}")
    s = load("e10_confirm", e10_run)
    if s:
        v = s["results"]["20000"]["validation data loss"]["pinc"]
        macro("TfSeedSpread", f"{100*(max(v)/min(v) - 1):.1f}")
        macro("TfSeedCount", str(len(v)))
    mp = os.path.join(RESULTS_DIR, "e3_closed_loop", e3_run, "meta.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            macro("EthreeTf", json.load(fh)["versions"]["tensorflow"])


def main():
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    e1(a.e1); e2(a.e2); e3(a.e3); e4(a.e4); e5(a.e5); e6(a.e6); e6_no_scaling(a.e6nod); e7(a.e7); e9(a.e9); e10(a.e10)
    # model / config facts
    cfgp = os.path.join(RESULTS_DIR, "models", a.model, "config.json")
    if os.path.exists(cfgp):
        with open(cfgp) as fh:
            c = json.load(fh)
        with open(os.path.join(RESULTS_DIR, "models", a.model, "summary.json")) as fh:
            sm = json.load(fh)
        with open(os.path.join(RESULTS_DIR, "models", a.model, "meta.json")) as fh:
            meta = json.load(fh)
        macro("CfgDepth", str(c["model"]["depth"])); macro("CfgWidth", str(c["model"]["width"]))
        macro("CfgLam", f"{c['loss']['lam']:g}"); macro("CfgNdata", f"{c['train']['n_data']:,}".replace(",", "\\,"))
        macro("CfgNval", f"{c['train']['n_val']:,}".replace(",", "\\,")); macro("CfgNcolloc", f"{c['train']['n_colloc']:,}".replace(",", "\\,"))
        macro("CfgEpochs", str(c["train"]["epochs"])); macro("CfgLbfgs", str(c["train"]["lbfgs_iters"])); macro("CfgLr", f"{c['train']['lr']:g}")
        macro("CfgBatch", str(c["train"]["batch_data"])); macro("CfgT", f"{c['T']:g}"); macro("CfgN", str(c["mpc"]["N"]))
        macro("CfgParams", f"{sm['n_params']:,}".replace(",", "\\,")); macro("CfgTrainMin", f"{sm['train_seconds']/60:.0f}")
        macro("CfgCpu", meta["cpu"].replace("(R)", "").replace("(TM)", "")); macro("CfgTf", meta["versions"]["tensorflow"])
        macro("CfgGit", meta["git"][:10]); macro("CfgSf", ", ".join(f"{v:g}" for v in c["scales"]["S_f"]))
        macro("CfgNoise", ", ".join(f"{v:g}" for v in c["sim"]["noise_sigma"]))
        macro("CfgQ", ", ".join(f"{v:g}" for v in c["mpc"]["Q"])); macro("CfgR", ", ".join(f"{v:g}" for v in c["mpc"]["R"]))
        macro("CfgRd", ", ".join(f"{v:g}" for v in c["mpc"]["R_delta"])); macro("CfgRmax", f"{c['mpc']['r_max']:g}"); macro("CfgWr", f"{c['mpc']['w_rmax']:g}")
    tf_stable(a.tfcheck, a.model, a.e10, a.e3)
    txt = "% generated by scripts/make_paper_tables.py -- do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    write("numbers.tex", txt)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- paper figures (readable labels)
def paper_figures(a):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    C = dict(nmpc_rk4="#1f77b4", pinc="#d62728", blackbox="#2ca02c", ltv="#9467bd", rk4="#1f77b4", linear="#9467bd")
    L = dict(nmpc_rk4="NMPC-RK4", pinc="PINC", blackbox="Black-box", ltv="LTV", rk4="RK4 ($\\Delta t$ = 0.01 s)", linear="Linearised (20 m/s)")

    def save(fig, name):
        fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print("  figure", name)

    s = load("e1_open_loop", a.e1)
    if s:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
        for ax, (reg, title) in zip(axes, (("in_box", "In-domain initial states"), ("extrap", "Extrapolation ($v_x$ = 25–30 m/s)"))):
            for arm in ("pinc", "blackbox", "linear", "rk4"):
                c = s[reg]["chain"][arm]["curve"]
                h = sorted(int(k) for k in c)
                m = np.array([c[str(k)]["all"]["mean"] for k in h])
                lo = np.array([c[str(k)]["all"]["lo"] for k in h]); hi = np.array([c[str(k)]["all"]["hi"] for k in h])
                ax.plot(h, m, color=C[arm], label=L[arm]); ax.fill_between(h, lo, hi, color=C[arm], alpha=0.2)
            ax.set(yscale="log", xlabel="prediction steps ($\\times T$)", title=title)
        axes[0].set_ylabel("NRMSE (all states)"); axes[0].legend(fontsize=7)
        save(fig, "fig_e1_horizon")

    s = load("e2_data_efficiency", a.e2)
    if s:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        for arm, lab in (("pinc", f"PINC ($\\lambda$ = {s['lam']:g})"), ("blackbox", "Black-box ($\\lambda$ = 0)")):
            n = [int(k) for k in s["sizes"]]
            cc = [s["curves"][arm][str(k)]["all"] for k in n]
            m = np.array([c["mean"] for c in cc])
            ax.errorbar(n, m, yerr=[m - [c["lo"] for c in cc], np.array([c["hi"] for c in cc]) - m], marker="o", capsize=3, color=C[arm], label=lab)
        ax.set(xscale="log", yscale="log", xlabel="training trajectories", ylabel="test NRMSE (all states)"); ax.legend(fontsize=7)
        save(fig, "fig_e2_nrmse")

    names = dict(speed_sin="speed sinusoid", speed_step="speed step", lane_change="single lane change", double_lane_change="double lane change (ISO 3888-2)")
    for ref in names:
        p = {arm: os.path.join(RESULTS_DIR, "e3_closed_loop", a.e3, f"log_{ref}_{arm}_s0.npz") for arm in ARMS}
        if not all(os.path.exists(v) for v in p.values()):
            continue
        logs = {arm: np.load(v) for arm, v in p.items()}
        t = logs["nmpc_rk4"]["t"]; ref_z = logs["nmpc_rk4"]["ref"]
        fig, axes = plt.subplots(5, 1, figsize=(6.2, 8), sharex=True)
        for i, (j, lab) in enumerate(((0, "$v_x$ [m/s]"), (3, "$\\psi$ [rad]"), (5, "$Y$ [m]"))):
            axes[i].plot(t, ref_z[:, j], "k--", lw=1, label="reference")
            for arm in ARMS:
                axes[i].plot(t, logs[arm]["x"][:, j], color=C[arm], lw=1, label=L[arm])
            axes[i].set_ylabel(lab)
        for i, (j, lab) in enumerate(((0, "$F_x$ [N]"), (1, "$\\delta$ [rad]")), start=3):
            for arm in ARMS:
                axes[i].step(t[:-1], logs[arm]["u"][:, j], where="post", color=C[arm], lw=1)
            axes[i].set_ylabel(lab)
        axes[0].legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.35))
        axes[-1].set_xlabel("time [s]")
        save(fig, f"fig_e3_{ref}")

    s = load("e4_timing", a.e4)
    if s:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        Ns = [int(n) for n in s["horizons"]]
        for arm in ARMS:
            ax.plot(Ns, [s["solve"][arm][str(n)]["median"]*1e3 for n in Ns], marker="o", color=C[arm], label=L[arm])
            ax.plot(Ns, [s["solve"][arm][str(n)]["p95"]*1e3 for n in Ns], ":", color=C[arm])
        ax.set(xlabel="horizon $N$", ylabel="solve time per step [ms]", yscale="log"); ax.legend(fontsize=7)
        save(fig, "fig_e4_solve")

    s = load("e5_robustness", a.e5)
    if s:
        def lab(n):
            return (n.replace("fiala mu=", "Fiala $\\mu$=").replace("Ca", "$C_\\alpha$").replace("Fx step -1000 N @ 4 s", "$-1$ kN force step")
                    .replace("%", "\\%").replace("m-", "mass $-$").replace("m+", "mass +"))
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
        xs = np.arange(len(s["settings"]))
        for ax, (k, t_) in zip(axes, (("rmse_Y", "lateral RMSE [m]"), ("rmse_psi", "heading RMSE [rad]"), ("rmse_vx", "speed RMSE [m/s]"))):
            for i, arm in enumerate(ARMS):
                cc = [s["per_setting"][n][arm][k] for n in s["settings"]]
                m = np.array([c["mean"] for c in cc])
                ax.errorbar(xs + 0.12*(i - 1.5), m, yerr=[m - [c["lo"] for c in cc], np.array([c["hi"] for c in cc]) - m],
                            fmt="o", ms=3, capsize=2, color=C[arm], label=L[arm])
            ax.set_xticks(xs); ax.set_xticklabels([lab(n) for n in s["settings"]], rotation=60, ha="right", fontsize=7); ax.set_title(t_)
        axes[0].legend(fontsize=7)
        save(fig, "fig_e5_degradation")


if __name__ == "__main__" and True:
    _ap = argparse.ArgumentParser()
    add_run_args(_ap)
    paper_figures(_ap.parse_args())
