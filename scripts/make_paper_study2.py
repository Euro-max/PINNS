"""Generate the Study 2 (and anchored-network) tables, figures and number macros for LATEX/main.tex from
results/ (no hand-typed numbers).  Companion of scripts/make_paper_tables.py, whose helpers it reuses; writes
LATEX/generated/numbers_s2.tex, tab_s*.tex, fig_s*.pdf and macro_index_s2.md (a list of every macro and its
value, for writing).

    python scripts/make_paper_study2.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_tables import OUT, sci, sig2, tabular, write  # noqa: E402
from pinc.config import RESULTS_DIR, ROOT  # noqa: E402
from pinc.metrics import bootstrap_ci  # noqa: E402

macros = {}
WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
VARIANTS = ("m0", "m1")
VTAG = dict(m0="Mzero", m1="Mone")
NTAG = {100: "Hundred", 1000: "Thousand", 20000: "TwentyK"}
ATAG = {"data-only": "Data", "PINC": "Pinc", "PINC-theta": "PincTheta", "grey-box": "Grey", "grey-box-qs": "GreyQs",
        "distilled": "Distil", "A8 data-only": "AnchData", "A8 PINC": "Anch", "A8 PINC-theta": "AnchTheta",
        "PINC-ablation": "PincAbl"}
MTAG = dict(one_step="One", h10="Ten", h50="Fifty", h50_extrap="Extrap")
LABEL = {"data-only": "data-only network", "PINC": "PINC network", "PINC-theta": "PINC network, learned $\\theta$",
         "grey-box": "grey-box, full prior", "grey-box-qs": "grey-box, quasi-steady prior", "distilled": "distilled network",
         "A8 data-only": "anchored data-only network", "A8 PINC": "anchored PINC network",
         "A8 PINC-theta": "anchored PINC network, learned $\\theta$",
         "PINC-ablation": "PINC network, $\\lambda = 10^{-3}$ (ablation)"}


def load(exp, run):
    with open(os.path.join(RESULTS_DIR, exp, run, "summary.json")) as fh:
        return json.load(fh)


def macro(name, value):
    if not name.isalpha():
        raise ValueError(f"macro name must be letters only: {name}")
    macros[name] = str(value)


def pval(p):
    """p-value as bare math (no dollar signs): 0.0064 or 8\\times10^{-4}."""
    return "--" if p != p else (sci(p, 1).strip("$") if p < 1e-3 else f"{p:.2g}")


def compare(a, b):
    """a against b, both per-seed errors of paired models (lower is better).  ratio: geometric mean of b/a over
    seeds, consistent with the paired t-test on log errors; wins: seeds where a < b."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.log(b) - np.log(a)
    p = stats.ttest_rel(np.log(a), np.log(b)).pvalue if len(a) > 1 else np.nan
    return dict(ratio=float(np.exp(np.mean(d))), wins=int(np.sum(a < b)), n=int(len(a)), p=float(p))


def cmp_macros(name, c):
    """<name>Ratio (b/a), <name>Factor and <name>Dir (a factor above one with its direction: lower = a better),
    <name>Seeds (seeds agreeing with that direction), <name>N, <name>P, and <name>Wins (seeds where a < b)."""
    up = c["ratio"] >= 1.0
    macro(name + "Ratio", sig2(c["ratio"]))
    macro(name + "Factor", sig2(c["ratio"] if up else 1.0/c["ratio"]))
    macro(name + "Dir", "lower" if up else "higher")
    macro(name + "Seeds", str(c["wins"] if up else c["n"] - c["wins"]))
    macro(name + "N", str(c["n"]))
    macro(name + "SeedsWord", WORDS[c["wins"] if up else c["n"] - c["wins"]])      # prose: seeds as words up to ten
    macro(name + "NWord", WORDS[c["n"]])
    macro(name + "Wins", f"{c['wins']}/{c['n']}")
    macro(name + "P", pval(c["p"]))


def gmean(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def geo_ci(v):
    """Geometric mean of positive per-seed values with a 95 % bootstrap interval (bootstrap on the logs)."""
    c = bootstrap_ci(np.log(np.asarray(v, float)))
    return dict(mean=float(np.exp(c["mean"])), lo=float(np.exp(c["lo"])), hi=float(np.exp(c["hi"])))


def ci_cell(v):
    c = geo_ci(v)
    e = int(np.floor(np.log10(abs(c["mean"]))))
    f = lambda x: f"{x/10**e:.1f}"
    return f"${f(c['mean'])}\\,[{f(c['lo'])}, {f(c['hi'])}]\\times10^{{{e}}}$"


# ---------------------------------------------------------------- per-seed values of every learned arm, pooled
E28_NAMES = {"anchored PINC": "A8 PINC", "anchored data-only": "A8 data-only"}


def pooled_main():
    """Main results (E28, seeds 5-9, never used for any selection): arms[variant][N][arm][metric] = list over seeds.
    Where the selected lambda is 0 the PINC network is the data-only network; it is then left out and the
    lambda = 1e-3 run appears as 'PINC-ablation'."""
    out, lams = {}, {}
    for v in VARIANTS:
        s = load("e28_main_fresh", f"e28_{v}")
        out[v], lams[v] = {}, {int(n): l for n, l in s["lambdas"].items()}
        for n, arms in s["results"].items():
            reg = s["registry"][n]
            for arm, val in arms.items():
                if arm == "PINC" and reg["PINC"] == reg["data-only"]:
                    continue
                out[v].setdefault(int(n), {})[E28_NAMES.get(arm, arm)] = {m: val[m] for m in MTAG}
    return out, lams


def pooled():
    """Secondary results on seeds 0-4 (E17, E19, E20, E22, E23): the distilled network and the learned physics
    parameters; compared only with arms of the same seeds.  arms[variant][N][arm][metric] = list over seeds."""
    out = {}
    for v in VARIANTS:
        out[v] = {}
        sources = [("e17_hf_compare", f"e17_{v}", None), ("e20_hf_greybox_qs", f"e20_{v}", ["grey-box-qs"]),
                   ("e22_anchored_confirm", f"e22_{v}", None), ("e19_hf_distill", f"e19_{v}", ["distilled"])]
        if v == "m0":
            sources.append(("e23_anchored_extend", "e23_n20k", ["A8 PINC", "A8 data-only"]))
        for exp, run, keep in sources:
            for n, arms in load(exp, run)["results"].items():
                for arm, val in arms.items():
                    if " vs " in arm or " learned" in arm or (keep and arm not in keep) or not isinstance(val, dict):
                        continue
                    if not all(isinstance(val.get(m), list) for m in MTAG):
                        continue
                    out[v].setdefault(int(n), {})[arm] = {m: val[m] for m in MTAG}
    return out


PAIRS = [("PINC-ablation", "data-only"), ("A8 PINC", "PINC-ablation"), ("PINC", "data-only"), ("grey-box", "data-only"), ("grey-box-qs", "data-only"), ("PINC", "grey-box"),
         ("PINC", "grey-box-qs"), ("A8 PINC", "PINC"), ("A8 PINC", "A8 data-only"), ("A8 PINC", "data-only"),
         ("A8 data-only", "data-only"), ("A8 PINC", "grey-box-qs"), ("A8 PINC", "grey-box"), ("grey-box-qs", "grey-box"),
         ("PINC-theta", "data-only"), ("PINC-theta", "PINC"), ("A8 PINC-theta", "A8 PINC"), ("A8 PINC-theta", "PINC-theta"),
         ("A8 PINC-theta", "data-only"), ("A8 PINC-theta", "grey-box-qs"), ("distilled", "PINC"), ("distilled", "data-only"),
         ("distilled", "grey-box"), ("distilled", "A8 PINC")]


SECONDARY = ("PINC-theta", "A8 PINC-theta", "distilled")


def comparisons(arms, prefix=""):
    """Macros \\<prefix>Cmp<V><N><X>Vs<Y><metric>{Ratio,Wins,P}: error of Y divided by error of X (above 1: X
    better).  Main results without prefix (E28, seeds 5-9); prefix 'Sec' for the secondary arms (seeds 0-4), whose
    comparisons only involve arms of the same seeds."""
    for v, byn in arms.items():
        for n, a in byn.items():
            for arm, val in a.items():
                for m, t in MTAG.items():
                    macro(f"{prefix}Val{VTAG[v]}{NTAG[n]}{ATAG[arm]}{t}", sci(float(gmean(val[m]))).strip("$"))
            for x, y in PAIRS:
                if prefix and not (x in SECONDARY or y in SECONDARY):
                    continue
                if x in a and y in a:
                    for m, t in MTAG.items():
                        cmp_macros(f"{prefix}Cmp{VTAG[v]}{NTAG[n]}{ATAG[x]}Vs{ATAG[y]}{t}", compare(a[x][m], a[y][m]))


def table_main(arms):
    order = ["data-only", "PINC", "PINC-ablation", "A8 data-only", "A8 PINC", "grey-box-qs", "grey-box"]
    for v in VARIANTS:
        rows = []
        for n in sorted(arms[v]):
            first = True
            for arm in [a for a in order if a in arms[v][n]]:
                val = arms[v][n][arm]
                rows.append([f"{n:,}".replace(",", "\\,") if first else "", LABEL[arm]] + [ci_cell(val[m]) for m in ("one_step", "h10", "h50")])
                first = False
        cap = (f"Test error of the learned models on {v.upper()} (NRMSE of the body states $v_x, v_y, r, \\psi$), geometric mean [95\\,\\% "
               "confidence interval] over five training seeds (5 to 9, not used in any selection); one step and chained predictions "
               "of 10 and 50 control periods. $N$: training trajectories. Physics weights as selected on the validation set "
               "(Table~\\ref{tab:lam-sel}).")
        write(f"tab_s2_main_{v}.tex", tabular("llccc", ["$N$", "model", "one step", "10 steps", "50 steps"], rows, cap,
                                               f"tab:s2-main-{v}", wide=True))


# ---------------------------------------------------------------- Study 1 with the anchored network (E23)
def study1():
    s = load("e23_anchored_extend", "e23_study1")["results"]
    t = {"MLP data-only": "MlpData", "MLP PINC": "MlpPinc", "A8 data-only": "AnchData", "A8 PINC": "Anch"}
    mt = dict(one_step="One", deriv="Deriv", h10="Ten", h50="Fifty", h50_extrap="Extrap")
    rows = []
    for n in ("20000", "100"):
        a = s[n]
        for arm in t:
            for m, tag in mt.items():
                macro(f"SoneVal{NTAG[int(n)]}{t[arm]}{tag}", sci(float(gmean(a[arm][m]))).strip("$"))
        for x, y in (("A8 PINC", "MLP PINC"), ("A8 PINC", "A8 data-only"), ("MLP PINC", "MLP data-only"), ("A8 data-only", "MLP data-only")):
            for m, tag in mt.items():
                cmp_macros(f"SoneCmp{NTAG[int(n)]}{t[x]}Vs{t[y]}{tag}", compare(a[x][m], a[y][m]))
        for i, m in enumerate(("one_step", "deriv", "h10", "h50", "h50_extrap")):
            c = compare(a["A8 PINC"][m], a["MLP PINC"][m])
            rows.append([(f"{int(n):,}".replace(",", "\\,") if i == 0 else ""),
                         {"one_step": "one-step test error", "deriv": "derivative error", "h10": "10-step error",
                          "h50": "50-step error", "h50_extrap": "50-step error, extrapolation"}[m]] +
                        [sci(float(gmean(a[arm][m]))) for arm in t] + [sig2(c["ratio"]), f"{c['wins']}/{c['n']}", f"${pval(c['p'])}$"])
    write("tab_s1_anchor.tex", tabular("llccccccc", ["$N$", "metric", "MLP, data only", "MLP PINC", "anchored, data only",
                                                       "anchored PINC", "ratio", "better", "$p$"], rows,
                                       "Study 1 (exact physics): the anchored network against the MLP of Table~\\ref{tab:e10}, with and without the "
                                       "physics loss, mean over five seeds. Ratio: error of the MLP PINC divided by the error of the anchored PINC; "
                                       "better: seeds in which the anchored PINC was better; $p$: paired $t$-test on log values.",
                                       "tab:s1-anchor", wide=True))


# ---------------------------------------------------------------- Study 1 rerun with the Study 2 protocol (E27, E18 st, E4)
S1_ARMS = ("data-only", "PINC", "anchored data-only", "anchored PINC")
S1_TAG = {"data-only": "Data", "PINC": "Pinc", "anchored data-only": "AnchData", "anchored PINC": "Anch",
          "NMPC-exact": "NmpcExact", "LTV": "Ltv"}
S1_N = {100: "Hundred", 1000: "Thousand", 10000: "TenK", 20000: "TwentyK"}
S1_MT = dict(one_step="One", deriv="Deriv", h10="Ten", h50="Fifty", h50_extrap="Extrap")


def study1_rerun():
    """Study 1 (single-track vehicle, exact physics) under the Study 2 protocol: lambda selected on validation (E27,
    seeds 0-2), main comparison on seeds 5-9 (E27), closed loop (E18 --variant st) and solve time (E4)."""
    lam = load("e27_study1_rerun", "e27_lambda")
    fl = lambda l: f"{l:g}" if l >= 0.1 or l == 0 else f"10^{{{int(round(np.log10(l)))}}}"
    for k, l in lam["best"].items():
        net, n = k.split("_")
        macro(f"SoneLamSel{'Plain' if net == 'plain' else 'Anch'}{S1_N[int(n)]}", fl(float(l)))
    for k, d in lam["results"].items():                      # validation 50-step error at lambda = 0 and at the selection
        net, n = k.split("_")
        tag = f"{'Plain' if net == 'plain' else 'Anch'}{S1_N[int(n)]}"
        best = str(lam["best"][k]) if str(lam["best"][k]) in d else f"{float(lam['best'][k]):g}"
        macro(f"SoneLamVal{tag}Zero", sci(float(np.mean(d["0"]))).strip("$"))
        macro(f"SoneLamVal{tag}Sel", sci(float(np.mean(d[best]))).strip("$"))
        macro(f"SoneLamGain{tag}", f"{np.mean(d['0'])/np.mean(d[best]):.0f}")
    m = load("e27_study1_rerun", "e27_main")
    res, lams = m["results"], m["lambdas"]
    rows = []
    for n in sorted(res, key=int):
        a = res[n]
        for arm in S1_ARMS:
            for mt, tag in S1_MT.items():
                macro(f"SoneVal{S1_N[int(n)]}{S1_TAG[arm]}{tag}", sci(float(gmean(a[arm][mt]))).strip("$"))
            l = 0.0 if "data-only" in arm else lams["anchored" if "anchored" in arm else "plain"][n]
            rows.append([f"{int(n):,}".replace(",", "\\,") if arm == S1_ARMS[0] else "", arm.replace("PINC", "PINC network").replace("data-only", "data-only network"),
                         f"${fl(float(l))}$"] + [ci_cell(a[arm][mt]) for mt in ("one_step", "deriv", "h10", "h50")])
        for x, y in (("PINC", "data-only"), ("anchored PINC", "anchored data-only"), ("anchored PINC", "PINC"),
                     ("anchored data-only", "data-only"), ("anchored PINC", "data-only")):
            for mt, tag in S1_MT.items():
                cmp_macros(f"SoneCmp{S1_N[int(n)]}{S1_TAG[x]}Vs{S1_TAG[y]}{tag}", compare(a[x][mt], a[y][mt]))
    write("tab_s1_main.tex", tabular("lllcccc", ["$N$", "model", "$\\lambda$", "one step", "derivative", "10 steps", "50 steps"], rows,
                                     "Study 1 (single-track vehicle, exact physics): test error (NRMSE of all four states), geometric mean "
                                     "[95\\,\\% confidence interval] over five training seeds (5 to 9). Derivative: error of the network's "
                                     "time derivative against the true dynamics, in units of $S_f$. Physics weights selected on the validation "
                                     "set with seeds 0 to 2.", "tab:s1-main", wide=True))
    # closed loop, against NMPC with the exact model
    R = cl_runs("e18_s1")
    rows = []
    for arm, n in [("NMPC-exact", 0), ("LTV", 0)] + [(a_, n_) for n_ in (100, 20000) for a_ in S1_ARMS]:
        runs = [x for ref in CL_REFS for x in R[(ref, arm, n)]]
        lost = sum(x["max_Y"] > 1.0 for x in runs)
        med = {mt: geo_ci(list(cl_metric(R, arm, n, mt).values()))["mean"] for mt in ("lane", "sin", "iso")}
        tag = f"SoneCl{S1_TAG[arm]}" + (S1_N[n] if n else "")
        macro(tag + "Lost", str(lost)); macro(tag + "Lane", f"{med['lane']:.3f}"); macro(tag + "Sin", f"{med['sin']:.3f}"); macro(tag + "Iso", f"{med['iso']:.2f}")
        cells = []
        if n:
            for mt, t in (("lane", "Lane"), ("sin", "Sin"), ("iso", "Iso")):
                c = cl_compare(R, (arm, n), ("NMPC-exact", 0), mt)
                cmp_macros(f"SoneClCmp{S1_N[n]}{S1_TAG[arm]}VsNmpcExact{t}", c)
                cells.append(f"{sig2(c['ratio'])} ({c['wins']}/{c['n']})")
        rows.append([arm if not n else arm.replace("PINC", "PINC network").replace("data-only", "data-only network"),
                     "--" if n == 0 else f"{n:,}".replace(",", "\\,"), f"{lost}/{len(runs)}",
                     f"{med['lane']:.3f}", f"{med['sin']:.3f}", f"{med['iso']:.2f}"] + (cells if cells else ["--"]*3))
    write("tab_s1_closed.tex", tabular("llcccc|ccc", ["controller", "$N$", "lost", "lane $Y$", "sin. $v_x$", "ISO peak",
                                                        "lane", "sin.", "ISO"], rows,
                                       "Study 1 closed loop on the single-track vehicle with exact physics. Columns as in "
                                       "Table~\\ref{tab:s2-closed}; right: NMPC with the exact model divided by the controller's error "
                                       "(above 1: the controller is better), geometric mean over training seeds, seeds better.",
                                       "tab:s1-closed", wide=True))
    # solve time
    t1 = {"NMPC-exact": ("e4_s1", "nmpc_rk4"), "LTV": ("e4_s1", "ltv"), "PINC network": ("e4_s1", "pinc"),
          "data-only network": ("e4_s1", "blackbox"), "anchored PINC network": ("e4_s1_anchored", "pinc")}
    rows = []
    for lab, (run, arm) in t1.items():
        sv = load("e4_timing", run)["solve"][arm]
        rows.append([lab] + [f"{sv[str(n)]['median']*1e3:.1f} [{sv[str(n)]['p95']*1e3:.1f}]" for n in (5, 10, 20, 40)])
        for n, w in ((10, "Ten"), (40, "Forty")):
            macro(f"SoneSolve{''.join(x.capitalize() for x in lab.replace('-', ' ').split()[:2])}{w}", f"{sv[str(n)]['median']*1e3:.1f}")
    write("tab_s1_timing.tex", tabular("lcccc", ["controller", "$N_\\mathrm{h}=5$", "10", "20", "40"], rows,
                                       "Study 1 solve time per step [ms]: median [95th percentile], one CPU thread with the GPU hidden, "
                                       "learned models trained on 1000 trajectories (seed 5); conditions as in Table~\\ref{tab:s2-timing}.",
                                       "tab:s1-timing"))


# ---------------------------------------------------------------- Blockset vehicle: E30 (trained on it) and E29 (transfer)
VB_ARMS = ("data-only", "anchored data-only", "anchored PINC", "grey-box-qs")
VB_TAG = {"data-only": "Data", "anchored data-only": "AnchData", "anchored PINC": "Anch", "grey-box-qs": "GreyQs",
          "PINC": "Pinc", "grey-box": "Grey"}
VB_LABEL = {"data-only": "data-only network", "anchored data-only": "anchored data-only network",
            "anchored PINC": "anchored PINC network", "grey-box-qs": "grey-box, quasi-steady prior"}


def blockset():
    """E30 (main text): trained and tested on the Blockset 14-DOF vehicle; E29 (appendix): trained on our plant."""
    e30 = load("e30_vdbs_retrain", "e30_eval")
    e29 = load("e29_vdbs_transfer", "e29_eval")
    chk = load("e30_vdbs_retrain", "e30_assemble")["checks"]
    cell = lambda c: f"{sig2(c['ratio'])} ({c['wins']}/5)"
    macro("VbPriorOneStep", f"{chk['prior_qs_one_step_body']:.4f}")
    macro("VbPriorFifty", f"{e30['prior']['50']:.3f}"); macro("VbPriorTen", f"{e30['prior']['10']:.3f}")
    macro("VbPriorOne", f"{e30['prior']['1']:.4f}")
    # the same metric (E9: mean over test starting states of the body-state RMS) for the prior on M0 and M1 (E31)
    for v in VARIANTS:
        pq = load("e31_speed_prediction", f"e31_priors_{v}")["priors"]["quasi-steady prior"]
        for h, t in (("1", "One"), ("10", "Ten"), ("50", "Fifty")):
            macro(f"PriorTest{VTAG[v]}{t}", f"{pq[h]:.4f}")
    m0 = load("e31_speed_prediction", "e31_priors_m0")["priors"]["quasi-steady prior"]
    macro("VbPriorOverMzeroOne", f"{e30['prior']['1']/m0['1']:.1f}")
    macro("VbPriorOverMzeroTen", f"{e30['prior']['10']/m0['10']:.1f}")
    rows = [["simplified physics alone", "--", f"{e30['prior']['50']:.3f}", "--", "--", f"{e30['prior']['50']:.3f}"]]
    for n in ("100", "1000"):
        for arm in VB_ARMS:
            r, c = e30["results"][n][arm], e30["comparisons"][n][arm]
            tag = f"Vb{VB_TAG[arm]}{NTAG[int(n)]}"
            macro(tag + "Fifty", sci(float(gmean(r["err"]["50"]))).strip("$"))
            macro(tag + "TransferFifty", sci(float(gmean(r["e29_h50"]))).strip("$"))
            cmp_macros(tag + "VsPrior", dict(ratio=c["vs_prior"]["50"]["ratio"], wins=c["vs_prior"]["50"]["wins"], n=5, p=c["vs_prior"]["50"]["p"]))
            cmp_macros(tag + "VsTransfer", dict(c["vs_e29"], n=5))
            if c["vs_data"]:
                cmp_macros(tag + "VsData", dict(c["vs_data"], n=5))
            rows.append([VB_LABEL[arm], f"{int(n):,}".replace(",", "\\,"), ci_cell(r["err"]["50"]), cell(c["vs_prior"]["50"]),
                         cell(c["vs_data"]) if c["vs_data"] else "--", ci_cell(r["e29_h50"])])
        for k, v in e30["comparisons"][n].items():
            if " vs " in k:
                x, y = k.split(" vs ")
                for h, t in (("1", "One"), ("10", "Ten"), ("50", "Fifty")):
                    cmp_macros(f"VbCmp{NTAG[int(n)]}{VB_TAG[x]}Vs{VB_TAG[y]}{t}", dict(v[h], n=5))
    write("tab_vb_main.tex", tabular("llcccc", ["model", "$N$", "50 steps", "vs prior", "vs data-only", "trained on our plant"], rows,
                                     "Trained and tested on the Blockset 14-DOF vehicle (E30): 50-step test error of the body states, "
                                     "geometric mean [95\\,\\% confidence interval] over five training seeds (5 to 9). vs prior: error of "
                                     "the simplified physics alone divided by the model's error (above 1: the model is better), geometric "
                                     "mean over seeds and seeds better; vs data-only: the same against the data-only network. Last column: "
                                     "the same models trained on our double-track plant and tested on the Blockset vehicle without retraining "
                                     "(E29). Physics weights as selected on M0.", "tab:vb-main", wide=True))
    # E29 appendix table
    rows = []
    for name in ("our plant (M0)", "quasi-steady prior"):
        rf = e29["references"][name]
        rows.append([name, "--", f"{rf['10']['mean']:.3f}", f"{rf['50']['mean']:.3f}", "--"])
    for n in ("100", "1000", "20000"):
        for arm, r in e29["models"][n].items():
            vp = r["vs_prior"]["50"]
            tag = f"Tr{VB_TAG.get(arm, ATAG.get(E28_NAMES.get(arm, arm), arm.replace('-', '')))}{NTAG[int(n)]}"
            macro(tag + "Fifty", sci(float(gmean(r["h50"]))).strip("$"))
            cmp_macros(tag + "VsPrior", dict(ratio=vp["ratio"], wins=vp["wins"], n=5, p=vp["p"]))
            rows.append([LABEL.get(E28_NAMES.get(arm, arm), arm), f"{int(n):,}".replace(",", "\\,"), sci(float(gmean(r["h10"]))),
                         sci(float(gmean(r["h50"]))), f"{sig2(vp['ratio'])} ({vp['wins']}/5)"])
    write("tab_vb_transfer.tex", tabular("llccc", ["predictor", "$N$", "10 steps", "50 steps", "vs prior"], rows,
                                         "Models trained on our double-track plant (M0, seeds 5 to 9) and tested without retraining on "
                                         "the Blockset 14-DOF vehicle (E29); references are single predictors from the same starting states. "
                                         "Columns as in Table~\\ref{tab:vb-main}.", "tab:vb-transfer", wide=True))
    rs = e29["resistance"]
    macro("VbResistConst", f"{rs['blockset']['c0']:.0f}"); macro("VbResistQuad", f"{rs['blockset']['c2']:.3f}")
    macro("VbResistConstOurs", f"{rs['our_plant']['c0']:.0f}"); macro("VbResistQuadOurs", f"{rs['our_plant']['c2']:.3f}")
    macro("VbResistDiff", f"{rs['our_plant']['c0'] - rs['blockset']['c0']:.0f}")


def blockset_setup():
    """Setup facts of the Blockset vehicle: the M0-rule calibration (results/e29_vdbs_transfer/data/calibration_m0_rule.txt,
    output of scripts/vdbs/calibrate14.m) and a 0.03 rad steer step at 15 m/s against our plant (step_steer_test.mat)."""
    import re
    import scipy.io as sio
    from pinc import plant_hf
    from pinc.config import load_config
    d = os.path.join(RESULTS_DIR, "e29_vdbs_transfer", "data")
    txt = open(os.path.join(d, "calibration_m0_rule.txt")).read()
    lam = float(re.findall(r"final lam_Kya ([\d.]+)", txt)[0])
    cf, cr = [float(x) for x in re.findall(r"Cf (\d+), Cr (\d+)", txt)[-1]]
    macro("VbLamKya", f"{lam:.3f}"); macro("VbStiffFront", f"{cf/1e3:.1f}"); macro("VbStiffRear", f"{cr/1e3:.1f}")
    S = sio.loadmat(os.path.join(d, "step_steer_test.mat"), squeeze_me=True)
    cfg = load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"))
    p = plant_hf.make_params(cfg.params, "M0")
    x = plant_hf.free_rolling_state(15.0, p, F=200.0)
    for k in range(40):
        x = plant_hf.simulate(x, np.array([200.0, 0.0 if 0.1*k < 2 - 1e-9 else 0.03]), 0.1, plant_hf.DT_PLANT, p)
    r_ours, r_vb = float(x[2]), float(S["r"][-1])
    macro("VbYawGainLower", f"{100*(1 - r_vb/r_ours):.0f}")
    macro("VbStepSteer", "0.03"); macro("VbStepSpeed", "15")


# ---------------------------------------------------------------- T-IV floats (main_tiv.tex)
def tiv_table(cols, header, rows, caption, label, wide=False, scale=True):
    """IEEE floats: caption above; one column (table) or both (table*), scaled to fit when needed."""
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    tab = f"\\begin{{tabular}}{{{cols}}}\\toprule\n{' & '.join(header)} \\\\\\midrule\n{body} \\\\\n\\bottomrule\\end{{tabular}}"
    if scale:
        tab = f"\\resizebox{{{'\\textwidth' if wide else '\\columnwidth'}}}{{!}}{{{tab}}}"
    env = "table*" if wide else "table"
    return f"\\begin{{{env}}}[t]\\centering\\footnotesize\n\\caption{{{caption}}}\\label{{{label}}}\n{tab}\n\\end{{{env}}}\n"


def tiv_floats(arms):
    fl = lambda l: "--" if l is None else (f"${l:g}$" if l >= 0.1 or l == 0 else f"$10^{{{int(round(np.log10(l)))}}}$")
    # ---- Table I: physics weights
    s1 = load("e27_study1_rerun", "e27_main")["lambdas"]
    rows = [["single-track (exact)", "PINC"] + [fl(float(s1["plain"][str(n)])) for n in (100, 1000, 20000)],
            ["", "anchored PINC"] + [fl(float(s1["anchored"][str(n)])) for n in (100, 1000, 20000)]]
    for v in VARIANTS:
        lam = load("e28_main_fresh", f"e28_{v}")["lambdas"]
        rows.append([f"double-track, {v.upper()}", "PINC"] + [fl(float(lam[str(n)][0])) for n in (100, 1000, 20000)])
        rows.append(["", "anchored PINC"] + [fl(float(lam[str(n)][1])) for n in (100, 1000, 20000)])
    vb = json.load(open(os.path.join(RESULTS_DIR, "e30_vdbs_retrain", "e30_train", "registry.json")))["lambdas"]
    rows.append(["Blockset (M0 value)", "anchored PINC"] + [fl(float(vb.get(str(n)))) if str(n) in vb else "--" for n in (100, 1000, 20000)])
    sel_f = os.path.join(RESULTS_DIR, "e30_vdbs_retrain", "e30_lambda", "summary.json")
    if os.path.exists(sel_f):
        sel = json.load(open(sel_f))["selected"]
        rows.append(["Blockset, sensitivity$^\\dagger$", "anchored PINC"] + [fl(float(sel[str(n)])) if str(n) in sel else "--" for n in (100, 1000, 20000)])
    write("tiv_tab_lambda.tex", tiv_table("llccc", ["vehicle", "network", "$N=100$", "1000", "20\\,000"], rows,
          "Physics weight $\\lambda$, selected on the validation 50-step error with seeds 0 to 2 before the reporting runs "
          "(seeds 5 to 9). The Blockset vehicle uses the M0 values. $^\\dagger$Selected on the Blockset vehicle in a "
          "follow-up run; sensitivity analysis only (Section~\\ref{sec:res-blockset}).", "tab:lambda"))
    # ---- Table II: Study 1 compact
    m = load("e27_study1_rerun", "e27_main")["results"]
    rows = []
    for n in ("100", "20000"):
        for arm in S1_ARMS:
            rows.append([f"{int(n):,}".replace(",", "\\,") if arm == S1_ARMS[0] else "", arm.replace("anchored ", "anch.\\ ")] +
                        [sci(float(gmean(m[n][arm][k]))) for k in ("one_step", "h10", "h50")])
    write("tiv_tab_s1.tex", tiv_table("llccc", ["$N$", "network", "one step", "10 steps", "50 steps"], rows,
          "Exact physics (single-track vehicle): test error of the four states, geometric mean over five training seeds. "
          "Confidence intervals and all sizes are in the supplement.", "tab:s1"))
    # ---- Table III: closed loop compact (reference NMPC with the quasi-steady prior)
    R = {v: cl_runs(f"e18_s2_{v}") for v in VARIANTS}
    order = [("NMPC-true", 0), ("NMPC-qs", 0)] + [(a, n) for n in (100, 1000) for a in ("data-only", "PINC", "PINC-ablation", "anchored data-only", "anchored PINC", "grey-box-qs")]
    lab = {"NMPC-true": "NMPC, true model", "NMPC-qs": "NMPC, quasi-steady prior", "data-only": "data-only", "PINC": "PINC",
           "PINC-ablation": "PINC, $\\lambda=10^{-3}$", "anchored data-only": "anchored data-only", "anchored PINC": "anchored PINC",
           "grey-box-qs": "grey-box (q.-s.\\ prior)"}
    rows = []
    for arm, n in order:
        if not any((CL_LANES[0], arm, n) in R[v] for v in VARIANTS):
            continue
        r = [lab[arm], "--" if n == 0 else f"{n:,}".replace(",", "\\,")]
        for v in VARIANTS:
            if (CL_LANES[0], arm, n) not in R[v]:
                r += ["--"]*4
                continue
            runs = [x for ref in CL_REFS for x in R[v][(ref, arm, n)]]
            med = {mt: geo_ci(list(cl_metric(R[v], arm, n, mt).values()))["mean"] for mt in ("lane", "iso", "sin")}
            r += [str(sum(x["max_Y"] > 1.0 for x in runs)), f"{med['lane']:.3f}", f"{med['iso']:.2f}", f"{med['sin']:.3f}"]
        rows.append(r)
    write("tiv_tab_closed.tex", tiv_table("llcccc|cccc", ["controller", "$N$", "lost", "lane $Y$", "ISO $Y$", "$v_x$",
                                                            "lost", "lane $Y$", "ISO $Y$", "$v_x$"], rows,
          "Closed loop on the double-track vehicle, M0 (left) and M1 (right). Lost: runs with a lateral error above "
          "1\\,m, of 50 (ten for the true model). Lane $Y$: lateral RMSE [m], mean over the two lane changes and the ISO path; ISO $Y$: peak "
          "lateral error [m] on the ISO 3888-2 path; $v_x$: speed RMSE [m/s] on the speed sinusoid. Learned controllers: "
          "geometric mean over five training seeds, each the mean of two noise seeds; NMPC: over ten noise seeds (two for "
          "the true model). --: not run. On M1 the selected weight at $N = 1000$ is 0, so the PINC network is the data-only "
          "network and the PINC row is replaced by the ablation with $\\lambda = 10^{-3}$.", "tab:closed", wide=True))
    # ---- Table IV: Blockset vehicle (E30), with the sensitivity rows when available
    e30 = load("e30_vdbs_retrain", "e30_eval")
    cell = lambda c: f"{sig2(c['ratio'])} ({c['wins']}/5)"
    rows = [["simplified physics alone", "--"] + [f"{e30['prior'][h]:.3f}" for h in ("10", "50")] + ["--", "--", "--"]]
    sens_f = os.path.join(RESULTS_DIR, "e30_vdbs_retrain", "e30_lambda_main", "summary.json")
    sens = json.load(open(sens_f))["results"] if os.path.exists(sens_f) else {}
    for n in ("100", "1000"):
        for arm in VB_ARMS:
            r, c = e30["results"][n][arm], e30["comparisons"][n][arm]
            rows.append([VB_LABEL[arm].replace(" network", ""), f"{int(n):,}".replace(",", "\\,"),
                         sci(float(gmean(r["err"]["10"]))), sci(float(gmean(r["err"]["50"]))), cell(c["vs_prior"]["50"]),
                         cell(c["vs_data"]) if c["vs_data"] else "--", sci(float(gmean(r["e29_h50"])))])
        if n in sens:
            r = sens[n]
            cv = r["comparisons"]
            rows.append([f"anchored PINC, $\\lambda={r['lam']:g}^\\dagger$", f"{int(n):,}".replace(",", "\\,"),
                         sci(float(gmean(r["err"]["10"]))), sci(float(gmean(r["err"]["50"]))), cell(cv["vs prior"]["50"]),
                         cell(cv["vs data-only"]["50"]), "--"])
    write("tiv_tab_blockset.tex", tiv_table("llccccc", ["model", "$N$", "10 steps", "50 steps", "vs prior", "vs data-only",
                                                           "trained on our plant"], rows,
          "Trained and tested on the Blockset 14-DOF vehicle: test error of the body states, geometric mean over five "
          "training seeds. vs prior / vs data-only (50 steps): the other model's error divided by this model's, above 1 "
          "this model is better, with seeds better. Last column: 50-step error of the same models trained on our double-track "
          "plant, without retraining. Physics weights as on M0; $^\\dagger$weight selected on the Blockset vehicle "
          "(sensitivity analysis).", "tab:blockset", wide=True))


def tiv_fig_data(arms):
    """Fig. 2: 10- and 50-step test error against N on M0 and M1, with the quasi-steady prior as a reference line."""
    plt = _plt()
    pri = {v: load("e31_speed_prediction", f"e31_priors_{v}")["priors"]["quasi-steady prior"] for v in VARIANTS}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for j, v in enumerate(VARIANTS):
        for i, (m, h) in enumerate((("h10", "10"), ("h50", "50"))):
            ax = axes[i, j]
            for arm in ("data-only", "PINC", "PINC-ablation", "A8 data-only", "A8 PINC", "grey-box-qs", "grey-box"):
                ns = [n for n in sorted(arms[v]) if arm in arms[v][n]]
                if not ns:
                    continue
                cc = [geo_ci(arms[v][n][arm][m]) for n in ns]
                mm = np.array([c["mean"] for c in cc])
                col, mk, ls = STYLE.get(arm, ("#6b6b6b", "x", "-"))
                ax.errorbar(ns, mm, yerr=[mm - [c["lo"] for c in cc], np.array([c["hi"] for c in cc]) - mm], color=col, marker=mk,
                            ls=ls, capsize=2, label=LABEL[arm].replace(" network", ""), ms=4)
            ax.axhline(pri[v][h], color=INK, lw=0.9, ls="--", label="simplified physics alone")
            ax.set(xscale="log", yscale="log")
            ax.set_title(f"{v.upper()}, {h} steps", color=INK)
            if i == 1:
                ax.set_xlabel("training trajectories $N$")
            if j == 0:
                ax.set_ylabel("test error (body states)")
    seen = {}
    for ax in axes.flat:                                   # every arm once, whichever panel has it
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l_, h_)
    fig.legend(list(seen.values()), list(seen.keys()), fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "tiv_fig_data")
    plt.close(fig)


def tiv_fig_speed(arms):
    """Fig. 3: 10-step test error against MPC solve time at a horizon of 10 (M0 at N = 1000, M1 at N = 100)."""
    plt = _plt()
    s = {r: load("e4_timing", r)["solve"] for r in ("e4_s2", "e4_s2_anchored", "e4_s2_gbfull")}
    t10 = {"A8 PINC": s["e4_s2_anchored"]["pinc"], "A8 data-only": s["e4_s2_anchored"]["pinc"], "PINC": s["e4_s2"]["pinc"],
           "data-only": s["e4_s2"]["blackbox"], "grey-box-qs": s["e4_s2"]["greybox"], "grey-box": s["e4_s2_gbfull"]["greybox"],
           "NMPC-qs": s["e4_s2"]["nmpc_qs"], "NMPC-prior": s["e4_s2"]["nmpc_rk4"]}
    pri = {v: load("e31_speed_prediction", f"e31_priors_{v}")["priors"] for v in VARIANTS}
    OFFSETS = {("m0", "PINC"): ((6, 2), "left"), ("m0", "data-only"): ((-6, -2), "right"), ("m0", "A8 data-only"): ((6, 2), "left"),
               ("m0", "A8 PINC"): ((6, -6), "left"), ("m1", "A8 PINC"): ((-6, 0), "right"), ("m1", "A8 data-only"): ((6, -6), "left"),
               ("m1", "grey-box-qs"): ((6, 2), "left"), ("m1", "PINC"): ((6, 0), "left"), ("m1", "data-only"): ((6, 0), "left")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (v, n) in zip(axes, (("m0", 1000), ("m1", 100))):
        for arm, sv in t10.items():
            t = sv["10"]["median"]*1e3
            if arm.startswith("NMPC"):
                e = pri[v]["quasi-steady prior" if arm == "NMPC-qs" else "full prior"]["10"]
                col, mk = {"NMPC-qs": ("#6b6b6b", "P"), "NMPC-prior": ("#008300", "P")}[arm]
                ax.plot(t, e, color=col, marker=mk, ls="none", ms=6)
                ax.annotate({"NMPC-qs": "NMPC, q.-s. prior", "NMPC-prior": "NMPC, full prior"}[arm], (t, e), textcoords="offset points",
                            xytext=(5, 5), fontsize=6.5, color=INK)
                continue
            if arm not in arms[v].get(n, {}):
                continue
            c = geo_ci(arms[v][n][arm]["h10"])
            col, mk, _ = STYLE.get(arm, ("#6b6b6b", "x", "-"))
            ax.errorbar(t, c["mean"], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], color=col, marker=mk, capsize=2, ls="none", ms=5)
            off, ha = OFFSETS.get((v, arm), ((5, 3), "left"))
            ax.annotate(LABEL[arm].replace(" network", "").replace(", quasi-steady prior", ", q.-s."), (t, c["mean"]),
                        textcoords="offset points", xytext=off, fontsize=6.5, color=INK, ha=ha)
        ax.axvline(100, color=MUTED, lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", xlabel="solve time per step at $N_\\mathrm{h}=10$ [ms]")
        ax.set_title(f"{v.upper()}, $N$ = " + f"{n:,}".replace(",", "\u2009"), color=INK)
        ax.set_xlim(5, 1000)
    axes[0].set_ylabel("10-step test error")
    fig.tight_layout()
    _save(fig, "tiv_fig_speed")
    plt.close(fig)


# ---------------------------------------------------------------- prior error (E13, E25)
def prior():
    s = load("e13_prior_error", "e13_v2")["variants"]
    rows = []
    bins = list(s["m0"]["by_ay"])
    lab = {"0-1": "0--1", "1-2": "1--2", "2-4": "2--4", "4-6": "4--6", "6-inf": "$>6$"}
    for b in bins:
        r0, r1 = np.asarray(s["m0"]["by_ay"][b]["rms"]), np.asarray(s["m1"]["by_ay"][b]["rms"])
        rows.append([lab.get(b, b), str(s["m0"]["by_ay"][b]["n"])] + [f"{x:.2f}" for x in r0[[0, 1, 2]]] + [f"{np.mean(r0[6:]):.2f}"] +
                    [f"{x:.2f}" for x in r1[[0, 1, 2, 4, 5]]] + [f"{np.mean(r1[6:]):.2f}"])
    write("tab_s2_prior.tex", tabular("lrcccc|cccccc", ["$|a_y|$ [m/s$^2$]", "states", "$v_x$", "$v_y$", "$r$", "wheels", "$v_x$", "$v_y$",
                                                          "$r$", "$F$", "$\\delta$", "wheels"], rows,
                                       "Error of the simplified physics model in the rates of change, $|f_\\mathrm{true} - f_P|/S_f$ (RMS over "
                                       "held-out driving states), by lateral acceleration; left M0, right M1. Wheels: mean of the four wheel-slip "
                                       "states. Heading and, on M0, the actuator states have no error.", "tab:s2-prior", wide=True))
    for v in VARIANTS:
        lo, hi = np.asarray(s[v]["by_ay"][bins[0]]["rms"]), np.asarray(s[v]["by_ay"][bins[-1]]["rms"])
        macro(f"Prior{VTAG[v]}BodyLow", f"{np.min(lo[:3]):.2f}--{np.max(lo[:3]):.2f}")
        macro(f"Prior{VTAG[v]}BodyHigh", f"{np.min(hi[:3]):.2f}--{np.max(hi[:3]):.2f}")
        macro(f"Prior{VTAG[v]}WheelLow", f"{np.mean(lo[6:]):.2f}")
        macro(f"Prior{VTAG[v]}WheelHigh", f"{np.mean(hi[6:]):.2f}")
    p = load("e25_prior_accuracy", "e25")["results"]
    for v in VARIANTS:
        for name, tag in (("full", "Full"), ("quasi-steady", "Qs")):
            for g in ("body", "actuators", "wheels"):
                macro(f"PriorAcc{VTAG[v]}{tag}{g.capitalize()}", f"{p[v][name]['all_t'][g]:.4f}")


# ---------------------------------------------------------------- architecture screen (E21)
def architectures():
    s = load("e21_architectures", "e21_m0")
    names = {"": "PINC network, 8$\\times$64 (baseline)", "wide": "wide MLP 6$\\times$256", "modmlp": "modified MLP", "fourier": "Fourier features",
             "adaptive": "adaptive activation", "tbasis": "multi-rate time basis", "deeponet": "DeepONet", "split": "split output heads",
             "anchored": "prior-anchored", "kan": "Chebyshev KAN"}
    tags = {"": "Mlp", "wide": "Wide", "modmlp": "ModMlp", "fourier": "Fourier", "adaptive": "Adaptive", "tbasis": "TimeBasis",
            "deeponet": "DeepOnet", "split": "Split", "anchored": "Anch", "kan": "Kan"}
    rows = []
    for tag in names:
        r = [names[tag]]
        for n in ("100", "1000"):
            a = s["results"][n]
            r.append(sci(float(gmean(a[tag]["h50"]))))
            if tag:
                c = a[f"{tag} vs A0"]["h50"]
                r.append(f"{sig2(c['ratio'])} ({c['wins']}/{c['n']}, ${pval(c['p'])}$)")
                cmp_macros(f"Arch{NTAG[int(n)]}{tags[tag]}Fifty", c)
            else:
                r.append("--")
        t = s["timing"][tag]
        r += [f"{t['n_params']:,}".replace(",", "\\,"), f"{t['solve_median']*1e3:.1f}"]
        macro(f"Arch{tags[tag]}Solve", f"{t['solve_median']*1e3:.1f}")
        macro(f"Arch{tags[tag]}Params", f"{t['n_params']:,}".replace(",", "\\,"))
        rows.append(r)
    write("tab_s2_arch.tex", tabular("lccccrr", ["architecture", "50 steps, $N$=100", "against MLP", "50 steps, $N$=1000", "against MLP",
                                                   "parameters", "solve [ms]"], rows,
                                     "Architecture screen on M0 (three seeds, physics loss with the $\\lambda$ selected for the MLP). 50-step "
                                     "test error of the body states; against MLP: error of the MLP divided by the error of the architecture "
                                     "(seeds better, paired $t$-test $p$); solve: MPC solve time per step at $N_\\mathrm{h} = 10$, one CPU thread.",
                                     "tab:s2-arch", wide=True))


# ---------------------------------------------------------------- learned physics parameters (E16 MLP, E22 anchored)
def theta():
    s = load("e16_hf_theta", "e16_m1")
    nom, tru = s["nominal"], s["truth"]
    sym = dict(Caf="$C_{\\alpha f}$ [kN/rad]", Car="$C_{\\alpha r}$ [kN/rad]", C_kappa="$C_\\kappa$ [kN]", tau_F="$\\tau_F$ [s]",
               tau_delta="$\\tau_\\delta$ [s]")
    scale = dict(Caf=1e-3, Car=1e-3, C_kappa=1e-3, tau_F=1, tau_delta=1)
    rows = []
    for k in sym:
        r = [sym[k], f"{nom[k]*scale[k]:.3g}", f"{tru[k]*scale[k]:.3g}"]
        for n in (100, 1000):
            for arch, tmpl in (("mlp", "hfm1_n{n}_lam0.001_theta_s{s}"), ("anch", "hfm1_n{n}_lam0.001_anchored_theta_s{s}")):
                vals = []
                for sd in range(3 if arch == "mlp" else 5):
                    p = os.path.join(RESULTS_DIR, "models", tmpl.format(n=n, s=sd), "summary.json")
                    if os.path.exists(p):
                        with open(p) as fh:
                            vals.append(json.load(fh)["theta"][k])
                lo, hi = min(vals)*scale[k], max(vals)*scale[k]
                r.append(f"{lo:.3g}--{hi:.3g}")
                macro(f"Theta{'Mlp' if arch == 'mlp' else 'Anch'}{NTAG[n]}{k.replace('_', '').capitalize().replace('Ckappa', 'Ckappa')}",
                      f"{lo:.3g}--{hi:.3g}")
        rows.append(r)
        macro(f"ThetaNominal{k.replace('_', '').capitalize()}", f"{nom[k]*scale[k]:.3g}")
        macro(f"ThetaTrue{k.replace('_', '').capitalize()}", f"{tru[k]*scale[k]:.3g}")
    write("tab_s2_theta.tex", tabular("lccccccc", ["parameter", "nominal", "true (M1)", "MLP, $N$=100", "anchored, $N$=100",
                                                     "MLP, $N$=1000", "anchored, $N$=1000"][:3] + ["MLP, $N$=100", "anchored, $N$=100",
                                                     "MLP, $N$=1000", "anchored, $N$=1000"], rows,
                                      "Physics parameters learned with the physics loss on M1 (range over seeds: three for the MLP, five for the "
                                      "anchored network). Cornering stiffness per axle; the slip stiffness is the same in the nominal model and M1.",
                                      "tab:s2-theta", wide=True))


# ---------------------------------------------------------------- closed loop (E18) and solve time (E4)
CL_REFS = ("speed_sin", "speed_step", "lane_change", "lane_change_short", "iso_lane_change")
CL_LANES = ("lane_change", "lane_change_short", "iso_lane_change")
CL_DUR = {"speed_sin": 10.0, "speed_step": 10.0, "lane_change": 6.0, "lane_change_short": 6.0, "iso_lane_change": 6.5}
R_MAX = 0.5                                    # soft yaw-rate limit of the MPC [rad/s] (configs, mpc.r_max)


def cl_runs(run):
    """{(manoeuvre, arm, N): [runs]} of an E18 run directory (Study 2: e18_s2_<v>, seeds 5-9)."""
    import glob
    out = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "e18_hf_closed_loop", run, "runs", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        out.setdefault((r["ref"], r["arm"], r["N"]), []).append(r)
    return out


def per_seed(runs, key):
    """Learned controllers: {training seed: mean of `key` over its two noise seeds}; NMPC: {noise seed: value}."""
    by = {}
    for r in runs:
        by.setdefault(r["model_seed"] if r["model_seed"] is not None else r["noise_seed"], []).append(r[key])
    return {k: float(np.mean(v)) for k, v in by.items()}


def cl_metric(R, arm, n, metric):
    """Per-seed values of an aggregate closed-loop metric: lane = mean lateral RMSE over the three lane changes,
    sin = speed RMSE on the sinusoid, iso = peak lateral error on the ISO 3888-2 path."""
    if metric == "lane":
        parts = [per_seed(R[(ref, arm, n)], "rmse_Y") for ref in CL_LANES]
        return {k: float(np.mean([p_[k] for p_ in parts])) for k in parts[0]}
    ref, key = {"sin": ("speed_sin", "rmse_vx"), "iso": ("iso_lane_change", "max_Y")}[metric]
    return per_seed(R[(ref, arm, n)], key)


def cl_noise_seeds(R, arm, n):
    """{training seed: [noise seeds]} of a learned controller (the NMPC runs it is paired with)."""
    by = {}
    for r in R[("speed_sin", arm, n)]:
        by.setdefault(r["model_seed"], []).append(r["noise_seed"])
    return by


def cl_compare(R, x, y, metric):
    """x, y: (arm, N).  Learned vs learned: paired by training seed (mean over its two noise seeds); learned vs
    NMPC: the NMPC value for a training seed is its mean over the same two noise seeds (Section 5.3)."""
    vx_ = cl_metric(R, *x, metric)
    vy_ = cl_metric(R, *y, metric)
    if y[0].startswith("NMPC") and not x[0].startswith("NMPC"):
        seeds = cl_noise_seeds(R, *x)
        vy_ = {m: float(np.mean([vy_[k] for k in ks])) for m, ks in seeds.items()}
    common = sorted(set(vx_) & set(vy_))
    return compare([vx_[m] for m in common], [vy_[m] for m in common])


CL_ROWS = [("NMPC-true", 0), ("NMPC-prior", 0), ("NMPC-qs", 0),
           ("data-only", 100), ("PINC", 100), ("anchored data-only", 100), ("anchored PINC", 100), ("grey-box-qs", 100), ("grey-box", 100),
           ("data-only", 1000), ("PINC", 1000), ("PINC-ablation", 1000), ("anchored data-only", 1000), ("anchored PINC", 1000),
           ("grey-box-qs", 1000), ("grey-box", 1000)]
CL_LABEL = {"NMPC-true": "NMPC, true model", "NMPC-prior": "NMPC, full prior", "NMPC-qs": "NMPC, quasi-steady prior",
            "data-only": "data-only network", "PINC": "PINC network", "PINC-ablation": "PINC network, $\\lambda = 10^{-3}$",
            "anchored data-only": "anchored data-only network", "anchored PINC": "anchored PINC network",
            "grey-box-qs": "grey-box, quasi-steady prior", "grey-box": "grey-box, full prior"}
CL_TAG = {"NMPC-true": "NmpcTrue", "NMPC-prior": "NmpcPrior", "NMPC-qs": "NmpcQs", "data-only": "Data", "PINC": "Pinc",
          "PINC-ablation": "PincAbl", "anchored data-only": "AnchData", "anchored PINC": "Anch", "grey-box-qs": "GreyQs",
          "grey-box": "Grey"}
CL_PAIRS = [(a, "NMPC-qs") for a in ("data-only", "PINC", "PINC-ablation", "anchored data-only", "anchored PINC", "grey-box-qs", "grey-box")] + \
           [("anchored PINC", "anchored data-only"), ("anchored PINC", "PINC"), ("anchored PINC", "data-only"), ("anchored PINC", "grey-box-qs"),
            ("PINC", "data-only"), ("grey-box-qs", "data-only"), ("anchored data-only", "data-only")]


def closed_loop():
    """Study 2 closed loop (E18 with the E28 models, seeds 5-9).  Reference: NMPC with the quasi-steady prior."""
    rows = []
    R = {v: cl_runs(f"e18_s2_{v}") for v in VARIANTS}
    for arm, n in CL_ROWS:
        r = [CL_LABEL[arm], "--" if n == 0 else f"{n:,}".replace(",", "\\,")]
        for v in VARIANTS:
            if (CL_LANES[0], arm, n) not in R[v]:
                r += ["--"]*5
                continue
            runs = [x for ref in CL_REFS for x in R[v][(ref, arm, n)]]
            lost = sum(x["max_Y"] > 1.0 for x in runs)
            yaw_t = float(np.mean([x["viol_time_r"]/CL_DUR[x["ref"]] for x in runs]))
            yaw_r = float(np.mean([x["viol_time_r"] > 0 for x in runs]))
            med = {m: geo_ci(list(cl_metric(R[v], arm, n, m).values()))["mean"] for m in ("lane", "sin", "iso")}
            r += [f"{lost}/{len(runs)}", f"{100*yaw_t:.0f}", f"{med['lane']:.3f}", f"{med['sin']:.3f}", f"{med['iso']:.2f}"]
            tag = f"Cl{VTAG[v]}{CL_TAG[arm]}" + (NTAG[n] if n else "")
            macro(tag + "Lost", str(lost)); macro(tag + "Runs", str(len(runs)))
            macro(tag + "YawTime", f"{100*yaw_t:.0f}"); macro(tag + "YawRuns", f"{100*yaw_r:.0f}")
            macro(tag + "Lane", f"{med['lane']:.3f}"); macro(tag + "Sin", f"{med['sin']:.3f}"); macro(tag + "Iso", f"{med['iso']:.2f}")
        rows.append(r)
    for v in VARIANTS:
        for x, y in CL_PAIRS:
            for n in (100, 1000):
                xn, yn = (x, n), (y, 0 if y.startswith("NMPC") else n)
                if (CL_LANES[0], *xn) not in R[v] or (CL_LANES[0], *yn) not in R[v]:
                    continue
                for m, t in (("lane", "Lane"), ("sin", "Sin"), ("iso", "Iso")):
                    cmp_macros(f"ClCmp{VTAG[v]}{NTAG[n]}{CL_TAG[x]}Vs{CL_TAG[y]}{t}", cl_compare(R[v], xn, yn, m))
        for m, t in (("lane", "Lane"), ("sin", "Sin"), ("iso", "Iso")):          # NMPC full prior vs quasi-steady
            cmp_macros(f"ClCmp{VTAG[v]}NmpcQsVsNmpcPrior{t}", cl_compare(R[v], ("NMPC-qs", 0), ("NMPC-prior", 0), m))
    write("tab_s2_closed.tex", tabular("llccccc|ccccc", ["controller", "$N$", "lost", "yaw", "lane $Y$", "sin. $v_x$", "ISO peak",
                                                          "lost", "yaw", "lane $Y$", "sin. $v_x$", "ISO peak"], rows,
                                       "Closed loop on the double-track plant, M0 (left) and M1 (right), with the models of the main "
                                       "comparison (seeds 5 to 9). Lost: runs whose lateral error exceeded 1\\,m, over all five manoeuvres. "
                                       "Yaw: share of time [\\%] above the soft yaw-rate limit of 0.5\\,rad/s. Lane $Y$: lateral RMSE [m], "
                                       "mean over the three lane changes; sin.\\ $v_x$: speed RMSE [m/s] on the speed sinusoid; ISO peak: peak "
                                       "lateral error [m] on the smooth path through the ISO 3888-2 cone layout. Learned controllers: ten runs "
                                       "per manoeuvre (five training seeds $\\times$ two noise seeds); each entry is the geometric mean over "
                                       "training seeds of the mean of the two runs of a seed. NMPC: geometric mean over noise seeds (ten; "
                                       "two for the true model).", "tab:s2-closed", wide=True))


def references_and_checks():
    from pinc.config import ROOT, load_config
    from pinc.refs import make_reference
    from pinc import plant_hf, tyre_mf
    from pinc.system import get_system
    cfg = load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"))
    for name, ov, tag in (("lane_change", {}, "Lc"), ("lane_change", {"refs.slc_length": 30.0}, "LcShort"), ("iso_lane_change", {}, "Dlc")):
        c = cfg.with_overrides(ov)
        ref = make_reference(name, c)
        X = np.linspace(0.0, 200.0, 200001)
        Y, dY = ref.path(X)
        ddY = np.gradient(dY, X)
        kappa = ddY/(1 + dY**2)**1.5
        macro(f"Ref{tag}Ay", f"{np.max(np.abs(kappa))*ref.v0**2:.1f}")
        macro(f"Ref{tag}Speed", f"{ref.v0:g}")
    from pinc.refs import iso3888_lanes
    lanes = iso3888_lanes(float(cfg.refs.iso_car_width))
    macro("IsoCarWidth", f"{cfg.refs.iso_car_width:g}")
    macro("IsoShiftIn", f"{lanes[1][2]:.2f}"); macro("IsoShiftOut", f"{lanes[1][2] - lanes[2][2]:.2f}")
    macro("IsoSlackMin", f"{min(sl for *_, sl in lanes):.2f}"); macro("IsoSlackMax", f"{max(sl for *_, sl in lanes):.2f}")
    p = get_system(cfg).truth
    Fz = plant_hf.static_loads(p)[0]
    a = np.linspace(0, 0.4, 4001)
    Fx, Fy = tyre_mf.forces(np.zeros_like(a), a, np.full_like(a, Fz), p["tyre"])
    mu = float(np.max(np.abs(Fy))/Fz)
    macro("TyreMu", f"{mu:.2f}")
    macro("TyreMuG", f"{mu*p['g']:.1f}")
    e = load("e26_model_checks", "e26")
    macro("CheckTyrePoints", str(e["tyre"]["n"]))
    macro("CheckTyreErrFx", f"{e['tyre']['max_err_Fx']:.2f}")
    macro("CheckTyreErrFy", sci(e["tyre"]["max_err_Fy"], 1).strip("$"))
    macro("CheckTyreMaxF", f"{max(e['tyre']['max_Fx'], e['tyre']['max_Fy'])/1e3:.1f}")
    macro("CheckGentleAy", f"{e['gentle']['gentle_ay']:g}")
    macro("CheckGentleAyReached", f"{e['gentle']['ay_max']:.1f}")
    macro("CheckGentleNrmse", f"{e['gentle']['nrmse_body']:.4f}")
    macro("CheckGentleDrives", str(e["gentle"]["n_drives"]))
    for v, t in e["wheel_time_constant_s"].items():
        macro(f"WheelTau{ {'5': 'Five', '10': 'Ten', '20': 'Twenty', '30': 'Thirty'}[v] }".replace(" ", ""), f"{t*1e3:.1f}")
    with open(os.path.join(RESULTS_DIR, "models", "hfm0_n100_lam0.01_s0", "meta.json")) as fh:
        macro("HfTrainTf", json.load(fh)["versions"]["tensorflow"])


def timing():
    """E4 on one CPU thread with the GPU hidden (Study 2: e4_s2, e4_s2_anchored, e4_s2_gbfull; N = 1000 models of seed 5)."""
    runs = {"e4_s2": {"pinc": "PINC network", "blackbox": "data-only network", "greybox": "grey-box, quasi-steady prior",
                      "nmpc_rk4": "NMPC, full prior", "nmpc_qs": "NMPC, quasi-steady prior", "nmpc_true": "NMPC, true model"},
            "e4_s2_gbfull": {"greybox": "grey-box, full prior"},
            "e4_s2_anchored": {"pinc": "anchored PINC network"}}
    tags = {"PINC network": "Pinc", "data-only network": "Data", "grey-box, quasi-steady prior": "GreyQs", "NMPC, full prior": "NmpcPrior",
            "NMPC, quasi-steady prior": "NmpcQs", "grey-box, full prior": "Grey", "anchored PINC network": "Anch", "NMPC, true model": "NmpcTrue"}
    order = ["anchored PINC network", "PINC network", "data-only network", "grey-box, quasi-steady prior", "grey-box, full prior",
             "NMPC, quasi-steady prior", "NMPC, full prior", "NMPC, true model"]
    W = {5: "Five", 10: "Ten", 20: "Twenty", 40: "Forty"}
    fmt = lambda x: f"{x:.1f}" if x < 1000 else f"{x/1e3:.2f}\\,s"
    vals, p95 = {}, {}
    for run, arms in runs.items():
        s = load("e4_timing", run)
        for arm, lab in arms.items():
            vals[lab] = {int(n): v["median"]*1e3 for n, v in s["solve"][arm].items()}
            p95[lab] = {int(n): v["p95"]*1e3 for n, v in s["solve"][arm].items()}
    rows = []
    for lab in order:
        r = [lab]
        for n in (5, 10, 20, 40):
            x = vals[lab].get(n)
            r.append("--" if x is None else f"{fmt(x)} [{fmt(p95[lab][n])}]")
            if x is not None:
                macro(f"Solve{tags[lab]}{W[n]}", f"{x:.1f}" if x < 1000 else f"{x/1e3:.2f}")
                macro(f"SolvePninetyFive{tags[lab]}{W[n]}", f"{p95[lab][n]:.1f}" if p95[lab][n] < 1000 else f"{p95[lab][n]/1e3:.2f}")
        rows.append(r)
    h = json.load(open(os.path.join(RESULTS_DIR, "e4_timing", "host_2026-10-09.json")))
    macro("HostCpu", h["cpu"].replace("(R)", "").replace("(TM)", "")); macro("HostPowerPlan", h["power_plan"])
    write("tab_s2_timing.tex", tabular("lcccc", ["prediction model", "$N_\\mathrm{h}=5$", "10", "20", "40"], rows,
                                       "MPC solve time per step [ms] on the double-track plant (M0) against the prediction horizon: median "
                                       "[95th percentile] over a 3\\,s lane change, after a warm-up solve; one CPU thread with the GPU hidden "
                                       f"({h['cpu'].replace('(R)', '').replace('(TM)', '')}, Windows power plan {h['power_plan']}, on mains). "
                                       "Learned models trained on 1000 trajectories, seed 5. NMPC with the true model was run up to "
                                       "$N_\\mathrm{h} = 10$.", "tab:s2-timing", wide=True))
    for n in (5, 10, 20, 40):
        w = W[n]
        a = vals["anchored PINC network"][n]
        macro(f"SolveRatioAnchOverPinc{w}", f"{a/vals['PINC network'][n]:.1f}")
        macro(f"SolveRatioNmpcQsOverAnch{w}", f"{vals['NMPC, quasi-steady prior'][n]/a:.1f}")
        macro(f"SolveRatioGreyQsOverAnch{w}", f"{vals['grey-box, quasi-steady prior'][n]/a:.1f}")
        macro(f"SolveRatioGreyOverGreyQs{w}", f"{vals['grey-box, full prior'][n]/vals['grey-box, quasi-steady prior'][n]:.0f}")
        macro(f"SolveRatioNmpcPriorOverNmpcQs{w}", f"{vals['NMPC, full prior'][n]/vals['NMPC, quasi-steady prior'][n]:.0f}")
    budget = 1e3*0.1
    within = {lab: max([n for n in (5, 10, 20, 40) if vals[lab].get(n, np.inf) <= budget], default=0) for lab in order}
    for lab, n in within.items():
        macro(f"SolveMaxHorizon{tags[lab]}", str(n) if n else "none")


# ---------------------------------------------------------------- setup facts of Study 2 (configs and vehicle model)
def hf_config():
    from pinc.config import ROOT, load_config
    from pinc import plant_hf, tyre_mf
    from pinc.greybox import QS_DT
    from pinc.system import HighFidelity, get_system
    cfg = load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"))
    sysm = get_system(cfg)
    p0, p1 = sysm.truth, get_system(load_config(os.path.join(ROOT, "configs", "hf_m1.yaml"))).truth
    th = lambda x: f"{x:,}".replace(",", "\\,")
    macro("HfT", f"{cfg.T:g}")
    macro("HfNh", str(cfg.mpc.N))
    macro("HfDtPlant", f"{cfg.sim.dt_plant*1e3:g}")
    macro("HfDtPred", f"{cfg.mpc.dt_pred*1e3:g}")
    macro("HfQsStep", f"{QS_DT*1e3:g}")
    macro("HfAnchorTauW", f"{HighFidelity.ANCHOR_TAU_W*1e3:g}")
    macro("HfNval", th(cfg.train.n_val)); macro("HfNtest", th(cfg.train.n_test))
    macro("HfNcolloc", th(cfg.train.n_colloc)); macro("HfLogFrac", f"{100*cfg.train.colloc_log_frac:g}")
    macro("HfLbfgs", th(cfg.train.lbfgs_iters)); macro("HfLr", f"{cfg.train.lr:g}")
    e15 = load("e15_hf_data", "e15_m0")
    from experiments.e15_hf_data import overrides
    macro("HfSteps", th(overrides(100, 0.0, 0, cfg)["train.steps"]))      # the fixed Adam budget of every Study 2 training
    macro("HfSx", ", ".join(f"{v:g}" for v in cfg.scales.S_x))
    macro("HfNoise", ", ".join(f"{v:g}" for v in cfg.sim.noise_sigma))
    macro("HfBoxVx", f"{cfg.box_train.vx[0]:g}, {cfg.box_train.vx[1]:g}")
    macro("HfTrack", f"{p0['t_f']:g}"); macro("HfCgHeight", f"{p0['h']:g}"); macro("HfRollShare", f"{p0['chi_f']:g}")
    macro("HfBrakeShare", f"{p0['beta_f']:g}"); macro("HfRw", f"{p0['R_w']:.3f}"); macro("HfIw", f"{p0['I_w']:.3g}")
    macro("HfFnom", th(int(p0['tyre']['FNOMIN'])))
    macro("HfAxleStiffMzero", f"{2*tyre_mf.cornering_stiffness(plant_hf.static_loads(p0)[0], p0['tyre'])/1e3:.0f}")
    macro("HfAxleStiffMone", f"{2*tyre_mf.cornering_stiffness(plant_hf.static_loads(p1)[0], p1['tyre'])/1e3:.0f}")
    macro("HfTauFNom", f"{p0['tau_F']:g}"); macro("HfTauDNom", f"{p0['tau_delta']:.2f}")
    macro("HfTauFTrue", f"{p1['tau_F']:.2f}"); macro("HfTauDTrue", f"{p1['tau_delta']:.2f}")
    lam = e15["best_lambda"]
    macro("HfLamHundred", f"{lam['100']:g}"); macro("HfLamThousand", f"{lam['1000']:g}"); macro("HfLamTwentyK", f"{lam['20000']:g}")
    macro("HfSeeds", "five")
    macro("HfBoxExtrapVx", f"{cfg.box_extrap.vx[0]:g} to {cfg.box_extrap.vx[1]:g}")
    from pinc import plant_hf as _ph
    macro("HfVeps", f"{_ph.V_EPS:g}")
    macro("HfLogMin", f"10^{{{-3}}}")                     # log-spaced collocation times T 10^U(-3, 0) (pinc/data.py)
    with open(os.path.join(RESULTS_DIR, "e4_timing", "e4_hf_m0_v2", "meta.json")) as fh:
        meta = json.load(fh)
    macro("HfCpu", meta["cpu"].replace("(R)", "").replace("(TM)", ""))
    macro("HfTf", meta["versions"]["tensorflow"])


def convergence_and_tuning():
    """E11 (training budget) macros and the E24 anchor-setting table (appendix)."""
    r = load("e11_convergence", "e11")["runs"]["long_8x64"]["val_data_at_lbfgs"]
    macro("ConvLossFiveHundred", sci(r["500"]).strip("$"))
    macro("ConvLossTwoThousand", sci(r["2000"]).strip("$"))
    macro("ConvRatio", f"{r['500']/r['2000']:.1f}")
    names = {"A8": "one Euler step (used)", "atw": "learned wheel time constant", "aexp": "exact actuator lag",
             "aend": "slip target at the end of the step", "again": "learned gain on the anchor", "aall": "all four together"}
    tags = {"A8": "Base", "atw": "TauW", "aexp": "Exp", "aend": "End", "again": "Gain", "aall": "All"}
    s0, s1 = load("e24_anchor_tuning", "e24_m0")["results"], load("e24_anchor_tuning", "e24_m1")["results"]
    rows = []
    for t in names:
        row = [names[t]]
        for res, v in ((s0, "m0"), (s1, "m1")):
            for n in ("100", "1000"):
                if t in res[n]:
                    row.append(ci_cell(res[n][t]["h50"]))
                    if t != "A8":
                        c = res[n][f"{t} vs A8"]["h50"]
                        cmp_macros(f"Tune{VTAG[v]}{NTAG[int(n)]}{tags[t]}Fifty", c)
                else:
                    row.append("--")
        rows.append(row)
    write("tab_s2_tuning.tex", tabular("lcccc", ["anchor", "M0, $N$=100", "M0, $N$=1000", "M1, $N$=100", "M1, $N$=1000"], rows,
                                       "Settings of the anchor (three seeds). 50-step test error of the body states, mean [95\\,\\% "
                                       "confidence interval]; on M1 only the learned gain and the combination were trained.",
                                       "tab:s2-tuning", wide=True))


def params_table():
    """One table with the vehicle, training and MPC settings of both studies (read from the configs)."""
    from pinc.config import ROOT, load_config
    from pinc.system import HighFidelity, get_system
    c1 = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    c2 = load_config(os.path.join(ROOT, "configs", "hf_m0.yaml"))
    v = c1.params
    p0 = get_system(c2).truth
    p1 = get_system(load_config(os.path.join(ROOT, "configs", "hf_m1.yaml"))).truth
    from experiments.e15_hf_data import overrides
    L = lambda xs: "[" + ", ".join(f"{x:g}" for x in xs) + "]"
    rows = [
        ["\\multicolumn{3}{l}{\\emph{Vehicle (both studies)}}"],
        ["mass, yaw inertia", "$m$, $I_z$", f"{v['m']:g}\\,kg, {v['Iz']:g}\\,kg\\,m$^2$"],
        ["axle distances", "$l_f$, $l_r$", f"{v['lf']:g}\\,m, {v['lr']:g}\\,m"],
        ["drag, rolling resistance", "$C_d A$, $F_{rr}$", f"{v['Cd']:g} $\\times$ {v['A']:g}\\,m$^2$, {v['Frr']:g}\\,N"],
        ["cornering stiffness per axle", "$C_{\\alpha f}$, $C_{\\alpha r}$", f"{v['Caf']/1e3:g}, {v['Car']/1e3:g}\\,kN/rad (single-track, prior, M0)"],
        ["slip stiffness per wheel (prior)", "$C_\\kappa$", f"{get_system(c2).prior['C_kappa']/1e3:.1f}\\,kN"],
        ["\\multicolumn{3}{l}{\\emph{Double-track plant (Study 2)}}"],
        ["track width, CG height", "$t$, $h$", f"{p0['t_f']:g}\\,m, {p0['h']:g}\\,m"],
        ["front roll share, front brake share", "$\\chi_f$, $\\beta_f$", f"{p0['chi_f']:g}, {p0['beta_f']:g}"],
        ["wheel radius, wheel inertia", "$R_w$, $I_w$", f"{p0['R_w']:.3f}\\,m, {p0['I_w']:.3g}\\,kg\\,m$^2$"],
        ["actuator lags, nominal / M1", "$\\tau_F$, $\\tau_\\delta$", f"{p0['tau_F']:g}, {p0['tau_delta']:.2f}\\,s / {p1['tau_F']:.2f}, {p1['tau_delta']:.2f}\\,s"],
        ["\\multicolumn{3}{l}{\\emph{Network and training}}"],
        ["control period", "$T$", f"{c1.T:g}\\,s (both)"],
        ["state scales, Study 1", "$S_x$", L(c1.scales.S_x)],
        ["state scales, Study 2", "$S_x$", L(c2.scales.S_x)],
        ["input scales", "$S_u$", L(c2.scales.S_u)],
        ["rate scales, Study 1", "$S_f$", "[" + ", ".join(f"{x:.3g}" for x in c1.scales.S_f) + "]"],
        ["rate scales, Study 2", "$S_f$", "[" + ", ".join(f"{x:.3g}" for x in c2.scales.S_f).replace("e+04", "$\\times10^4$") + "]"],
        ["anchor wheel time constant", "$\\tau_w$", f"{HighFidelity.ANCHOR_TAU_W*1e3:g}\\,ms"],
        ["network", "", f"{c2.model.depth} $\\times$ {c2.model.width}, $\\tanh$"],
        ["budget, Study 1", "", f"{c1.train.epochs} Adam epochs (cosine decay), {c1.train.lbfgs_iters} L-BFGS"],
        ["budget, Study 2", "", f"{overrides(100, 0.0, 0, c2)['train.steps']:,}".replace(",", "\\,") + " Adam steps (constant), " +
         f"{c2.train.lbfgs_iters:,}".replace(",", "\\,") + " L-BFGS"],
        ["\\multicolumn{3}{l}{\\emph{MPC}}"],
        ["horizon", "$N_\\mathrm{h}$", str(c1.mpc.N)],
        ["weights", "$Q = P$, $R$, $R_\\Delta$", f"{L(c1.mpc.Q)}, {L(c1.mpc.R)}, {L(c1.mpc.R_delta)}"],
        ["yaw-rate limit", "$r_{\\max}$, $w_r$", f"{c1.mpc.r_max:g}\\,rad/s, {c1.mpc.w_rmax:g}"],
    ]
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    write("tab_params.tex", "\\begin{table}[htbp]\\centering\\small\n\\caption{Parameters of the plants, networks and "
          "controllers.}\\label{tab:params}\n\\begin{tabular}{p{5.0cm}lp{5.4cm}}\\toprule\nquantity & symbol & value \\\\\\midrule\n" +
          body + " \\\\\n\\bottomrule\\end{tabular}\\end{table}\n")


WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
LAM_SIZES = (100, 1000, 20000)
LAM_CLOSE = 1.5                                  # "second-best lambda within this factor of the selected one"


def lambda_grids():
    """{(network, variant): {N: {lambda (float): [validation 50-step body error per seed 0-2]}}} from E15 and E23."""
    g = {}
    for v in VARIANTS:
        runs = load("e15_hf_data", f"e15_{v}_grid")["runs"]
        g[("plain", v)] = {n: {float(k.split("_")[1]): x["h50_val"] for k, x in runs.items() if k.split("_")[0] == str(n)}
                           for n in LAM_SIZES}
        res = load("e23_anchored_extend", f"e23_lambda_{v}_grid")["results"]
        g[("anchored", v)] = {n: {float(a.split()[-1]): x["h50_val"] for a, x in res[str(n)].items()} for n in LAM_SIZES}
    return g


def lambda_facts():
    """Lambda selection (E15 plain, E23 anchored; seeds 0-2, validation 50-step body error, mean over seeds)."""
    from scipy import stats
    g = lambda_grids()
    NET = dict(plain="Plain", anchored="Anch")
    close, cells = 0, []
    for (net, v), grid in g.items():
        for n, by_lam in grid.items():
            sel = min(by_lam, key=lambda l: np.mean(by_lam[l]))                     # the rule E15 / E23 apply
            macro(f"LamSel{NET[net]}{VTAG[v]}{NTAG[n]}", f"{sel:g}" if sel >= 0.1 or sel == 0 else f"10^{{{int(round(np.log10(sel)))}}}")
            gm = sorted(gmean(x) for x in by_lam.values())
            close += gm[1]/gm[0] < LAM_CLOSE
            cells += list(by_lam.values())
    macro("LamGrids", WORDS[len(g)*len(LAM_SIZES)])
    macro("LamGridsClose", WORDS[close])
    macro("LamSeedSpread", f"{np.median([max(c)/min(c) for c in cells]):.1f}")
    f = [np.exp(stats.t.ppf(0.975, len(c) - 1)*np.std(np.log(c), ddof=1)/np.sqrt(len(c))) for c in cells]
    macro("LamCiFactor", f"{np.median(f):.1f}")
    # examples: anchored network on M0
    a = g[("anchored", "m0")]
    macro("LamExThousandBest", sci(gmean(a[1000][1.0])).strip("$")); macro("LamExThousandSecond", sci(gmean(a[1000][1e-4])).strip("$"))
    macro("LamExThousandGap", f"{100*(gmean(a[1000][1e-4])/gmean(a[1000][1.0]) - 1):.0f}")
    macro("LamExThousandPointOne", f"{gmean(a[1000][0.1])/gmean(a[1000][1.0]):.1f}")
    seeds = a[100][0.1]
    macro("LamExHundredSeedLo", f"{min(seeds):.3f}"); macro("LamExHundredSeedHi", f"{max(seeds):.2f}")
    # plain network on M1 with 100 trajectories: diverges for every lambda
    p = g[("plain", "m1")][100]
    means = [np.mean(x) for x in p.values()]
    macro("LamPlainMoneHundredMin", f"{min(means):.1f}"); macro("LamPlainMoneHundredMax", f"{max(means):.1f}")
    macro("LamPlainMoneHundredSeedMin", f"{min(min(x) for x in p.values()):.2f}")
    macro("LamGridMax", "10"); macro("LamGridMaxAnchMzero", "100"); macro("LamCloseFactor", f"{LAM_CLOSE:g}")
    macro("LamAblation", "10^{-3}")                                                # E28 ABLATION_LAM
    macro("LamAnchMzeroHundredGain", f"{gmean(a[100][0.0])/gmean(a[100][10.0]):.1f}")
    return g


def lambda_criterion():
    """Sensitivity of the lambda selection to the criterion: the lambda the validation 10-step error would select,
    next to the one used (validation 50-step error), from the E9 validation runs of the E15 / E23 grids (seeds 0-2).
    Also one-step test error against lambda (E15 / E23, seeds 0-2) with the prior's one-step error (E25)."""
    rows, one = [], {}
    pa = load("e25_prior_accuracy", "e25")["results"]
    for net, exp, fmt in (("plain", "e15_{v}_grid_e9val", "hf{v}_n{n}_lam{l}_s{k}"),
                          ("anchored", "e23_lambda_{v}_grid_e9val", "hf{v}_n{n}_lam{l}_anchored_s{k}")):
        for v in VARIANTS:
            ev = load("e9_physics_learning", exp.format(v=v))["models"]
            for n in LAM_SIZES:
                hz = {}
                for k_ in ev:
                    head = fmt.format(v=v, n=n, l="", k="").split("_lam")[0] + "_lam"
                    if not k_.startswith(head) or (net == "plain" and "anchored" in k_):
                        continue
                    lam = float(k_[len(head):].split("_")[0])
                    hz.setdefault(lam, {10: [], 50: []})
                    for h in (10, 50):
                        hz[lam][h].append(ev[k_]["horizon_body"]["in_domain"][str(h)]["mean"])
                fmt_l = lambda l: f"{l:g}" if l >= 0.1 or l == 0 else f"10^{{{int(round(np.log10(l)))}}}"
                sel = {h: min(hz, key=lambda l: np.mean(hz[l][h])) for h in (10, 50)}
                tag = f"{'Plain' if net == 'plain' else 'Anch'}{VTAG[v]}{NTAG[n]}"
                macro(f"LamTenSel{tag}", fmt_l(sel[10]))
                for l in (sel[10], sel[50]):
                    macro(f"LamTenVal{tag}{'Ten' if l == sel[10] else 'Fifty'}Sel", sci(float(np.mean(hz[l][10]))).strip("$"))
                rows.append([("PINC" if net == "plain" else "anchored PINC") if (v == "m0" and n == LAM_SIZES[0]) else "",
                             v.upper() if n == LAM_SIZES[0] else "", f"{n:,}".replace(",", "\\,"), f"${fmt_l(sel[50])}$", f"${fmt_l(sel[10])}$"])
    write("tab_lam_criterion.tex", tabular("lllcc", ["network", "variant", "$N$", "50-step (used)", "10-step"], rows,
                                           "Physics weight $\\lambda$ selected by the mean validation error over three seeds (0 to 2) of chained "
                                           "50-step predictions (used for all main results) and the weight the 10-step error would select. "
                                           "M0: matched stiffness; M1: mismatched stiffness and actuator lag.", "tab:lam-criterion"))
    for v in VARIANTS:
        macro(f"PriorOneStep{VTAG[v]}", f"{pa[v]['quasi-steady']['all_t']['body']:.4f}")
    return pa


def lambda_edges(g):
    """Selections at the top of a grid: how far the selected value is from its neighbours (validation 50-step, mean of
    three seeds).  Study 1 anchored at N = 20 000 (E27) and M1 anchored at N = 100 (E23)."""
    l = load("e27_study1_rerun", "e27_lambda")["results"]["anchored_20000"]
    top = [float(np.mean(l[k])) for k in ("0.1", "1", "10")]
    macro("SoneLamEdgeSpread", f"{100*(max(top)/min(top) - 1):.0f}")
    m1 = g[("anchored", "m1")][100]
    v = sorted((float(np.mean(x)), lam) for lam, x in m1.items())
    macro("LamEdgeMoneGap", f"{100*(v[1][0]/v[0][0] - 1):.0f}")
    macro("LamEdgeMoneSecond", f"{v[1][1]:g}" if v[1][1] >= 0.1 else f"10^{{{int(round(np.log10(v[1][1])))}}}")


def fig_lambda_onestep(g, pa):
    """One-step test error against lambda (seeds 0-2) with the prior's one-step error (dashed)."""
    plt = _plt()
    e15 = {v: load("e15_hf_data", f"e15_{v}_grid")["runs"] for v in VARIANTS}
    e23 = {v: load("e23_anchored_extend", f"e23_lambda_{v}_grid")["results"] for v in VARIANTS}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    col = {100: "#2a78d6", 1000: "#eb6834", 20000: "#1baf7a"}
    mk = {100: "o", 1000: "s", 20000: "^"}
    for (net, v), ax in zip([(n_, v_) for n_ in ("plain", "anchored") for v_ in VARIANTS], axes.flat):
        for n in LAM_SIZES:
            if net == "plain":
                by = {float(k.split("_")[1]): x["test"]["body"] for k, x in e15[v].items() if k.split("_")[0] == str(n)}
            else:
                by = {float(a.split()[-1]): x["one_step"] for a, x in e23[v][str(n)].items()}
            lams = sorted(by)
            xs = [np.log10(l) if l > 0 else -5 for l in lams]
            m = [gmean(by[l]) for l in lams]
            ax.plot(xs[1:], m[1:], color=col[n], marker=mk[n], ls="-", label=f"$N$ = {n:,}".replace(",", "\u2009"))
            ax.plot(xs[:1], m[:1], color=col[n], marker=mk[n], ls="none")
            tag = f"{'Plain' if net == 'plain' else 'Anch'}{VTAG[v]}{NTAG[n]}"
            macro(f"OneStepLamMax{tag}", sci(float(m[-1])).strip("$"))
        ax.axhline(pa[v]["quasi-steady"]["all_t"]["body"], color=INK, lw=0.9, ls="--", label="prior (quasi-steady)")
        ax.axvline(-4.5, color=MUTED, lw=0.6, ls=":")
        ax.set(yscale="log", title={"plain": "PINC network", "anchored": "anchored PINC network"}[net] + f", {v.upper()}")
        ax.title.set_color(INK)
    ticks = [-5, -4, -3, -2, -1, 0, 1, 2]
    for ax in axes[1]:
        ax.set_xticks(ticks, ["0"] + [f"$10^{{{t}}}$" for t in ticks[1:]])
        ax.set_xlabel("physics weight $\\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel("one-step test error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig_s2_lambda_onestep")
    plt.close(fig)


# ---------------------------------------------------------------- figures
# One fixed colour per model across every figure (dataviz reference palette, fixed order), plus a marker and a
# line style, so no figure relies on colour alone.
STYLE = {"A8 PINC": ("#2a78d6", "o", "-"), "PINC": ("#eb6834", "s", "--"), "data-only": ("#1baf7a", "^", ":"),
         "PINC-ablation": ("#eb6834", "s", ":"),
         "grey-box-qs": ("#eda100", "D", "-."), "grey-box": ("#e87ba4", "v", (0, (3, 1, 1, 1, 1, 1))),
         "NMPC-prior": ("#008300", "P", "--"), "NMPC-true": ("#4a3aa7", "X", "-")}
INK, MUTED = "#222222", "#6b6b6b"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 1.4, "lines.markersize": 5})
    return plt


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=160)
    print("  figure", name)


def fig_data(arms):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        for arm in ("data-only", "PINC", "A8 PINC", "grey-box-qs", "grey-box"):
            ns = [n for n in sorted(arms[v]) if arm in arms[v][n]]
            if not ns:
                continue
            cc = [geo_ci(arms[v][n][arm]["h50"]) for n in ns]
            m = np.array([c["mean"] for c in cc])
            col, mk, ls = STYLE[arm]
            ax.errorbar(ns, m, yerr=[m - [c["lo"] for c in cc], np.array([c["hi"] for c in cc]) - m], color=col, marker=mk,
                        ls=ls, capsize=2.5, label=LABEL[arm])
        ax.set(xscale="log", yscale="log", xlabel="training trajectories $N$",
               title={"m0": "M0: nominal parameters right", "m1": "M1: nominal parameters wrong"}[v])
        ax.title.set_color(INK)
    axes[0].set_ylabel("50-step test error (body states)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    _save(fig, "fig_s2_data")
    plt.close(fig)


def fig_lambda(g):
    """Validation 50-step error against lambda: every seed (points) and the mean (line); lambda = 0 on the left."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    col = {100: "#2a78d6", 1000: "#eb6834", 20000: "#1baf7a"}
    mk = {100: "o", 1000: "s", 20000: "^"}
    for (net, v), ax in zip([(n_, v_) for n_ in ("plain", "anchored") for v_ in VARIANTS], axes.flat):
        for n, by_lam in g[(net, v)].items():
            lams = sorted(by_lam)
            xs = [np.log10(l) if l > 0 else -5 for l in lams]
            m = [np.mean(by_lam[l]) for l in lams]
            lab = f"$N$ = {n:,}".replace(",", "\u2009")
            ax.plot(xs[1:], m[1:], color=col[n], marker=mk[n], ls="-", label=lab)      # lambda > 0 on the log axis
            ax.plot(xs[:1], m[:1], color=col[n], marker=mk[n], ls="none")              # lambda = 0, apart from the line
            for x, l in zip(xs, lams):
                ax.scatter([x]*len(by_lam[l]), by_lam[l], color=col[n], s=7, alpha=0.35, lw=0)
            best = lams[int(np.argmin(m))]
            bx = np.log10(best) if best > 0 else -5
            ax.scatter([bx], [min(m)], s=70, facecolors="none", edgecolors=INK, lw=0.9, zorder=5)
        ax.set(yscale="log", title={"plain": "PINC network", "anchored": "anchored PINC network"}[net] + f", {v.upper()}")
        ax.title.set_color(INK)
    ticks = [-5, -4, -3, -2, -1, 0, 1, 2]
    for ax in axes.flat:
        ax.axvline(-4.5, color=MUTED, lw=0.6, ls=":")
    for ax in axes[1]:
        ax.set_xticks(ticks, ["0"] + [f"$10^{{{t}}}$" for t in ticks[1:]])
        ax.set_xlabel("physics weight $\\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation 50-step error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig_s2_lambda")
    plt.close(fig)


def fig_speed(arms):
    plt = _plt()
    s = {r: load("e4_timing", r)["solve"] for r in ("e4_hf_m0_v2", "e4_hf_m0_gbfull", "e4_hf_m0_a8")}
    t10 = {"A8 PINC": s["e4_hf_m0_a8"]["pinc"]["10"]["median"], "PINC": s["e4_hf_m0_v2"]["pinc"]["10"]["median"],
           "data-only": s["e4_hf_m0_v2"]["blackbox"]["10"]["median"], "grey-box-qs": s["e4_hf_m0_v2"]["greybox"]["10"]["median"],
           "grey-box": s["e4_hf_m0_gbfull"]["greybox"]["10"]["median"]}
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for arm, t in t10.items():
        c = geo_ci(arms["m0"][1000][arm]["h50"])
        col, mk, _ = STYLE[arm]
        ax.errorbar(t*1e3, c["mean"], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], color=col, marker=mk, capsize=2.5, ls="none")
        off = {"A8 PINC": (-6, -12), "grey-box-qs": (7, 2)}.get(arm, (6, 2))
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(LABEL[arm].replace(", ", ",\n"), (t*1e3, c["mean"]), textcoords="offset points", xytext=off, fontsize=7,
                    color=INK, ha=ha)
    ax.axvline(100, color=MUTED, lw=0.8, ls=":")
    ax.annotate("0.1 s budget", (100, 1), xycoords=("data", "axes fraction"), xytext=(3, -10), textcoords="offset points",
                fontsize=7, color=MUTED)
    ax.set(xscale="log", yscale="log", xlabel="MPC solve time per step at $N_\\mathrm{h} = 10$ [ms]",
           ylabel="50-step test error (M0, $N$ = 1000)")
    ax.set_xlim(5, 1000)
    _save(fig, "fig_s2_speed")
    plt.close(fig)


def fig_dlc():
    plt = _plt()
    d = os.path.join(RESULTS_DIR, "e18_hf_closed_loop", "e18_m1", "runs")
    files = {"NMPC-true": "NMPC-true_N0_mNone_k0", "NMPC-prior": "NMPC-prior_N0_mNone_k0", "PINC": "PINC_N100_m0_k0",
             "A8 PINC": "A8-PINC_N100_m0_k0", "grey-box-qs": "grey-box-qs_N100_m0_k0"}
    lab = {"NMPC-true": "NMPC, true model", "NMPC-prior": "NMPC, full prior", "PINC": "PINC network",
           "A8 PINC": "anchored PINC network", "grey-box-qs": "grey-box, quasi-steady prior"}
    logs = {k: np.load(os.path.join(d, f"double_lane_change_{f}.npz")) for k, f in files.items()}
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.2), sharex=True)
    ref = logs["NMPC-true"]
    X = ref["ref"][:, 4]
    axes[0].plot(ref["t"], ref["ref"][:, 5], color=INK, lw=1, ls="--", label="reference")
    for k, lg in logs.items():
        col, _, ls = STYLE[k]
        axes[0].plot(lg["t"], lg["x"][:, 5], color=col, ls=ls, label=lab[k])
        axes[1].plot(lg["t"], lg["x"][:, 0] - lg["ref"][:, 0], color=col, ls=ls)
        axes[2].step(lg["t"][:-1], lg["u"][:, 1], where="post", color=col, ls=ls)
    axes[0].set_ylabel("$Y$ [m]"); axes[1].set_ylabel("$v_x$ error [m/s]"); axes[2].set_ylabel("$\\delta$ command [rad]")
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    _save(fig, "fig_s2_dlc")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    arms, _ = pooled_main()
    comparisons(arms)
    comparisons(pooled(), prefix="Sec")
    table_main(arms)
    study1_rerun()
    blockset()
    blockset_setup()
    prior()
    architectures()
    theta()
    closed_loop()
    timing()
    hf_config()
    references_and_checks()
    convergence_and_tuning()
    params_table()
    g = lambda_facts()
    fig_lambda(g)
    pa = lambda_criterion()
    fig_lambda_onestep(g, pa)
    lambda_edges(g)
    fig_data(arms)
    fig_speed(arms)
    fig_dlc()
    tiv_floats(arms)
    tiv_fig_data(arms)
    tiv_fig_speed(arms)
    txt = "% generated by scripts/make_paper_study2.py -- do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    write("numbers_s2.tex", txt)
    write("macro_index_s2.md", "\n".join(f"{k} = {v}" for k, v in sorted(macros.items())) + "\n")
    print(f"  {len(macros)} macros")


if __name__ == "__main__":
    main()
