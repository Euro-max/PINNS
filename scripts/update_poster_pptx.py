"""Update the Canva-exported poster (poster.pptx) in place with the current results.

Text boxes keep Canva's fonts, sizes and paragraph styles (paragraphs are cloned from the originals);
figure and equation frames get new images rendered at each frame's exact aspect ratio.  Every number is
read from results/ or the model config.  Requires python-pptx (not in requirements.txt):

    pip install python-pptx lxml
    python scripts/update_poster_pptx.py poster_canva_original.pptx poster.pptx
"""
import copy
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinc.config import RESULTS_DIR  # noqa: E402

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS = {"a": A[1:-1], "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
C = dict(nmpc_rk4="#1f77b4", pinc="#d62728", blackbox="#2ca02c", linear="#9467bd", ref="#222222")
EMU = 914400


def load(exp, run):
    with open(os.path.join(RESULTS_DIR, exp, run, "summary.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- numbers
def numbers():
    e1, e2, e3, e4, e10 = (load("e1_open_loop", "e1_v2"), load("e2_data_efficiency", "e2_v2"), load("e3_closed_loop", "e3_v2"),
                           load("e4_timing", "e4_v2"), load("e10_confirm", "e10"))
    cfg = json.load(open(os.path.join(RESULTS_DIR, "models", "pinc_v2_s0", "config.json")))
    big, small = e10["results"]["20000"], e10["results"]["100"]
    raw = e3["raw"]["speed_sin"]
    rk = np.array([r["rmse_vx"] for r in raw["nmpc_rk4"]]); bb = np.array([r["rmse_vx"] for r in raw["blackbox"]])
    lane = e3["per_ref"]["lane_change"]
    s4 = e4["solve"]
    return dict(
        deriv=f"{big['dx/dt error vs truth']['ratio_geomean']:.0f}", h50=f"{big['50-step error, in-domain']['ratio_geomean']:.1f}",
        h50x=f"{big['50-step error, extrapolation']['ratio_geomean']:.1f}", h50small=f"{small['50-step error, in-domain']['ratio_geomean']:.0f}",
        n100=f"{e2['curves']['blackbox']['100']['all']['mean']/e2['curves']['pinc']['100']['all']['mean']:.0f}",
        seeds=str(len(e3["seeds"])), lane_p=f"{lane['pinc']['rmse_Y']['mean']*100:.1f}", lane_r=f"{lane['nmpc_rk4']['rmse_Y']['mean']*100:.1f}",
        bbmed=f"{np.median(bb/rk):.1f}", bbworse=str(int((bb > rk).sum())), seed=int(np.argmin(np.abs(bb/rk - np.median(bb/rk)))),
        t40=f"{s4['nmpc_rk4']['40']['median']/s4['pinc']['40']['median']:.1f}",
        depth=str(cfg["model"]["depth"]), width=str(cfg["model"]["width"]), lam=f"{cfg['loss']['lam']:g}",
        ndata=f"{cfg['train']['n_data']:,}".replace(",", " "), lr=f"{cfg['train']['lr']:g}",
        Sx=cfg["scales"]["S_x"], Su=cfg["scales"]["S_u"], Sf=cfg["scales"]["S_f"], e1=e1)


# ---------------------------------------------------------------- shape helpers
def find(shapes, sid):
    for sh in shapes:
        if sh.shape_id == sid:
            return sh
        if sh.shape_type == 6:
            r = find(sh.shapes, sid)
            if r is not None:
                return r
    return None


def absbox(sh):
    x, y, w, h = sh.left, sh.top, sh.width, sh.height
    el = sh._element.getparent()
    while el is not None and el.tag.endswith("}grpSp"):
        xf = el.find("p:grpSpPr/a:xfrm", NS)
        off, ext, co, ce = (xf.find(f"a:{t}", NS) for t in ("off", "ext", "chOff", "chExt"))
        sx = int(ext.get("cx"))/int(ce.get("cx")); sy = int(ext.get("cy"))/int(ce.get("cy"))
        x = int(off.get("x")) + (x - int(co.get("x")))*sx; y = int(off.get("y")) + (y - int(co.get("y")))*sy
        w, h = w*sx, h*sy
        el = el.getparent()
    return x/EMU, y/EMU, w/EMU, h/EMU


def run_templates(txBody):
    """Regular and bold rPr found anywhere in the text body (Canva uses separate 'Canva Sans Bold' faces)."""
    reg = bold = None
    for r in txBody.iter(A + "r"):
        rp = r.find(A + "rPr")
        if rp is None:
            continue
        if rp.get("b") in ("true", "1"):
            bold = bold if bold is not None else rp
        else:
            reg = reg if reg is not None else rp
    return reg, bold


def variant(rpr, make_bold):
    """Copy a run template, switching between Canva Sans and Canva Sans Bold."""
    r = copy.deepcopy(rpr.getparent())
    rp = r.find(A + "rPr")
    if make_bold:
        rp.set("b", "true")
    elif "b" in rp.attrib:
        del rp.attrib["b"]
    for tag in ("latin", "ea", "cs", "sym"):
        f = rp.find(A + tag)
        if f is not None:
            face = f.get("typeface").replace(" Bold", "")
            f.set("typeface", face + (" Bold" if make_bold else ""))
    return rp


def make_run(rpr_tpl, text, size):
    r = copy.deepcopy(rpr_tpl.getparent()) if rpr_tpl is not None else None
    for t in r.findall(A + "t"):
        r.remove(t)
    rp = r.find(A + "rPr")
    if size is not None:
        rp.set("sz", str(size))
    t = r.makeelement(A + "t", {}); t.text = text
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def set_text(slide, sid, paras, reg_from=None, bold_from=None):
    """paras: list of (template_paragraph_index, runs) with runs = str ('**bold**' segments allowed) or None for an empty line."""
    sh = find(slide.shapes, sid)
    tb = sh.text_frame._txBody
    olds = tb.findall(A + "p")
    reg, bold = run_templates(tb)
    if reg is None and bold is not None:
        reg = variant(bold, False)
    if bold is None and reg is not None:
        bold = variant(reg, True)
    newps = []
    for idx, text in paras:
        tpl = olds[idx]
        p = copy.deepcopy(tpl)
        size = None
        for r in list(p):
            if r.tag in (A + "r", A + "br", A + "fld"):
                if r.tag == A + "r" and size is None and r.find(A + "rPr") is not None:
                    size = r.find(A + "rPr").get("sz")
                p.remove(r)
        if size is None:  # empty template paragraph: take size from its endParaRPr
            e = tpl.find(A + "endParaRPr")
            size = e.get("sz") if e is not None else None
        end = p.find(A + "endParaRPr")
        if text:
            parts = text.split("**")
            for k, seg in enumerate(parts):
                if not seg:
                    continue
                r = make_run(bold if k % 2 else reg, seg, size)
                if end is not None:
                    end.addprevious(r)
                else:
                    p.append(r)
        newps.append(p)
    for p in olds:
        tb.remove(p)
    for p in newps:
        tb.append(p)
    return sh


def set_image(slide, sid, png_bytes):
    sh = find(slide.shapes, sid)
    blip = sh._element.find(".//" + A + "blip")
    _, rid = slide.part.get_or_add_image_part(io.BytesIO(png_bytes))
    blip.set(R + "embed", rid)
    ext = blip.find(A + "extLst")
    if ext is not None:  # drop an SVG alternative that would still show the old picture
        for e in list(ext):
            if "svgBlip" in str(e.tag) or any("svgBlip" in str(c.tag) for c in e.iter()):
                ext.remove(e)
    # stretch the new picture exactly onto the frame: Canva crops with srcRect / negative fillRect offsets
    sr = sh._element.find(".//" + A + "srcRect")
    if sr is not None:
        sr.getparent().remove(sr)
    fr = sh._element.find(".//" + A + "fillRect")
    if fr is not None:
        for k in ("l", "t", "r", "b"):
            if k in fr.attrib:
                del fr.attrib[k]
    return absbox(sh)


def fix_pill(slide, gid):
    """Section-title pills: Canva shifts the text box up and relies on an oversized fixed line spacing, which only
    PowerPoint honours.  Make the text box coincide with the pill and centre the text, so every viewer agrees."""
    g = find(slide.shapes, gid)._element
    sps = g.findall("p:sp", NS)
    pill = next(sp for sp in sps if sp.find("p:txBody", NS) is None)
    tbox = next(sp for sp in sps if sp.find("p:txBody", NS) is not None)
    pxf, txf = pill.find("p:spPr/a:xfrm", NS), tbox.find("p:spPr/a:xfrm", NS)
    for tag in ("off", "ext"):
        src, dst = pxf.find(f"a:{tag}", NS), txf.find(f"a:{tag}", NS)
        for k, v in src.attrib.items():
            dst.set(k, v)
    bp = tbox.find("p:txBody/a:bodyPr", NS)
    for k in ("tIns", "bIns"):
        bp.set(k, "0")
    bp.set("anchor", "ctr")
    for ln in tbox.findall(".//a:lnSpc", NS):
        ln.getparent().remove(ln)


def anchor_top(slide, sid, start_y):
    """Top-align a text box so its first line starts at absolute height start_y (inches)."""
    sh = find(slide.shapes, sid)
    bp = sh.text_frame._txBody.find(A + "bodyPr")
    _, y, _, _ = absbox(sh)
    bp.set("anchor", "t")
    bp.set("tIns", str(int(max(0.0, start_y - y)*EMU)))   # insets are not scaled by the enclosing group


# ---------------------------------------------------------------- image rendering
def fig_bytes(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    return b.getvalue()


def equation(lines, w, h, max_pt=30, x0=0.02, yc=None, transparent=True):
    """Render math lines left-aligned in a w x h inch frame (figure size = frame size, so fontsize is the printed
    size in points); shrink to fit, never larger than max_pt."""
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_alpha(0 if transparent else 1)
    fs = float(max_pt)
    for _ in range(40):
        fig.clf()
        ys = [yc] if (yc is not None and len(lines) == 1) else [1 - (i + 0.5)/len(lines) for i in range(len(lines))]
        texts = [fig.text(x0, ys[i], l, fontsize=fs, va="center", ha="left") for i, l in enumerate(lines)]
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        bbs = [t.get_window_extent(rend) for t in texts]
        W = (max(b.x1 for b in bbs)/(fig.dpi*w) - x0)/(1 - x0) + 0.02; Hl = max(b.height for b in bbs)/(fig.dpi*h/len(lines))
        if W <= 0.97 and Hl <= 0.85:
            break
        fs *= min(0.97/W, 0.85/Hl, 0.95)
    return fig_bytes(fig)


def plot_horizon(e1, w, h):
    plt.rcParams.update({"font.size": 15, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    for arm, lab in (("linear", "Linearised model"), ("blackbox", "Data-only network"), ("pinc", "PINC")):
        c = e1["in_box"]["chain"][arm]["curve"]; k = sorted(int(x) for x in c)
        ax.plot(k, [c[str(i)]["all"]["mean"] for i in k], color=C[arm], lw=2.6, label=lab)
    ax.set(yscale="log", xlabel="prediction steps (× 0.1 s)", ylabel="prediction error")
    ax.set_title("Prediction error vs horizon", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    return fig_bytes(fig)


def plot_speed(seed, w, h):
    logs = {a: np.load(os.path.join(RESULTS_DIR, "e3_closed_loop", "e3_v2", f"log_speed_sin_{a}_s{seed}.npz")) for a in ("nmpc_rk4", "pinc", "blackbox")}
    t = logs["pinc"]["t"]; ref = logs["pinc"]["ref"][:, 0]
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    for a, lab in (("blackbox", "Data-only MPC"), ("nmpc_rk4", "Exact-model MPC"), ("pinc", "PINC-MPC")):
        ax.plot(t, np.abs(logs[a]["x"][:, 0] - ref), color=C[a], lw=2.2, label=lab)
    ax.set(xlabel="time [s]", ylabel="|speed error| [m/s]", ylim=(0, None))
    ax.set_title("Speed tracking error (sine reference)", fontsize=16)
    ax.legend(fontsize=12)
    return fig_bytes(fig)


def plot_data(w, h):
    e2 = load("e2_data_efficiency", "e2_v2")
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    ns = [int(k) for k in e2["sizes"]]
    for arm, lab in (("blackbox", "Data-only network"), ("pinc", "PINC")):
        c = [e2["curves"][arm][str(k)]["all"] for k in ns]
        m = np.array([v["mean"] for v in c])
        ax.errorbar(ns, m, yerr=[m - [v["lo"] for v in c], np.array([v["hi"] for v in c]) - m], color=C[arm], marker="o", ms=7,
                    capsize=4, lw=2.4, label=lab)
    ax.set(xscale="log", yscale="log", xlabel="training trajectories", ylabel="test error")
    ax.set_title("Figure 5: prediction error vs training data (5 seeds)", fontsize=15)
    ax.legend(fontsize=12)
    return fig_bytes(fig)


def diagram(n, w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, 22.4); ax.set_ylim(0, 10); ax.axis("off")
    fig.patch.set_alpha(0)
    ins = ["$t$", "$v_x$", "$v_y$", "$r$", r"$\psi$", "$F_x$", r"$\delta$"]
    for i, s in enumerate(ins):
        y = 9.0 - i*1.25
        ax.add_patch(plt.Circle((1.0, y), 0.5, color="#1f4e8c")); ax.text(1.0, y, s, color="white", ha="center", va="center", fontsize=15)
    depth = int(n["depth"])
    for j in range(depth):
        x = 3.2 + j*1.45
        ax.add_patch(plt.Rectangle((x, 1.2), 1.0, 7.6, color="#4a7fc1", ec="none"))
        ax.text(x + 0.5, 5.0, f"{n['width']}\ntanh", color="white", ha="center", va="center", fontsize=11, rotation=90)
    xs = 3.2 + depth*1.45 + 0.2
    for i in range(7):
        ax.plot([1.5, 3.2], [9.0 - i*1.25, 5.0 + (i - 3)*1.0], color="#9bb7dd", lw=0.8)
    ax.add_patch(plt.Rectangle((xs, 3.6), 1.6, 2.8, color="#d62728", ec="none"))
    ax.text(xs + 0.8, 5.0, "×D", color="white", ha="center", va="center", fontsize=16, weight="bold")
    outs = ["$v_x$", "$v_y$", "$r$", r"$\psi$"]
    for i, s in enumerate(outs):
        y = 7.6 - i*1.75
        ax.plot([xs + 1.6, xs + 2.6], [5.0, y], color="#9bb7dd", lw=0.8)
        ax.add_patch(plt.Circle((xs + 3.1, y), 0.5, color="#1f4e8c")); ax.text(xs + 3.1, y, s, color="white", ha="center", va="center", fontsize=15)
    ax.text(1.0, 0.25, "inputs", ha="center", fontsize=12); ax.text(3.2 + depth*0.72, 0.25, f"{depth} hidden layers", ha="center", fontsize=12)
    ax.text(xs + 3.1, 0.25, "state at t", ha="center", fontsize=12)
    return fig_bytes(fig)


def scale_table(n, w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.axis("off"); fig.patch.set_alpha(0)
    rows = [["$v_x$", f"{n['Sx'][0]:g} m/s", f"{n['Sf'][0]:.2g} m/s²"], ["$v_y$", f"{n['Sx'][1]:g} m/s", f"{n['Sf'][1]:.2g} m/s²"],
            ["$r$", f"{n['Sx'][2]:g} rad/s", f"{n['Sf'][2]:.2g} rad/s²"], [r"$\psi$", f"{n['Sx'][3]:g} rad", f"{n['Sf'][3]:.2g} rad/s"],
            ["$F_x$, $\\delta$", f"{n['Su'][0]:g} N, {n['Su'][1]:g} rad", "–"]]
    t = ax.table(cellText=rows, colLabels=["", "scale $S_x$, $S_u$", "rate $S_f$"], loc="center", cellLoc="center", colLoc="center",
                 bbox=[0, 0, 1, 1], edges="horizontal")
    t.auto_set_font_size(False); t.set_fontsize(13)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#888888")
        if r == 0:
            cell.set_text_props(weight="bold")
    return fig_bytes(fig)


# ---------------------------------------------------------------- main
def main(src, dst):
    n = numbers()
    prs = Presentation(src); s = prs.slides[0]

    set_text(s, 148, [(0, "Physics-Informed Neural Networks"), (1, "for Predictive Vehicle Control")])
    set_text(s, 11, [(0, "Model predictive control (**MPC**) needs a prediction model that is both fast and accurate. We train a "
                        "physics-informed neural network for control (**PINC**) on a vehicle model with coupled speed and steering "
                        f"dynamics and use it inside MPC. Against the same network trained on data alone, the physics term made the learned "
                        f"dynamics **{n['deriv']}×** more accurate and cut 50-step prediction errors **{n['h50']}×** (**{n['h50small']}×** with only "
                        f"100 training trajectories). In closed loop, PINC-MPC tracked as well as an MPC using the exact model over four "
                        f"manoeuvres and **{n['seeds']}** noisy runs each.")])
    set_text(s, 83, [(0, "We describe the vehicle with a single-track (bicycle) model: longitudinal speed vx, lateral speed vy, yaw rate r and "
                        "heading ψ, driven by a longitudinal force Fx and steering angle δ. The tyre forces depend on the front and rear slip "
                        "angles, so speed and steering are coupled. **PINC** learns this model and replaces it inside the **MPC**.")])
    set_text(s, 80, [(0, "**Longitudinal Dynamics:**")])
    set_text(s, 87, [(0, "**Lateral and Yaw Dynamics:**")])
    set_text(s, 110, [(0, "**Our Contribution: Physics-Informed Model**"), (1, "1."),
                      (2, "A neural network trained with a physics-informed loss maps time t, initial state and input u to the predicted "
                          "state; chaining it over the horizon gives the MPC its model."), (3, None)])
    set_text(s, 115, [(0, "2."), (1, "Exact initial condition and a per-state scaling D, which keeps the four physics terms balanced:"),
                      (2, None), (3, None), (4, None)])
    set_text(s, 94, [(0, "Approach:"), (1, "One **MPC**, four prediction models: exact model, **PINC**, data-only network, linearised model."),
                     (2, "Data-only network: the same network trained without the physics term (λ = 0)."), (3, None)])
    set_text(s, 97, [(0, "Neural Network Architecture"), (1, "**7** inputs: time, initial state (vx, vy, r, ψ), force and steering"),
                     (2, f"**{n['depth']}** hidden layers × **{n['width']}** neurons, tanh activation"),
                     (3, "Exact initial condition, per-state output scaling D"), (4, "**4** outputs: vx, vy, yaw rate r and heading ψ"), (5, None)])
    set_text(s, 102, [(0, "Training Protocol"), (1, "Loss: "), (2, f"Optimizers: **Adam** then **L-BFGS** (lr = **{n['lr']}**)"),
                      (3, f"Data: **{n['ndata']}** simulated trajectories"), (4, "Normalization:"), (5, None), (6, None), (7, None)])
    set_text(s, 106, [(0, "Test manoeuvres"), (1, None)])
    set_text(s, 145, [
        (0, "**Does PINC learn the physics?**"), (1, "**Prediction (5 training seeds):**"),
        (2, f"Learned dynamics **{n['deriv']}×** more accurate than the data-only network."),
        (3, f"50-step prediction error **{n['h50']}×** lower, and **{n['h50x']}×** lower outside the training range (Figure 3, left)."),
        (5, f"With only **100** training trajectories, **{n['n100']}×** more accurate (Figure 5)."), (7, None),
        (8, "**Closed-Loop Control:**"), (9, f"**4 manoeuvres × {n['seeds']} noisy runs:**"),
        (10, f"**PINC-MPC** matches the exact-model MPC (lane change: **{n['lane_p']}** cm vs **{n['lane_r']}** cm error)."),
        (11, f"The data-only MPC drifts in speed: median error **{n['bbmed']}×** higher, in all **{n['bbworse']}** runs (Figure 3, right)."),
        (13, f"Not faster than the exact model at short horizons; **{n['t40']}×** faster at **40** steps."),
    ])
    set_text(s, 67, [(0, "Physics-informed training made the network learn the vehicle dynamics instead of only fitting data. Next steps:"),
                     (1, "Unknown vehicles: learn from measured data, with physics as a prior."),
                     (2, "Higher-fidelity vehicle models, where the surrogate's speed advantage grows."),
                     (3, "Hardware-in-the-loop and on-vehicle tests."),
                     (4, "Key lesson: scale the physics loss per state; without it, physics made the model less accurate."), (5, None)])
    set_text(s, 142, [(1, "[1] S. Wei et al. An integrated longitudinal and lateral vehicle-following control system with radar and V2V "
                          "communication. IEEE Trans. Veh. Technol., 68(2):1116–1127, 2019."),
                      (2, "[2] A. Farea et al. Understanding physics-informed neural networks: techniques, applications, trends, and "
                          "challenges. AI, 5(3):1534–1557, 2024."),
                      (3, "[3] E. A. Antonelo et al. Physics-informed neural nets for control of dynamical systems. Neurocomputing, "
                          "579:127419, 2024.")])

    def box(sid):
        return absbox(find(s.shapes, sid))
    cap = {74: 30, 84: 30, 146: 26, 147: 26, 112: 28, 99: 22, 103: 22}
    pos = {99: dict(x0=0.12, yc=0.75)}   # loss equation: right of the "Loss:" label, on its line
    eq = {
        74: [r"$m(\dot v_x - v_y r) = F_x - \frac{1}{2}\rho C_d A v_x^2 - F_{rr} - F_{yf}\sin\delta$"],
        84: [r"$m(\dot v_y + v_x r) = F_{yf}\cos\delta + F_{yr},\quad I_z\,\dot r = l_f F_{yf}\cos\delta - l_r F_{yr}$"],
        146: [r"$J = \sum_{k=1}^{N} e_k^\top Q\, e_k + \sum_{k=0}^{N-1} \tilde u_k^\top R\,\tilde u_k + \Delta\tilde u_k^\top R_\Delta \Delta\tilde u_k$"],
        147: [r"$s_{k+1} = f_w(T, s_k, u_k),\quad u_{\min} \leq u_k \leq u_{\max}$"],
        112: [r"$\hat s(t) = \hat s_0 + \frac{t}{T}\,D\,f_w(t, s_0, u)$"],
        99: [r"$\mathcal{L} = \mathcal{L}_{data} + \lambda\,\mathcal{L}_{phys},\ \lambda = " + n["lam"] + "$"],
        103: [r"$\mathrm{speed\ sine \cdot speed\ step \cdot lane\ change \cdot double\ lane\ change}$"],
    }
    for sid, lines in eq.items():
        _, _, w, h = box(sid)
        set_image(s, sid, equation(lines, w, h, cap[sid], **pos.get(sid, {})))
    for sid in (107, 111):   # symbol images beside "1." / "2." no longer referenced by the text
        el = find(s.shapes, sid)._element
        el.getparent().remove(el)
    _, _, w, h = box(119); set_image(s, 119, plot_horizon(n["e1"], w, h))
    _, _, w, h = box(120); set_image(s, 120, plot_speed(n["seed"], w, h))
    _, _, w, h = box(88); set_image(s, 88, diagram(n, w, h))
    _, _, w, h = box(98); set_image(s, 98, scale_table(n, w, h))
    # data-efficiency figure in the free space at the bottom of the Conclusion panel (26.7-38.2 in)
    fx, fy, fw, fh = 23.55, 33.3, 8.7, 4.7
    s.shapes.add_picture(io.BytesIO(plot_data(fw, fh)), int(fx*EMU), int(fy*EMU), int(fw*EMU), int(fh*EMU))
    for gid in (12, 15, 21, 31, 34, 37, 40, 62, 43):   # section titles and the tagline banner
        fix_pill(s, gid)
    sup = find(s.shapes, 150)                            # "Supervised by ..." wrapped in non-PowerPoint viewers
    sup.left, sup.width = int(10.9*EMU), int(11.8*EMU)
    anchor_top(s, 127, 19.35)   # Problem Definition (original text): start below its title pill in every viewer
    anchor_top(s, 145, 31.3)   # Results text: just below Figure 3 and its caption
    anchor_top(s, 67, 27.2)    # Conclusion: just below its heading
    anchor_top(s, 142, 40.2)   # References: just below its heading
    prs.save(dst)
    print("wrote", dst, "| speed-figure run:", n["seed"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
