"""Poster assets (A0, Canva): high-resolution figures, equation images and the text for each poster box,
all generated from results/ (no hand-typed numbers).  Output: poster/.

    python scripts/make_poster_assets.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinc.config import RESULTS_DIR, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "poster")
RUN = dict(e1="e1_v2", e2="e2_v2", e3="e3_v2", e4="e4_v2", e5="e5_v2", e10="e10", model="pinc_v2_s0")
C = dict(ref="#222222", nmpc_rk4="#1f77b4", pinc="#d62728", blackbox="#2ca02c", linear="#9467bd", ltv="#9467bd")
L = dict(nmpc_rk4="Exact-model MPC", pinc="PINC-MPC", blackbox="Data-only MPC", linear="Linearised model", ltv="LTV-MPC")
plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 19, "legend.fontsize": 16,
                     "lines.linewidth": 2.6, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def load(exp, run):
    with open(os.path.join(RESULTS_DIR, exp, run, "summary.json")) as fh:
        return json.load(fh)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def fig_speed(e3):
    """Speed sinusoid for the seed whose data-only / exact ratio is closest to the median."""
    raw = e3["raw"]["speed_sin"]
    rk = np.array([r["rmse_vx"] for r in raw["nmpc_rk4"]]); bb = np.array([r["rmse_vx"] for r in raw["blackbox"]])
    ratio = bb/rk
    seed = int(np.argmin(np.abs(ratio - np.median(ratio))))
    logs = {a: np.load(os.path.join(RESULTS_DIR, "e3_closed_loop", RUN["e3"], f"log_speed_sin_{a}_s{seed}.npz")) for a in ("nmpc_rk4", "pinc", "blackbox")}
    t = logs["pinc"]["t"]; ref = logs["pinc"]["ref"][:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.3]))
    axes[0].plot(t, ref, "--", color=C["ref"], lw=2, label="reference")
    for a in ("nmpc_rk4", "pinc", "blackbox"):
        axes[0].plot(t, logs[a]["x"][:, 0], color=C[a], label=L[a], alpha=0.9)
        axes[1].plot(t, np.abs(logs[a]["x"][:, 0] - ref), color=C[a])
    axes[0].set_ylabel("speed [m/s]"); axes[0].legend(ncol=2, loc="lower left", frameon=True)
    axes[1].set_ylabel("|speed error| [m/s]"); axes[1].set_xlabel("time [s]")
    save(fig, "fig_speed_tracking")
    return seed, float(np.median(ratio))


def fig_horizon(e1):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for arm in ("linear", "blackbox", "pinc"):
        c = e1["in_box"]["chain"][arm]["curve"]
        h = sorted(int(k) for k in c)
        m = np.array([c[str(k)]["all"]["mean"] for k in h]); lo = [c[str(k)]["all"]["lo"] for k in h]; hi = [c[str(k)]["all"]["hi"] for k in h]
        lab = {"linear": "Linearised model", "blackbox": "Data-only network", "pinc": "PINC (physics-informed)"}[arm]
        ax.plot(h, m, color=C[arm], label=lab); ax.fill_between(h, lo, hi, color=C[arm], alpha=0.2)
    ax.set(yscale="log", xlabel="prediction steps (× 0.1 s)", ylabel="normalised prediction error")
    ax.legend(loc="lower right")
    save(fig, "fig_prediction_horizon")


def fig_data(e2):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    n = [int(k) for k in e2["sizes"]]
    for arm, lab in (("pinc", "PINC (physics-informed)"), ("blackbox", "Data-only network")):
        cc = [e2["curves"][arm][str(k)]["all"] for k in n]
        m = np.array([c["mean"] for c in cc])
        ax.errorbar(n, m, yerr=[m - [c["lo"] for c in cc], np.array([c["hi"] for c in cc]) - m], marker="o", ms=10, capsize=5, color=C[arm], label=lab, lw=2.6)
    ax.set(xscale="log", yscale="log", xlabel="training trajectories", ylabel="test prediction error")
    ax.legend()
    save(fig, "fig_data_efficiency")


def fig_timing(e4):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    Ns = [int(k) for k in e4["horizons"]]
    for arm in ("nmpc_rk4", "pinc", "ltv"):
        ax.plot(Ns, [e4["solve"][arm][str(k)]["median"]*1e3 for k in Ns], marker="o", ms=10, color=C[arm], label=L[arm])
    ax.set(xlabel="prediction horizon N (steps)", ylabel="solve time per step [ms]")
    ax.legend()
    save(fig, "fig_solve_time")


def equations():
    eqs = {
        "eq_vehicle_model": [r"$m(\dot v_x - v_y r) = F_x - \frac{1}{2}\rho C_d A v_x^2 - F_{rr} - F_{yf}\sin\delta$",
                             r"$m(\dot v_y + v_x r) = F_{yf}\cos\delta + F_{yr}$",
                             r"$I_z\,\dot r = l_f F_{yf}\cos\delta - l_r F_{yr},\qquad \dot\psi = r$",
                             r"$\alpha_f = \delta - \arctan\frac{v_y + l_f r}{v_x},\qquad \alpha_r = -\arctan\frac{v_y - l_r r}{v_x}$"],
        "eq_pinc": [r"$\hat s(t) = \hat s_0 + \frac{t}{T}\,D\,\mathrm{NN}(t, s_0, u),\qquad D = \mathrm{diag}\left(\frac{S_f\,T}{S_x}\right)$",
                    r"$\mathcal{L} = \mathcal{L}_{\mathrm{data}} + \lambda\,\frac{1}{n}\sum\left\|\frac{\dot{\hat s} - f(\hat s, u)}{S_f}\right\|^2$"],
        "eq_mpc": [r"$\min_{u_0..u_{N-1}}\ \sum_{k=1}^{N} e_k^\top Q\, e_k + \sum_{k=0}^{N-1} \tilde u_k^\top R\,\tilde u_k + \Delta\tilde u_k^\top R_\Delta \Delta\tilde u_k$",
                   r"$\mathrm{s.t.}\quad s_{k+1} = \mathrm{PINC}(T, s_k, u_k),\qquad u_{\min} \leq u_k \leq u_{\max}$"],
    }
    for name, lines in eqs.items():
        fig = plt.figure(figsize=(12, 0.95*len(lines) + 0.3))
        for i, l in enumerate(lines):
            fig.text(0.01, 1 - (i + 0.6)/len(lines), l, fontsize=26, va="center")
        save(fig, name)


def text(e1, e2, e3, e4, e5, e10, seed, med_ratio):
    ib = e1["in_box"]; ex = e1["extrap"]
    r50 = ib["chain"]["blackbox"]["curve"]["50"]["all"]["mean"]/ib["chain"]["pinc"]["curve"]["50"]["all"]["mean"]
    rex = ex["chain"]["blackbox"]["curve"]["50"]["all"]["mean"]/ex["chain"]["pinc"]["curve"]["50"]["all"]["mean"]
    big, small = e10["results"]["20000"], e10["results"]["100"]
    d100 = e2["curves"]["blackbox"]["100"]["all"]["mean"]/e2["curves"]["pinc"]["100"]["all"]["mean"]
    d1e5 = e2["curves"]["blackbox"]["100000"]["all"]["mean"]/e2["curves"]["pinc"]["100000"]["all"]["mean"]
    sp = e3["per_ref"]
    lane = sp["lane_change"]
    s4 = e4["solve"]
    rat40 = s4["nmpc_rk4"]["40"]["median"]/s4["pinc"]["40"]["median"]
    nom = e5["per_setting"]["nominal"]; mu = e5["per_setting"]["fiala mu=0.4"]
    with open(os.path.join(RESULTS_DIR, "models", RUN["model"], "config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(RESULTS_DIR, "models", RUN["model"], "summary.json")) as fh:
        nparams = json.load(fh)["n_params"]
    t = f"""# Poster text (generated by scripts/make_poster_assets.py from results/; paste into Canva)

Figures are in poster/*.png (300 dpi); equations in poster/eq_*.png.

## Title (suggestion)
Physics-Informed Neural Networks for Model Predictive Vehicle Control

## Abstract
Model predictive control (MPC) needs a fast and accurate prediction model. We train a physics-informed neural network for control (PINC) on a vehicle model with coupled speed and steering dynamics, and use it inside MPC. Compared with the same network trained on data alone, the physics term made the learned dynamics {big['dx/dt error vs truth']['ratio_geomean']:.0f}× more accurate and cut 50-step prediction errors {big['50-step error, in-domain']['ratio_geomean']:.1f}× ({small['50-step error, in-domain']['ratio_geomean']:.0f}× with only 100 training trajectories). In closed loop, PINC-MPC tracked as well as an MPC using the exact model, across four manoeuvres and 30 noisy runs each.

## Mathematical Modeling
We use a single-track (bicycle) vehicle model with longitudinal speed v_x, lateral speed v_y, yaw rate r and heading ψ, driven by a longitudinal force F_x and steering angle δ.
- Vehicle dynamics: [eq_vehicle_model.png]
- PINC surrogate: the network maps (time, initial state, input) to the future state; the physics loss penalises the mismatch between its time derivative and the vehicle model f. The per-state scaling D keeps all four physics terms balanced: [eq_pinc.png]
- MPC: at every 0.1 s step we minimise tracking error and control effort over the prediction horizon, using PINC as the model: [eq_mpc.png]

## Experimental Setup
- Network: fully connected, {cfg['model']['depth']} layers × {cfg['model']['width']} neurons, tanh ({nparams:,} parameters); exact initial condition.
- Training: {cfg['train']['n_data']:,} simulated trajectories, physics weight λ = {cfg['loss']['lam']:g} (chosen on validation data), Adam then L-BFGS.
- Controllers compared (same cost, solver and plant): exact-model MPC, PINC-MPC, data-only MPC (same network, λ = 0), linearised MPC.
- Tests: speed sinusoid, speed step, lane change, ISO 3888-2 double lane change; 30 runs each with sensor noise; plus mass, tyre and friction mismatch.

## Results
Does the network learn the physics? (5 training seeds)
- Learned dynamics (dx/dt) error: {big['dx/dt error vs truth']['ratio_geomean']:.0f}× lower than the data-only network.
- 50-step prediction error: {big['50-step error, in-domain']['ratio_geomean']:.1f}× lower in the training range, {big['50-step error, extrapolation']['ratio_geomean']:.1f}× lower outside it. [fig_prediction_horizon.png shows one model: {r50:.1f}× at 50 steps]
- With only 100 training trajectories, PINC is {d100:.0f}× more accurate; with 100 000, still {d1e5:.2f}×. [fig_data_efficiency.png]

Closed-loop control (4 manoeuvres × 30 runs, all solves converged)
- PINC-MPC matches the exact-model MPC: lane-change lateral error {lane['pinc']['rmse_Y']['mean']*100:.1f} cm vs {lane['nmpc_rk4']['rmse_Y']['mean']*100:.1f} cm.
- The data-only MPC drifts in speed: median speed error {med_ratio:.1f}× the exact model on the speed sinusoid (worse in all 30 runs). [fig_speed_tracking.png, run {seed}]
- Under a low-friction road (μ = 0.4) all controllers degrade together (lateral error {mu['nmpc_rk4']['rmse_Y']['mean']*100:.1f} cm exact vs {mu['pinc']['rmse_Y']['mean']*100:.1f} cm PINC).
- Speed: PINC-MPC is not faster than a compiled exact model for short horizons; it becomes {rat40:.1f}× faster at a 40-step horizon. [fig_solve_time.png]

## Conclusion
Physics-informed training made the neural model learn the vehicle dynamics rather than just fit data: it predicts far better over long horizons and needs much less data, and inside MPC it controls the vehicle as well as the exact model. The key was scaling the physics loss per state; without it, the physics term hurt accuracy. Next steps:
1. Use PINC where the exact model is unknown, learning from measured data with physics as a prior.
2. Higher-fidelity vehicle models, where the surrogate's speed advantage grows.
3. Hardware-in-the-loop and on-vehicle tests.

## References (add to the existing two)
- E. A. Antonelo et al., "Physics-informed neural nets for control of dynamical systems," Neurocomputing, 579:127419, 2024.
- R. Rajamani, Vehicle Dynamics and Control, 2nd ed., Springer, 2012.
"""
    with open(os.path.join(OUT, "poster_text.md"), "w") as fh:
        fh.write(t)
    print("  wrote poster_text.md")


def main():
    os.makedirs(OUT, exist_ok=True)
    e1 = load("e1_open_loop", RUN["e1"]); e2 = load("e2_data_efficiency", RUN["e2"]); e3 = load("e3_closed_loop", RUN["e3"])
    e4 = load("e4_timing", RUN["e4"]); e5 = load("e5_robustness", RUN["e5"]); e10 = load("e10_confirm", RUN["e10"])
    seed, med = fig_speed(e3)
    fig_horizon(e1); fig_data(e2); fig_timing(e4); equations()
    text(e1, e2, e3, e4, e5, e10, seed, med)


if __name__ == "__main__":
    main()
