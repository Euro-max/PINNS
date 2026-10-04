"""Metrics for surrogate accuracy and closed-loop tracking, with bootstrap CIs
and paired Wilcoxon tests (ground rule 4: every number is computed here)."""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---- surrogate accuracy ------------------------------------------------------
def rmse(pred, true, axis=0):
    return np.sqrt(np.mean((np.asarray(pred) - np.asarray(true))**2, axis=axis))


def nrmse(pred, true, scale, axis=0):
    """RMSE divided by the per-state scale S_x (dimensionless)."""
    return rmse(pred, true, axis)/np.asarray(scale)


# ---- closed-loop -------------------------------------------------------------
def iae(e, T):
    """Integral of absolute error: sum |e_k| * T  (fixes D18)."""
    return float(np.sum(np.abs(e))*T)


def max_abs(e):
    return float(np.max(np.abs(e)))


def p95_abs(e):
    return float(np.percentile(np.abs(e), 95))


def control_effort(u_tilde, R, T):
    """sum_k u~_k^T R u~_k * T with u~ = u/S_u (normalised)."""
    u_tilde = np.asarray(u_tilde)
    return float(np.sum((u_tilde**2)*np.asarray(R))*T)


def violation_time(r, r_max, T):
    return float(np.sum(np.abs(r) > r_max)*T)


def rise_time(t, y, y0, y1, lo=0.1, hi=0.9):
    """10-90 % rise time of y from level y0 to y1; NaN if never reached."""
    t, y = np.asarray(t), np.asarray(y)
    d = y1 - y0
    if d == 0:
        return float("nan")
    frac = (y - y0)/d
    i_lo = np.argmax(frac >= lo) if np.any(frac >= lo) else None
    i_hi = np.argmax(frac >= hi) if np.any(frac >= hi) else None
    if i_lo is None or i_hi is None:
        return float("nan")
    return float(t[i_hi] - t[i_lo])


def settling_time(t, e, band):
    """Time after which |e| stays within `band` for the rest of the record."""
    t, e = np.asarray(t), np.asarray(e)
    outside = np.abs(e) > band
    if not np.any(outside):
        return float(t[0])
    last = np.max(np.nonzero(outside)[0])
    if last + 1 >= len(t):
        return float("nan")
    return float(t[last + 1])


def solve_time_stats(times):
    times = np.asarray(times)
    return dict(mean=float(times.mean()), median=float(np.median(times)),
                p95=float(np.percentile(times, 95)), max=float(times.max()))


# ---- statistics across seeds ------------------------------------------------
def bootstrap_ci(x, n_boot=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(means, 100*alpha/2)),
                hi=float(np.percentile(means, 100*(1 - alpha/2))), n=int(x.size))


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test a vs b (same seeds).  Effect size is the
    matched-pairs rank-biserial correlation r = (W+ - W-)/(W+ + W-)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    d = d[d != 0]
    if d.size < 2:
        return dict(p=float("nan"), effect=float("nan"), n=int(d.size))
    res = stats.wilcoxon(d)
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    r = (w_pos - w_neg)/(w_pos + w_neg)
    return dict(p=float(res.pvalue), effect=float(r), n=int(d.size))
