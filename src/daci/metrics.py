"""
daci.metrics
============
Evaluation suite. The headline metrics for the paper:
  - marginal coverage gap        (validity)
  - mean width / MSIW            (efficiency)
  - Winkler interval score       (proper scoring rule for intervals; lower better)
  - windowed coverage error & worst-slab coverage  (conditional validity)
  - post-change-point coverage error & recovery lag (adaptivity -- DACI's selling point)
  - runtime per 1k steps         (overhead; DACI's internal signal must be ~free)
"""
import numpy as np


def _rolling_cov(covered, win):
    c = covered.astype(float)
    if len(c) < win:
        return np.full(len(c), c.mean())
    rc = np.convolve(c, np.ones(win) / win, mode="valid")
    pad = np.full(win - 1, rc[0])
    return np.concatenate([pad, rc])


def winkler(y, lo, up, alpha):
    w = up - lo
    below = y < lo
    above = y > up
    s = w.copy().astype(float)
    s[below] += (2 / alpha) * (lo[below] - y[below])
    s[above] += (2 / alpha) * (y[above] - up[above])
    return s


def evaluate(result, trace, alpha=0.10, cov_win=150, tol=0.05, post_H=150):
    m = result.eval_mask
    y = trace.y_true
    lo, up = result.lower, result.upper
    covered = (y >= lo) & (y <= up)
    target = 1 - alpha

    cov = covered[m].mean()
    width = (up - lo)[m]
    out = {
        "method": result.name,
        "coverage": float(cov),
        "cov_gap": float(abs(cov - target)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "MSIW": float(np.mean(width) / (np.std(y[m]) + 1e-9)),
        "winkler": float(np.mean(winkler(y, lo, up, alpha)[m])),
        "runtime_ms_per1k": float(1000 * result.runtime / max(1, m.sum()) * 1000),
    }

    # conditional validity: rolling coverage error & worst slab
    rc = _rolling_cov(covered, cov_win)
    out["local_cov_err"] = float(np.mean(np.abs(rc[m] - target)))
    out["worst_slab_cov"] = float(np.min(rc[m]))  # lowest local coverage seen

    # post-change-point adaptivity
    cps = trace.change_points
    if cps:
        pos = np.searchsorted(trace.t_index, cps)
        errs, lags = [], []
        for p in pos:
            seg = slice(p, min(len(trace), p + post_H))
            sm = result.eval_mask[seg]
            if sm.sum() > 10:
                errs.append(abs(covered[seg][sm].mean() - target))
            # recovery lag: steps after CP until trailing rolling coverage re-enters band
            lag = post_H
            for k in range(p, min(len(trace), p + post_H)):
                if abs(rc[k] - target) <= tol:
                    lag = k - p; break
            lags.append(lag)
        out["postCP_cov_err"] = float(np.mean(errs)) if errs else np.nan
        out["recovery_lag"] = float(np.mean(lags)) if lags else np.nan
    else:
        out["postCP_cov_err"] = np.nan
        out["recovery_lag"] = np.nan
    return out
