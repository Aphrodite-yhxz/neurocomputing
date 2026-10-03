"""Research prototype: temporary pending-error hints from same-target revisions.

Not a claimed new conformal theorem or a validated submission method. All labels enter
the state only after their reveal time. The protocol and novelty exclusions live in
docs/INNOVATION_PROTOCOL_V1.md. No legacy benchmark files are modified.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .revision import load_archive, band, past_error_scale


@dataclass
class RevisionPanel:
    horizons: np.ndarray
    y: np.ndarray
    left: np.ndarray
    right: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    residuals: np.ndarray
    history: np.ndarray


@njit(cache=True)
def _histories(residuals, horizons, window):
    k, n = residuals.shape
    out = np.full((k, n, window), np.nan)
    for a in range(k):
        for t in range(window + horizons[a] - 1, n):
            end = t - horizons[a] + 1
            out[a, t] = np.sort(residuals[a, end-window:end])
    return out


def prepare(path, window=450):
    horizons = np.array([1, 6, 24], dtype=np.int64)
    aa = [load_archive(path, int(h)) for h in horizons]
    for a in aa:
        # Forecast at row t/h targets the same observation as t+h-k/k.
        np.testing.assert_allclose(a.y[:len(a.y)-a.delay+1],
                                   aa[0].y[a.delay-1:], rtol=0, atol=1e-8)
    bands = [band(a) for a in aa]
    left = np.array([b[0] for b in bands]); right = np.array([b[1] for b in bands])
    center = np.array([np.sort(a.quantiles, axis=1)[:, 2] for a in aa])
    scale = []
    for i, a in enumerate(aa):
        es = past_error_scale(a.y, center[i], left[i], right[i], a.delay)
        scale.append(np.maximum((right[i]-left[i])/2, .05*es))
    scale = np.maximum(np.array(scale), 1e-12)
    y = np.array([a.y for a in aa]); residuals = (y-center)/scale
    return RevisionPanel(horizons, y, left, right, center, scale, residuals,
                         _histories(residuals, horizons, window))


@njit(cache=True)
def _probability(hist, low, high):
    # Strict lower miss and strict upper miss, including point-mass/tie cases.
    nl = np.searchsorted(hist, low, side='left')
    nu = np.searchsorted(hist, high, side='right')
    return (nl + len(hist) - nu) / len(hist)


@njit(cache=True)
def _run(y, left, right, center, scale, histories, horizons, k, mode,
         alpha, gamma, weight, prefix):
    n = y.shape[1]; h = horizons[k]; start = prefix+h-1
    scores = np.maximum(left[k]-y[k], y[k]-right[k])
    unit = max(np.quantile(np.abs(scores[:prefix]), .9),
               np.quantile(right[k, :prefix]-left[k, :prefix], .5), 1e-12)
    eta = gamma*unit
    initial = np.quantile(scores[:prefix], 1-alpha)
    state = initial
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    emitted = np.full(n, np.nan); states = np.full(n, np.nan)
    debt = np.zeros(n); reflection = np.zeros(n)
    errors = np.full(n, np.nan)
    # Records estimate-event pairs, not independent data points; assessed after run.
    probs = np.full((n, max(h-1, 1)), np.nan)
    indices = np.full((n, max(h-1, 1)), -1, dtype=np.int64)
    repairs = 0
    for t in range(start, n):
        j = t-h
        if j >= start:
            err = 1. if y[k,j] < lo[j] or y[k,j] > hi[j] else 0.
            errors[j] = err
            raw = state + eta*(err-alpha)
            state = max(-unit, raw)
            reflection[t] = state-raw
        states[t] = state
        hint = 0.
        if mode != 0 and weight > 0:
            for j in range(max(start, t-h+1), t):
                if mode == 1:
                    end = t-h+1
                    begin = max(start, end-20)
                    p = alpha if end <= begin else np.mean(errors[begin:end])
                else:
                    kk = k; origin = j
                    if mode >= 3:
                        for v in range(len(horizons)):
                            u = j+h-horizons[v]
                            if horizons[v] <= h and u <= t:
                                kk = v; origin = u; break
                    if mode == 4:
                        origin = max(0, origin-24)
                    low = (lo[j]-center[kk,origin])/scale[kk,origin]
                    high = (hi[j]-center[kk,origin])/scale[kk,origin]
                    p = _probability(histories[kk,t], low, high)
                slot = t-j-1
                probs[t,slot] = p; indices[t,slot] = j
                hint += p-alpha
        debt[t] = weight*hint
        q = state+eta*debt[t]
        emitted[t] = q
        lo[t] = left[k,t]-q; hi[t] = right[k,t]+q
        if lo[t] > hi[t]:
            lo[t] = hi[t] = (left[k,t]+right[k,t])/2
            repairs += 1
    # Hindsight diagnostics are separate from prediction/state construction.
    for j in range(start, n):
        errors[j] = 1. if y[k,j] < lo[j] or y[k,j] > hi[j] else 0.
    return lo, hi, emitted, states, debt, reflection, errors, probs, indices, unit, repairs


MODES = {'none': 0, 'recent': 1, 'original': 2, 'revised': 3, 'stale': 4}


def calibrate_pending(panel, horizon, mode='revised', alpha=.1, gamma=.02,
                      weight=1., prefix=600):
    if mode not in MODES or horizon not in panel.horizons:
        raise ValueError('Unknown mode/horizon')
    if not 0 < alpha < 1 or gamma <= 0 or not 0 <= weight <= 1:
        raise ValueError('Invalid controller configuration')
    k = int(np.flatnonzero(panel.horizons == horizon)[0])
    if prefix < panel.history.shape[2]+max(panel.horizons) or prefix+horizon >= panel.y.shape[1]:
        raise ValueError('Insufficient causal initialization/history')
    r = _run(panel.y, panel.left, panel.right, panel.center, panel.scale,
             panel.history, panel.horizons, k, MODES[mode], alpha, gamma, weight, prefix)
    lo,hi,q,z,debt,reflection,errors,probs,indices,unit,repairs = r
    return lo,hi,dict(q=q,state=z,debt=debt,reflection=reflection,errors=errors,
                     probabilities=probs,indices=indices,unit=unit,repairs=repairs,
                     start=prefix+horizon-1,eta=gamma*unit)


def hint_diagnostics(diag, start=1200, stop=None, alpha=.1):
    p = diag['probabilities'][start:stop]; jj = diag['indices'][start:stop]
    mask = jj >= 0
    if not np.any(mask):
        return dict(hint_count=0,brier=np.nan,brier_skill=np.nan,debt_mae=np.nan)
    observed = diag['errors'][np.maximum(jj,0)]
    mse = float(np.mean((p[mask]-observed[mask])**2))
    base = float(np.mean((alpha-observed[mask])**2))
    true_debt = np.where(mask,observed-alpha,0).sum(axis=1)
    estimated_debt = np.where(mask,p-alpha,0).sum(axis=1)
    return dict(hint_count=int(mask.sum()),brier=mse,
                brier_skill=1-mse/base if base>0 else np.nan,
                debt_mae=float(np.mean(abs(estimated_debt-true_debt))))
