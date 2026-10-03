"""Audited retrospective benchmark primitives for the Neurocomputing revision.

Legacy modules are deliberately unchanged. All inputs are archived forecasts, all
methods share an excluded prefix, and score features use only observed outcomes.
No finite-sample conditional validity is claimed for these time-series experiments.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numba import njit


@dataclass
class Archive:
    backbone: str
    dataset: str
    origin: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    quantiles: np.ndarray
    levels: np.ndarray
    delay: int = 1


def load_archive(path, horizon=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if horizon is None:
            a = Archive(str(z["backbone"]), str(z["dataset"]), z["t_index"],
                        z["y_true"].astype(float), z["mu"].astype(float),
                        z["quantiles"].astype(float), z["q_levels"].astype(float))
        else:
            bb, ds = path.stem.split("__", 1)
            a = Archive(bb, ds, z["t_index"], z[f"y_h{horizon}"].astype(float),
                        z[f"mu_h{horizon}"].astype(float),
                        z[f"q_h{horizon}"].astype(float),
                        np.array([.1, .3, .5, .7, .9]), horizon)
    n = len(a.y)
    if not (len(a.mu) == len(a.origin) == n and a.quantiles.shape == (n, len(a.levels))):
        raise ValueError(f"Inconsistent archive shape: {path}")
    if not all(np.isfinite(x).all() for x in (a.y, a.mu, a.quantiles, a.levels)):
        raise ValueError(f"Nonfinite archive: {path}")
    if not np.all(np.diff(a.origin) == 1) or not np.all(np.diff(a.levels) > 0):
        raise ValueError(f"Unordered times/levels: {path}")
    return a


def band(a, lower=.1, upper=.9, rearrange=True):
    """Require stored levels; never extrapolate unavailable .05/.95 quantiles."""
    il, iu = np.flatnonzero(np.isclose(a.levels, lower)), np.flatnonzero(np.isclose(a.levels, upper))
    if len(il) != 1 or len(iu) != 1:
        raise ValueError(f"Requested {lower, upper} absent from {a.levels}")
    q = np.sort(a.quantiles, axis=1) if rearrange else a.quantiles
    return q[:, il[0]].copy(), q[:, iu[0]].copy()


@njit(cache=True)
def past_error_scale(y, mu, qlo, qhi, delay, decay=.05):
    """EWMA absolute error, updated when an outcome becomes observable."""
    n = len(y)
    out = np.empty(n)
    # Forecast-only initialization; no unseen target participates.
    state = max(qhi[0] - qlo[0], abs(mu[0]) * .001, 1e-12)
    for i in range(n):
        j = i - delay
        if j >= 0:
            state = (1 - decay) * state + decay * abs(y[j] - mu[j])
        out[i] = max(state, 1e-12)
    return out


def score_components(a, score, regularization=.1, rearrange=True):
    qlo, qhi = band(a, rearrange=rearrange)
    es = past_error_scale(a.y, a.mu, qlo, qhi, a.delay)
    width = np.maximum(qhi - qlo, 0)
    if score == "residual":
        left, right, scale = a.mu.copy(), a.mu.copy(), np.ones(len(a.y))
    elif score == "native_scale":
        left, right = a.mu.copy(), a.mu.copy()
        scale = np.maximum(width, .001 * es)
    elif score == "error_scale":
        left, right, scale = a.mu.copy(), a.mu.copy(), es
    elif score == "cqr":
        left, right, scale = qlo, qhi, np.ones(len(a.y))
    elif score == "regularized_cqr":
        left, right = qlo, qhi
        scale = np.maximum(width + regularization * es, .001 * es)
    else:
        raise ValueError(score)
    scores = np.maximum(left - a.y, a.y - right) / scale
    return left, right, scale, scores


@njit(cache=True)
def _pinball(s, q, alpha):
    return (1 - alpha) * (s - q) if s > q else alpha * (q - s)


@njit(cache=True)
def _quantile(x, p):
    # NumPy's linear interpolation, checked against the noncompiled reference.
    return np.quantile(x, min(max(p, 0.0), 1.0))


@njit(cache=True)
def _calibrate(left, right, scale, scores, y, rule, alpha, gamma, window, prefix, delay):
    """Rules 0=split, 1=rolling, 2=clipped ACI, 3=SF-OGD, 4=PI (log).

    SF-OGD uses the published unprojected signed-score update. PI uses the
    tangent integrator of Angelopoulos et al.; no scorecaster is claimed.
    Both initialize from the excluded calibration prefix. They are explicit
    signed-score adaptations, not bitwise copies of a package's forecaster wrapper.
    """
    n = len(y)
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    issued = np.full(n, np.nan)
    start = prefix + delay - 1
    fixed = _quantile(scores[:prefix], 1 - alpha)
    qt = fixed; at = alpha; g2 = 0.; cumerr = 0.; updates = 0
    unit = max(np.max(np.abs(scores[:prefix])), 1e-12)
    clips = 0; empty = 0; nonfinite = 0
    for i in range(start, n):
        j = i - delay
        if j >= start:
            err = 0. if lo[j] <= y[j] <= hi[j] else 1.
            if rule == 2:
                raw = at + gamma * (alpha - err)
                at = min(max(raw, .001), .999)
                clips += int(at != raw)
            elif rule == 3:
                grad = alpha - err
                g2 += grad * grad
                qt -= unit * grad / np.sqrt(g2)
            elif rule == 4:
                qt += gamma * unit * (err - alpha)
                cumerr += err - alpha
                updates += 1
        end = i - delay + 1  # exclusive: y[end-1] is the last observed outcome
        if rule == 0:
            q = fixed
        elif rule in (1, 2):
            q = _quantile(scores[max(0, end - window):end], 1 - (at if rule == 2 else alpha))
        elif rule == 3:
            q = qt
        else:
            arg = cumerr * np.log(updates + 2.) / (updates + 1.)
            if abs(arg) >= np.pi / 2:
                integ = np.inf if arg > 0 else -np.inf
            else:
                integ = unit * np.tan(arg)
            q = qt + integ
        issued[i] = q
        lo[i] = left[i] - q * scale[i]
        hi[i] = right[i] + q * scale[i]
        if lo[i] > hi[i]:
            empty += 1
            # Explicit practical repair: issued interval is a singleton, not an
            # invalid negative width. Feedback uses this same emitted interval.
            lo[i] = hi[i] = (left[i] + right[i]) / 2
        if not np.isfinite(lo[i]) or not np.isfinite(hi[i]):
            nonfinite += 1
    return lo, hi, issued, clips, empty, nonfinite


RULES = {"split": 0, "rolling": 1, "aci": 2, "sfogd": 3, "pi_log": 4}


def calibrate(a, score="cqr", rule="aci", alpha=.1, gamma=.02, window=450,
              prefix=600, regularization=.1, rearrange=True):
    if not 0 < alpha < 1 or prefix < 20 or prefix + a.delay >= len(a.y):
        raise ValueError("Invalid alpha or insufficient calibration prefix")
    left, right, scale, scores = score_components(a, score, regularization, rearrange)
    lo, hi, q, clips, empty, nonfinite = _calibrate(
        left, right, scale, scores, a.y, RULES[rule], alpha, gamma, window, prefix, a.delay)
    return lo, hi, dict(clips=clips, singleton_repairs=empty, nonfinite=nonfinite)


def interval_scores(y, lo, hi, alpha=.1):
    if np.any(lo > hi):
        raise ValueError("Interval score undefined for reversed endpoints")
    return hi - lo + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def metrics(y, lo, hi, start=1200, stop=None, alpha=.1, rolling=150):
    y, lo, hi = y[start:stop], lo[start:stop], hi[start:stop]
    if not len(y) or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("Missing predictions in common evaluation interval")
    covered = (lo <= y) & (y <= hi)
    cov = float(covered.mean())
    scores = interval_scores(y, lo, hi, alpha)
    local = np.convolve(covered.astype(float), np.ones(min(rolling, len(y))) / min(rolling, len(y)), "valid")
    return dict(n=len(y), coverage=cov, coverage_gap=abs(cov - (1-alpha)),
                lower_miss=float((y < lo).mean()), upper_miss=float((y > hi).mean()),
                width=float(np.mean(hi - lo)), interval_score=float(scores.mean()),
                local_error=float(np.mean(abs(local - (1-alpha)))),
                worst_window=float(local.min()), singleton_rate=float((lo == hi).mean()),
                infinite_rate=float((~np.isfinite(lo) | ~np.isfinite(hi)).mean()))


def paired_bootstrap(a, b, block=200, resamples=1000, seed=1729):
    """Positive difference means a is better than b. Both include identical times."""
    d = np.asarray(b) - np.asarray(a)
    if not np.isfinite(d).all():
        return dict(mean_gain=float(np.mean(d)), ci_low=np.nan, ci_high=np.nan, p=np.nan)
    block = min(block, len(d))
    nblocks = max(1, int(np.ceil(len(d) / block)))
    cs = np.concatenate(([0.], np.cumsum(d)))
    means = (cs[block:] - cs[:-block]) / block
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(resamples, nblocks))].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    # Centered bootstrap two-sided test, with plus-one Monte Carlo correction.
    p = (1 + np.sum(abs(draws-d.mean()) >= abs(d.mean()))) / (resamples + 1)
    return dict(mean_gain=float(d.mean()), ci_low=float(low), ci_high=float(high), p=float(p))
