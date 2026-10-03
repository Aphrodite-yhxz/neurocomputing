"""
daci.forecasters
================
A forecaster is run ONCE per series and produces a `ForecastTrace` (causal, one-step-ahead):
point forecast mu_t, native spread sigma_t, optional native quantiles, and an optional
internal-signal vector per step (e.g. a TSFM's last-layer hidden state). All calibrators
then operate on this cached trace -- so an expensive TSFM is never re-run per method.

To plug in a REAL frozen TSFM (Chronos / TimesFM / Moirai) on a GPU, subclass BaseForecaster
and fill `run()` (see HFTSFMForecaster template at the bottom).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ForecastTrace:
    name: str
    t_index: np.ndarray                       # times t at which y_{t+1} is predicted
    mu: np.ndarray                            # point forecast of y_{t+1}
    sigma: np.ndarray                         # native spread (e.g. IQR/1.349 for a real TSFM)
    y_true: np.ndarray                        # realized y_{t+1}
    internal: Optional[np.ndarray] = None     # (L, d) internal signal per step (hidden state)
    quantiles: Optional[np.ndarray] = None    # (L, Q) native predictive quantiles of y_{t+1}
    q_levels: Optional[np.ndarray] = None     # (Q,)
    change_points: Optional[list] = None      # ground-truth CPs in ORIGINAL series index
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.mu)


class BaseForecaster:
    name = "base"

    def run(self, series: np.ndarray, warmup: int) -> ForecastTrace:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Frozen Echo State Network surrogate (no training of recurrent weights; the
# readout is fit ONCE on warmup -> "zero-shot"-like, mirroring a frozen TSFM).
# ---------------------------------------------------------------------------
class ESNForecaster(BaseForecaster):
    name = "ESN-surrogate"

    def __init__(self, n_res=200, rho=0.9, in_scale=0.5, leak=0.3,
                 sigma_win=40, ridge=1e-2, seed=0):
        g = np.random.default_rng(seed)
        self.n_res, self.leak, self.sigma_win, self.ridge = n_res, leak, sigma_win, ridge
        self.Win = g.uniform(-in_scale, in_scale, size=(n_res, 2))
        W = g.uniform(-1, 1, size=(n_res, n_res)) * (g.random((n_res, n_res)) < 0.1)
        self.W = W * (rho / np.max(np.abs(np.linalg.eigvals(W))))

    def _states(self, series):
        h = np.zeros(self.n_res)
        H = np.empty((len(series), self.n_res))
        for t, u in enumerate(series):
            h = (1 - self.leak) * h + self.leak * np.tanh(
                self.Win @ np.array([1.0, u]) + self.W @ h)
            H[t] = h
        return H

    def run(self, series, warmup):
        series = np.asarray(series, float)
        T = len(series)
        self._sig_floor = max(1e-6, 1e-3 * np.std(series))  # relative floor, scale-robust
        H = self._states(series)
        X = np.column_stack([np.ones(T), series, H])
        # fit readout on warmup, frozen thereafter
        Xw, Yw = X[:warmup - 1], series[1:warmup]
        A = Xw.T @ Xw + self.ridge * np.eye(Xw.shape[1])
        wout = np.linalg.solve(A, Xw.T @ Yw)
        mu_all = X @ wout

        t_idx = np.arange(warmup, T - 1)
        mu = mu_all[t_idx]
        y_true = series[t_idx + 1]
        # causal rolling residual std -> native sigma
        sigma = np.empty(len(t_idx))
        for i, t in enumerate(t_idx):
            lo = max(warmup, t - self.sigma_win)
            res = series[lo + 1:t + 1] - mu_all[lo:t]
            sigma[i] = max(np.std(res) if len(res) > 5 else np.std(series[:warmup]),
                           self._sig_floor)
        # native Gaussian quantiles (a real TSFM would supply its own)
        qs = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        from scipy.stats import norm  # local import; scipy is available
        z = norm.ppf(qs)
        quantiles = mu[:, None] + sigma[:, None] * z[None, :]
        return ForecastTrace(self.name, t_idx, mu, sigma, y_true,
                             internal=H[t_idx], quantiles=quantiles, q_levels=qs,
                             change_points=None)


# ---------------------------------------------------------------------------
# Template adapter for a REAL frozen TSFM (fill in on the GPU machine).
# ---------------------------------------------------------------------------
class HFTSFMForecaster(BaseForecaster):
    """Skeleton for Chronos / TimesFM / Moirai. Run on a machine with HF access + GPU.

    Implement `run` to slide a context window of length `context` over the series and, at
    each step t (predicting y_{t+1}):
      mu_t       = median of the native predictive distribution
      sigma_t    = (q0.75 - q0.25) / 1.349         # native spread, breathes with the model
      quantiles  = the model's native quantiles
      internal_t = last-layer hidden state of the final context token  (the DACI signal)
                   and/or attention-distribution summary; obtain via output_hidden_states=True
    Cache everything into a ForecastTrace; calibrators do the rest.
    """
    name = "TSFM"

    def __init__(self, model, context=512, name="TSFM"):
        self.model, self.context, self.name = model, context, name

    def run(self, series, warmup):  # pragma: no cover - user fills on GPU box
        raise NotImplementedError(
            "Fill HFTSFMForecaster.run on a GPU/HF-enabled machine. "
            "See the docstring; the same calibrators then apply unchanged.")
