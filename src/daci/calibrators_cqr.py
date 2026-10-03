"""
daci.calibrators_cqr
====================
CQR-SCORE variants of the adaptive baselines, for the "score vs. mechanism" decomposition
(reviewer control). Each method keeps its native adaptation rule but replaces the
absolute-residual score with the CQR score on the model's native band. This isolates how much
of NQ-CP's advantage comes from the CQR score (shared by all *-CQR variants) versus from the
specific ACI-style level rule.

All three consume trace.quantiles (outermost native quantiles q_lo, q_hi) and emit
Gamma_t = [q_lo - Q_t, q_hi + Q_t] with the CQR nonconformity s = max(q_lo - y, y - q_hi).
"""
import time
import numpy as np

from .calibrators import BaseCalibrator, CalibrationResult, _clip_a, Z90


def _native(trace):
    if trace.quantiles is None:
        raise ValueError("CQR-variant needs trace.quantiles")
    return trace.quantiles[:, 0], trace.quantiles[:, -1]


class DtACICQR(BaseCalibrator):
    """DtACI expert pool, CQR score on the native band."""
    name = "DtACI-CQR"

    def __init__(self, gammas=(0.001, 0.005, 0.02, 0.05, 0.1), cal_win=450, eta=2.0):
        self.gammas = np.array(gammas); self.cal_win, self.eta = cal_win, eta

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        qlo, qhi = _native(trace)
        L = len(trace); K = len(self.gammas)
        lo = np.empty(L); up = np.empty(L)
        a = np.full(K, alpha); w = np.ones(K) / K; scores = []
        for i in range(L):
            y = trace.y_true[i]
            a_bar = _clip_a(float(w @ a))
            if len(scores) >= 20:
                arr = np.array(scores[-self.cal_win:])
                Q = np.quantile(arr, _clip_a(1 - a_bar))
                Q_k = np.array([np.quantile(arr, _clip_a(1 - a[k])) for k in range(K)])
            else:
                Q = 0.0; Q_k = np.zeros(K)
            lo[i], up[i] = qlo[i] - Q, qhi[i] + Q
            s = max(qlo[i] - y, y - qhi[i])
            err_k = (s > Q_k).astype(float)
            a = np.clip(a + self.gammas * (alpha - err_k), 1e-3, 0.999)
            loss_k = np.abs(err_k - alpha)
            w = w * np.exp(-self.eta * loss_k); w = 0.99 * w / w.sum() + 0.01 / K
            scores.append(s)
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class ConformalPIDCQR(BaseCalibrator):
    """Conformal PID (P + saturated I + EWMA scorecaster) tracking the CQR margin directly."""
    name = "PID-CQR"

    def __init__(self, eta=None, ki=0.5, ewma=0.05):
        self.eta, self.ki, self.ewma = eta, ki, ewma

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        qlo, qhi = _native(trace)
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        s_all = np.maximum(qlo - trace.y_true, trace.y_true - qhi)
        scale = np.median(np.abs(s_all[:200])) + 1e-9
        eta = self.eta if self.eta is not None else 0.2 * scale
        Q = float(np.quantile(np.maximum(s_all[:50], 0), 0.9)) if L > 50 else scale
        sc = 0.0; integ = 0.0
        for i in range(L):
            y = trace.y_true[i]
            Q_eff = max(Q + self.ki * float(np.tanh(integ / 20.0)) * scale + sc, 0.0)
            lo[i], up[i] = qlo[i] - Q_eff, qhi[i] + Q_eff
            s = max(qlo[i] - y, y - qhi[i])
            err = 0 if s <= Q_eff else 1
            Q = max(Q + eta * (err - alpha), -scale)
            integ = float(np.clip(integ + (err - alpha), -60, 60))
            sc = (1 - self.ewma) * sc + self.ewma * (s - Q)
            sc = float(np.clip(sc, -5 * scale, 5 * scale))
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class SAOCPCQR(BaseCalibrator):
    """SAOCP expert ladder (SF-OGD on the pinball loss), CQR score on the native band."""
    name = "SAOCP-CQR"

    def __init__(self, lifetimes=(32, 64, 128, 256, 512), eta_scale=0.05, meta_eta=2.0):
        self.lifetimes, self.eta_scale, self.meta_eta = lifetimes, eta_scale, meta_eta

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        qlo, qhi = _native(trace)
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        s_all = np.maximum(qlo - trace.y_true, trace.y_true - qhi)
        scale = np.median(np.abs(s_all[:200])) + 1e-9
        K = len(self.lifetimes)
        Q = np.full(K, float(np.quantile(np.maximum(s_all[:50], 0), 0.9)) if L > 50 else scale)
        w = np.ones(K) / K
        eta = self.eta_scale * scale * np.sqrt(
            np.array([max(self.lifetimes) / l for l in self.lifetimes]))
        for i in range(L):
            y = trace.y_true[i]
            Q_agg = float(w @ Q)
            lo[i], up[i] = qlo[i] - Q_agg, qhi[i] + Q_agg
            s = max(qlo[i] - y, y - qhi[i])
            grad = np.where(s > Q, -(1 - alpha), alpha)
            Q = Q - eta * grad
            for k, l in enumerate(self.lifetimes):
                if (i + 1) % l == 0:
                    Q[k] = Q_agg
            loss_k = np.abs((s > Q).astype(float) - alpha)
            w = w * np.exp(-self.meta_eta * loss_k); w = 0.98 * w / w.sum() + 0.02 / K
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)
