"""
daci.calibrators
================
Every calibrator consumes a cached ForecastTrace and emits causal online prediction
intervals. All share the SAME marginal-coverage machinery family so comparisons are fair;
DACI's novelty is (a) a native-spread-normalized score and (b) internal-signal-driven
*anticipatory* learning-rate (lever-isolated -- see Stage-0 lesson in the handoff doc).
"""
import time
import numpy as np

Z90 = 1.6448536  # normal quantile for a central 90% interval half-width


def _clip_a(a):
    return float(np.clip(a, 1e-3, 0.999))


class CalibrationResult:
    def __init__(self, name, lower, upper, eval_mask, runtime, diag=None):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.eval_mask = eval_mask
        self.runtime = runtime
        self.diag = diag or {}


class BaseCalibrator:
    name = "base"

    def calibrate(self, trace, alpha=0.10) -> CalibrationResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class Native(BaseCalibrator):
    """Uncalibrated: trust the model's own spread. mu +/- z*sigma."""
    name = "Native"

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        z = Z90 if abs(alpha - 0.10) < 1e-9 else _z(1 - alpha / 2)
        lo = trace.mu - z * trace.sigma
        up = trace.mu + z * trace.sigma
        mask = np.ones(len(trace), bool)
        return CalibrationResult(self.name, lo, up, mask, time.perf_counter() - t0)


class SplitConformal(BaseCalibrator):
    """Studentized split conformal: fixed normalized quantile from a calibration prefix."""
    name = "SplitConformal"

    def __init__(self, cal_frac=0.3):
        self.cal_frac = cal_frac

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace)
        ncal = int(self.cal_frac * L)
        res = np.abs(trace.y_true - trace.mu) / trace.sigma
        qhat = np.quantile(res[:ncal], min(1.0, (1 - alpha) * (1 + 1 / ncal)))
        lo = trace.mu - qhat * trace.sigma
        up = trace.mu + qhat * trace.sigma
        mask = np.zeros(L, bool); mask[ncal:] = True  # evaluate only on the test part
        return CalibrationResult(self.name, lo, up, mask, time.perf_counter() - t0,
                                 diag={"qhat": qhat, "ncal": ncal})


class ACI(BaseCalibrator):
    """Adaptive Conformal Inference (Gibbs & Candes 2021), abs-residual score."""
    name = "ACI"

    def __init__(self, gamma=0.02, cal_win=450):
        self.gamma, self.cal_win = gamma, cal_win

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        a_t = alpha; scores = []
        for i in range(L):
            mu, s = trace.mu[i], trace.sigma[i]
            q = np.quantile(scores[-self.cal_win:], _clip_a(1 - a_t)) if len(scores) >= 20 else Z90 * s
            lo[i], up[i] = mu - q, mu + q
            err = 0 if abs(trace.y_true[i] - mu) <= q else 1
            a_t = _clip_a(a_t + self.gamma * (alpha - err))
            scores.append(abs(trace.y_true[i] - mu))
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class DtACI(BaseCalibrator):
    """Faithful-in-spirit FACI/DtACI (Gibbs & Candes 2022/2024): a pool of ACI experts with
    different step sizes, aggregated by exponential weights on coverage accuracy. (Exact
    reference impl can be swapped for camera-ready.)"""
    name = "DtACI"

    def __init__(self, gammas=(0.001, 0.005, 0.02, 0.05, 0.1), cal_win=450, eta=2.0):
        self.gammas = np.array(gammas); self.cal_win, self.eta = cal_win, eta

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); K = len(self.gammas)
        lo = np.empty(L); up = np.empty(L)
        a = np.full(K, alpha); w = np.ones(K) / K; scores = []
        for i in range(L):
            mu, s = trace.mu[i], trace.sigma[i]
            a_bar = _clip_a(float(w @ a))
            if len(scores) >= 20:
                arr = np.array(scores[-self.cal_win:])
                q = np.quantile(arr, _clip_a(1 - a_bar))
                q_k = np.array([np.quantile(arr, _clip_a(1 - a[k])) for k in range(K)])
            else:
                q = Z90 * s; q_k = np.full(K, q)
            lo[i], up[i] = mu - q, mu + q
            d = abs(trace.y_true[i] - mu)
            err = 0 if d <= q else 1
            err_k = (d > q_k).astype(float)
            # expert updates + reweight by coverage-gap loss
            a = np.clip(a + self.gammas * (alpha - err_k), 1e-3, 0.999)
            loss_k = np.abs(err_k - alpha)
            w = w * np.exp(-self.eta * loss_k); w = 0.99 * w / w.sum() + 0.01 / K
            scores.append(d)
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class SPCI(BaseCalibrator):
    """SPCI-style (Xu & Xie 2023): predict the residual quantile from recent residual
    features with a quantile regressor, refit periodically. Optional (needs sklearn)."""
    name = "SPCI"

    def __init__(self, win=300, n_lags=8, refit=100):
        self.win, self.n_lags, self.refit = win, n_lags, refit

    def calibrate(self, trace, alpha=0.10):
        from sklearn.ensemble import GradientBoostingRegressor
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        res = np.abs(trace.y_true - trace.mu)
        nl = self.n_lags
        model_hi = None
        for i in range(L):
            mu, s = trace.mu[i], trace.sigma[i]
            if i > self.win and (model_hi is None or i % self.refit == 0):
                # build features = last nl residuals -> target current residual
                rows, ys = [], []
                for j in range(i - self.win, i):
                    if j - nl >= 0:
                        rows.append(res[j - nl:j]); ys.append(res[j])
                if len(rows) > 30:
                    model_hi = GradientBoostingRegressor(
                        loss="quantile", alpha=1 - alpha, n_estimators=60,
                        max_depth=3, learning_rate=0.1)
                    model_hi.fit(np.array(rows), np.array(ys))
            if model_hi is not None and i - nl >= 0:
                q = float(model_hi.predict(res[i - nl:i][None, :])[0])
            else:
                q = Z90 * s
            lo[i], up[i] = mu - q, mu + q
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class ECI(BaseCalibrator):
    """Error-quantified Conformal Inference (Wang et al. 2025, arXiv:2502.00818), faithful-in-
    spirit: replaces ACI's binary miscoverage feedback with a CONTINUOUS error-magnitude term
    (smoothed quantile-loss gradient), reacting faster to shifts. Still purely REACTIVE
    (consumes realized errors only) -> our sharpest 'fast adaptation' baseline."""
    name = "ECI"

    def __init__(self, gamma=0.02, cal_win=450, tau=0.5):
        self.gamma, self.cal_win, self.tau = gamma, cal_win, tau

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        a_t = alpha; scores = []
        for i in range(L):
            mu, s = trace.mu[i], trace.sigma[i]
            if len(scores) >= 20:
                arr = np.array(scores[-self.cal_win:])
                q = np.quantile(arr, _clip_a(1 - a_t))
                sc = np.std(arr) + 1e-9
            else:
                q = Z90 * s; sc = s + 1e-9
            lo[i], up[i] = mu - q, mu + q
            e = abs(trace.y_true[i] - mu)
            err = 0 if e <= q else 1
            # continuous error-quantification feedback: sigmoid of signed margin (scale-free)
            eq = 1.0 / (1.0 + np.exp(-(e - q) / (self.tau * sc)))   # ~err but smooth
            a_t = _clip_a(a_t + self.gamma * (alpha - eq))
            scores.append(e)
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class DACI(BaseCalibrator):
    """Drift-Anticipatory Conformal Inference (ours).
    Lever 1: native-spread-normalized score r=|res|/sigma  -> width breathes with the model.
    Lever 2: internal-signal drift d_t drives a CENTERED time-varying target coverage
             alpha_tar_t = alpha*(1 - c*(d_t - dbar_t)), so the method is proactively
             conservative exactly when a shift is anticipated, while the long-run mean target
             stays ~alpha (marginal coverage preserved). An optional LR boost speeds tracking.
    mode: 'tv' (time-varying target, default), 'lr' (Stage-1 LR-boost), or 'both'.
    Falls back to plain online conformal if no internal signal is present.
    """
    name = "DACI"

    def __init__(self, gamma0=0.02, lam=3.0, ki=0.1, c_tv=2.5, mode="lr", score="studentized",
                 feedback="binary", recenter=False, rc_ema=0.05, rc_gate=0.5,
                 drift_mode="real", drift_seed=0, win_shrink=0.0,
                 cal_win=450, ema=0.02, dbar_ema=0.01, dwin=400):
        self.gamma0, self.lam, self.ki, self.c_tv, self.mode = gamma0, lam, ki, c_tv, mode
        self.score = score       # "studentized" (mu +/- q*sigma) or "cqr" (asymmetric native band)
        self.feedback = feedback # "binary" (ACI-style) or "eci" (continuous error magnitude)
        self.recenter = recenter # drift-GATED bias re-centering (BC-ACI idea, gated by d_t)
        self.rc_ema, self.rc_gate = rc_ema, rc_gate
        self.drift_mode, self.drift_seed = drift_mode, drift_seed
        self.win_shrink = win_shrink   # >0: shrink calibration window when drift is high
                                       # (drop stale pre-shift scores -> the direct fix)
        self.cal_win, self.ema, self.dbar_ema, self.dwin = cal_win, ema, dbar_ema, dwin

    def _drift_signal(self, internal):
        if internal is None:
            return None
        L = len(internal); d = np.zeros(L)
        ref = internal[0].copy(); hist = []
        for i in range(L):
            ref = (1 - self.ema) * ref + self.ema * internal[i]
            raw = np.linalg.norm(internal[i] - ref)
            hist.append(raw)
            if len(hist) > 30:
                h = np.array(hist[-self.dwin:])
                med = np.median(h); mad = np.median(np.abs(h - med)) + 1e-9
                d[i] = np.clip((raw - med) / (3 * mad), 0, 1.5)
        return d

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        d = self._drift_signal(trace.internal)
        if d is None:
            d = np.zeros(L)
        # --- placebo controls (isolate the internal signal's causal contribution) ---
        # "shuffled": same marginal distribution, temporal alignment DESTROYED
        # "ungated" : constant signal at the mean level (machinery on, information off)
        # "zero"    : no signal at all
        if self.drift_mode == "shuffled":
            d = np.random.default_rng(self.drift_seed).permutation(d)
        elif self.drift_mode == "ungated":
            d = np.full(L, float(np.mean(d)))
        elif self.drift_mode == "zero":
            d = np.zeros(L)
        # efficiency lever: asymmetric CQR on the model's native band (needs native quantiles)
        use_cqr = (self.score == "cqr" and trace.quantiles is not None)
        if use_cqr:
            qlo = trace.quantiles[:, 0]; qhi = trace.quantiles[:, -1]   # e.g. q0.1, q0.9
        a_t = alpha; scores = []; dbar = 0.0; mbar = alpha; n = 0; bias = 0.0
        for i in range(L):
            y = trace.y_true[i]
            mu0 = trace.mu[i]
            mu = mu0 + (bias if self.recenter else 0.0)     # drift-gated re-centering
            # drift-driven calibration window: shrink to drop stale pre-shift scores
            if self.win_shrink > 0:
                w_t = int(max(60, self.cal_win * (1.0 - self.win_shrink * min(d[i], 1.0))))
            else:
                w_t = self.cal_win
            if use_cqr:
                Q = np.quantile(scores[-w_t:], _clip_a(1 - a_t)) if len(scores) >= 20 else 0.0
                off = (bias if self.recenter else 0.0)
                lo[i], up[i] = qlo[i] + off - Q, qhi[i] + off + Q  # asymmetric, respects native skew
            else:
                s = trace.sigma[i]
                qr = np.quantile(scores[-w_t:], _clip_a(1 - a_t)) if len(scores) >= 20 else Z90
                half = qr * s                              # breathes with sigma
                lo[i], up[i] = mu - half, mu + half
            err = 0 if (lo[i] <= y <= up[i]) else 1
            dbar = (1 - self.dbar_ema) * dbar + self.dbar_ema * d[i]
            n += 1; mbar += (err - mbar) / n               # running miscoverage rate
            if self.mode in ("tv", "both"):
                a_tar = float(np.clip(alpha * (1 - self.c_tv * (d[i] - dbar)),
                                      max(0.01, 0.25 * alpha), min(0.5, 1.75 * alpha)))
            else:
                a_tar = alpha
            gamma_t = self.gamma0 * (1.0 + (self.lam * d[i] if self.mode in ("lr", "both") else 0.0))
            # feedback signal: binary miscoverage (ACI) or continuous error magnitude (ECI-style)
            if self.feedback == "eci" and len(scores) >= 20:
                arr = np.array(scores[-w_t:]); sc = np.std(arr) + 1e-9
                cur = (max(qlo[i] - y, y - qhi[i]) if use_cqr
                       else abs(y - mu) / trace.sigma[i])
                thr = np.quantile(arr, _clip_a(1 - a_t))
                fb = 1.0 / (1.0 + np.exp(-(cur - thr) / (0.5 * sc)))
            else:
                fb = err
            # PI control: proportional (anticipatory, on fb) + integral (exact long-run calibration)
            a_t = _clip_a(a_t + gamma_t * (a_tar - fb) + self.ki * (alpha - mbar))
            # drift-GATED online bias estimate (BC-ACI idea; only trusted when drift is high)
            if self.recenter:
                gate = 1.0 if d[i] > self.rc_gate else 0.15
                bias = (1 - self.rc_ema * gate) * bias + self.rc_ema * gate * (y - mu0)
            if use_cqr:
                scores.append(max(qlo[i] - y, y - qhi[i]))   # CQR nonconformity (can be <0)
            else:
                scores.append(min(abs(y - mu) / trace.sigma[i], 50.0))
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0,
                                 diag={"drift": d})


def _z(p):
    from scipy.stats import norm
    return float(norm.ppf(p))




class RACI(BaseCalibrator):
    """Regime-Aware Conformal Inference (option B).

    Idea: a frozen TSFM's context embedding is a poor *predictor of future shifts* but an
    excellent *identifier of which regime we are in*. So we don't try to anticipate --
    we RETRIEVE. Online-cluster the context embeddings into k regimes and keep a SEPARATE
    calibration-score buffer per regime. At time t we calibrate with scores collected in the
    SAME regime, which (a) gives regime-conditional validity, and (b) means a recurring regime
    (weekday/weekend, summer/winter, storm/no-storm) is instantly calibrated -- no relearning.

    A global PI-controlled level keeps marginal coverage on target.

    cluster_mode: "real"     -> embeddings drive the regime assignment
                  "shuffled" -> PLACEBO: assignments permuted (same buffer sizes, no information)
                  "none"     -> single global buffer (= standard online CP)
    """
    name = "RACI"

    def __init__(self, k=4, score="cqr", gamma0=0.02, ki=0.1, cal_win=450,
                 warm=400, min_buf=40, cluster_mode="real", seed=0, online_lr=0.05):
        self.k, self.score, self.gamma0, self.ki = k, score, gamma0, ki
        self.cal_win, self.warm, self.min_buf = cal_win, warm, min_buf
        self.cluster_mode, self.seed, self.online_lr = cluster_mode, seed, online_lr

    def _assign(self, internal, L):
        """Online regime assignment from context embeddings (k-means seeded on warmup)."""
        if internal is None or self.cluster_mode == "none":
            return np.zeros(L, int), 1
        X = np.asarray(internal, float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)          # standardize dims
        from sklearn.cluster import KMeans
        w = min(self.warm, L)
        km = KMeans(n_clusters=self.k, n_init=4, random_state=self.seed).fit(X[:w])
        proto = km.cluster_centers_.copy()
        lab = np.empty(L, int)
        for i in range(L):                               # streaming assign + prototype update
            d = np.linalg.norm(proto - X[i], axis=1)
            c = int(np.argmin(d)); lab[i] = c
            proto[c] += self.online_lr * (X[i] - proto[c])
        if self.cluster_mode == "shuffled":              # PLACEBO: keep label stats, kill meaning
            lab = np.random.default_rng(self.seed).permutation(lab)
        return lab, self.k

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        lab, K = self._assign(trace.internal, L)
        use_cqr = (self.score == "cqr" and trace.quantiles is not None)
        if use_cqr:
            qlo = trace.quantiles[:, 0]; qhi = trace.quantiles[:, -1]
        bufs = {c: [] for c in range(max(K, 1))}
        glob = []
        a_t = alpha; mbar = alpha; n = 0
        for i in range(L):
            y = trace.y_true[i]; c = lab[i]
            b = bufs[c] if len(bufs[c]) >= self.min_buf else glob   # fall back until warm
            if use_cqr:
                Q = np.quantile(b[-self.cal_win:], _clip_a(1 - a_t)) if len(b) >= 20 else 0.0
                lo[i], up[i] = qlo[i] - Q, qhi[i] + Q
                sc = max(qlo[i] - y, y - qhi[i])
            else:
                mu, s = trace.mu[i], trace.sigma[i]
                qr = np.quantile(b[-self.cal_win:], _clip_a(1 - a_t)) if len(b) >= 20 else Z90
                lo[i], up[i] = mu - qr * s, mu + qr * s
                sc = min(abs(y - mu) / s, 50.0)
            err = 0 if (lo[i] <= y <= up[i]) else 1
            n += 1; mbar += (err - mbar) / n
            a_t = _clip_a(a_t + self.gamma0 * (alpha - err) + self.ki * (alpha - mbar))
            bufs[c].append(sc); glob.append(sc)
        return CalibrationResult(self.name, lo, up, np.ones(L, bool),
                                 time.perf_counter() - t0, diag={"labels": lab})


class NQCP(BaseCalibrator):
    """NQ-CP — Native-Quantile Conformal Prediction (THE PAPER'S METHOD, final form).

    Wrap a FROZEN time series foundation model online, using ONLY what a black-box API returns:
    its native predictive quantiles. The native band has (approximately) the right SHAPE --
    it encodes the model's heteroscedasticity -- but the wrong LEVEL. So we conformalize the
    band CQR-style (Romano et al. 2019) and let an ACI-style adaptive level (Gibbs & Candes 2021)
    fix the level online under distribution shift:

        s_t     = max(q_lo_t - y_t,  y_t - q_hi_t)                (CQR nonconformity)
        Gamma_t = [q_lo_t - Q_t,  q_hi_t + Q_t],  Q_t = Quantile_{1-alpha_t}(recent scores)
        alpha_{t+1} = clip(alpha_t + gamma (alpha - err_t))

    Deliberately simple. Re-centering and an integral (PI) term were implemented and ABLATED:
    on 10 real energy series with a frozen Chronos-Bolt neither helped (both were marginally
    WORSE), so they are not part of the method. See DACI (this module) to reproduce those
    ablations, and its drift_mode= placebo controls for the internal-signal study.

    Requires trace.quantiles (native). Falls back to a studentized score if absent.
    """
    name = "NQ-CP"

    def __init__(self, gamma=0.02, cal_win=450):
        self.gamma, self.cal_win = gamma, cal_win

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        use_native = trace.quantiles is not None
        if use_native:
            qlo, qhi = trace.quantiles[:, 0], trace.quantiles[:, -1]
        a_t = alpha; scores = []
        for i in range(L):
            y = trace.y_true[i]
            if use_native:
                Q = np.quantile(scores[-self.cal_win:], _clip_a(1 - a_t)) if len(scores) >= 20 else 0.0
                lo[i], up[i] = qlo[i] - Q, qhi[i] + Q
                s = max(qlo[i] - y, y - qhi[i])
            else:
                mu, sg = trace.mu[i], trace.sigma[i]
                Q = np.quantile(scores[-self.cal_win:], _clip_a(1 - a_t)) if len(scores) >= 20 else Z90
                lo[i], up[i] = mu - Q * sg, mu + Q * sg
                s = min(abs(y - mu) / sg, 50.0)
            err = 0 if (lo[i] <= y <= up[i]) else 1
            a_t = _clip_a(a_t + self.gamma * (alpha - err))
            scores.append(s)
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class SplitCQR(BaseCalibrator):
    """Split-conformal CQR on the NATIVE band (Romano et al. 2019, applied to a frozen TSFM).

    Control baseline for NQ-CP: same score, but the conformal margin Q is FIXED from a
    calibration prefix (no online level adaptation). Separates the two ingredients:
    if NQ-CP >> SplitCQR under shift, the ONLINE level is doing real work, not just CQR."""
    name = "SplitCQR"

    def __init__(self, cal_frac=0.3):
        self.cal_frac = cal_frac

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        if trace.quantiles is None:
            raise ValueError("SplitCQR needs native quantiles.")
        L = len(trace); ncal = int(self.cal_frac * L)
        qlo, qhi = trace.quantiles[:, 0], trace.quantiles[:, -1]
        s = np.maximum(qlo - trace.y_true, trace.y_true - qhi)
        Q = np.quantile(s[:ncal], min(1.0, (1 - alpha) * (1 + 1 / ncal)))
        lo, up = qlo - Q, qhi + Q
        mask = np.zeros(L, bool); mask[ncal:] = True
        return CalibrationResult(self.name, lo, up, mask, time.perf_counter() - t0,
                                 diag={"Q": float(Q), "ncal": ncal})


class ConformalPID(BaseCalibrator):
    """Conformal PID control (Angelopoulos, Candes & Tibshirani 2023), faithful-in-spirit:
    track the conformal QUANTILE q_t directly with P (instantaneous coverage error),
    I (integrated coverage error, saturated), and a scorecaster D-term (EWMA score forecast).
    Consumes abs-residual scores of the point forecast (black-box baseline class).
    Exact reference implementation can be swapped for camera-ready."""
    name = "PID"

    def __init__(self, eta=None, ki=None, ewma=0.05, cal_win=450):
        self.eta, self.ki, self.ewma, self.cal_win = eta, ki, ewma, cal_win

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        res = np.abs(trace.y_true - trace.mu)
        scale = np.median(res[:200]) + 1e-9
        eta = self.eta if self.eta is not None else 0.2 * scale      # units of the score
        ki = self.ki if self.ki is not None else 0.5                 # dimensionless
        q_track = Z90 * float(np.median(trace.sigma[:20]))
        sc = 0.0; integ = 0.0
        for i in range(L):
            mu = trace.mu[i]
            # output = quantile tracker (P) + saturated integral (I) + scorecaster (D)
            q_eff = max(q_track + ki * float(np.tanh(integ / 20.0)) * scale + sc, 0.0)
            lo[i], up[i] = mu - q_eff, mu + q_eff
            err = 0 if res[i] <= q_eff else 1
            q_track = max(q_track + eta * (err - alpha), 0.0)       # P: quantile tracking
            integ = float(np.clip(integ + (err - alpha), -60, 60))  # I: coverage-error sum
            # D / scorecaster: EWMA of scores anticipates slow drift in the score scale
            sc = (1 - self.ewma) * sc + self.ewma * (res[i] - q_track)
            sc = float(np.clip(sc, -5 * scale, 5 * scale))
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class SAOCP(BaseCalibrator):
    """Strongly Adaptive Online Conformal Prediction (Bhatnagar et al. 2023),
    faithful-in-spirit: a ladder of SF-OGD experts with geometric lifetimes; each expert
    gradient-steps the interval RADIUS on the pinball loss; meta-aggregation weights experts
    by recent coverage accuracy on their active window. Abs-residual score (black-box class).
    Exact reference implementation can be swapped for camera-ready."""
    name = "SAOCP"

    def __init__(self, lifetimes=(32, 64, 128, 256, 512), eta_scale=0.05, meta_eta=2.0):
        self.lifetimes, self.eta_scale, self.meta_eta = lifetimes, eta_scale, meta_eta

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        res = np.abs(trace.y_true - trace.mu)
        scale = np.median(res[:200]) + 1e-9
        K = len(self.lifetimes)
        q = np.full(K, Z90 * float(np.median(trace.sigma[:20])))    # expert radii
        w = np.ones(K) / K
        eta = self.eta_scale * scale * np.sqrt(
            np.array([max(self.lifetimes) / l for l in self.lifetimes]))
        for i in range(L):
            mu = trace.mu[i]
            q_agg = float(w @ q)
            lo[i], up[i] = mu - q_agg, mu + q_agg
            # SF-OGD step on the pinball loss of each expert's own radius
            grad = np.where(res[i] > q, -(1 - alpha), alpha)        # d pinball / d q
            q = np.maximum(q - eta * grad, 0.0)
            # restart experts at the end of their lifetime (strong adaptivity ladder)
            for k, l in enumerate(self.lifetimes):
                if (i + 1) % l == 0:
                    q[k] = q_agg
            # meta-weights: exponential on each expert's own coverage-gap loss
            loss_k = np.abs((res[i] > q).astype(float) - alpha)
            w = w * np.exp(-self.meta_eta * loss_k)
            w = 0.98 * w / w.sum() + 0.02 / K
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


class NQCPn(BaseCalibrator):
    """NQ-CPn — WIDTH-NORMALIZED native-quantile conformal prediction (extension, Stage 3).

    NQ-CP adds an ADDITIVE margin Q to the native band. Under a location-scale data process
    y = m(x) + s(x)*eps with SHAPE-consistent native quantiles (q_hat = m + s*c_tau, i.e. the
    band is right up to a level factor), the required margin is PROPORTIONAL to s(x), not
    constant: an additive Q is simultaneously too wide where s(x) is small (PV nights) and too
    narrow where it is large (PV noon). Normalizing the CQR score by the band width makes the
    score scale-free,

        s_t = max(q_lo_t - y_t, y_t - q_hi_t) / w_t,    w_t = max(q_hi_t - q_lo_t, floor_t),
        Gamma_t = [q_lo_t - Q_t * w_t,  q_hi_t + Q_t * w_t],

    so the conformal correction expands the band MULTIPLICATIVELY -- and, under the
    shape-consistency assumption above, the same Q_t is the right correction in every regime
    (approximate conditional validity, not just marginal). floor_t is a causal scale floor
    (1e-3 * expanding std of observed y) so PV nights cannot produce degenerate scores.

    Expected effect vs NQ-CP: same marginal coverage, better local/worst-slab coverage, and a
    Winkler improvement wherever the native band's LEVEL error is multiplicative rather than
    additive. Also enables cross-series score pooling (scores are scale-free) -> cold-start.

    REAL-DATA VERDICT (2026-07, 54 backbone-series pairs, pre-registered rule): NOT adopted
    as the headline method — Winkler is a statistical tie with NQ-CP (0 sig wins / 20 on the
    extensions grid). Secondary findings kept for the paper: best conditional validity of all
    11 methods (local rank 1.4-1.6, postCP 0.0156 vs 0.0178) and the lowest reserve cost at
    high scarcity ratios (rho=50: 87.9 vs NQ-CP 89.3, ACI=100).
    WARNING — MULTI-HORIZON: do NOT use at h>1 on near-zero-width regimes (PV nights):
    the width-normalized score explodes when the native band collapses while h-step
    volatility does not (observed 4-8x Winkler blow-ups at h=24). The additive NQ-CP is the
    multi-horizon recommendation; a horizon-aware floor would be needed first.
    """
    name = "NQ-CPn"

    def __init__(self, gamma=0.02, cal_win=450, cap=50.0):
        self.gamma, self.cal_win, self.cap = gamma, cal_win, cap

    def calibrate(self, trace, alpha=0.10):
        t0 = time.perf_counter()
        L = len(trace); lo = np.empty(L); up = np.empty(L)
        if trace.quantiles is None:
            raise ValueError("NQ-CPn needs native quantiles (trace.quantiles).")
        qlo, qhi = trace.quantiles[:, 0], trace.quantiles[:, -1]
        a_t = alpha; scores = []
        y_sum = 0.0; y_sq = 0.0; n_y = 0          # causal expanding std of y
        for i in range(L):
            y = trace.y_true[i]
            ystd = np.sqrt(max(y_sq / n_y - (y_sum / n_y) ** 2, 0.0)) if n_y > 5 else 1.0
            w = max(qhi[i] - qlo[i], 1e-6 + 1e-3 * ystd)
            Q = np.quantile(scores[-self.cal_win:], _clip_a(1 - a_t)) if len(scores) >= 20 else 0.0
            lo[i], up[i] = qlo[i] - Q * w, qhi[i] + Q * w
            err = 0 if (lo[i] <= y <= up[i]) else 1
            a_t = _clip_a(a_t + self.gamma * (alpha - err))
            scores.append(min(max(qlo[i] - y, y - qhi[i]) / w, self.cap))
            y_sum += y; y_sq += y * y; n_y += 1
        return CalibrationResult(self.name, lo, up, np.ones(L, bool), time.perf_counter() - t0)


# Registry (defined last so every calibrator above exists)
ALL_CALIBRATORS = [Native(), SplitConformal(), ACI(), DtACI(), SPCI(), ECI(), NQCP()]
