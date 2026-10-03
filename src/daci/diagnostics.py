"""
daci.diagnostics
================
Pre-deployment diagnostic: predict, BEFORE running anything, how much NQ-CP will improve
over residual-based online conformal (ACI) on a given series -- from a model-free property
of the data (the paper's Sec. "Mechanism").

    from daci.diagnostics import het_index, expected_gain, should_use_native_band
    het_index(x)            -> the heteroscedasticity index HET (W=24 blocks, log-vol std)
    expected_gain(x)        -> (predicted Winkler gain %, 1-sigma band)
    should_use_native_band(x) -> ("yes"|"marginal"|"no", explanation)

IMPORTANT -- read this before trusting expected_gain() as a point estimate:
The gain-vs-HET relationship is a TWO-REGIME effect, not a fine linear law. Over the paper's
124 points the Pearson r is 0.94 but the Spearman rho is only 0.57, and with the strongly
heteroscedastic PV cluster removed the correlation vanishes (r=0.13, rho~0). In other words HET
predicts the MAGNITUDE CLASS of the benefit (large for volatile renewables, modest for smooth
aggregate demand) but NOT its precise percentage within the low-HET majority. Prefer
`should_use_native_band()` (a coarse yes/marginal/no tier) over `expected_gain()` (a point
estimate kept only as a rough order-of-magnitude guide). The linear constants below are a crude
pooled OLS fit (results/real/main_v2_significance.csv, 4 backbones x 31 series, 2026-07); refit
with `refit(significance_csv)`.
CAVEAT 2: even the tier is reliable only when the native band has approximately the right SHAPE.
A backbone whose native quantiles are badly shaped on a dataset (high native local-coverage
error) can violate it downward -- e.g. TimesFM on ERCOT (native local err ~0.13) is the one
place NQ-CP loses. Read the diagnostic together with the native local-coverage error.
"""
import numpy as np

_INTERCEPT = 4.51
_SLOPE = 3.12
_RESID_SD = 4.6
_W = 24


def het_index(x, W=_W):
    """std of log block-volatility of the differenced series (scale-free, model-free).
    W must resolve the dominant volatility cycle (~one day of samples or less)."""
    dx = np.diff(np.asarray(x, float))
    n = len(dx) // W
    if n < 8:
        raise ValueError(f"series too short for W={W} ({n} blocks; need >= 8)")
    vols = np.array([dx[i * W:(i + 1) * W].std() for i in range(n)])
    return float(np.std(np.log(np.maximum(vols, 1e-9))))


def expected_gain(x, W=_W):
    """Predicted NQ-CP-over-ACI Winkler gain (%) with a 1-sigma uncertainty band."""
    g = _INTERCEPT + _SLOPE * het_index(x, W)
    return g, (g - _RESID_SD, g + _RESID_SD)


def should_use_native_band(x, W=_W):
    h = het_index(x, W)
    g, (glo, ghi) = expected_gain(x, W)
    if glo > 5.0:
        return "yes", (f"HET={h:.2f}: strongly heteroscedastic; expect ~{g:.0f}% "
                       f"[{glo:.0f}, {ghi:.0f}] Winkler gain from NQ-CP.")
    if ghi < 5.0:
        return "no", (f"HET={h:.2f}: near-homoscedastic; expected gain ~{g:.0f}% -- "
                      f"plain ACI is likely adequate.")
    return "marginal", (f"HET={h:.2f}: expected gain ~{g:.0f}% [{glo:.0f}, {ghi:.0f}]; "
                        f"NQ-CP is never worse in our study, so default to it.")


def refit(significance_csv, loaders=None, W=_W):
    """Refit the constants from a significance CSV (paper pipeline). Returns dict; also
    prints the new constants to paste above."""
    import pandas as pd
    if loaders is None:
        from experiments.make_fig_main import LOADERS as loaders  # type: ignore
    sig = pd.read_csv(significance_csv)
    sig = sig[sig.get("comparison", "NQ-CP vs ACI") == "NQ-CP vs ACI"] \
        if "comparison" in sig.columns else sig
    h = np.array([het_index(loaders[d]()[0], W) for d in sig.dataset])
    g = sig.rel_gain_pct.values
    slope, intercept = np.polyfit(h, g, 1)
    resid_sd = float((g - (intercept + slope * h)).std())
    out = dict(intercept=float(intercept), slope=float(slope), resid_sd=resid_sd,
               r=float(np.corrcoef(h, g)[0, 1]), n=len(g))
    print(f"_INTERCEPT = {intercept:.2f}; _SLOPE = {slope:.2f}; _RESID_SD = {resid_sd:.1f}"
          f"   # r={out['r']:.3f}, n={out['n']}")
    return out
