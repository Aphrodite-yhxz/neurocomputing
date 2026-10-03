"""
daci.runner
===========
Orchestrates forecaster x dataset x calibrator grids. Each (forecaster, dataset) trace is
computed ONCE and reused across all calibrators; results collected into a tidy DataFrame.
"""
import numpy as np
import pandas as pd

from .calibrators import ALL_CALIBRATORS
from .metrics import evaluate


def run_grid(forecaster, datasets, calibrators=None, alpha=0.10, warmup=600, verbose=True):
    calibrators = calibrators or ALL_CALIBRATORS
    rows = []
    traces = {}
    for load in datasets:
        series, cps, name = load()
        wu = warmup if len(series) > 2 * warmup else max(100, len(series) // 4)
        trace = forecaster.run(series, wu)
        trace.change_points = cps
        traces[name] = trace
        if verbose:
            print(f"\n# dataset={name}  (T={len(series)}, eval_steps={len(trace)})")
            print(f"  {'method':15s} {'cov':>6s} {'width':>7s} {'winkler':>8s} "
                  f"{'localErr':>8s} {'postCP':>7s} {'reclag':>7s}")
        for cal in calibrators:
            try:
                res = cal.calibrate(trace, alpha=alpha)
                ev = evaluate(res, trace, alpha=alpha)
            except Exception as e:
                if verbose:
                    print(f"  {cal.name:15s}  SKIPPED ({type(e).__name__}: {e})")
                continue
            ev["dataset"] = name
            ev["forecaster"] = forecaster.name
            rows.append(ev)
            if verbose:
                pc = ev["postCP_cov_err"]; rl = ev["recovery_lag"]
                print(f"  {ev['method']:15s} {ev['coverage']:6.3f} {ev['mean_width']:7.2f} "
                      f"{ev['winkler']:8.2f} {ev['local_cov_err']:8.3f} "
                      f"{(pc if pc==pc else float('nan')):7.3f} {(rl if rl==rl else float('nan')):7.1f}")
    return pd.DataFrame(rows), traces


def aggregate_ranks(df, metrics=("cov_gap", "mean_width", "winkler",
                                 "local_cov_err", "postCP_cov_err", "recovery_lag")):
    """Average rank per method across datasets for each metric (1 = best, lower metric better)."""
    out = {}
    for metric in metrics:
        sub = df.dropna(subset=[metric])
        if sub.empty:
            continue
        piv = sub.pivot_table(index="method", columns="dataset", values=metric)
        ranks = piv.rank(axis=0, method="min")     # rank methods within each dataset
        out[metric] = ranks.mean(axis=1)
    return pd.DataFrame(out)
