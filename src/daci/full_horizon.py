"""Checked conversion from issue-time neural forecasts to author R target time."""
from pathlib import Path
import numpy as np
from .revision import Archive


def load_full(path,horizon):
    with np.load(path,allow_pickle=False) as z:
        if not 1<=horizon<=z['mu'].shape[1]:raise ValueError('Unavailable horizon')
        i=horizon-1
        return Archive(str(z['backbone']),str(z['dataset']),z['t_index'].copy(),z['y'][:,i].astype(float),
            z['mu'][:,i].astype(float),z['quantiles'][:,i].astype(float),z['q_levels'].copy(),horizon)


def to_target_time(mu,y):
    """R row r (1-based) refers to target at origin[0]+r.

    Prediction from issue row i at horizon h goes to R row i+h.
    ERROR stops at the final issue time; evaluation labels after that time are
    never passed to the calibrator. MEAN contains the already-issued future fan.
    """
    mu=np.asarray(mu,float);y=np.asarray(y,float)
    if mu.shape!=y.shape or mu.ndim!=2:raise ValueError('Wrong forecast shape')
    n,H=mu.shape
    if n<=H:raise ValueError('Too few forecast origins')
    pf=np.full((n+H-1,H),np.nan);errors=np.full((n-1,H),np.nan)
    for h in range(1,H+1):
        pf[h-1:h-1+n,h-1]=mu[:,h-1]
        errors[h-1:,h-1]=y[:n-h,h-1]-mu[:n-h,h-1]
    return pf,errors


def from_target_time(values,n):
    values=np.asarray(values,float);H=values.shape[1]
    if len(values)<n+H-1:raise ValueError('Incomplete target-time output')
    return np.column_stack([values[h-1:h-1+n,h-1] for h in range(1,H+1)])
