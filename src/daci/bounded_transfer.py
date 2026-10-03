"""Exploratory bounded pairwise scale transfer, with explicit validity limitations."""
import numpy as np
from numba import njit
from .revision import band,past_error_scale


@njit(cache=True)
def _bounded(y,left,right,width,raw,delay,prefix,window,alpha,gamma,cap):
    n=len(y);start=prefix+delay-1
    lo=np.full(n,np.nan);hi=np.full(n,np.nan)
    at=alpha;clips=0;repairs=0;max_ratio=0.;bound_violation=0.
    cap_hits=0;transfers=0
    for t in range(start,n):
        j=t-delay
        if j>=start:
            err=0. if lo[j]<=y[j]<=hi[j] else 1.
            candidate=at+gamma*(alpha-err)
            at=min(.999,max(.001,candidate));clips+=int(at!=candidate)
        end=t-delay+1;begin=max(0,end-window)
        values=np.empty(end-begin)
        for j in range(begin,end):
            ratio=width[t]/width[j]
            max_ratio=max(max_ratio,ratio)
            cap_hits+=int(ratio>cap or ratio<1/cap);transfers+=1
            values[j-begin]=raw[j]*min(cap,max(1/cap,ratio))
        q=np.quantile(values,1-at)
        if np.isfinite(cap):
            bound_violation=max(bound_violation,abs(q)-cap*np.max(np.abs(raw[begin:end])))
        lo[t]=left[t]-q;hi[t]=right[t]+q
        if lo[t]>hi[t]:
            lo[t]=hi[t]=(left[t]+right[t])/2;repairs+=1
    return lo,hi,clips,repairs,max_ratio,bound_violation,cap_hits/transfers


def calibrate_transfer(a,score='cqr',cap=4.,gamma=.02,alpha=.1,prefix=600,window=450):
    if cap<1 or not np.isfinite(gamma) or gamma<=0:raise ValueError('Invalid cap/rate')
    if score not in ('residual','cqr'):raise ValueError(score)
    ql,qu=band(a)
    es=past_error_scale(a.y,a.mu,ql,qu,a.delay)
    width=np.maximum(qu-ql,.001*es)
    left,right=(a.mu,a.mu) if score=='residual' else (ql,qu)
    raw=np.maximum(left-a.y,a.y-right)
    r=_bounded(a.y,left,right,width,raw,a.delay,prefix,window,alpha,gamma,cap)
    lo,hi,clips,repairs,max_ratio,violation,hit_rate=r
    return lo,hi,dict(clips=clips,singleton_repairs=repairs,max_raw_ratio=max_ratio,
                     bound_violation=violation,cap_hit_rate=hit_rate)
