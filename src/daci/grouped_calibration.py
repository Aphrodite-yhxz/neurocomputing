"""Established groupwise calibration used as a diagnostic factorial baseline."""
import numpy as np
from numba import njit
from .revision import score_components,band


@njit(cache=True)
def _run(left,right,y,scores,groups,delay,prefix,window,alpha,gamma,group_memory,group_state):
    n=len(y);start=prefix+delay-1;state=np.full(2,alpha)
    lo=np.full(n,np.nan);hi=np.full(n,np.nan);levels=np.full(n,np.nan)
    history=np.empty((2,n));counts=np.zeros(2,np.int64);fallback=0;repairs=0
    for t in range(n):
        j=t-delay
        if j>=0:
            group=groups[j];history[group,counts[group]]=scores[j];counts[group]+=1
            if j>=start:
                g=group if group_state else 0
                err=0. if lo[j]<=y[j]<=hi[j] else 1.
                state[g]=min(.999,max(.001,state[g]+gamma*(alpha-err)))
        if t<start:continue
        group=groups[t];g=group if group_state else 0;at=state[g];levels[t]=at
        if group_memory and counts[group]>=20:
            q=np.quantile(history[group,max(0,counts[group]-window):counts[group]],1-at)
        else:
            fallback+=int(group_memory)
            end=t-delay+1;q=np.quantile(scores[max(0,end-window):end],1-at)
        lo[t]=left[t]-q;hi[t]=right[t]+q
        if lo[t]>hi[t]:lo[t]=hi[t]=(left[t]+right[t])/2;repairs+=1
    return lo,hi,levels,fallback,repairs


def calibrate_grouped(a,score='cqr',mode='both',gamma=.02,alpha=.1,prefix=600,window=450):
    if score not in ('cqr','residual'):raise ValueError('Unsupported score')
    if mode not in ('none','memory','state','both'):raise ValueError('Unknown factorial cell')
    qlo,qhi=band(a);width=qhi-qlo;threshold=float(np.median(width[:prefix]))
    groups=(width>threshold).astype(np.int64)
    left,right,_,scores=score_components(a,score)
    lo,hi,levels,fallback,repairs=_run(left,right,a.y,scores,groups,a.delay,prefix,window,alpha,gamma,
        mode in ('memory','both'),mode in ('state','both'))
    return lo,hi,dict(groups=groups,threshold=threshold,levels=levels,fallbacks=fallback,singleton_repairs=repairs)
