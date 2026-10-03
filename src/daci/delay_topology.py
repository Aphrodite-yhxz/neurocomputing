"""Explicit shared/issued/interpolated delayed ACI; exploratory stability screen."""
import numpy as np
from numba import njit
from .revision import score_components


@njit(cache=True)
def _run(y,left,right,scores,delay,prefix,window,alpha,gamma,rho):
    n=len(y);start=prefix+delay-1
    lo=np.full(n,np.nan);hi=np.full(n,np.nan);levels=np.full(n,alpha)
    corrections=np.zeros(n);at=alpha;clips=0;repairs=0
    for t in range(start,n):
        j=t-delay
        if j>=start:
            err=0. if lo[j]<=y[j]<=hi[j] else 1.
            raw=(1-rho)*at+rho*levels[j]+gamma*(alpha-err)
            at=min(.999,max(.001,raw));corrections[t]=at-raw;clips+=int(at!=raw)
        levels[t]=at
        end=t-delay+1
        q=np.quantile(scores[max(0,end-window):end],1-at)
        lo[t]=left[t]-q;hi[t]=right[t]+q
        if lo[t]>hi[t]:lo[t]=hi[t]=(left[t]+right[t])/2;repairs+=1
    return lo,hi,levels,corrections,clips,repairs


def calibrate_topology(a,score='cqr',mode='compensated',gamma=.02,alpha=.1,prefix=600,window=450):
    rho={'shared':0.,'compensated':gamma,'quarter':.25,'half':.5,'issued':1.}[mode]
    if score not in ('residual','cqr'):raise ValueError(score)
    left,right,_,scores=score_components(a,score)
    lo,hi,levels,corr,clips,repairs=_run(a.y,left,right,scores,a.delay,prefix,window,alpha,gamma,rho)
    return lo,hi,dict(levels=levels,corrections=corr,clips=clips,singleton_repairs=repairs,rho=rho)


def spectral_radius(h,gamma,g=1.,rho=0.):
    if h==1:return abs(1-gamma*g)
    coefficients=np.zeros(h+1)
    coefficients[0]=1;coefficients[1]=-(1-rho);coefficients[-1]=gamma*g-rho
    return float(np.max(np.abs(np.roots(coefficients))))
