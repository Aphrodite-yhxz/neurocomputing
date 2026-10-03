"""Matched level controllers with observed-only memory and signed-error controls."""
import numpy as np
from numba import njit
from .revision import band,interval_scores

@njit(cache=True)
def _level(y,left,right,scores,delay,prefix,window,alpha,gamma,issued,signed):
    n=len(y);start=prefix+delay-1
    lo=np.full(n,np.nan);hi=np.full(n,np.nan);levels=np.full(n,alpha)
    valid=np.flatnonzero(np.isfinite(scores));position=0;at=alpha
    for t in range(start,n):
        j=t-delay
        if j>=start:
            raw=levels[j] if issued else at
            if np.isfinite(y[j]) and np.isfinite(lo[j]) and np.isfinite(hi[j]):
                raw+=gamma*(alpha-(0. if lo[j]<=y[j]<=hi[j] else 1.))
            at=min(.999,max(.001,raw))
        levels[t]=at
        while position<len(valid) and valid[position]<=j:position+=1
        hist=scores[valid[max(0,position-window):position]]
        if len(hist)==0:continue
        if signed:
            lo[t]=left[t]+np.quantile(hist,at/2)
            hi[t]=right[t]+np.quantile(hist,1-at/2)
        else:
            q=np.quantile(hist,1-at);lo[t]=left[t]-q;hi[t]=right[t]+q
            if lo[t]>hi[t]:lo[t]=hi[t]=(left[t]+right[t])/2
    return lo,hi,levels

def calibrate_level(a,score='cqr',mode='shared',gamma=.02,window=450,prefix=600,alpha=.1):
    assert mode in ['shared','issued']
    if score=='cqr':
        left,right=band(a);scores=np.maximum(left-a.y,a.y-right)
    elif score=='residual':left=right=a.mu;scores=abs(a.y-a.mu)
    elif score=='signed':left=right=a.mu;scores=a.y-a.mu
    else:raise ValueError(score)
    return _level(a.y,left,right,scores,a.delay,prefix,window,alpha,gamma,mode=='issued',score=='signed')

def observed_metrics(y,lo,hi,start=1200,stop=None,alpha=.1):
    y=y[start:stop];lo=lo[start:stop];hi=hi[start:stop];mask=np.isfinite(y)
    yy=y[mask];ll=lo[mask];uu=hi[mask]
    if not len(yy):raise ValueError('Empty observed evaluation')
    bad=~np.isfinite(ll)|~np.isfinite(uu)|(ll>uu)
    loss=interval_scores(yy,ll,uu,alpha);loss[bad]=np.inf
    covered=(yy>=ll)&(yy<=uu)&~bad
    # A 150-calendar-origin window, not 150 possibly compressed observed targets.
    c=np.zeros(len(y));c[mask]=covered
    cc=np.r_[0,np.cumsum(c)];nn=np.r_[0,np.cumsum(mask)]
    if len(y)>=150:
        numer=cc[150:]-cc[:-150];denom=nn[150:]-nn[:-150]
        local=np.mean(abs(numer[denom>0]/denom[denom>0]-(1-alpha)))
    else:local=abs(covered.mean()-(1-alpha))
    return dict(interval_score=float(loss.mean()),coverage=float(covered.mean()),width=float(np.mean(uu-ll)),local_error=float(local),lower_miss=float(np.mean(yy<ll)),upper_miss=float(np.mean(yy>uu)),n=int(mask.sum()),missing=int((~mask).sum()),invalid_intervals=int(bad.sum()))

def nonnegative(lo,hi):return np.maximum(0,lo),np.maximum(0,hi)
