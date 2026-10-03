"""COP reference update and explicitly causal delayed-score adaptation.

Based on Hu et al., ICLR 2026, author's MIT-licensed implementation at
creator-xi/Conformal-Optimistic-Prediction, a41c0dc. The untouched source and
license are archived in docs/references/v4/cop. This is an existing baseline.
"""
import numpy as np
from numba import njit
from .revision import score_components


@njit(cache=True)
def reference_cop(scores, alpha=.1, lr=.1, burnin=300, proportional=True):
    """Exact author h=1 recursion, including zero initialization and ECDF lag."""
    n=len(scores);q=np.zeros(n);state=0.
    for t in range(n-1):
        step=lr
        if proportional and t>0:
            hist=scores[max(0,t-burnin):t]
            step=lr*(np.max(hist)-np.min(hist))
        grad=alpha if q[t]>=scores[t] else -(1-alpha)
        state-=step*grad
        hint=0. if t<burnin else np.mean(scores[t-burnin:t]<=state)-(1-alpha)
        q[t+1]=state-.5*step*hint
    return q


@njit(cache=True)
def _delayed(left,right,y,scores,delay,prefix,burnin,alpha,lr):
    n=len(y);start=prefix+delay-1
    lo=np.full(n,np.nan);hi=np.full(n,np.nan);qs=np.full(n,np.nan)
    state=np.quantile(scores[:prefix],1-alpha);repairs=0
    for t in range(start,n):
        j=t-delay;threshold=state
        if j>=start:
            # Exclude j itself, matching the author's h=1 ECDF/range convention.
            hist=scores[max(0,j-burnin):j]
            step=lr*(np.max(hist)-np.min(hist))
            error=0. if lo[j]<=y[j]<=hi[j] else 1.
            state+=step*(error-alpha)
            hint=np.mean(hist<=state)-(1-alpha)
            threshold=state-.5*step*hint
        qs[t]=threshold;lo[t]=left[t]-threshold;hi[t]=right[t]+threshold
        if lo[t]>hi[t]:
            lo[t]=hi[t]=(left[t]+right[t])/2;repairs+=1
    return lo,hi,qs,repairs


def calibrate_cop(a,score='cqr',lr=.1,alpha=.1,prefix=600,burnin=300):
    if score not in ('cqr','residual') or not 0<alpha<1 or lr<=0:
        raise ValueError('Invalid COP configuration')
    if prefix<burnin or prefix+a.delay>=len(a.y):raise ValueError('Insufficient prefix')
    left,right,_,scores=score_components(a,score)
    lo,hi,q,repairs=_delayed(left,right,a.y,scores,a.delay,prefix,burnin,alpha,lr)
    return lo,hi,dict(threshold=q,singleton_repairs=repairs)
