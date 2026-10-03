"""Feasibility study of COP's implicit refinement under delayed feedback.

The proximal objective is prior work (Hu et al., 2026), not a new invention here.
This module examines explicit versus implicit solutions and delay scaling.
"""
import numpy as np
from numba import njit
from .revision import score_components


@njit(cache=True)
def empirical_prox(sorted_scores,b,lam,alpha=.1):
    if lam==0:return b
    n=len(sorted_scores);p=1-alpha
    # Find the first knot whose right derivative is nonnegative.
    left=0;right=n
    while left<right:
        mid=(left+right)//2
        derivative=sorted_scores[mid]-b+lam*((mid+1)/n-p)
        if derivative>=0:right=mid
        else:left=mid+1
    k=left
    # Between knot k-1 and k, the ECDF equals k/n. Clip its smooth root to
    # the interval; this also handles a root at a knot or a repeated atom.
    q=b+lam*(p-k/n)
    if k>0:q=max(q,sorted_scores[k-1])
    if k<n:q=min(q,sorted_scores[k])
    return q


@njit(cache=True)
def _run(left,right,y,scores,delay,prefix,window,alpha,gamma,implicit,k):
    n=len(y);start=prefix+delay-1
    lo=np.full(n,np.nan);hi=np.full(n,np.nan);states=np.full(n,np.nan)
    qs=np.full(n,np.nan);unit=max(np.max(scores[:prefix])-np.min(scores[:prefix]),1e-12)
    eta=gamma*unit;b=np.quantile(scores[:prefix],1-alpha);repairs=0
    for t in range(start,n):
        j=t-delay
        if j>=start:
            error=0. if lo[j]<=y[j]<=hi[j] else 1.
            b+=eta*(error-alpha)
        hist=scores[max(0,j-window):j]
        if k==0:q=b
        elif implicit:q=empirical_prox(np.sort(hist),b,k*eta,alpha)
        else:q=b-k*eta*(np.mean(hist<=b)-(1-alpha))
        states[t]=b;qs[t]=q;lo[t]=left[t]-q;hi[t]=right[t]+q
        if lo[t]>hi[t]:lo[t]=hi[t]=(left[t]+right[t])/2;repairs+=1
    return lo,hi,states,qs,eta,repairs


def calibrate_implicit(a,score='cqr',mode='implicit_delay',gamma=.02,alpha=.1,prefix=600,window=300):
    if score not in ('cqr','residual'):raise ValueError('Unsupported score')
    options={'ogd':(False,0.),'explicit_half':(False,.5),'explicit_delay':(False,float(a.delay)),
             'implicit_half':(True,.5),'implicit_delay':(True,float(a.delay))}
    implicit,k=options[mode];left,right,_,scores=score_components(a,score)
    lo,hi,states,q,eta,repairs=_run(left,right,a.y,scores,a.delay,prefix,window,alpha,gamma,implicit,k)
    return lo,hi,dict(states=states,threshold=q,eta=eta,singleton_repairs=repairs,k=k)
