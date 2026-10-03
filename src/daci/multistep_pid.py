"""Asymmetric author multi-step PID algebra, accelerated for fixed forecasts.

Algorithm source: Wang/Hyndman conformalForecast, commit 0fe81e6; GPL-3.0.
Source and original license retained in docs/references/v4/acmcp.
The custom least-squares AR(1) scorecaster is explicitly declared in the V8 addendum.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _sat(x,n,csat,ki):
    if n<=1 or ki==0:return 0.
    z=x*np.log(n)/(csat*n)
    if z>=np.pi/2:return np.inf
    if z<=-np.pi/2:return -np.inf
    return ki*np.tan(z)

@njit(cache=True)
def _scorecast(hist,h,mode):
    if mode==0:return 0.
    if mode==1:return np.mean(hist)
    if mode==2:return hist[-1]
    x=hist[:-1];y=hist[1:];xm=np.mean(x);ym=np.mean(y)
    denom=np.sum((x-xm)**2)
    phi=0. if denom<=1e-20 else np.sum((x-xm)*(y-ym))/denom
    phi=min(.995,max(-.995,phi));intercept=ym-phi*xm
    pred=hist[-1]
    for _ in range(h):pred=intercept+phi*pred
    return pred

@njit(cache=True)
def _run(mu,y,h,lr,ki,mode,ncal,csat,alpha):
    n=len(y);errors=y-mu
    qlo=np.zeros(n);qhi=np.zeros(n);base_lo=0.;base_hi=0.;count_lo=0.;count_hi=0.
    lo=np.full(n,np.nan);hi=np.full(n,np.nan)
    for t in range(h,n):
        j=t-h;err=errors[j]
        el=1. if -err>qlo[j] else 0.;eu=1. if err>qhi[j] else 0.
        hist=errors[max(0,j-ncal+1):j+1]
        eta=lr if len(hist)<=1 else lr*(np.max(hist)-np.min(hist))
        base_lo+=eta*(el-alpha/2);base_hi+=eta*(eu-alpha/2)
        count_lo+=el;count_hi+=eu;count=j+1
        il=_sat(count_lo-count*alpha/2,count,csat,ki)
        iu=_sat(count_hi-count*alpha/2,count,csat,ki)
        sc=_scorecast(hist,h,mode) if t>=ncal+h-1 else 0.
        qlo[t]=base_lo+il-sc;qhi[t]=base_hi+iu+sc
        if t>=ncal+h-1:
            lo[t]=mu[t]-qlo[t];hi[t]=mu[t]+qhi[t]
    return lo,hi

def calibrate_multistep_pid(a,lr=.01,ki=1.,scorecaster='none',ncal=600,csat=.5,alpha=.1):
    modes={'none':0,'mean':1,'naive':2,'ar1':3}
    lo,hi=_run(a.mu,a.y,a.delay,lr,ki,modes[scorecaster],ncal,csat,alpha)
    crossed=lo>hi
    # Feedback in _run remains author's tail-score feedback; repairs are output-only.
    lo[crossed]=hi[crossed]=(lo[crossed]+hi[crossed])/2
    return lo,hi,dict(crossed=crossed,singleton_repairs=int(crossed.sum()))
