"""Replay the pinned author's AcMCP interval recursion with fixed scorecasts.

Adapted from Wang/Hyndman conformalForecast, GPL-3.0; original source and license
remain in docs/references/v4/acmcp. Scorecasts must come from that source.
"""
import numpy as np
from numba import njit
from .multistep_pid import _sat

@njit(cache=True)
def replay_h(pf,errors,scorecasts,h,lr,ki,ncal=600,csat=.5):
    n=len(pf);ql=np.zeros(n);qu=np.zeros(n);bl=np.zeros(n);bu=np.zeros(n)
    lower=np.full(n,np.nan);upper=np.full(n,np.nan);count_l=0.;count_u=0.
    for t in range(h-1,len(errors)):
        err=errors[t];el=1. if -err>ql[t] else 0.;eu=1. if err>qu[t] else 0.
        count_l+=el;count_u+=eu
        hist=errors[max(t-ncal+1,h-1):t+1]
        step=lr if len(hist)==1 else lr*(np.max(hist)-np.min(hist))
        bl[t+h]=bl[t+h-1]+step*(el-.05);bu[t+h]=bu[t+h-1]+step*(eu-.05)
        count=t-h+2
        il=_sat(count_l-.05*count,count,csat,ki);iu=_sat(count_u-.05*count,count,csat,ki)
        score=scorecasts[t+h] if np.isfinite(scorecasts[t+h]) else 0.
        ql[t+h]=bl[t+h]+il-score;qu[t+h]=bu[t+h]+iu+score
        if t>=ncal+h-2:
            lower[t+h]=pf[t+h]-ql[t+h];upper[t+h]=pf[t+h]+qu[t+h]
    return lower,upper
