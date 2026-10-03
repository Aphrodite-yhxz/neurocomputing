"""Score-space reference rules, audited against archived author implementations.

SAOCP follows Salesforce's Apache-2.0 implementation in docs/references/saocp.py
(copyright Salesforce 2023); its license is docs/references/saocp_LICENSE.txt.
Changes: array-based execution, archived forecasts, signed-score projection switch,
and explicit delayed feedback. Forecast-model training wrappers are not used.
FACI follows docs/references/faci.py (same license), with a declared rolling history.
"""
import numpy as np
from numba import njit
from .revision import score_components


@njit(cache=True)
def _loss(s,q,alpha):
    return max((1-alpha)*(s-q),alpha*(q-s))


@njit(cache=True)
def saocp_thresholds(scores, prefix=600, delay=1, alpha=.1, project=True, lifetime=8):
    n=len(scores)
    q=np.zeros(n+1); gn=np.zeros(n+1); z=np.zeros(n+1); wz=np.zeros(n+1)
    age=np.zeros(n+1,np.int64); life=np.zeros(n+1,np.int64)
    active=np.empty(n+1,np.int64); count=0
    out=np.empty(n); weight=np.empty(n+1); prior=np.empty(n+1)
    scale=max(np.max(np.abs(scores[:prefix]))*np.sqrt(3),1e-12)
    pred=0.
    for i in range(n):
        j=i-delay
        if j>=0:
            t=j+1
            previous=pred
            kept=0
            for k in range(count):
                e=active[k]
                if age[e]<=life[e]:
                    active[kept]=e; kept+=1
            count=kept
            active[count]=t; count+=1
            twopower=1; v=t
            while v%2==0:
                twopower*=2; v//=2
            life[t]=lifetime*twopower; q[t]=previous
            # Reference computes meta-loss after adding the new, zero-weight expert.
            sw=0.; sp=0.
            for k in range(count):
                e=active[k]
                prior[k]=1/(e*e*(1+np.floor(np.log2(e))))
                w=0. if age[e]==0 else z[e]/age[e]*(1+wz[e])
                weight[k]=prior[k]*max(w,0.); sw+=weight[k]; sp+=prior[k]
            meta=0.
            for k in range(count):
                p=weight[k]/sw if sw>0 else prior[k]/sp
                meta+=p*q[active[k]]
            ml=_loss(scores[j],meta,alpha)
            for k in range(count):
                e=active[k]
                w=0. if age[e]==0 else z[e]/age[e]*(1+wz[e])
                g=(ml-_loss(scores[j],q[e],alpha))/scale/max(alpha,1-alpha)
                g=min(max(g,(-1. if w>0 else 0.)),1.)
                z[e]+=g; wz[e]+=g*w; age[e]+=1
                grad=-(1-alpha) if scores[j]>q[e] else (alpha if scores[j]<q[e] else 0.)
                gn[e]+=grad*grad
                if gn[e]>0:
                    q[e]-=scale/np.sqrt(3*gn[e])*grad
                    if project: q[e]=max(q[e],0.)
            # Next issued prediction uses post-update expert weights.
            sw=0.;sp=0.
            for k in range(count):
                e=active[k]
                prior[k]=1/(e*e*(1+np.floor(np.log2(e))))
                w=z[e]/age[e]*(1+wz[e])
                weight[k]=prior[k]*max(w,0.);sw+=weight[k];sp+=prior[k]
            pred=0.
            for k in range(count):
                p=weight[k]/sw if sw>0 else prior[k]/sp
                pred+=p*q[active[k]]
        out[i]=pred
    return out


@njit(cache=True)
def faci_thresholds(scores,prefix=600,delay=1,alpha=.1,window=450):
    n=len(scores);k=8
    alphas=np.full(k,alpha);w=np.ones(k)/k
    gammas=np.array([.001*2**i for i in range(k)])
    mix=.005
    denom=((1-alpha)**2*alpha**3+alpha**2*(1-alpha)**3)/3
    eta=np.sqrt(3/100)*np.sqrt((np.log(100*k)+2)/denom)
    out=np.zeros(n)
    for i in range(n):
        j=i-delay
        if j>=prefix:
            old=scores[max(0,j-window):j]
            beta=np.mean(old>=scores[j])
            losses=np.empty(k)
            for e in range(k): losses[e]=max(alpha*(beta-alphas[e]),(1-alpha)*(alphas[e]-beta))
            # Stabilized multiplicative update followed by fixed share.
            logs=np.log(w)-eta*losses; logs-=np.max(logs)
            wb=np.exp(logs);wb/=np.sum(wb)
            w=(1-mix)*wb+mix/k
            for e in range(k):
                alphas[e]=min(max(alphas[e]+gammas[e]*(alpha-(1. if alphas[e]>beta else 0.)),0.),1.)
        end=i-delay+1
        if end>0:
            arr=np.sort(scores[max(0,end-window):end])
            lev=1-np.dot(w,alphas)
            idx=min(max(int(np.ceil(lev*len(arr)))-1,0),len(arr)-1)
            out[i]=arr[idx]
    return out


def calibrate_reference(a,score="residual",rule="saocp",alpha=.1,prefix=600,window=450):
    left,right,scale,scores=score_components(a,score)
    if rule=="saocp":
        q=saocp_thresholds(scores,prefix,a.delay,alpha,score in ("residual","native_scale","error_scale"))
    elif rule=="faci":
        q=faci_thresholds(scores,prefix,a.delay,alpha,window)
    else: raise ValueError(rule)
    lo=left-q*scale;hi=right+q*scale
    bad=lo>hi;mid=(left+right)/2
    lo[bad]=mid[bad];hi[bad]=mid[bad]
    return lo,hi,dict(singleton_repairs=int(bad[prefix+a.delay-1:].sum()),
                      nonfinite=int((~np.isfinite(lo[prefix+a.delay-1:])).sum()),clips=0)
