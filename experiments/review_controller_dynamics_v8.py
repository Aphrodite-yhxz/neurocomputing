"""Prescribed-response controller simulations, separate from empirical forecasting."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results/review_v8/dynamics'

@njit(cache=True)
def simulate(eq,draws,h,g,gamma,issued,stochastic):
    n=len(eq);a=np.full(n,.105);feedback=np.empty(n);clipped=np.zeros(n)
    for t in range(n):
        if t>=h:
            j=t-h;base=a[j] if issued else a[t-1]
            raw=base+gamma*(.1-feedback[j]);a[t]=min(.999,max(.001,raw));clipped[t]=a[t]-raw
        p=min(1.,max(0.,.1+g*(a[t]-eq[t])))
        feedback[t]=(1. if draws[t]<p else 0.) if stochastic else p
    return a,feedback,clipped

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];traces={};n=12000
    for kind in ['stationary','tracking']:
        eq=np.full(n,.1)
        if kind=='tracking':eq[4000:8000]=.2;eq[8000:]=.06
        for h in [1,6,24]:
            bound=2*np.sin(np.pi/(4*h-2))
            for g in [1.,4.]:
                settings=[('ratio_'+str(r),r*bound/g) for r in [.5,1.2]] if kind=='stationary' else [(f'gamma_{v:g}',v) for v in sorted(set([.005,.02,.05]+[x/h for x in [.005,.02,.05]]))]
                for label,gamma in settings:
                    for stochastic in [False,True]:
                        for seed in (range(20) if stochastic else [0]):
                            draws=np.random.default_rng(seed).random(n)
                            for mode in ['shared','issued']:
                                a,e,c=simulate(eq,draws,h,g,gamma,mode=='issued',stochastic)
                                early=np.r_[np.arange(4000,4500),np.arange(8000,8500)]
                                rows.append(dict(kind=kind,horizon=h,slope=g,gamma=gamma,setting=label,
                                    stochastic=stochastic,seed=seed,mode=mode,gain_ratio=gamma*g/bound,
                                    stationary_state_sd=float(np.std(a[2000:4000])),
                                    stationary_rmse=float(np.sqrt(np.mean((a[2000:4000]-eq[2000:4000])**2))),
                                    transition_mae=float(np.mean(abs(a[early]-eq[early]))),
                                    mean_miscoverage=float(e[2000:].mean()),clip_fraction=float(np.mean(c[2000:]!=0))))
                                if not stochastic and h==24 and g==4 and (kind=='stationary' or gamma==.005):
                                    traces[kind+'/'+label+'/'+mode]=a
    d=pd.DataFrame(rows);d.to_csv(OUT/'results.csv',index=False)
    d.groupby(['kind','horizon','slope','gamma','setting','stochastic','mode']).mean(numeric_only=True).to_csv(OUT/'summary.csv')
    np.savez_compressed(OUT/'illustrative_traces.npz',**traces)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    for mode,color in [('shared','#305f91'),('issued','#c57530')]:
        axes[0].plot(traces['stationary/ratio_1.2/'+mode][:2000],label=mode,color=color,lw=1)
        axes[1].plot(np.arange(-100,800),traces['tracking/gamma_0.005/'+mode][3900:4800],label=mode,color=color,lw=1.5)
    axes[0].axhline(.1,c='gray',ls='--',lw=.7);axes[0].set(title='Above shared local boundary',xlabel='Controller time',ylabel='State a')
    axes[1].plot([-100,0,0,799],[.1,.1,.2,.2],ls='--',c='gray',lw=.8,label='equilibrium')
    axes[1].set(title='Same small rate: tracking tradeoff',xlabel='Time since equilibrium change',ylabel='State a')
    for ax in axes:ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    fig.savefig(OUT/'controlled_dynamics.png',dpi=220);plt.close(fig)
    (OUT/'manifest.json').write_text(json.dumps(dict(status='prescribed_response_simulation_not_forecast_evidence',
        configuration_runs=len(d),bernoulli_seeds=list(range(20)),
        protocol_sha256=hashlib.sha256((ROOT/'docs/REVIEW_RESPONSE_PROTOCOL_V8.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('Completed prescribed-response runs:',len(d))
    print(d[(d.kind=='tracking')&(~d.stochastic)&(d.horizon==24)&(d.slope==4)&(d.gamma==.005)][['mode','transition_mae']].to_string(index=False))

if __name__=='__main__':main()
