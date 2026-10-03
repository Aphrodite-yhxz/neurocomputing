"""Single source of numerical claims and manuscript graphics for the revision."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/neurocomputing';PAPER=ROOT/'paper/neurocomputing'
PAPER.mkdir(parents=True,exist_ok=True);(PAPER/'figures').mkdir(exist_ok=True)
OUT=BASE/'report';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})

def family(x):
    for p,f in [('electricity','UCI load'),('solar','Alabama PV'),('ETT','ETT'),('ercot','ERCOT'),('spliced','Constructed switches'),('seasonal','Constructed seasons'),('synthetic','Synthetic'),('gefcom_solar','GEFCom solar'),('gefcom_wind','GEFCom wind')]:
        if x.startswith(p):return f
    raise ValueError(x)

def load(panel):
    d=pd.read_csv(BASE/panel/'results.csv');d['family']=d.dataset.map(family)
    b=d[d.method=='residual/aci'][['backbone','dataset','horizon','interval_score']].rename(columns={'interval_score':'base'})
    d=d.merge(b,on=['backbone','dataset','horizon'],validate='many_to_one')
    d['gain']=100*(1-d.interval_score/d.base)
    return d

def ci(x):
    x=np.asarray(x);r=np.random.default_rng(27)
    z=x[r.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return [float(t) for t in np.quantile(z,[.025,.975])]

panels={n:load(n) for n in ['main_a0.1','gefcom_a0.1','dense_a0.1','multihorizon_a0.1','main_a0.2','main_a0.05']}
main=panels['main_a0.1'];mh=panels['multihorizon_a0.1'];gef=panels['gefcom_a0.1']
stats={};agg=[]
for name,d in panels.items():
    for (h,m),g in d.groupby(['horizon','method']):
        series=g.groupby('dataset').gain.mean()
        agg.append(dict(panel=name,horizon=int(h),method=m,gain=float(series.mean()),
            coverage=g.coverage.mean(),coverage_gap=g.coverage_gap.mean(),local_error=g.local_error.mean(),
            singleton_rate=g.singleton_rate.mean(),n_pairs=len(g),n_series=len(series),
            wins=int((g.gain>0).sum()),ci_low=ci(series)[0],ci_high=ci(series)[1]))
agg=pd.DataFrame(agg);agg.to_csv(OUT/'aggregate.csv',index=False)
stats['primary']=agg[(agg.panel=='main_a0.1')&(agg.method=='cqr/aci')].iloc[0].to_dict()
g=main[main.method=='cqr/aci'].groupby(['dataset','family']).gain.mean().reset_index()
stats['without_pv']={'gain':g[g.family!='Alabama PV'].gain.mean(),'ci':ci(g[g.family!='Alabama PV'].gain)}
stats['observed_only']={'gain':g[~g.family.str.startswith(('Constructed','Synthetic'))].gain.mean(),'ci':ci(g[~g.family.str.startswith(('Constructed','Synthetic'))].gain)}
stats['leave_family_out']={f:float(g[g.family!=f].gain.mean()) for f in sorted(g.family.unique())}
fam=main.groupby(['family','method']).agg(gain=('gain','mean'),coverage=('coverage','mean'),n=('gain','size')).reset_index()
fam.to_csv(OUT/'family_results.csv',index=False)
main.groupby(['backbone','method'])[['gain','coverage','coverage_gap','local_error']].mean().to_csv(OUT/'backbone_results.csv')
gef.groupby(['family','method'])[['gain','coverage','coverage_gap','local_error']].mean().to_csv(OUT/'gefcom_family_results.csv')
native=pd.read_csv(BASE/'main_a0.1/native_coverage.csv').groupby(['backbone','nominal'])[['coverage','width','interval_score']].mean()
native.to_csv(OUT/'native.csv')
stats['native_coverage']=native.reset_index().to_dict('records')
stats['tuned_paired_gain']=float(pd.read_csv(BASE/'main_a0.1/paired_comparisons.csv').query("method=='cqr/aci/tuned'").relative_gain.mean())
stats['significance']={}
for block in [100,200,400]:
    s=pd.read_csv(BASE/'main_a0.1/paired_comparisons.csv')
    s=s[(s.method=='cqr/aci')&(s.block==block)]
    p=s.p.to_numpy();order=np.argsort(p);threshold=.05*np.arange(1,len(p)+1)/len(p)
    passing=np.flatnonzero(p[order]<=threshold);cut=p[order][passing[-1]] if len(passing) else -1
    stats['significance'][str(block)]=dict(ci_positive=int((s.ci_low>0).sum()),ci_negative=int((s.ci_high<0).sum()),bh_discoveries=int((p<=cut).sum()),bh_positive=int(((p<=cut)&(s.mean_gain>0)).sum()))
# Bootstrap uncertainty here resamples series, not correlated backbones as independent.
for m in ['prefix_selected/aci','cqr/aci/tuned','cqr/aci/unsorted']:
    stats[m]=agg[(agg.panel=='main_a0.1')&(agg.method==m)].iloc[0].to_dict()
(OUT/'claims.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')

# Score x rule: all bars are relative to the same residual-ACI reference.
rules=['split','rolling','aci','sfogd','pi_log','saocp','faci']
scores=['residual','error_scale','native_scale','cqr']
labels=['Residual','Past-error scale','Native scale','CQR']
fig,ax=plt.subplots(figsize=(7.5,3.7),layout='constrained')
x=np.arange(len(rules));colors=['#6C7A89','#D5A53B','#319799','#354F9D']
for k,(score,label) in enumerate(zip(scores,labels)):
    vals=[agg.query("panel=='main_a0.1'").set_index('method').loc[score+'/'+r,'gain'] for r in rules]
    ax.bar(x+(k-1.5)*.19,vals,.19,label=label,color=colors[k])
ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,['Split','Rolling','ACI','SF-OGD*','PI-log*','SAOCP*','FACI-R*'])
ax.set_ylabel('Interval-score improvement (%)');ax.legend(ncol=4,fontsize=10,loc='upper left')
ax.set_ylim(ax.get_ylim()[0],20)
fig.savefig(PAPER/'figures/score_rule.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.5,3.6),layout='constrained')
for score,rule,label,c in [('cqr','aci','CQR + ACI','#354F9D'),('cqr','pi_log','CQR + PI-log','#168B88'),('native_scale','aci','Native scale + ACI','#BB7635'),('residual','aci','Residual + ACI','#6C7A89')]:
    sub=mh[mh.method==score+'/'+rule].groupby('horizon')[['coverage','gain']].mean()
    axes[0].plot(sub.index,sub.coverage*100,'o-',label=label,color=c)
    axes[1].plot(sub.index,sub.gain,'o-',label=label,color=c)
axes[0].axhline(90,color='black',ls=':',lw=1);axes[0].set_ylabel('Coverage (%)')
axes[1].axhline(0,color='black',lw=.6);axes[1].set_ylabel('Interval-score improvement (%)')
for a in axes:a.set_xlabel('Horizon / lag (steps)');a.set_xticks([1,6,24])
axes[0].legend(fontsize=10);fig.savefig(PAPER/'figures/horizon.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,3.6),layout='constrained')
fams=list(g.groupby('family').gain.mean().sort_values().index)
for i,f in enumerate(fams):
    v=g[g.family==f].gain.to_numpy();ax.scatter(np.full(len(v),i)+np.linspace(-.18,.18,len(v)),v,color='#354F9D',s=20,alpha=.8)
    ax.plot([i-.25,i+.25],[v.mean()]*2,color='#CB6528',lw=2)
ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(len(fams)),[f.replace(' ','\n') for f in fams]);ax.set_ylabel('CQR-ACI improvement (%)')
fig.savefig(PAPER/'figures/families.png');plt.close(fig)

dc=pd.read_csv(BASE/'delay_control/results.csv')
fig,axes=plt.subplots(1,2,figsize=(7.5,3.6),layout='constrained')
for method,label,c in [('cqr/aci','CQR + ACI','#354F9D'),('cqr/pi_log','CQR + PI-log','#168B88'),('residual/aci','Residual + ACI','#6C7A89')]:
    sub=dc[dc.method==method].groupby('feedback_lag')[['coverage','coverage_gap','local_error']].mean()
    axes[0].plot(sub.index,100*sub.coverage,'o-',label=label,color=c)
    axes[1].plot(sub.index,100*sub.local_error,'o-',label=label,color=c)
axes[0].axhline(90,color='black',ls=':',lw=1);axes[0].set_ylabel('Coverage (%)');axes[1].set_ylabel('150-step coverage error (pp)')
for a in axes:a.set_xlabel('Artificial lag (fixed h=1)');a.set_xticks([1,6,24])
axes[1].legend(fontsize=10);fig.savefig(PAPER/'figures/delay_control.png');plt.close(fig)

# Machine-generated TeX tables, kept compact enough for a single-column preprint.
def tex_table(name,cols,rows):
    lines=['\\begin{tabular}{'+'l'+'r'*(len(cols)-1)+'}','\\toprule',' & '.join(cols)+r' \\','\\midrule']
    lines += [' & '.join(map(str,r))+r' \\' for r in rows]
    lines+=['\\bottomrule','\\end{tabular}']
    (PAPER/(name+'.tex')).write_text('\n'.join(lines),encoding='utf-8')

chosen=['residual/aci','error_scale/aci','native_scale/aci','cqr/aci','residual/faci','cqr/faci','residual/saocp','cqr/saocp','cqr/pi_log','cqr/sfogd','prefix_selected/aci','residual/aci/tuned','cqr/aci/tuned']
table=[]
for m in chosen:
    r=agg[(agg.panel=='main_a0.1')&(agg.method==m)].iloc[0]
    table.append([m.replace('_','-'),f'{r.gain:.2f}',f'{100*r.coverage:.2f}',f'{100*r.coverage_gap:.2f}',f'{100*r.local_error:.2f}'])
tex_table('table_main',['Score / rule','Gain (\\%)','Cov. (\\%)','Gap (pp)','Local (pp)'],table)
table=[]
for h in [1,6,24]:
    for m in ['residual/aci','native_scale/aci','cqr/aci','cqr/pi_log','cqr/saocp','cqr/faci','cqr/aci/tuned']:
        r=agg[(agg.panel=='multihorizon_a0.1')&(agg.horizon==h)&(agg.method==m)].iloc[0]
        table.append([str(h)+' / '+m.replace('_','-'),f'{r.gain:.2f}',f'{100*r.coverage:.2f}',f'{100*r.local_error:.2f}'])
tex_table('table_horizon',['Horizon / method','Gain (\\%)','Cov. (\\%)','Local (pp)'],table)
table=[]
for f,sub in g.groupby('family'):
    c=ci(sub.gain);table.append([f,len(sub),f'{sub.gain.mean():.2f}',f'[{c[0]:.2f}, {c[1]:.2f}]'])
tex_table('table_family',['Family','Series','Gain (\\%)','Descriptive 95\\% CI'],table)
table=[]
for name in ['main_a0.2','main_a0.1','main_a0.05','dense_a0.1','gefcom_a0.1']:
    r=agg[(agg.panel==name)&(agg.method=='cqr/aci')].iloc[0]
    table.append([name.replace('_','-'),int(r.n_pairs),f'{r.gain:.2f}',f'{100*r.coverage:.2f}'])
tex_table('table_sensitivity',['Panel / miscoverage','Pairs','Gain (\\%)','Cov. (\\%)'],table)
series=main[main.method=='cqr/aci'].groupby('dataset')[['gain','coverage']].mean()
tex_table('table_series',['Series','Gain (\\%)','Coverage (\\%)'],[[i.replace('_',r'\_'),f'{r.gain:.2f}',f'{100*r.coverage:.2f}'] for i,r in series.iterrows()])
print(json.dumps(stats,indent=2))
