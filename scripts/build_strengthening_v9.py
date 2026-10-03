"""Build figures and tables from the locked confirmation and explanation controls."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/strengthening_v9'
PAPER=ROOT/'paper/neurocomputing_strengthened'
def table(name,header,rows,spec):
    (PAPER/name).write_text('\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+[r+r' \\' for r in rows]+[r'\bottomrule',r'\end{tabular}'])+'\n')
def main():
    s=pd.read_csv(OUT/'grid/selected.csv');c=pd.read_csv(OUT/'grid/state_contrasts.csv')
    p=c[(c.horizon==24)&(c.period=='validation')]
    stages=['original','wide','zero_small','window'];labels=['Original','Wide','Add zero / small','Add window']
    rows=[];summary=[]
    for family in ['air','weather']:
        for score in ['residual','cqr']:
            vals=[p[(p.family==family)&(p.score==score)&(p.grid==g)].state_gain.mean() for g in stages]
            rows.append(('Air quality' if family=='air' else 'Weather')+' & '+('Residual' if score=='residual' else 'CQR')+' & '+' & '.join(f'{v:.2f}' for v in vals))
            summary.append(dict(family=family,score=score,**dict(zip(stages,vals))))
    table('table_new_confirmation.tex','Source & Score & Original & Wide & Zero/small & Window',rows,'llrrrr')
    pd.DataFrame(summary).to_csv(OUT/'confirmation_headlines.csv',index=False)
    # Direct comparisons retain per-pair denominators.
    primary=s[(s.period=='validation')&(s.horizon==24)]
    roll=primary[(primary.grid=='rolling_tuned')&(primary.score=='cqr')&(primary['mode']=='shared')]
    rolling_rows=[];direct=[]
    for family in ['air','weather']:
        for mode in ['shared','issued']:
            a=primary[(primary.family==family)&(primary.grid=='window')&(primary.score=='cqr')&(primary['mode']==mode)]
            b=roll[roll.family==family]
            d=a.merge(b,on=['backbone','dataset'],suffixes=('_aci','_roll'),validate='one_to_one')
            d['gain']=100*(1-d.interval_score_aci/d.interval_score_roll);d['family']=family;d['state']=mode;direct.append(d)
            rolling_rows.append(f'{family.title()} & {mode.title()} & {d.gain.mean():+.2f} & {100*d.coverage_aci.mean():.2f} & {100*d.coverage_roll.mean():.2f} & {100*d.local_error_aci.mean():.2f} & {100*d.local_error_roll.mean():.2f}')
    pd.concat(direct).to_csv(OUT/'rolling_direct.csv',index=False)
    table('table_new_rolling.tex','Source & State & Gain & Cov. ACI & Cov. roll & Local ACI & Local roll',rolling_rows,'llrrrrr')
    pv=pd.read_csv(OUT/'pv/summary.csv');pv=pv[(pv.settings=='tuned')&pv.group.isin(['all','positive','forecast_high'])]
    rows=[]
    for proj,baseline in [('raw','residual'),('nonnegative','residual'),('nonnegative','signed')]:
        for group in ['all','positive','forecast_high']:
            r=pv[(pv.projection==proj)&(pv.baseline==baseline)&(pv.group==group)].iloc[0]
            rows.append(f'{"Raw" if proj=="raw" else "Nonnegative"} & {baseline.title()} & {group.replace("_"," ")} & {r.gain:.2f} & {100*r.coverage_cqr:.2f} & {100*r.coverage_baseline:.2f}')
    table('table_pv_controls.tex','Support & Baseline & Group & CQR gain & Cov. CQR & Cov. base',rows,'lllrrr')
    # Correctly calculate validation sensitivity from the archived selectors.
    keys=['backbone','dataset','family','score','mode','grid','horizon']
    v=s[s.period=='early'].merge(s[s.period=='late'],on=keys,suffixes=('_early','_late'),validate='one_to_one')
    v['same_configuration']=(v.gamma_early==v.gamma_late)&(v.window_early==v.window_late)
    v.to_csv(OUT/'validation_selection_pairs.csv',index=False)
    v.groupby(['horizon','score','mode','grid']).same_configuration.mean().to_csv(OUT/'grid/validation_stability.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8),layout='constrained')
    for ax,family,title in zip(axes,['air','weather'],['New source: air quality','New source: weather']):
        for score,color in [('residual','#305f91'),('cqr','#c57530')]:
            vals=[p[(p.family==family)&(p.score==score)&(p.grid==g)].state_gain.mean() for g in stages]
            ax.plot(range(4),vals,'o-',label=score.upper() if score=='cqr' else 'Residual',color=color)
        ax.axhline(0,color='gray',lw=.7);ax.set(title=title,xticks=range(4),xticklabels=['Original','Wide','+ zero','+ window'],ylabel='Issued/shared score gain (%)')
        ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    fig.savefig(PAPER/'figures/new_confirmation.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8),layout='constrained')
    for group,label,color in [('all','All outputs','#305f91'),('positive','Actual positive','#c57530'),('forecast_high','High forecast','#b25f91')]:
        vals=[];cov=[]
        for proj,base in [('raw','residual'),('nonnegative','residual'),('nonnegative','signed')]:
            r=pv[(pv.projection==proj)&(pv.baseline==base)&(pv.group==group)].iloc[0];vals.append(r.gain);cov.append(r.coverage_cqr*100)
        axes[0].plot(range(3),vals,'o-',label=label,color=color)
        axes[1].plot(range(3),cov,'o-',label=label,color=color)
    for ax in axes:ax.set(xticks=range(3),xticklabels=['Raw / symmetric','Support / symmetric','Support / signed']);ax.tick_params(axis='x',labelsize=7);ax.grid(alpha=.15)
    axes[0].set(title='Native-band benefit after controls',ylabel='CQR score gain (%)');axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(title='Coverage of selected CQR',ylabel='Coverage (%)');axes[1].axhline(90,color='gray',ls='--',lw=.8)
    fig.savefig(PAPER/'figures/pv_controls.png',dpi=220);plt.close(fig)
    ac=pd.read_csv(OUT/'acmcp/selected.csv');keys=['backbone','dataset','horizon']
    base=s[(s.family=='weather')&(s.period=='validation')&(s.score=='cqr')&(s['mode']=='shared')&(s.grid=='rolling_tuned')]
    allcontrasts=[];rows=[]
    for h in [1,6,24]:
        br=base[base.horizon==h]
        for name,frame in [('Rolling CQR',br),('Issued CQR',s[(s.family=='weather')&(s.period=='validation')&(s.score=='cqr')&(s['mode']=='issued')&(s.grid=='window')&(s.horizon==h)]),('AcMCP-CSS',ac[(ac.variant=='AcMCP-CSS')&(ac.horizon==h)]),('PID, no scorecast',ac[(ac.variant=='PID-no-scorecast')&(ac.horizon==h)])]:
            d=frame.merge(br,on=keys,suffixes=('','_rolling'),validate='one_to_one')
            d['score_ratio']=d.interval_score/d.interval_score_rolling;d['method']=name
            allcontrasts.append(d)
            rows.append(f'{h} & {name} & {d.score_ratio.mean():.3f} & {100*d.coverage.mean():.2f} & {100*d.local_error.mean():.2f}')
    pd.concat(allcontrasts).to_csv(OUT/'acmcp/direct_comparison.csv',index=False)
    table('table_acmcp_full.tex','Horizon & Method & IS ratio & Coverage & Local',rows,'rlrrr')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
