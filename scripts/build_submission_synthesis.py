"""Direct paired rolling comparisons and final-precision submission displays.

This summarizes saved results; it does not tune or recompute a controller.
"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper/neurocomputing_submission'
OUT=ROOT/'results/submission_synthesis'

def table(name,header,rows,spec):
    (PAPER/name).write_text('\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+
        [r+r' \\' for r in rows]+[r'\bottomrule',r'\end{tabular}'])+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    legacy=ROOT/'paper/neurocomputing_review_v8'
    for name in ['table_rate_sensitivity.tex','table_clip_accounting.tex',
                 'table_confirmation_controls.tex','table_solar_components.tex']:
        (PAPER/name).write_text((legacy/name).read_text().replace('Revisited V7','Traffic/FX'))
    shutil.copyfile(ROOT/'results/review_v8/dynamics/controlled_dynamics.png',PAPER/'figures/controlled_dynamics.png')
    old=pd.read_csv(ROOT/'results/confirmation_v7/calibration/selected.csv')
    wide=pd.read_csv(ROOT/'results/review_v8/state_grid/revisit/selected.csv')
    rolling=old[(old.score=='cqr')&(old['mode']=='rolling')].copy()
    keys=['backbone','dataset','horizon']
    records=[]
    for name,frame in [('Rolling CQR',rolling),
            ('Shared ACI',wide[(wide.score=='cqr')&(wide['mode']=='shared')&(wide.selection=='wide')]),
            ('Issued ACI',wide[(wide.score=='cqr')&(wide['mode']=='issued')&(wide.selection=='wide')]),
            ('COP',old[(old.score=='cqr')&(old['mode']=='cop')])]:
        fields=keys+['interval_score','coverage','local_error','width','lower_miss','upper_miss']
        c=frame[fields].merge(rolling[fields],on=keys,suffixes=('','_rolling'),validate='one_to_one')
        c['method']=name;c['family']=c.dataset.str.rsplit('_c',n=1).str[0]
        c['paired_gain']=100*(1-c.interval_score/c.interval_score_rolling)
        c['coverage_delta_pp']=100*(c.coverage-c.coverage_rolling)
        c['local_delta_pp']=100*(c.local_error-c.local_error_rolling)
        c['positive_score_gain']=c.paired_gain>1e-10
        records.append(c)
    full=pd.concat(records,ignore_index=True);full.to_csv(OUT/'rolling_pairs.csv',index=False)
    summary=full.groupby(['horizon','method']).agg(gain=('paired_gain','mean'),median=('paired_gain','median'),
        wins=('positive_score_gain','sum'),pairs=('paired_gain','size'),coverage=('coverage','mean'),
        local_error=('local_error','mean'),coverage_delta_pp=('coverage_delta_pp','mean'),local_delta_pp=('local_delta_pp','mean'))
    summary.to_csv(OUT/'rolling_summary.csv')
    full.groupby(['horizon','method','family']).agg(gain=('paired_gain','mean'),median=('paired_gain','median'),
        wins=('positive_score_gain','sum'),pairs=('paired_gain','size'),coverage=('coverage','mean'),local_error=('local_error','mean')).to_csv(OUT/'rolling_family_summary.csv')
    rows=[]
    for name in ['Rolling CQR','Shared ACI','Issued ACI','COP']:
        r=summary.loc[(24,name)]
        sign='--' if name=='Rolling CQR' else f'{int(r.wins)}/16'
        rows.append(f'{name} & {r.gain:+.2f} & {sign} & {100*r.coverage:.2f} & {100*r.local_error:.2f} & {r.coverage_delta_pp:+.2f} & {r.local_delta_pp:+.2f}')
    table('table_rolling_direct.tex','CQR controller & Gain & Wins & Coverage & Local & $\\Delta$ cov. & $\\Delta$ local',rows,'lrrrrrr')
    # All headline values and table cells use unrounded pair-averaged values.
    solar=pd.read_csv(ROOT/'results/review_v8/solar_components/summary.csv')
    s=solar.set_index(['settings','regime'])
    values={'PVDefaultPositiveCoverage':100*s.loc[('default','positive'),'coverage_cqr'],
            'PVDefaultOverallCoverage':100*s.loc[('default','all'),'coverage_cqr'],
            'PVTunedPositiveCoverage':100*s.loc[('tuned','positive'),'coverage_cqr'],
            'PVTunedPositiveGain':s.loc[('tuned','positive'),'conditional_gain'],
            'PVDefaultPositiveGain':s.loc[('default','positive'),'conditional_gain']}
    for name,method in [('Rolling','Rolling CQR'),('Shared','Shared ACI'),('Issued','Issued ACI'),('COP','COP')]:
        r=summary.loc[(24,method)]
        values[name+'RollingGain']=r.gain
        values[name+'Coverage']=100*r.coverage
        values[name+'Local']=100*r.local_error
    (PAPER/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.2f}'+'}' for k,v in values.items())+'\n')
    (OUT/'headline_values.json').write_text(json.dumps(values,indent=2)+'\n')
    rows=[]
    for _,r in solar.iterrows():
        rows.append(f'{r.settings} & {r.regime} & {100*r.fraction_cqr:.2f} & {r.conditional_gain:.2f} & {r.overall_gain_contribution:.2f} & {100*r.coverage_residual:.2f} & {100*r.coverage_cqr:.2f}')
    table('table_solar_decomposition.tex','Settings & Group & Share & Gain & Contribution & Cov. R & Cov. CQR',rows,'llrrrrr')
    # Remove internal phase identifiers from scientific figure labels.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    common=[.0001,.0002,.0005,.001,.002,.005,.01,.02,.05,.1];matrices=[]
    for panel in ['development','revisit']:
        frame=pd.read_csv(ROOT/f'results/review_v8/state_grid/{panel}/results.csv')
        ref=frame[(frame.score=='cqr')&(frame['mode']=='shared')&(frame.selection=='original')][keys+['interval_score']].rename(columns={'interval_score':'denom'})
        for mode in ['shared','issued']:
            d=frame[(frame.score=='cqr')&(frame['mode']==mode)&(frame.selection=='configuration')&frame.gamma.isin(common)]
            d=d.merge(ref,on=keys,validate='many_to_one');d['ratio']=d.interval_score/d.denom
            matrices.append(np.log2(d.groupby(['gamma','horizon']).ratio.mean().unstack().reindex(index=common,columns=[1,6,24]).values))
    b=max(abs(v).max() for v in matrices);norm=TwoSlopeNorm(vmin=-b,vcenter=0,vmax=b)
    fig,axes=plt.subplots(2,2,figsize=(6.8,5.6),layout='constrained')
    for ax,v,title in zip(axes.flat,matrices,['Development: shared','Development: issued','Traffic/FX: shared','Traffic/FX: issued']):
        im=ax.imshow(v,aspect='auto',cmap='RdBu_r',norm=norm);ax.set_yticks(range(10),[f'{x:g}' for x in common]);ax.set_xticks(range(3),['1','6','24']);ax.set(xlabel='Horizon / natural delay',ylabel='Learning rate',title=title)
    fig.colorbar(im,ax=axes,shrink=.85,label='log2(mean paired score ratio to original shared ACI)')
    fig.savefig(PAPER/'figures/rate_delay_surface.png',dpi=220);plt.close(fig)
    # A compact final-finding plot is intentionally descriptive, not a CI.
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8),layout='constrained')
    for panel,label,color in [('development','Development','#305f91'),('revisit','Traffic / FX','#c57530')]:
        d=pd.read_csv(ROOT/f'results/review_v8/state_grid/{panel}/summary.csv')
        d=d[(d.horizon==24)&(d.score=='cqr')].set_index('selection')
        vals=d.loc[['original','scaled','wide'],'gain'].to_numpy()
        axes[0].plot(range(3),vals,'o-',label=label,color=color)
    axes[0].set(xticks=range(3),xticklabels=['Original','Add /h','Wide'],ylabel='Issued/shared score gain (%)',title='CQR state gain depends on tuning');axes[0].axhline(0,color='gray',lw=.7);axes[0].legend(frameon=False,fontsize=8)
    for method,color in [('Rolling CQR','#444444'),('Shared ACI','#305f91'),('Issued ACI','#b25f91'),('COP','#c57530')]:
        r=summary.loc[(24,method)];axes[1].scatter(100*r.coverage,r.gain,color=color,s=38)
        offsets={'Rolling CQR':(-48,-15),'Shared ACI':(6,-13),'Issued ACI':(5,5),'COP':(-26,8)}
        axes[1].annotate(method,(100*r.coverage,r.gain),xytext=offsets[method],textcoords='offset points',fontsize=8)
    axes[1].axvline(90,color='gray',ls='--',lw=.8);axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set(xlabel='Mean coverage (%)',ylabel='Paired score gain vs Rolling (%)',title='Traffic / FX: score--coverage tradeoff',xlim=(88.1,90.4))
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(PAPER/'figures/core_findings.png',dpi=220);plt.close(fig)
    paths=[ROOT/'results/confirmation_v7/calibration/selected.csv',ROOT/'results/review_v8/state_grid/revisit/selected.csv',ROOT/'results/review_v8/solar_components/summary.csv']
    (OUT/'provenance.json').write_text(json.dumps({'status':'descriptive_synthesis_of_saved_results_no_retuning',
        'denominator':'each matched pair CQR rolling interval score','pairs_per_horizon':16,
        'rounding':'percentages formatted directly from unrounded pair-level aggregate; no intermediate rounding',
        'sources':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
    print(summary.loc[24].round(6).to_string());print(json.dumps(values,indent=2))

if __name__=='__main__':main()
