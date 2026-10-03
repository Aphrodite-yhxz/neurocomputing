"""Tables and scientific figures for the post-review sensitivity revision."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/review_v8'
PAPER=ROOT/'paper/neurocomputing_review_v8'

def table(name,header,rows,columns):
    (PAPER/name).write_text('\n'.join([r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']+
        [r+r' \\' for r in rows]+[r'\bottomrule',r'\end{tabular}'])+'\n',encoding='utf-8')

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows=[];clips=[];native=[];raw=[];selected=[]
    for panel,label in [('development','Development'),('revisit','Revisited V7')]:
        s=pd.read_csv(BASE/f'state_grid/{panel}/summary.csv')
        for grid in ['original','scaled','wide']:
            r=s[(s.horizon==24)&(s.score=='cqr')&(s.selection==grid)].iloc[0]
            rows.append(f'{label} & {grid} & {r.gain:.2f} & {int(r.wins)}/16 & {100*r.coverage_shared:.2f} & {100*r.coverage_issued:.2f} & {100*r.local_shared:.2f} & {100*r.local_issued:.2f}')
            for mode in ['shared','issued']:
                clips.append(f'{label} & {grid} & {mode} & {100*r["clip_"+mode]:.3f} & {100*r["endpoint_"+mode]:.3f} & {100*(r["clip_"+mode]+r["endpoint_"+mode]):.3f}')
        sel=pd.read_csv(BASE/f'state_grid/{panel}/selected.csv');selected.append(sel)
        keys=['backbone','dataset','horizon','mode','selection']
        m=sel[sel.score=='cqr'].merge(sel[sel.score=='residual'],on=keys,suffixes=('_cqr','_residual'),validate='one_to_one')
        m['gain']=100*(1-m.interval_score_cqr/m.interval_score_residual);m['panel']=panel;native.append(m)
        raw.append(pd.read_csv(BASE/f'state_grid/{panel}/results.csv'))
    table('table_rate_sensitivity.tex','Panel & Grid & Gain & Wins & Cov. S & Cov. I & Local S & Local I',rows,'llrrrrrr')
    table('table_clip_accounting.tex','Panel & Grid & State & Clip & Endpoint & Excess',clips,'lllrrr')
    n=pd.concat(native,ignore_index=True);n.to_csv(BASE/'native_state_contrasts.csv',index=False)
    n.groupby(['panel','horizon','mode','selection']).agg(gain=('gain','mean'),coverage_cqr=('coverage_cqr','mean'),coverage_residual=('coverage_residual','mean')).to_csv(BASE/'native_state_summary.csv')
    allsel=pd.concat(selected,ignore_index=True)
    allsel.groupby(['panel','horizon','score','mode','selection','gamma']).size().to_csv(BASE/'selected_rate_counts.csv',header=['count'])

    common=[.0001,.0002,.0005,.001,.002,.005,.01,.02,.05,.1]
    matrices=[]
    for panel in ['development','revisit']:
        frame=next(d for d in raw if d.panel.iloc[0]==panel)
        ref=frame[(frame.score=='cqr')&(frame['mode']=='shared')&(frame.selection=='original')][['backbone','dataset','horizon','interval_score']].rename(columns={'interval_score':'denom'})
        for mode in ['shared','issued']:
            f=frame[(frame.score=='cqr')&(frame['mode']==mode)&(frame.selection=='configuration')&frame.gamma.isin(common)]
            f=f.merge(ref,on=['backbone','dataset','horizon'],validate='many_to_one');f['ratio']=f.interval_score/f.denom
            matrix=f.groupby(['gamma','horizon']).ratio.mean().unstack().reindex(index=common,columns=[1,6,24])
            assert matrix.notna().all().all();matrices.append(np.log2(matrix.values))
    bound=max(abs(x).max() for x in matrices);norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    fig,axes=plt.subplots(2,2,figsize=(6.8,5.6),layout='constrained')
    for ax,values,title in zip(axes.flat,matrices,['Development: shared','Development: issued','Revisited V7: shared','Revisited V7: issued']):
        im=ax.imshow(values,aspect='auto',cmap='RdBu_r',norm=norm)
        ax.set_yticks(range(len(common)),[f'{x:g}' for x in common]);ax.set_xticks(range(3),['1','6','24'])
        ax.set(xlabel='Horizon / natural delay',ylabel='Learning rate',title=title)
    fig.colorbar(im,ax=axes,shrink=.85,label='log2(mean paired score ratio to original shared ACI)')
    fig.savefig(PAPER/'figures/rate_delay_surface.png',dpi=220);plt.close(fig)
    shutil.copyfile(BASE/'dynamics/controlled_dynamics.png',PAPER/'figures/controlled_dynamics.png')

    solar=pd.read_csv(BASE/'solar_components/summary.csv');rows=[]
    for _,r in solar.iterrows():
        rows.append(f'{r.settings} & {r.regime} & {100*r.fraction_cqr:.2f} & {r.conditional_gain:.2f} & {r.overall_gain_contribution:.2f} & {100*r.coverage_residual:.2f} & {100*r.coverage_cqr:.2f}')
    table('table_solar_decomposition.tex','Settings & Group & Share & Gain & Contribution & Cov. R & Cov. CQR',rows,'llrrrrr')
    comps=pd.read_csv(BASE/'solar_components/components.csv');rows=[]
    for _,r in comps.iterrows():
        rows.append(f'{r.settings} & {r.score} & {r.regime} & {r.interval_score:.3f} & {r.width:.3f} & {r.lower_penalty:.3f} & {r.upper_penalty:.3f}')
    table('table_solar_components.tex','Settings & Score & Group & IS & Width & Lower & Upper',rows,'lllrrrr')

    old=pd.read_csv(ROOT/'results/confirmation_v7/calibration/selected.csv')
    ref=old[(old.score=='residual')&(old['mode']=='rolling')][['backbone','dataset','horizon','interval_score']].rename(columns={'interval_score':'denom'})
    top=allsel[(allsel.panel=='revisit')&(allsel.selection=='wide')].copy();top['mode']=top['mode']+'_wide'
    pid=pd.read_csv(BASE/'multistep_pid/revisit/selected.csv');pid['mode']=pid.selection.map({'selected':'Multi-step PID','no_scorecast':'Multi-step PI'})
    full=pd.concat([old,top,pid],ignore_index=True);full=full[full.horizon==24].copy()
    full=full.merge(ref,on=['backbone','dataset','horizon'],validate='many_to_one');full['ratio']=full.interval_score/full.denom
    full['family']=full.dataset.str.rsplit('_c',n=1).str[0]
    full.to_csv(BASE/'unified_controller_rows.csv',index=False)
    names={'rolling':'Rolling','shared':'Shared, original','issued':'Issued, original','shared_wide':'Shared, wide','issued_wide':'Issued, wide',
           'pi_log':'PI-log','sfogd':'SF-OGD','cop':'COP','grouped':'Grouped ACI','ogd':'Fixed-scale OGD','implicit_delay':'Implicit-delay'}
    rows=[]
    for score in ['residual','cqr','signed_error']:
        for mode in (list(names) if score!='signed_error' else ['Multi-step PI','Multi-step PID']):
            f=full[(full.score==score)&(full['mode']==mode)];assert len(f)==16,(score,mode,len(f))
            fam=f.groupby('family').mean(numeric_only=True)
            rows.append(f'{dict(residual="Residual",cqr="CQR",signed_error="Signed error")[score]} & {names.get(mode,mode)} & {fam.ratio.mean():.3f} & {100*fam.loc["exchange_rate","coverage"]:.2f} & {100*fam.loc["traffic","coverage"]:.2f} & {100*fam.local_error.mean():.2f} & {100*fam.singleton_rate.mean():.3f}')
    table('table_confirmation_controls.tex','Score & Controller & IS ratio & FX cov. & Traffic cov. & Local & Singleton',rows,'llrrrrr')
    sources=list((BASE/'state_grid').glob('*/*.csv'))+list((BASE/'solar_components').glob('*.csv'))+list((BASE/'multistep_pid').glob('*/*.csv'))
    (BASE/'paper_sources.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
    print(n[(n.horizon==24)&(n.selection=='wide')].groupby(['panel','mode']).gain.mean().to_string())
    print('All V8 tables and figures generated.')

if __name__=='__main__':main()
