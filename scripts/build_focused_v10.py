"""Generate concise scientific displays from the post-review paired analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/focused_v10';PAPER=ROOT/'paper/neurocomputing_focused'
def table(name,header,rows,spec):
    (PAPER/name).write_text('\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+[r+r' \\' for r in rows]+[r'\bottomrule',r'\end{tabular}'])+'\n')
def rate(x):
    if x==0:return '0'
    return f'{x:.3g}'
def main():
    (PAPER/'figures').mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(OUT/'positive_summary.csv');s=s[s.horizon==24];rows=[]
    for family in ['air','weather']:
        for score in ['cqr','residual']:
            a=s[(s.family==family)&(s.score==score)].set_index('search')
            vals=[a.loc[g,'gain'] for g in ['positive450','positive900','positive_window','including_zero']]
            rows.append(f'{family.title()} & {score.upper() if score=="cqr" else "Residual"} & '+' & '.join(f'{v:.3f}' for v in vals))
    table('table_positive_rates.tex',r'Source & Score & $W=450$ & $W=900$ & Select $W$ & Add zero',rows,'llrrrr')
    pairs=pd.read_csv(OUT/'final_pair_choices.csv');rows=[]
    names={'air_aotizhongxin':'Air A','air_dingling':'Air D','weather_pressure':'Pressure','weather_temperature':'Temperature'}
    for r in pairs.sort_values(['dataset','backbone']).itertuples():
        model='C2' if r.backbone=='Chronos-2enc-only' else 'Bolt'
        rows.append(f'{names[r.dataset]} / {model} & {rate(r.gamma_s)} & {r.window_s} & {r.interval_score_s:.3f} & {rate(r.gamma_i)} & {r.window_i} & {r.interval_score_i:.3f} & '+('Same rolling' if r.same_rolling else 'Distinct'))
    table('table_final_choices.tex',r'Pair & $\gamma_S$ & $W_S$ & IS$_S$ & $\gamma_I$ & $W_I$ & IS$_I$ & Selection',rows,'lrrrrrrl')
    t=pd.read_csv(OUT/'temporal_pairs.csv');a=t[(t.family=='weather')&(t.horizon==1)&(t.method=='Issued')]
    rows=[]
    for (dataset,backbone),sub in a.groupby(['dataset','backbone']):
        r=sub[sub.block=='whole'].iloc[0];bs=sub.set_index('block')
        rows.append(f'{names[dataset]} / '+('C2' if backbone=='Chronos-2enc-only' else 'Bolt')+f' & {r.gain:+.2f} & {100*r.local_error_rolling:.2f} & {100*r.local_error:.2f} & '+' & '.join(f'{bs.loc[b,"local_improvement_pp"]:+.2f}' for b in ['early','middle','late']))
    table('table_local_pairs.tex','Pair & Score gain & Local roll & Local ACI & Early & Middle & Late',rows,'lrrrrrr')
    summary=pd.read_csv(OUT/'temporal_summary.csv');rows=[]
    for r in summary[(summary.method=='Issued')&(summary.block=='whole')].itertuples():
        rows.append(f'{r.family.title()} & {r.horizon} & {r.gain:+.2f} & {100*r.coverage_rolling:.2f} & {100*r.coverage:.2f} & {100*r.local_rolling:.2f} & {100*r.local:.2f}')
    table('table_horizon_tradeoff.tex','Source & $h$ & Gain & Cov. roll & Cov. ACI & Local roll & Local ACI',rows,'lrrrrrr')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),layout='constrained')
    for ax,family in zip(axes,['air','weather']):
        for score,col in [('cqr','#bb7132'),('residual','#35678f')]:
            sub=s[(s.family==family)&(s.score==score)].set_index('search')
            ax.plot(range(3),[sub.loc[g,'gain'] for g in ['positive450','positive_window','including_zero']],'o-',color=col,label=score.upper() if score=='cqr' else 'Residual')
        ax.set(title=family.title(),xticks=range(3),xticklabels=['Positive / 450','Positive / select W','Include zero'],ylabel='Issued/shared score gain (%)');ax.tick_params(axis='x',labelsize=7);ax.axhline(0,color='gray',lw=.6);ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.savefig(PAPER/'figures/positive_rates.png',dpi=220);plt.close(fig)
    # All pair/block results, without a test-selected highlight or drift attribution.
    fig,axes=plt.subplots(2,3,figsize=(8,5.3),layout='constrained')
    for col,h in enumerate([1,6,24]):
        sub=t[(t.horizon==h)&(t.method=='Issued')&t.block.isin(['early','middle','late'])].copy()
        sub['pair']=sub.dataset.map(names)+' / '+sub.backbone.map({'Chronos-2enc-only':'C2','Chronos-Boltenc-dec':'Bolt'})
        for row,(metric,title) in enumerate([('gain','Score gain (%)'),('local_improvement_pp','Local-error improvement (pp)')]):
            values=sub.pivot(index='pair',columns='block',values=metric).reindex(columns=['early','middle','late'])
            ax=axes[row,col];limit=max(float(np.abs(values.to_numpy()).max()),.01)
            im=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto');ax.set(xticks=range(3),xticklabels=['Early','Middle','Late'],yticks=range(len(values)),yticklabels=values.index if col==0 else [],title=f'h={h}\n{title}');ax.tick_params(labelsize=7)
            for i in range(len(values)):
                for j in range(3):ax.text(j,i,f'{values.iloc[i,j]:.1f}',ha='center',va='center',fontsize=6,color='white' if abs(values.iloc[i,j])>.65*limit else 'black')
            fig.colorbar(im,ax=ax,shrink=.7,pad=.02)
    fig.savefig(PAPER/'figures/temporal_pairs.png',dpi=230);plt.close(fig)
    if (OUT/'history_summary.csv').exists():
        hs=pd.read_csv(OUT/'history_summary.csv');rows=[]
        for h in [1,6,24]:
            for search in ['fixed450','fixed600','selected']:
                for method in ['Rolling-CQR','CQR-shared','CQR-issued','AcMCP-CSS','PID-no-scorecast']:
                    r=hs[(hs.horizon==h)&(hs.search==search)&(hs.method==method)].iloc[0]
                    rows.append(f'{h} & {search[-3:] if search.startswith("fixed") else "450/600"} & {method} & {r.ratio:.3f} & {100*r.coverage:.2f} & {100*r.local:.2f}')
        table('table_history_control.tex','Horizon & History & Method & IS ratio & Coverage & Local',rows,'rllrrr')
        table('table_history_main.tex','History & Method & IS ratio & Coverage & Local',[r.split(' & ',1)[1] for r in rows if r.startswith('24 & ')],'llrrr')
    values=dict(weather_positive450=float(s[(s.family=='weather')&(s.score=='cqr')&(s.search=='positive450')].gain.iloc[0]),weather_positive_window=float(s[(s.family=='weather')&(s.score=='cqr')&(s.search=='positive_window')].gain.iloc[0]),same_rolling_pairs=int(pairs.same_rolling.sum()))
    (OUT/'focused_numbers.json').write_text(json.dumps(values,indent=2)+'\n')
    print(json.dumps(values,indent=2))
if __name__=='__main__':main()
