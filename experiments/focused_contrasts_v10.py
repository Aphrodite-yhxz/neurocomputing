"""Post-review positive-rate contrasts and prespecified consecutive test blocks."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.full_horizon import load_full
from daci.strengthening import calibrate_level,observed_metrics
OUT=ROOT/'results/focused_v10';OLD=ROOT/'results/strengthening_v9'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    grid=pd.read_csv(OLD/'grid/results.csv');rows=[]
    keys=['backbone','dataset','family','horizon','score','mode']
    for _,sub in grid.groupby(keys):
        positive=sub[sub.gamma>0]
        pools={f'positive{w}':positive[positive.window==w] for w in [200,450,900]}
        pools.update(positive_window=positive,including_zero=sub)
        for name,pool in pools.items():
            best=pool.sort_values(['validation','gamma','window']).iloc[0].to_dict()
            best['search']=name;rows.append(best)
    selected=pd.DataFrame(rows);selected.to_csv(OUT/'positive_selected.csv',index=False)
    keys=['backbone','dataset','family','horizon','score','search']
    c=selected[selected['mode']=='issued'].merge(selected[selected['mode']=='shared'],on=keys,suffixes=('_i','_s'),validate='one_to_one')
    c['gain']=100*(1-c.interval_score_i/c.interval_score_s)
    c['same_rolling']=(c.gamma_i==0)&(c.gamma_s==0)&(c.window_i==c.window_s)
    c['same_positive_setting']=(c.gamma_i>0)&(c.gamma_i==c.gamma_s)&(c.window_i==c.window_s)
    c.to_csv(OUT/'positive_pairs.csv',index=False)
    c.groupby(['family','horizon','score','search']).agg(gain=('gain','mean'),median=('gain','median'),same_rolling=('same_rolling','sum'),pairs=('gain','size')).to_csv(OUT/'positive_summary.csv')
    final=c[(c.horizon==24)&(c.score=='cqr')&(c.search=='including_zero')].copy()
    final.to_csv(OUT/'final_pair_choices.csv',index=False)
    old=pd.read_csv(OLD/'grid/selected.csv');old=old[(old.period=='validation')&(old.score=='cqr')]
    trajectory_rows=[]
    blocks={'whole':(1200,2976),'early':(1200,1792),'middle':(1792,2384),'late':(2384,2976)}
    for p in sorted((OLD/'forecasts').glob('*.npz')):
        for h in [1,6,24]:
            a=load_full(p,h)
            sub=old[(old.backbone==a.backbone)&(old.dataset==a.dataset)&(old.horizon==h)]
            for method,mode,name in [('Rolling','shared','rolling_tuned'),('Shared','shared','window'),('Issued','issued','window')]:
                row=sub[(sub['mode']==mode)&(sub.grid==name)].iloc[0]
                lo,hi,_=calibrate_level(a,'cqr',mode,float(row.gamma),int(row.window))
                for block,(start,stop) in blocks.items():
                    metric=observed_metrics(a.y,lo,hi,start,stop)
                    if block=='whole':
                        for k in ['interval_score','coverage','local_error']:np.testing.assert_allclose(metric[k],row[k],rtol=1e-11,atol=1e-11)
                    trajectory_rows.append(dict(backbone=a.backbone,dataset=a.dataset,family=a.dataset.split('_')[0],horizon=h,method=method,block=block,start=start,stop=stop,gamma=float(row.gamma),window=int(row.window),**metric))
    df=pd.DataFrame(trajectory_rows);df.to_csv(OUT/'temporal_results.csv',index=False)
    keys=['backbone','dataset','family','horizon','block','start','stop']
    rolling=df[df.method=='Rolling'];c=df[df.method!='Rolling'].merge(rolling,on=keys,suffixes=('','_rolling'),validate='many_to_one')
    c['gain']=100*(1-c.interval_score/c.interval_score_rolling)
    c['local_improvement_pp']=100*(c.local_error_rolling-c.local_error)
    c['coverage_change_pp']=100*(c.coverage-c.coverage_rolling)
    c.to_csv(OUT/'temporal_pairs.csv',index=False)
    c.groupby(['family','horizon','method','block']).agg(gain=('gain','mean'),local_gain_pp=('local_improvement_pp','mean'),coverage=('coverage','mean'),coverage_rolling=('coverage_rolling','mean'),local=('local_error','mean'),local_rolling=('local_error_rolling','mean'),positive_local_pairs=('local_improvement_pp',lambda x:int((x>1e-12).sum()))).to_csv(OUT/'temporal_summary.csv')
    print(pd.read_csv(OUT/'positive_summary.csv').query('horizon==24').to_string(index=False))
    print(c[(c.family=='weather')&(c.method=='Issued')&(c.horizon==1)][['backbone','dataset','block','gain','local_improvement_pp']].to_string(index=False))
    (OUT/'contrast_provenance.json').write_text(json.dumps(dict(stage='Exploratory post-review analysis on inspected data',protocol_sha256=hashlib.sha256((ROOT/'docs/FOCUSED_FINISH_PROTOCOL_V10.md').read_bytes()).hexdigest(),selected_rows=len(selected),temporal_rows=len(df)),indent=2)+'\n')
if __name__=='__main__':main()
