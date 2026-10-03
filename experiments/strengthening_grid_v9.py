"""Locked nested tuning, score/state interaction and validation-period controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.full_horizon import load_full
from daci.strengthening import calibrate_level,observed_metrics
OUT=ROOT/'results/strengthening_v9/grid'
def rates(h):
    base=[.005,.02,.05]
    wide=sorted(set([.0001,.0002,.0005,.001,.002,.005,.01,.02,.05,.1]+[g/h for g in base]))
    return base,wide,sorted(set(wide+[0,.000001,.000005,.00001,.00005]))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fingerprint():return hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),ROOT/'src/daci/strengthening.py',ROOT/'docs/STRENGTHENING_PROTOCOL_V9.md'])).hexdigest()
def job(path,h):
    p=Path(path);dest=OUT/'pairs'/f'{p.stem}__h{h}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        old=json.loads(dest.read_text());assert old['fingerprint']==fingerprint() and old['archive_sha256']==sha(p)
        return dest.name,'cached'
    a=load_full(p,h);base,wide,small=rates(h);rows=[]
    for score in ['residual','cqr']:
        for mode in ['shared','issued']:
            for window in [200,450,900]:
                for gamma in small:
                    lo,hi,_=calibrate_level(a,score,mode,gamma,window)
                    val=observed_metrics(a.y,lo,hi,start=600+h-1,stop=1200-h+1)
                    early=observed_metrics(a.y,lo,hi,start=600+h-1,stop=900-h+1)
                    late=observed_metrics(a.y,lo,hi,start=900,stop=1200-h+1)
                    rows.append(dict(backbone=a.backbone,dataset=a.dataset,family=a.dataset.split('_')[0],horizon=h,score=score,mode=mode,gamma=gamma,window=window,validation=val['interval_score'],early=early['interval_score'],late=late['interval_score'],**observed_metrics(a.y,lo,hi)))
    dest.write_text(json.dumps(dict(fingerprint=fingerprint(),archive_sha256=sha(p),rows=rows),indent=2)+'\n')
    return dest.name,len(rows)
def summarize():
    ps=sorted((OUT/'pairs').glob('*.json'));assert len(ps)==24
    records=[]
    for p in ps:
        r=json.loads(p.read_text());assert r['fingerprint']==fingerprint();records+=r['rows']
    df=pd.DataFrame(records);df.to_csv(OUT/'results.csv',index=False);selections=[]
    for keys,sub in df.groupby(['backbone','dataset','family','horizon','score','mode']):
        base,wide,small=rates(keys[3])
        pools={'original':sub[(sub.window==450)&sub.gamma.isin(base)],'wide':sub[(sub.window==450)&sub.gamma.isin(wide)],'zero_small':sub[sub.window==450],'window':sub,'rolling450':sub[(sub.window==450)&(sub.gamma==0)],'rolling_tuned':sub[sub.gamma==0]}
        for grid,candidates in pools.items():
            for period in ['validation','early','late']:
                best=candidates.sort_values([period,'gamma','window']).iloc[0].to_dict()
                best.update(grid=grid,period=period,zero_selected=best['gamma']==0,smallest_positive=best['gamma']==min(g for g in candidates.gamma.unique() if g>0) if (candidates.gamma>0).any() else False)
                selections.append(best)
    sel=pd.DataFrame(selections);sel.to_csv(OUT/'selected.csv',index=False)
    keys=['backbone','dataset','family','horizon','score','grid','period']
    c=sel[sel['mode']=='issued'].merge(sel[sel['mode']=='shared'],on=keys,suffixes=('_issued','_shared'),validate='one_to_one')
    c['state_gain']=100*(1-c.interval_score_issued/c.interval_score_shared)
    c['state_log_ratio']=np.log(c.interval_score_shared/c.interval_score_issued)
    c.to_csv(OUT/'state_contrasts.csv',index=False)
    c.groupby(['family','horizon','score','grid','period']).agg(gain=('state_gain','mean'),median=('state_gain','median'),pairs=('state_gain','size'),coverage_shared=('coverage_shared','mean'),coverage_issued=('coverage_issued','mean')).to_csv(OUT/'state_summary.csv')
    ik=['backbone','dataset','family','horizon','grid','period']
    inter=c[c.score=='cqr'].merge(c[c.score=='residual'],on=ik,suffixes=('_cqr','_residual'),validate='one_to_one')
    inter['interaction_log_ratio']=inter.state_log_ratio_residual-inter.state_log_ratio_cqr
    inter.to_csv(OUT/'score_state_interaction.csv',index=False)
    primary=sel[sel.period=='validation']
    primary.groupby(['family','horizon','score','mode','grid'])[['gamma','window','zero_selected','smallest_positive','coverage','local_error']].mean().to_csv(OUT/'selection_summary.csv')
    print(c[(c.horizon==24)&(c.period=='validation')].groupby(['family','score','grid']).state_gain.mean().round(4).to_string())
if __name__=='__main__':
    paths=sorted((ROOT/'results/strengthening_v9/forecasts').glob('*.npz'));assert len(paths)==8
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(job,str(p),h) for p in paths for h in [1,6,24]]
        for f in as_completed(fs):print(f.result(),flush=True)
    summarize()
