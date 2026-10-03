"""Exploratory support and signed-residual attribution on all original PV pairs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.revision import load_archive,interval_scores
from daci.strengthening import calibrate_level,nonnegative,observed_metrics
OUT=ROOT/'results/strengthening_v9/pv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp():return hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'src/daci/strengthening.py').read_bytes()).hexdigest()
def job(path):
    p=Path(path);dest=OUT/'pairs'/f'{p.stem}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        old=json.loads(dest.read_text());assert old['fingerprint']==fp() and old['archive_sha256']==sha(p)
        return p.name,'cached'
    a=load_archive(p);assert np.all(a.y>=0);rows=[];grid=[]
    threshold=float(np.median(a.mu[:600]));idx=np.arange(len(a.y))
    groups={'all':idx>=1200,'zero':(idx>=1200)&(a.y==0),'positive':(idx>=1200)&(a.y>0),'forecast_low':(idx>=1200)&(a.mu<=threshold),'forecast_high':(idx>=1200)&(a.mu>threshold)}
    for score in ['residual','signed','cqr']:
        trajectories={}
        for window in [200,450,900]:
            for gamma in [0,.005,.02,.05]:
                lo,hi,_=calibrate_level(a,score,'shared',gamma,window)
                trajectories[(gamma,window)]=(lo,hi)
                for projection in ['raw','nonnegative']:
                    l,u=(lo,hi) if projection=='raw' else nonnegative(lo,hi)
                    loss=interval_scores(a.y[600:1200],l[600:1200],u[600:1200])
                    grid.append(dict(backbone=a.backbone,dataset=a.dataset,score=score,projection=projection,gamma=gamma,window=window,validation=float(loss.mean()),**observed_metrics(a.y,l,u)))
        for projection in ['raw','nonnegative']:
            candidates=[r for r in grid if r['score']==score and r['projection']==projection]
            for settings in ['default','tuned','rolling']:
                eligible=[r for r in candidates if (r['gamma']>0 if settings=='tuned' else r['gamma']==0)]
                best=min(eligible,key=lambda r:(r['validation'],r['gamma'],r['window'])) if settings!='default' else next(r for r in candidates if r['gamma']==.02 and r['window']==450)
                lo,hi=trajectories[(best['gamma'],best['window'])]
                l,u=(lo,hi) if projection=='raw' else nonnegative(lo,hi)
                if projection=='nonnegative':
                    assert np.all(interval_scores(a.y[1200:],l[1200:],u[1200:])<=interval_scores(a.y[1200:],lo[1200:],hi[1200:])+1e-9)
                for group,mask in groups.items():
                    y=a.y[mask];ll=l[mask];uu=u[mask]
                    rows.append(dict(backbone=a.backbone,dataset=a.dataset,score=score,projection=projection,settings=settings,group=group,gamma=best['gamma'],window=best['window'],forecast_threshold=threshold,n=len(y),fraction=float(mask.sum()/(len(a.y)-1200)),coverage=float(np.mean((ll<=y)&(y<=uu))),interval_score=float(interval_scores(y,ll,uu).mean()),width=float(np.mean(uu-ll)),lower_penalty=float(20*np.maximum(ll-y,0).mean()),upper_penalty=float(20*np.maximum(y-uu,0).mean()),negative_upper_frequency=float(np.mean(hi[mask]<0))))
    dest.write_text(json.dumps(dict(fingerprint=fp(),archive_sha256=sha(p),rows=rows,grid=grid),indent=2)+'\n')
    return p.name,len(rows)
def summarize():
    ps=sorted((OUT/'pairs').glob('*.json'));assert len(ps)==32
    rows=[];grid=[]
    for p in ps:
        r=json.loads(p.read_text());assert r['fingerprint']==fp();rows+=r['rows'];grid+=r['grid']
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False);pd.DataFrame(grid).to_csv(OUT/'grid.csv',index=False)
    df.groupby(['settings','projection','score','group'])[['coverage','interval_score','width','lower_penalty','upper_penalty','fraction','negative_upper_frequency']].mean().to_csv(OUT/'components.csv')
    keys=['backbone','dataset','settings','projection','group'];contrasts=[]
    for baseline in ['residual','signed']:
        c=df[df.score=='cqr'].merge(df[df.score==baseline],on=keys,suffixes=('_cqr','_baseline'),validate='one_to_one')
        c['baseline']=baseline;c['gain']=np.where(c.interval_score_baseline>0,100*(1-c.interval_score_cqr/c.interval_score_baseline),np.nan)
        contrasts.append(c)
    c=pd.concat(contrasts,ignore_index=True);c.to_csv(OUT/'contrasts.csv',index=False)
    summary=c.groupby(['settings','projection','baseline','group']).agg(gain=('gain','mean'),pairs=('gain','size'),defined_gains=('gain','count'),coverage_cqr=('coverage_cqr','mean'),coverage_baseline=('coverage_baseline','mean'),fraction=('fraction_cqr','mean'))
    summary.to_csv(OUT/'summary.csv')
    np.testing.assert_allclose(summary.loc[('tuned','raw','residual','positive'),'gain'],12.769942460492455,rtol=1e-10)
    print(summary.loc['tuned'].round(5).to_string())
if __name__=='__main__':
    paths=sorted((ROOT/'results/real/traces').glob('*__solar_*.npz'));assert len(paths)==32
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(job,str(p)) for p in paths]
        for f in as_completed(fs):print(f.result(),flush=True)
    summarize()
