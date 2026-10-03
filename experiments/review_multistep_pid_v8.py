"""Full archived-panel comparison of the source-checked asymmetric multi-step PID."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from daci.revision import metrics
from daci.full_horizon import load_full,to_target_time
from daci.multistep_pid import calibrate_multistep_pid
OUT=ROOT/'results/review_v8/multistep_pid'

def fingerprint():
    paths=[Path(__file__),ROOT/'docs/V8_MULTISTEP_PID_ADDENDUM.md',ROOT/'docs/REVIEW_RESPONSE_PROTOCOL_V8.md',
           ROOT/'docs/references/v4/acmcp/R/pid.R']
    paths += [ROOT/'src/daci'/p for p in ['revision.py','full_horizon.py','multistep_pid.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

def measures(y,lo,hi,start,stop=None):
    yy=y[start:stop];l=lo[start:stop];u=hi[start:stop]
    if np.isfinite(l).all() and np.isfinite(u).all():return dict(metrics(y,lo,hi,start=start,stop=stop),invalid_rate=0.)
    # Infinite/invalid intervals are assigned infinite proper loss, never omitted.
    cov=(l<=yy)&(yy<=u);local=np.convolve(cov.astype(float),np.ones(min(150,len(yy)))/min(150,len(yy)),'valid')
    return dict(n=len(yy),coverage=float(cov.mean()),coverage_gap=abs(cov.mean()-.9),
        lower_miss=float(np.mean(yy<l)),upper_miss=float(np.mean(yy>u)),width=float('inf'),
        interval_score=float('inf'),local_error=float(np.mean(abs(local-.9))),worst_window=float(local.min()),
        singleton_rate=float(np.mean(l==u)),infinite_rate=float(np.mean(~np.isfinite(l)|~np.isfinite(u))),
        invalid_rate=float(np.mean(np.isnan(l)|np.isnan(u))))

def job(path,panel):
    p=Path(path);dest=OUT/panel/'pairs'/f'{p.stem}.json';fp=fingerprint();ah=hashlib.sha256(p.read_bytes()).hexdigest()
    if dest.exists():
        old=json.loads(dest.read_text());assert old['fingerprint']==fp and old['archive_sha256']==ah
        return p.name,'cached'
    with np.load(p) as z:_,errors=to_target_time(z['mu'],z['y'])
    ki=max(float(np.nanmax(abs(errors[:600]))),1e-12);rows=[]
    for h in [1,6,24]:
        a=load_full(p,h);rates=sorted(set([.005,.01,.05,.1]+[v/h for v in [.005,.01,.05,.1]]))
        candidates=[]
        for scorecaster in ['none','mean','naive','ar1']:
            for lr in rates:
                lo,hi,d=calibrate_multistep_pid(a,lr=lr,ki=ki,scorecaster=scorecaster)
                va=measures(a.y,lo,hi,600+h-1,1200-h+1);ev=measures(a.y,lo,hi,1200)
                r=dict(panel=panel,backbone=a.backbone,dataset=a.dataset,horizon=h,mode='multistep_pid',
                    score='signed_error',scorecaster=scorecaster,lr=lr,ki=ki,csat=.5,
                    method=f'pid/{scorecaster}/lr{lr:g}',selection='configuration',validation_score=va['interval_score'],
                    validation_infinite_rate=va['infinite_rate'],**ev,test_crossed_rate=float(d['crossed'][1200:].mean()))
                rows.append(r);candidates.append(r)
        for name,eligible in [('selected',candidates),('no_scorecast',[r for r in candidates if r['scorecaster']=='none'])]:
            best=min(eligible,key=lambda r:(r['validation_score'],r['method']))
            rows.append(dict(best,selection=name,selected_method=best['method']))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(fingerprint=fp,archive_sha256=ah,rows=rows),indent=2))
    return p.name,len(rows)

def summarize(panel):
    paths=sorted((OUT/panel/'pairs').glob('*.json'));assert len(paths)==16
    records=[]
    for p in paths:
        obj=json.loads(p.read_text());assert obj['fingerprint']==fingerprint();records+=obj['rows']
    df=pd.DataFrame(records);df.to_csv(OUT/panel/'results.csv',index=False)
    sel=df[df.selection!='configuration'].copy();sel.to_csv(OUT/panel/'selected.csv',index=False)
    top=pd.read_csv(ROOT/f'results/review_v8/state_grid/{panel}/selected.csv')
    top=top[top.selection=='wide'].copy();top['baseline']=top.score+'/'+top['mode']+'/wide'
    keys=['backbone','dataset','horizon']
    contrasts=[]
    for baseline,b in top.groupby('baseline'):
        c=sel.merge(b[keys+['interval_score','coverage']],on=keys,suffixes=('_pid','_base'),validate='many_to_one')
        c['baseline']=baseline;c['gain_pid_vs_base']=100*(1-c.interval_score_pid/c.interval_score_base);contrasts.append(c)
    c=pd.concat(contrasts,ignore_index=True);c.to_csv(OUT/panel/'contrasts.csv',index=False)
    ans=c.groupby(['horizon','selection','baseline']).agg(gain=('gain_pid_vs_base','mean'),
        coverage_pid=('coverage_pid','mean'),coverage_base=('coverage_base','mean'),local_pid=('local_error','mean'))
    ans.to_csv(OUT/panel/'summary.csv');print(panel,ans.loc[24].round(5).to_string())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    check=json.loads((ROOT/'results/review_v8/pid_source_checks/h24_n1000/checks.json').read_text())
    assert check['author_equivalence_passed'] and check['causality_passed']
    assert check['module_sha256']==hashlib.sha256((ROOT/'src/daci/multistep_pid.py').read_bytes()).hexdigest()
    for panel,folder in [('development','results/innovation_v4/full_horizon'),('revisit','results/confirmation_v7/forecasts')]:
        paths=sorted((ROOT/folder).glob('*.npz'));assert len(paths)==16
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fs=[pool.submit(job,str(p),panel) for p in paths]
            for f in as_completed(fs):print(panel,f.result(),flush=True)
        summarize(panel)
