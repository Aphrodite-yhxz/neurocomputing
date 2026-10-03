"""Post-review ACI rate sensitivity with exact test-feedback clipping accounting."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from daci.revision import metrics
from daci.full_horizon import load_full
from daci.delay_topology import calibrate_topology
OUT=ROOT/'results/review_v8/state_grid'
BASE=[.005,.02,.05]

def grids(h):
    return {'original':BASE,'scaled':sorted(set(BASE+[g/h for g in BASE])),
            'wide':sorted(set([.0001,.0002,.0005,.001,.002,.005,.01,.02,.05,.1]+[g/h for g in BASE]))}

def fingerprint():
    paths=[Path(__file__),ROOT/'docs/REVIEW_RESPONSE_PROTOCOL_V8.md']
    paths += [ROOT/'src/daci'/p for p in ['revision.py','full_horizon.py','delay_topology.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

def accounting(a,lo,hi,d,gamma,start=1200):
    h=a.delay;n=len(a.y);first=start+h;last=n-1;k=last-first+1
    levels=d['levels'];rho=d['rho']
    potential=lambda t:levels[t]+rho*levels[t-h+1:t].sum()
    errors=(a.y<lo)|(a.y>hi)
    clip=d['corrections'][first:last+1].sum()/(gamma*k)
    endpoint=-(potential(last)-potential(first-1))/(gamma*k)
    excess=float(errors[start:n-h].mean()-.1)
    discrepancy=clip+endpoint-excess
    assert abs(discrepancy)<1e-10,(a.dataset,h,gamma,discrepancy)
    issued=float(errors[start:].mean()-.1)
    assert abs(issued-excess)<=h/(n-start)+1e-12
    return dict(revealed_test_n=k,pending_test_n=h,clip_term=clip,endpoint_term=endpoint,
        revealed_excess=excess,issued_excess=issued,pending_difference=issued-excess,
        identity_error=discrepancy,test_clip_count=int(np.count_nonzero(d['corrections'][first:])))

def job(path,h,panel):
    p=Path(path);dest=OUT/panel/'pairs'/f'{p.stem}__h{h}.json'
    fp=fingerprint();ah=hashlib.sha256(p.read_bytes()).hexdigest()
    if dest.exists():
        old=json.loads(dest.read_text());assert old['fingerprint']==fp and old['archive_sha256']==ah
        return dest.name,'cached'
    a=load_full(p,h);rows=[];choices=grids(h)
    for score in ['residual','cqr']:
        for mode in ['shared','issued']:
            candidates=[]
            for gamma in choices['wide']:
                lo,hi,d=calibrate_topology(a,score,mode,gamma=gamma)
                val=metrics(a.y,lo,hi,start=600+h-1,stop=1200-h+1)
                r=dict(panel=panel,backbone=a.backbone,dataset=a.dataset,horizon=h,score=score,mode=mode,
                    gamma=gamma,method=f'{score}/{mode}/g{gamma:g}',selection='configuration',
                    validation_score=val['interval_score'],**metrics(a.y,lo,hi),**accounting(a,lo,hi,d,gamma))
                rows.append(r);candidates.append(r)
            for name,values in choices.items():
                eligible=[r for r in candidates if r['gamma'] in values]
                best=min(eligible,key=lambda r:(r['validation_score'],r['method']))
                rows.append(dict(best,selection=name,selected_method=best['method'],
                    at_lower_boundary=best['gamma']==min(values),at_upper_boundary=best['gamma']==max(values)))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(fingerprint=fp,archive_sha256=ah,rows=rows),indent=2))
    return dest.name,len(rows)

def summarize(panel):
    records=[]
    for p in sorted((OUT/panel/'pairs').glob('*.json')):
        x=json.loads(p.read_text());assert x['fingerprint']==fingerprint();records+=x['rows']
    df=pd.DataFrame(records);assert len(list((OUT/panel/'pairs').glob('*.json')))==48
    df.to_csv(OUT/panel/'results.csv',index=False)
    sel=df[df.selection!='configuration'].copy()
    keys=['panel','backbone','dataset','horizon','score','selection']
    a=sel[sel['mode']=='issued'];b=sel[sel['mode']=='shared']
    c=a.merge(b,on=keys,suffixes=('_issued','_shared'),validate='one_to_one')
    c['gain']=100*(1-c.interval_score_issued/c.interval_score_shared)
    c['win']=c.gain>1e-10
    assert np.max(np.abs(c[c.horizon==1].gain))<1e-10
    sel.to_csv(OUT/panel/'selected.csv',index=False);c.to_csv(OUT/panel/'contrasts.csv',index=False)
    ans=c.groupby(['horizon','score','selection']).agg(gain=('gain','mean'),median_gain=('gain','median'),
        wins=('win','sum'),pairs=('win','size'),coverage_shared=('coverage_shared','mean'),coverage_issued=('coverage_issued','mean'),
        local_shared=('local_error_shared','mean'),local_issued=('local_error_issued','mean'),
        clip_shared=('clip_term_shared','mean'),clip_issued=('clip_term_issued','mean'),
        endpoint_shared=('endpoint_term_shared','mean'),endpoint_issued=('endpoint_term_issued','mean'))
    ans.to_csv(OUT/panel/'summary.csv')
    print(ans.loc[24].round(6).to_string())
    if panel=='revisit':
        c['family']=c.dataset.str.rsplit('_c',n=1).str[0]
        c.groupby(['horizon','score','selection','family']).agg(gain=('gain','mean'),median=('gain','median'),wins=('win','sum'),
            coverage_shared=('coverage_shared','mean'),coverage_issued=('coverage_issued','mean')).to_csv(OUT/panel/'family_summary.csv')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--panel',choices=['development','revisit'],required=True)
    ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    folder=ROOT/('results/innovation_v4/full_horizon' if args.panel=='development' else 'results/confirmation_v7/forecasts')
    paths=sorted(folder.glob('*.npz'));assert len(paths)==16
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(job,str(p),h,args.panel) for p in paths for h in [1,6,24]]
        for f in as_completed(fs):print(f.result(),flush=True)
    summarize(args.panel)
