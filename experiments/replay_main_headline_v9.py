"""Rebuild the 124-pair native-information headline from predictions and tuning."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.revision import load_archive,calibrate,metrics
OUT=ROOT/'results/strengthening_v9/main_replay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp():return hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'src/daci/revision.py').read_bytes()).hexdigest()
def job(path):
    p=Path(path);out=OUT/'pairs'/f'{p.stem}.json';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        r=json.loads(out.read_text());assert r['fingerprint']==fp() and r['archive_sha256']==sha(p)
        return p.name,'cached'
    a=load_archive(p);rows=[]
    for score in ['residual','cqr']:
        candidates=[]
        for gamma in [.005,.02,.05]:
            for window in [200,450,900]:
                lo,hi,_=calibrate(a,score=score,rule='aci',gamma=gamma,window=window)
                method=f'{score}/aci' if gamma==.02 and window==450 else f'{score}/aci/g{gamma:g}/w{window}'
                r=dict(backbone=a.backbone,dataset=a.dataset,score=score,method=method,gamma=gamma,window=window,validation_score=metrics(a.y,lo,hi,start=600,stop=1200)['interval_score'],**metrics(a.y,lo,hi))
                candidates.append(r);rows.append(r)
        best=min(candidates,key=lambda r:(r['validation_score'],r['method']))
        rows.append(dict(best,method=f'{score}/aci/tuned',selected_method=best['method']))
    out.write_text(json.dumps(dict(archive_sha256=sha(p),fingerprint=fp(),rows=rows),indent=2)+'\n');return p.name,'computed'
def summarize():
    rows=[];paths=sorted((OUT/'pairs').glob('*.json'));assert len(paths)==124
    for p in paths:
        r=json.loads(p.read_text());assert r['fingerprint']==fp();rows+=r['rows']
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False)
    old=pd.read_csv(ROOT/'results/neurocomputing/main_a0.1/results.csv')
    c=df.merge(old,on=['backbone','dataset','method'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(c)==len(df)
    for metric in ['interval_score','coverage','validation_score']:
        np.testing.assert_allclose(c[metric+'_new'],c[metric+'_old'],rtol=1e-10,atol=1e-9)
    sub=df[df.method.str.endswith('/tuned')]
    pair=sub[sub.score=='cqr'].merge(sub[sub.score=='residual'],on=['backbone','dataset'],suffixes=('_cqr','_residual'),validate='one_to_one')
    pair['gain']=100*(1-pair.interval_score_cqr/pair.interval_score_residual)
    pair.to_csv(OUT/'headline_pairs.csv',index=False)
    report=dict(all_checks_passed=True,pairs=124,computed_configurations=124*18,selected_references=248,mean_paired_gain=float(pair.gain.mean()),max_score_difference=float(abs(c.interval_score_new-c.interval_score_old).max()),scope='Prediction arrays and validation tuning recomputed; neural inference not regenerated')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    paths=sorted((ROOT/'results/real/traces').glob('*.npz'));assert len(paths)==124
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(job,str(p)) for p in paths]
        for i,f in enumerate(as_completed(fs),1):
            r=f.result()
            if i%16==0:print(i,r,flush=True)
    summarize()
