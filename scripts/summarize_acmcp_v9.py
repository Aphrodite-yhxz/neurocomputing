"""Audit full R trajectories, then tune only its lr using cached fitted scorecasts."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.acmcp_cached import replay_h
from daci.strengthening import observed_metrics
OUT=ROOT/'results/strengthening_v9/acmcp'
def main():
    rows=[];checks=[];sources=[]
    for job in sorted(p for p in OUT.iterdir() if p.is_dir()):
        assert (job/'complete.json').exists(),f'Incomplete: {job.name}'
        def read(name):return pd.read_csv(job/name,header=None,na_values=['NA']).to_numpy()
        pf=read('mean.csv');errors=read('errors.csv');sc=read('scorecaster.csv');rl=read('lower.csv');ru=read('upper.csv')
        cfg=pd.read_csv(job/'config.csv').iloc[0]
        p=ROOT/'results/strengthening_v9/forecasts'/f'{job.name}.npz'
        with np.load(p) as z:ys=z['y'];backbone=str(z['backbone']);dataset=str(z['dataset'])
        n=len(ys)
        for h in range(1,25):
            lo,hi=replay_h(pf[:,h-1],errors[:,h-1],sc[:,h-1],h,.01,float(cfg.KI))
            np.testing.assert_allclose(lo,rl[:,h-1],rtol=2e-12,atol=2e-9,equal_nan=True)
            np.testing.assert_allclose(hi,ru[:,h-1],rtol=2e-12,atol=2e-9,equal_nan=True)
            diff=max(float(np.nanmax(abs(lo-rl[:,h-1]))),float(np.nanmax(abs(hi-ru[:,h-1]))))
            checks.append(dict(archive=job.name,horizon=h,max_endpoint_difference=diff))
            if h not in [1,6,24]:continue
            rates=sorted(set([.001,.005,.01,.05,.1]+[v/h for v in [.001,.005,.01,.05,.1]]))
            for variant in ['AcMCP-CSS','PID-no-scorecast']:
                scores=sc[:,h-1] if variant=='AcMCP-CSS' else np.zeros(len(sc))
                for lr in rates:
                    l,u=replay_h(pf[:,h-1],errors[:,h-1],scores,h,lr,float(cfg.KI))
                    l=l[h-1:h-1+n].copy();u=u[h-1:h-1+n].copy()
                    cross=l>u;l[cross]=u[cross]=(l[cross]+u[cross])/2
                    r=dict(backbone=backbone,dataset=dataset,horizon=h,variant=variant,lr=lr,crossed=int(cross[1200:].sum()),scorecast_successes=int(np.isfinite(sc[:,h-1]).sum()),validation=observed_metrics(ys[:,h-1],l,u,start=600+h-1,stop=1200-h+1)['interval_score'],**observed_metrics(ys[:,h-1],l,u))
                    rows.append(r)
        sources.append(dict(job=job.name,scorecasts_sha256=hashlib.sha256((job/'scorecaster.csv').read_bytes()).hexdigest(),author_manifest=json.loads((job/'complete.json').read_text())))
    assert len(sources)==4
    df=pd.DataFrame(rows);df.to_csv(OUT/'grid.csv',index=False)
    sel=df.sort_values(['validation','lr']).groupby(['backbone','dataset','horizon','variant'],as_index=False).first();sel.to_csv(OUT/'selected.csv',index=False)
    (OUT/'source_checks.json').write_text(json.dumps(dict(all_checks_passed=True,cases=checks,sources=sources),indent=2)+'\n')
    print(sel.groupby(['horizon','variant'])[['coverage','interval_score','local_error','lr']].mean().to_string())
if __name__=='__main__':main()
