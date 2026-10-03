"""Common-window, common-post-startup-validation complete-configuration comparison."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from daci.full_horizon import load_full
from daci.strengthening import calibrate_level,observed_metrics
from daci.acmcp_cached import replay_h
from experiments.strengthening_grid_v9 import rates
OUT=ROOT/'results/focused_v10';OLD=ROOT/'results/strengthening_v9'
def main():
    rows=[];checks=[];inputs=[]
    for path in sorted((OLD/'forecasts').glob('*__weather_*.npz')):
        for window,base in [(450,OUT/'acmcp_n450'),(600,OLD/'acmcp')]:
            job=base/path.stem;assert (job/'complete.json').exists()
            def read(name):return pd.read_csv(job/name,header=None,na_values=['NA']).to_numpy()
            pf,errors,sc,rl,ru=[read(n) for n in ['mean.csv','errors.csv','scorecaster.csv','lower.csv','upper.csv']]
            cfg=pd.read_csv(job/'config.csv').iloc[0];assert cfg.ncal==window
            inputs.append(dict(job=job.relative_to(ROOT).as_posix(),scorecaster_sha256=hashlib.sha256((job/'scorecaster.csv').read_bytes()).hexdigest()))
            for h in range(1,25):
                lo,hi=replay_h(pf[:,h-1],errors[:,h-1],sc[:,h-1],h,.01,float(cfg.KI),window)
                np.testing.assert_allclose(lo,rl[:,h-1],rtol=2e-12,atol=2e-9,equal_nan=True)
                np.testing.assert_allclose(hi,ru[:,h-1],rtol=2e-12,atol=2e-9,equal_nan=True)
                checks.append(dict(archive=path.stem,window=window,horizon=h,max_endpoint_difference=max(float(np.nanmax(abs(lo-rl[:,h-1]))),float(np.nanmax(abs(hi-ru[:,h-1]))))))
                if h not in [1,6,24]:continue
                a=load_full(path,h);n=len(a.y);start=900+h-1;stop=1200-h+1
                assert np.isfinite(sc[h-1+start:h-1+n,h-1]).all(),'Uninitialized scorecast in evaluated origins'
                for method in ['AcMCP-CSS','PID-no-scorecast']:
                    caster=sc[:,h-1] if method=='AcMCP-CSS' else np.zeros(len(sc))
                    for rate in sorted(set([.001,.005,.01,.05,.1]+[v/h for v in [.001,.005,.01,.05,.1]])):
                        l,u=replay_h(pf[:,h-1],errors[:,h-1],caster,h,rate,float(cfg.KI),window)
                        l=l[h-1:h-1+n].copy();u=u[h-1:h-1+n].copy();cross=l>u;l[cross]=u[cross]=(l[cross]+u[cross])/2
                        rows.append(dict(backbone=a.backbone,dataset=a.dataset,horizon=h,method=method,window=window,rate=rate,validation=observed_metrics(a.y,l,u,start,stop)['interval_score'],**observed_metrics(a.y,l,u)))
                for mode in ['shared','issued']:
                    for gamma in rates(h)[2]:
                        l,u,_=calibrate_level(a,'cqr',mode,gamma,window)
                        rows.append(dict(backbone=a.backbone,dataset=a.dataset,horizon=h,method='CQR-'+mode,window=window,rate=gamma,validation=observed_metrics(a.y,l,u,start,stop)['interval_score'],**observed_metrics(a.y,l,u)))
                        if gamma==0 and mode=='shared':
                            rows.append(dict(rows[-1],method='Rolling-CQR'))
    df=pd.DataFrame(rows);assert df.invalid_intervals.eq(0).all();df.to_csv(OUT/'history_grid.csv',index=False)
    chosen=[]
    for _,sub in df.groupby(['backbone','dataset','horizon','method']):
        for search in ['fixed450','fixed600','selected']:
            pool=sub if search=='selected' else sub[sub.window==int(search[-3:])]
            best=pool.sort_values(['validation','rate','window']).iloc[0].to_dict();best['search']=search;chosen.append(best)
    sel=pd.DataFrame(chosen);sel.to_csv(OUT/'history_selected.csv',index=False)
    keys=['backbone','dataset','horizon','search']
    pair=sel.merge(sel[sel.method=='Rolling-CQR'],on=keys,suffixes=('','_rolling'),validate='many_to_one')
    pair['ratio']=pair.interval_score/pair.interval_score_rolling
    pair['gain']=100*(1-pair.ratio)
    pair['local_improvement_pp']=100*(pair.local_error_rolling-pair.local_error)
    pair.to_csv(OUT/'history_pairs.csv',index=False)
    summary=pair.groupby(['horizon','search','method']).agg(ratio=('ratio','mean'),coverage=('coverage','mean'),local=('local_error','mean'),gain=('gain','mean'),window=('window','mean'))
    summary.to_csv(OUT/'history_summary.csv')
    (OUT/'history_source_checks.json').write_text(json.dumps(dict(all_checks_passed=True,cases=checks,inputs=inputs,all_evaluated_scorecasts_finite=True,configuration_rows=len(df),validation='[900+h-1,1200-h+1); test [1200,2976)'),indent=2)+'\n')
    print(summary.to_string())
if __name__=='__main__':main()
