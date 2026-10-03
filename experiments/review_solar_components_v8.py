"""Outcome-conditional loss decomposition of the original one-step PV result."""
from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from daci.revision import load_archive,calibrate,interval_scores
OUT=ROOT/'results/review_v8/solar_components'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(ROOT/'results/neurocomputing/main_a0.1/results.csv')
    paths=sorted((ROOT/'results/real/traces').glob('*__solar_*.npz'));assert len(paths)==32
    rows=[];sources=[]
    for path in paths:
        a=load_archive(path);y=a.y[1200:];assert np.all(y>=0)
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for settings in ['default','tuned']:
            for score in ['residual','cqr']:
                method=f'{score}/aci'+('/tuned' if settings=='tuned' else '')
                row=old[(old.backbone==a.backbone)&(old.dataset==a.dataset)&(old.method==method)].iloc[0]
                gamma=.02 if pd.isna(row.gamma) else float(row.gamma)
                window=450 if pd.isna(row.window) else int(row.window)
                lo,hi,_=calibrate(a,score=score,rule='aci',gamma=gamma,window=window)
                lo=lo[1200:];hi=hi[1200:]
                width=hi-lo;lower=20*np.maximum(lo-y,0);upper=20*np.maximum(y-hi,0)
                loss=width+lower+upper;cover=(lo<=y)&(y<=hi)
                np.testing.assert_allclose(loss,interval_scores(y,lo,hi),atol=1e-10)
                np.testing.assert_allclose(loss.mean(),row.interval_score,atol=1e-8,rtol=1e-10)
                pair=[]
                for regime,mask in [('all',np.ones(len(y),bool)),('zero',y==0),('positive',y>0)]:
                    r=dict(backbone=a.backbone,dataset=a.dataset,settings=settings,score=score,regime=regime,
                        gamma=gamma,window=window,n=int(mask.sum()),fraction=float(mask.mean()),
                        coverage=float(cover[mask].mean()),interval_score=float(loss[mask].mean()),
                        width=float(width[mask].mean()),lower_penalty=float(lower[mask].mean()),upper_penalty=float(upper[mask].mean()))
                    pair.append(r);rows.append(r)
                for metric in ['coverage','interval_score','width','lower_penalty','upper_penalty']:
                    np.testing.assert_allclose(sum(r['fraction']*r[metric] for r in pair[1:]),pair[0][metric],rtol=1e-10,atol=1e-10)
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False)
    keys=['backbone','dataset','settings','regime']
    c=df[df.score=='cqr'].merge(df[df.score=='residual'],on=keys,suffixes=('_cqr','_residual'),validate='one_to_one')
    bas=c[c.regime=='all'][['backbone','dataset','settings','interval_score_residual']].rename(columns={'interval_score_residual':'overall_residual'})
    c=c.merge(bas,on=['backbone','dataset','settings'],validate='many_to_one')
    c['conditional_gain']=100*(1-c.interval_score_cqr/c.interval_score_residual)
    c['overall_gain_contribution']=100*c.fraction_cqr*(c.interval_score_residual-c.interval_score_cqr)/c.overall_residual
    for metric in ['width','lower_penalty','upper_penalty']:
        c[metric+'_contribution']=100*c.fraction_cqr*(c[metric+'_residual']-c[metric+'_cqr'])/c.overall_residual
    c.to_csv(OUT/'contrasts.csv',index=False)
    columns=['conditional_gain','overall_gain_contribution','fraction_cqr','coverage_cqr','coverage_residual',
             'width_contribution','lower_penalty_contribution','upper_penalty_contribution']
    summary=c.groupby(['settings','regime'])[columns].mean()
    assert round(summary.loc[('default','all'),'conditional_gain'],2)==35.25
    summary.to_csv(OUT/'summary.csv')
    df.groupby(['settings','score','regime'])[['fraction','coverage','interval_score','width','lower_penalty','upper_penalty']].mean().to_csv(OUT/'components.csv')
    (OUT/'manifest.json').write_text(json.dumps(dict(status='post_review_evaluation_only_diagnostic',
        original_default_and_tuned_scores_reproduced=True,weighted_components_verified=True,
        protocol_sha256=hashlib.sha256((ROOT/'docs/REVIEW_RESPONSE_PROTOCOL_V8.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=sources),indent=2))
    print(summary.round(5).to_string())

if __name__=='__main__':main()
