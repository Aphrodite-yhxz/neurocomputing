"""Verify descriptive contrasts and the zero-rate rolling limit; no model tuning."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.full_horizon import load_full
from daci.revision import calibrate,metrics
from daci.delay_topology import calibrate_topology

def main():
    out=ROOT/'results/submission_synthesis'
    d=pd.read_csv(out/'rolling_pairs.csv');assert len(d)==192
    assert d.groupby(['method','horizon']).size().eq(16).all()
    np.testing.assert_allclose(d.paired_gain,100*(1-d.interval_score/d.interval_score_rolling),atol=1e-12)
    np.testing.assert_allclose(d.coverage_delta_pp,100*(d.coverage-d.coverage_rolling),atol=1e-12)
    np.testing.assert_allclose(d.local_delta_pp,100*(d.local_error-d.local_error_rolling),atol=1e-12)
    old=pd.read_csv(ROOT/'results/confirmation_v7/calibration/selected.csv')
    source=old[(old.score=='cqr')&(old['mode']=='rolling')]
    cases=0;maxdiff=0.
    for path in sorted((ROOT/'results/confirmation_v7/forecasts').glob('*.npz')):
        for h in [1,6,24]:
            a=load_full(path,h)
            l,u,_=calibrate(a,score='cqr',rule='rolling',window=450)
            v=metrics(a.y,l,u)
            r=source[(source.backbone==a.backbone)&(source.dataset==a.dataset)&(source.horizon==h)].iloc[0]
            for key in ['interval_score','coverage','local_error']:np.testing.assert_allclose(v[key],r[key],atol=1e-10,rtol=1e-12)
            for mode in ['shared','issued']:
                ll,uu,_=calibrate_topology(a,'cqr',mode,gamma=0.)
                np.testing.assert_allclose(ll[1200:],l[1200:],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(uu[1200:],u[1200:],atol=1e-12,rtol=1e-12)
                maxdiff=max(maxdiff,float(np.max(abs(ll[1200:]-l[1200:]))),float(np.max(abs(uu[1200:]-u[1200:]))));cases+=1
    assert cases==96
    p=ROOT/'paper/neurocomputing_submission/manuscript.tex'
    text=p.read_text();assert '84.97' not in text
    assert not any(x in text for x in ['Revisited V7','V8 analysis'])
    report={'all_checks_passed':True,'paired_rows':len(d),'zero_rate_rolling_cases':cases,
            'max_endpoint_difference':maxdiff,'existing_rolling_summaries_reproduced':True,
            'status':'implementation_identity_check_and_descriptive_synthesis_no_retuning',
            'manuscript_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
