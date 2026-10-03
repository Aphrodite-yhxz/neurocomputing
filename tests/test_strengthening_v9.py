import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from daci.revision import Archive,metrics,interval_scores
from daci.delay_topology import calibrate_topology
from daci.strengthening import calibrate_level,observed_metrics,nonnegative

def fixture(h):
    rng=np.random.default_rng(9030);y=rng.normal(size=1550);mu=np.sin(np.arange(1550)/45)
    q=mu[:,None]+np.array([-1.2,-.5,0,.5,1.2])
    return Archive('fixture','fixture',np.arange(1550),y,mu,q,np.array([.1,.3,.5,.7,.9]),h)

def test_legacy_and_rolling_identity():
    for h in [1,6,24]:
        a=fixture(h)
        for score in ['residual','cqr']:
            for mode in ['shared','issued']:
                for gamma in [0,.0001,.02]:
                    lo,hi,_=calibrate_level(a,score,mode,gamma)
                    l,u,_=calibrate_topology(a,score,mode,gamma)
                    np.testing.assert_allclose(lo,l,atol=1e-12,equal_nan=True)
                    np.testing.assert_allclose(hi,u,atol=1e-12,equal_nan=True)
                m=metrics(a.y,lo,hi);n=observed_metrics(a.y,lo,hi)
                for key in ['interval_score','coverage','local_error']:
                    np.testing.assert_allclose(m[key],n[key],atol=1e-12)

def test_delayed_causality_and_missing_labels():
    for h in [1,6,24]:
        for score in ['residual','cqr','signed']:
            for mode in ['shared','issued']:
                a=fixture(h);a.y[::19]=np.nan
                lo,hi,levels=calibrate_level(a,score,mode)
                a.y[1000:]+=100
                l,u,s=calibrate_level(a,score,mode)
                np.testing.assert_array_equal(lo[:1000+h],l[:1000+h])
                np.testing.assert_array_equal(hi[:1000+h],u[:1000+h])
                assert np.isfinite(lo[600+h-1:]).all()
                for t in range(600+2*h-1,len(a.y)):
                    j=t-h
                    if not np.isfinite(a.y[j]):
                        assert levels[t]==levels[j if mode=='issued' else t-1]

def test_support_projection_score_and_coverage():
    rng=np.random.default_rng(7201);bounds=np.sort(rng.normal(size=(10000,2)),axis=1)
    y=abs(rng.normal(size=10000));y[::3]=0
    lo,hi=bounds.T;l,u=nonnegative(lo,hi)
    assert np.all(interval_scores(y,l,u)<=interval_scores(y,lo,hi)+1e-12)
    assert np.all(~((lo<=y)&(y<=hi))|((l<=y)&(y<=u)))
    crossing=(lo<0)&(hi>=0)
    np.testing.assert_allclose(interval_scores(y,lo,hi)[crossing]-interval_scores(y,l,u)[crossing],-lo[crossing],atol=1e-12)
