"""Compare accelerated multi-step PID with unmodified author R, all stored horizons."""
from pathlib import Path
import sys,hashlib,json,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from dataclasses import replace
from daci.full_horizon import to_target_time,from_target_time,load_full
from daci.multistep_pid import calibrate_multistep_pid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--horizon',type=int,default=6);ap.add_argument('--n',type=int,default=1600)
    args=ap.parse_args();n=args.n;H=args.horizon
    path=ROOT/'results/innovation_v4/full_horizon/Chronos-Boltenc-dec__electricity_c20.npz'
    dest=ROOT/'results/review_v8/pid_source_checks'/f'h{H}_n{n}';dest.mkdir(parents=True,exist_ok=True)
    with np.load(path) as z:mu=z['mu'][:n,:H];y=z['y'][:n,:H]
    pf,errors=to_target_time(mu,y);ki=float(np.nanmax(abs(errors[:600])))
    np.savetxt(dest/'mean.csv',pf,delimiter=',');np.savetxt(dest/'errors.csv',errors,delimiter=',')
    pd.DataFrame([dict(ncal=600,lr=.01,KI=ki)]).to_csv(dest/'config.csv',index=False)
    wrapper=dest/'run_snapshot.R';wrapper.write_bytes((ROOT/'scripts/check_multistep_pid_v8.R').read_bytes())
    subprocess.run([str(ROOT/'.tools/R-4.6.1/bin/Rscript.exe'),'--vanilla',str(wrapper),str(ROOT),str(dest)],check=True)
    rows=[]
    for mode in ['none','mean','naive','ar1']:
        lower=from_target_time(pd.read_csv(dest/f'{mode}_lower.csv',header=None,na_values=['NA']).values,n)
        upper=from_target_time(pd.read_csv(dest/f'{mode}_upper.csv',header=None,na_values=['NA']).values,n)
        crossed=lower>upper;lower[crossed]=upper[crossed]=(lower[crossed]+upper[crossed])/2
        for h in range(1,H+1):
            a=load_full(path,h);a=replace(a,origin=a.origin[:n],y=a.y[:n],mu=a.mu[:n],quantiles=a.quantiles[:n])
            lo,hi,_=calibrate_multistep_pid(a,lr=.01,ki=ki,scorecaster=mode)
            np.testing.assert_allclose(lo,lower[:,h-1],atol=1e-8,rtol=1e-10,equal_nan=True)
            np.testing.assert_allclose(hi,upper[:,h-1],atol=1e-8,rtol=1e-10,equal_nan=True)
            cut=800;b=replace(a,y=a.y.copy());b.y[cut:]+=1000
            l,u,_=calibrate_multistep_pid(b,lr=.01,ki=ki,scorecaster=mode)
            np.testing.assert_allclose(lo[:cut+h],l[:cut+h],equal_nan=True)
            np.testing.assert_allclose(hi[:cut+h],u[:cut+h],equal_nan=True)
            rows.append(dict(horizon=h,scorecaster=mode,max_difference=float(max(np.nanmax(abs(lo-lower[:,h-1])),np.nanmax(abs(hi-upper[:,h-1]))))))
    report=dict(author_equivalence_passed=True,causality_passed=True,comparisons=rows,
        source_sha256=hashlib.sha256((ROOT/'docs/references/v4/acmcp/R/pid.R').read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256((ROOT/'src/daci/multistep_pid.py').read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
    (dest/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
