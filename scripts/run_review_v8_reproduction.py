"""Recompute every V8 configuration and compare with packaged reference CSVs."""
from pathlib import Path
import importlib.metadata, json, platform, subprocess, sys, time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def main():
    start = time.perf_counter()
    out = ROOT/'results/review_v8/reproduction'; out.mkdir(parents=True,exist_ok=True)
    commands = [
        ['experiments/review_state_grid_v8.py','--panel','development','--workers','2'],
        ['experiments/review_state_grid_v8.py','--panel','revisit','--workers','2'],
        ['experiments/review_solar_components_v8.py'],
        ['experiments/review_controller_dynamics_v8.py'],
        ['experiments/review_multistep_pid_v8.py','--workers','2'],
        ['scripts/build_review_v8_evidence.py'],
        ['scripts/audit_review_v8.py'],
    ]
    for i, command in enumerate(commands):
        print(f'[{i+1}/{len(commands)}] {command[0]}',flush=True)
        with (out/f'step-{i+1}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    comparisons=[]
    for ref in sorted((ROOT/'reference_results/review_v8').rglob('*.csv')):
        relative=ref.relative_to(ROOT/'reference_results')
        actual=ROOT/'results'/relative
        a=pd.read_csv(actual); b=pd.read_csv(ref)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=1e-10,rtol=1e-10)
        differences=[]
        for c in a.select_dtypes(include='number'):
            x=a[c].to_numpy(); y=b[c].to_numpy(); mask=np.isfinite(x)&np.isfinite(y)
            if mask.any(): differences.append(float(np.max(np.abs(x[mask]-y[mask]))))
        comparisons.append({'path':relative.as_posix(),'rows':len(a),
                            'max_absolute_numeric_difference':max(differences,default=0.)})
    assert len(comparisons)>=25
    report={'all_checks_passed':True,'scope':'Every V8 configuration; archived neural predictions are inputs, not regenerated',
            'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','numba','matplotlib']},
            'elapsed_seconds':time.perf_counter()-start,'csv_comparisons':comparisons,
            'max_absolute_numeric_difference':max(x['max_absolute_numeric_difference'] for x in comparisons)}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
