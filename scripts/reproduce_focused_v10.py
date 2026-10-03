"""Recompute earlier evidence plus focused post-review contrasts from supplied inputs."""
from pathlib import Path
import json,subprocess,sys,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/focused_v10'
def main():
    start=time.perf_counter();logs=OUT/'reproduction';logs.mkdir(parents=True,exist_ok=True)
    (ROOT/'paper/neurocomputing_focused/figures').mkdir(parents=True,exist_ok=True)
    scripts=['scripts/reproduce_strengthening_v9.py','experiments/focused_contrasts_v10.py',
             'scripts/summarize_focused_acmcp_v10.py','scripts/build_focused_v10.py','scripts/audit_focused_v10.py']
    for i,script in enumerate(scripts,1):
        print(f'[{i}/{len(scripts)}] {script}',flush=True)
        with (logs/f'step-{i}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',script],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    comparisons=[]
    for ref in sorted((ROOT/'reference_results/focused_v10').glob('*.csv')):
        a=pd.read_csv(OUT/ref.name);b=pd.read_csv(ref)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-10,atol=1e-10)
        comparisons.append(dict(path=ref.name,rows=len(a)))
    assert len(comparisons)==11
    old=json.loads((ROOT/'results/strengthening_v9/reproduction_report.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text());assert old['all_checks_passed'] and audit['all_checks_passed']
    report=dict(all_checks_passed=True,reference_csv_count=old['reference_csv_count']+len(comparisons),
                focused_comparisons=comparisons,elapsed_seconds=time.perf_counter()-start,
                abstract_values=audit['abstract_values'],environment=old['environment'],
                scope='All mapped calibration and focused analyses on supplied forecasts; AcMCP R fits and refit witnesses are supplied inputs, with optional full R refitting commands.')
    (OUT/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
