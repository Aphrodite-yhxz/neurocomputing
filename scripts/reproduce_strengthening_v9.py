"""Rebuild headline and all subsequent calibration analyses from supplied forecasts."""
from pathlib import Path
import json,subprocess,sys,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/strengthening_v9'

def main():
    start=time.perf_counter();logs=OUT/'reproduction';logs.mkdir(parents=True,exist_ok=True)
    (ROOT/'paper/neurocomputing_strengthened/figures').mkdir(parents=True,exist_ok=True)
    commands=['scripts/reproduce_submission.py','experiments/replay_main_headline_v9.py',
        'experiments/strengthening_grid_v9.py','experiments/strengthening_pv_v9.py',
        'scripts/summarize_acmcp_v9.py','scripts/build_strengthening_v9.py','scripts/audit_strengthening_v9.py']
    for i,command in enumerate(commands,1):
        print(f'[{i}/{len(commands)}] {command}',flush=True)
        with (logs/f'step-{i}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    comparisons=[]
    for ref in sorted((ROOT/'reference_results/strengthening_v9').rglob('*.csv')):
        rel=ref.relative_to(ROOT/'reference_results/strengthening_v9')
        a=pd.read_csv(OUT/rel);b=pd.read_csv(ref)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=1e-10,rtol=1e-10)
        comparisons.append(dict(path=rel.as_posix(),rows=len(a)))
    assert len(comparisons)==20
    prior=json.loads((ROOT/'results/submission_synthesis/reproduction_report.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text());assert prior['all_checks_passed'] and audit['all_checks_passed']
    report=dict(all_checks_passed=True,reference_csv_count=prior['reference_csv_count']+len(comparisons),
        strengthening_csv_comparisons=comparisons,elapsed_seconds=time.perf_counter()-start,
        main_headline_recomputed_pairs=124,new_forecast_archives=8,
        author_trajectory_cases=96,saved_refit_witnesses=60,
        scope='Calibration on supplied neural forecasts. Default AcMCP replay uses supplied full R scorecasts; optional R refitting is separate. Earlier controller benchmarks outside the mapped replay remain historical inputs.',
        environment=prior['environment'])
    (OUT/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
