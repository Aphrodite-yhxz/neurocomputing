"""One-command CPU replay for the submission repository, using supplied forecasts."""
from pathlib import Path
import json,subprocess,sys,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.perf_counter()
    for path in ['paper/neurocomputing_review_v8/figures','paper/neurocomputing_submission/figures']:
        (ROOT/path).mkdir(parents=True,exist_ok=True)
    for script in ['run_review_v8_reproduction.py','build_submission_synthesis.py','audit_submission_synthesis.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    comparisons=[]
    for ref in sorted((ROOT/'reference_results/submission_synthesis').glob('*.csv')):
        a=pd.read_csv(ROOT/'results/submission_synthesis'/ref.name);b=pd.read_csv(ref)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=1e-10,rtol=1e-10)
        comparisons.append(ref.name)
    assert len(comparisons)==3
    original=json.loads((ROOT/'results/review_v8/reproduction/report.json').read_text())
    audit=json.loads((ROOT/'results/submission_synthesis/audit.json').read_text())
    assert original['all_checks_passed'] and audit['all_checks_passed']
    report={'all_checks_passed':True,'scope':'subsequent calibration analyses and final paired synthesis; neural forecasts are supplied inputs',
            'reference_csv_count':len(original['csv_comparisons'])+len(comparisons),
            'new_csv_comparisons':comparisons,'zero_rate_rolling_cases':audit['zero_rate_rolling_cases'],
            'max_zero_rate_endpoint_difference':audit['max_endpoint_difference'],
            'elapsed_seconds':time.perf_counter()-start,'environment':{k:original[k] for k in ['python','platform','packages']}}
    (ROOT/'results/submission_synthesis/reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
