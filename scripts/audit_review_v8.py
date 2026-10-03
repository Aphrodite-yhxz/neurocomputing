"""Audit V8 provenance, validation selection and exact numerical identities."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/review_v8'
sys.path.insert(0, str(ROOT/'src'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'experiments/{name}.py')
    out = importlib.util.module_from_spec(spec); spec.loader.exec_module(out)
    return out

def main():
    report = {'panels': {}, 'all_checks_passed': False}
    state = module('review_state_grid_v8'); pid = module('review_multistep_pid_v8')
    for panel, folder in [('development','results/innovation_v4/full_horizon'),
                          ('revisit','results/confirmation_v7/forecasts')]:
        hashes = {p.stem: sha(p) for p in (ROOT/folder).glob('*.npz')}
        assert len(hashes) == 16
        counts = {}
        for kind, mod, nfiles, expected in [('state_grid',state,48,(2304,576)),
                                           ('multistep_pid',pid,16,(1280,96))]:
            files = sorted((BASE/kind/panel/'pairs').glob('*.json')); assert len(files) == nfiles
            allrows = []
            for path in files:
                obj = json.loads(path.read_text())
                stem = path.stem.rsplit('__h',1)[0] if kind == 'state_grid' else path.stem
                assert obj['archive_sha256'] == hashes[stem]
                assert obj['fingerprint'] == mod.fingerprint()
                rows = obj['rows']; allrows += rows
                configs = [r for r in rows if r['selection'] == 'configuration']
                for r in rows:
                    if r['selection'] == 'configuration': continue
                    if kind == 'state_grid':
                        grid = state.grids(r['horizon'])[r['selection']]
                        eligible = [x for x in configs if x['score'] == r['score'] and
                                    x['mode'] == r['mode'] and x['gamma'] in grid]
                    else:
                        eligible = [x for x in configs if x['horizon'] == r['horizon'] and
                                    (r['selection'] == 'selected' or x['scorecaster'] == 'none')]
                    best = min(eligible, key=lambda x: (x['validation_score'],x['method']))
                    assert r['selected_method'] == best['method']
                    for k in ['interval_score','coverage','validation_score','local_error']:
                        assert r[k] == best[k]
            frame = pd.DataFrame(allrows)
            pd.testing.assert_frame_equal(frame, pd.read_csv(BASE/kind/panel/'results.csv'),
                                          check_dtype=False, atol=1e-12, rtol=1e-12)
            configs = frame[frame.selection == 'configuration']; chosen = frame[frame.selection != 'configuration']
            assert (len(configs), len(chosen)) == expected
            counts[kind] = {'computed_configurations':len(configs),'selected_references':len(chosen),
                            'cache_hashes_match':True,'validation_minima_verified':True}
            if kind == 'state_grid':
                err = np.max(np.abs(frame.clip_term + frame.endpoint_term - frame.revealed_excess))
                assert err < 1e-10
                assert np.all(np.abs(frame.pending_difference) <= frame.pending_test_n/frame.n + 1e-12)
                counts[kind]['max_accounting_error'] = float(err)
                wide = chosen[(chosen.selection == 'wide') & (chosen.horizon == 24) & (chosen.score == 'cqr')]
                counts[kind]['wide_h24_cqr_boundaries'] = {
                    mode:{'lower':int(g.at_lower_boundary.sum()),'upper':int(g.at_upper_boundary.sum()),'pairs':len(g)}
                    for mode,g in wide.groupby('mode')}
                oldpath = 'results/innovation_v3/delay_topology/selected.csv' if panel == 'development' else 'results/confirmation_v7/calibration/selected.csv'
                old = pd.read_csv(ROOT/oldpath); old = old[old['mode'].isin(['shared','issued'])]
                keys = ['backbone','dataset','horizon','score','mode']
                merged = chosen[chosen.selection == 'original'].merge(old,on=keys,suffixes=('_v8','_old'),validate='one_to_one')
                assert len(merged) == 192
                for k in ['gamma','validation_score','interval_score','coverage','local_error']:
                    np.testing.assert_allclose(merged[k+'_v8'],merged[k+'_old'],atol=1e-10,rtol=1e-12)
                counts[kind]['original_grid_reproduced'] = True
            else:
                for label, part in [('configurations',configs),('selected',chosen)]:
                    counts[kind][label+'_nonfinite_test_scores'] = int((~np.isfinite(part.interval_score)).sum())
                    counts[kind][label+'_nonfinite_validation_scores'] = int((~np.isfinite(part.validation_score)).sum())
                    counts[kind][label+'_max_test_infinite_rate'] = float(part.infinite_rate.max())
                    counts[kind][label+'_max_test_invalid_rate'] = float(part.invalid_rate.max())
        report['panels'][panel] = counts
    source = json.loads((BASE/'pid_source_checks/h24_n1000/checks.json').read_text())
    assert source['author_equivalence_passed'] and source['causality_passed']
    assert source['module_sha256'] == sha(ROOT/'src/daci/multistep_pid.py')
    assert source['source_sha256'] == sha(ROOT/'docs/references/v4/acmcp/R/pid.R')
    assert source['wrapper_sha256'] == sha(ROOT/'scripts/check_multistep_pid_v8.R')
    assert len(source['comparisons']) == 96
    report['pid_source'] = {'comparisons':96,'causality_passed':True,
        'max_absolute_difference':max(x['max_difference'] for x in source['comparisons'])}
    solar = pd.read_csv(BASE/'solar_components/results.csv'); assert len(solar) == 384
    for _, g in solar.groupby(['backbone','dataset','settings','score']):
        a = g[g.regime == 'all'].iloc[0]; groups = g[g.regime != 'all']
        for k in ['coverage','interval_score','width','lower_penalty','upper_penalty']:
            np.testing.assert_allclose((groups.fraction*groups[k]).sum(),a[k],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(g.interval_score,g.width+g.lower_penalty+g.upper_penalty,atol=1e-10,rtol=1e-12)
    manifest = json.loads((BASE/'solar_components/manifest.json').read_text())
    for x in manifest['sources']: assert sha(ROOT/x['path']) == x['sha256']
    assert manifest['script_sha256'] == sha(ROOT/'experiments/review_solar_components_v8.py')
    assert manifest['protocol_sha256'] == sha(ROOT/'docs/REVIEW_RESPONSE_PROTOCOL_V8.md')
    assert manifest['original_default_and_tuned_scores_reproduced']
    report['solar'] = {'pairs':32,'rows':len(solar),'weighted_components_verified':True,'original_scores_reproduced':True}
    dynamic = json.loads((BASE/'dynamics/manifest.json').read_text())
    assert dynamic['configuration_runs'] == len(pd.read_csv(BASE/'dynamics/results.csv')) == 1764
    assert dynamic['script_sha256'] == sha(ROOT/'experiments/review_controller_dynamics_v8.py')
    assert dynamic['protocol_sha256'] == sha(ROOT/'docs/REVIEW_RESPONSE_PROTOCOL_V8.md')
    report['simulation_runs'] = 1764
    for path, value in json.loads((BASE/'paper_sources.json').read_text()).items():
        assert sha(ROOT/path) == value
    report['paper_csv_hashes_verified'] = True
    report['all_checks_passed'] = True
    out = BASE/'manuscript'; out.mkdir(exist_ok=True)
    (out/'numerical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
