"""Check chronology, matched contrasts, decompositions and source witnesses."""
from pathlib import Path
import hashlib,json,sys,runpy
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/strengthening_v9'
sys.path.insert(0,str(ROOT/'src'))
from daci.full_horizon import to_target_time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def main():
    protocol=sha(ROOT/'docs/STRENGTHENING_PROTOCOL_V9.md')
    lock=read(OUT/'protocol_lock.json');assert lock['protocol_sha256']==protocol
    manifest=read(ROOT/'data/strengthening_v9/manifest.json')
    assert manifest['protocol_sha256']==protocol
    for entry in manifest['sources']:
        assert sha(ROOT/'data/strengthening_v9'/f"{entry['name']}.zip")==entry['sha256']
    sources={x['name']:x for x in manifest['series']}
    for entry in sources.values():
        assert sha(ROOT/entry['path'])==entry['sha256']
    archives=sorted((OUT/'forecasts').glob('*.npz'));assert len(archives)==8
    observation_checks=[]
    for path in archives:
        meta=read(path.with_suffix('.json'))
        assert meta['archive_sha256']==sha(path) and meta['protocol_sha256']==protocol
        assert meta['generator_sha256']==sha(ROOT/'experiments/generate_strengthening_v9.py')
        with np.load(path) as z:
            entry=sources[str(z['dataset'])];assert meta['source_sha256']==entry['sha256']
            np.testing.assert_array_equal(z['target_index'],z['t_index'][:,None]+np.arange(1,25))
            np.testing.assert_allclose(z['y'],z['series'][z['target_index']],equal_nan=True)
            np.testing.assert_array_equal(z['observed'],np.isfinite(z['y']))
            assert z['y'].shape==(2976,24) and z['quantiles'].shape==(2976,24,5)
            assert np.isfinite(z['mu']).all() and np.isfinite(z['quantiles']).all()
            raw=z['series'];filled=pd.Series(raw).ffill().fillna(0).to_numpy()
            np.testing.assert_array_equal(z['context_series'],filled)
            observation_checks.append(dict(archive=path.name,missing_targets=int((~z['observed']).sum())))
    grid=pd.read_csv(OUT/'grid/results.csv');assert len(grid)==4896
    assert grid.invalid_intervals.eq(0).all()
    # Shared and issued recurrences coincide at one step, and at gamma=0.
    keys=['backbone','dataset','horizon','score','gamma','window']
    matched=grid[grid['mode']=='shared'].merge(grid[grid['mode']=='issued'],on=keys,suffixes=('_s','_i'),validate='one_to_one')
    equal=matched[(matched.horizon==1)|(matched.gamma==0)]
    np.testing.assert_allclose(equal.interval_score_s,equal.interval_score_i,rtol=1e-12,atol=1e-12)
    selection=pd.read_csv(OUT/'grid/selected.csv')
    # Verify every selected setting against its declared validation pool, not its test score.
    from experiments.strengthening_grid_v9 import rates
    for row in selection.itertuples(index=False):
        sub=grid[(grid.backbone==row.backbone)&(grid.dataset==row.dataset)&(grid.horizon==row.horizon)&(grid.score==row.score)&(grid['mode']==row.mode)]
        base,wide,_=rates(row.horizon)
        in_base=np.isclose(sub.gamma.to_numpy()[:,None],base,rtol=1e-12,atol=1e-18).any(axis=1)
        in_wide=np.isclose(sub.gamma.to_numpy()[:,None],wide,rtol=1e-12,atol=1e-18).any(axis=1)
        if row.grid=='original':sub=sub[(sub.window==450)&in_base]
        elif row.grid=='wide':sub=sub[(sub.window==450)&in_wide]
        elif row.grid=='zero_small':sub=sub[sub.window==450]
        elif row.grid=='rolling450':sub=sub[(sub.window==450)&sub.gamma.eq(0)]
        elif row.grid=='rolling_tuned':sub=sub[sub.gamma.eq(0)]
        else:assert row.grid=='window'
        best=sub.sort_values([row.period,'gamma','window']).iloc[0]
        assert np.isclose(row.gamma,best.gamma,rtol=1e-12,atol=1e-18) and row.window==best.window,(row,best)
    contrast=pd.read_csv(OUT/'grid/state_contrasts.csv')
    np.testing.assert_allclose(contrast.state_log_ratio,np.log(contrast.interval_score_shared/contrast.interval_score_issued),atol=1e-14)
    ck=['backbone','dataset','horizon','score','period']
    attenuation=contrast[contrast.grid=='original'].merge(contrast[contrast.grid=='window'],on=ck,suffixes=('_a','_b'),validate='one_to_one')
    left=attenuation.state_log_ratio_a-attenuation.state_log_ratio_b
    right=np.log(attenuation.interval_score_shared_a/attenuation.interval_score_shared_b)-np.log(attenuation.interval_score_issued_a/attenuation.interval_score_issued_b)
    np.testing.assert_allclose(left,right,atol=1e-14)
    pv=pd.read_csv(OUT/'pv/results.csv');assert len(pv)==2880
    np.testing.assert_allclose(pv.interval_score,pv.width+pv.lower_penalty+pv.upper_penalty,rtol=1e-12,atol=1e-10)
    group_checks=0
    for _,sub in pv.groupby(['backbone','dataset','score','projection','settings']):
        s=sub.set_index('group')
        for groups in [('zero','positive'),('forecast_low','forecast_high')]:
            parts=s.loc[list(groups)];np.testing.assert_allclose(parts.fraction.sum(),1,atol=1e-12)
            assert parts.n.sum()==s.loc['all','n']
            for key in ['coverage','interval_score','width','lower_penalty','upper_penalty']:
                np.testing.assert_allclose((parts[key]*parts.fraction).sum(),s.loc['all',key],rtol=1e-12,atol=1e-10)
            group_checks+=1
    main=read(OUT/'main_replay/audit.json');assert main['all_checks_passed'] and main['pairs']==124
    np.testing.assert_allclose(main['mean_paired_gain'],9.979980111694497,atol=1e-10)
    ac=read(OUT/'acmcp/source_checks.json');assert ac['all_checks_passed'] and len(ac['cases'])==96
    for source in ac['sources']:
        job=OUT/'acmcp'/source['job'];m=source['author_manifest']
        assert sha(job/'scorecaster.csv')==source['scorecasts_sha256']
        assert sha(job/'run_author_snapshot.R')==m['r_wrapper_sha256']
        assert sha(ROOT/'docs/references/v4/acmcp/R/acmcp.R')==m['author_acmcp_sha256']
        path=OUT/'forecasts'/f"{source['job']}.npz"
        assert sha(path)==m['archive_sha256'] and m['protocol_sha256']==protocol
        with np.load(path) as z:pf,errors=to_target_time(z['mu'],z['y'])
        for name,expected in [('mean.csv',pf),('errors.csv',errors)]:
            np.testing.assert_allclose(pd.read_csv(job/name,header=None).to_numpy(),expected,rtol=1e-13,atol=1e-12,equal_nan=True)
        counts=pd.read_csv(job/'counts.csv')
        assert len(counts)==24
    refits=pd.read_csv(OUT/'acmcp/scorecast_refits.csv');assert len(refits)==60 and refits.absolute_difference.max()<1e-8
    ag=pd.read_csv(OUT/'acmcp/grid.csv');assert len(ag)==200 and ag.invalid_intervals.eq(0).all()
    tests=runpy.run_path(str(ROOT/'tests/test_strengthening_v9.py'))
    test_names=[x for x in tests if x.startswith('test_')]
    for name in test_names:tests[name]()
    paper=(ROOT/'paper/neurocomputing_strengthened/manuscript.tex').read_text(encoding='utf-8')
    assert 'On the latter panel' not in paper and 'published at ICLR 2026' in paper
    report=dict(all_checks_passed=True,protocol_sha256=protocol,local_protocol_not_external_preregistration=True,
        new_forecast_archives=8,observation_checks=observation_checks,grid_configurations=len(grid),
        validation_selected_rows=len(selection),state_identity_cases=len(equal),attenuation_identity_cases=len(attenuation),
        pv_trajectories=1152,pv_grid_rows=2304,pv_group_recombinations=group_checks,arithmetic_causality_tests=test_names,
        main_headline=main,author_trajectory_cases=96,max_author_endpoint_difference=max(x['max_endpoint_difference'] for x in ac['cases']),
        saved_independent_scorecast_refits=60,max_saved_refit_difference=float(refits.absolute_difference.max()),
        scorecast_scope='Default replay uses supplied R fits; this audit verifies the saved refit witness. Full R refitting is a separate command.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    sys.path.insert(0,str(ROOT));main()
