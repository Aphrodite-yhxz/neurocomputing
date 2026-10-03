"""Audit final numerical claims directly against recomputed pair-level results."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/focused_v10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pos=pd.read_csv(OUT/'positive_selected.csv');assert len(pos)==480
    assert pos[pos.search.str.startswith('positive')].gamma.gt(0).all()
    old=pd.read_csv(ROOT/'results/strengthening_v9/grid/results.csv')
    for row in pos.itertuples():
        sub=old[(old.backbone==row.backbone)&(old.dataset==row.dataset)&(old.horizon==row.horizon)&(old.score==row.score)&(old['mode']==row.mode)]
        if row.search!='including_zero':sub=sub[sub.gamma>0]
        if row.search in ['positive200','positive450','positive900']:sub=sub[sub.window==int(row.search[-3:])]
        best=sub.sort_values(['validation','gamma','window']).iloc[0]
        assert row.gamma==best.gamma and row.window==best.window
        np.testing.assert_allclose(row.interval_score,best.interval_score,atol=1e-11)
    pairs=pd.read_csv(OUT/'final_pair_choices.csv');assert len(pairs)==8 and pairs.same_rolling.sum()==6
    same=pairs[pairs.same_rolling];np.testing.assert_array_equal(same.interval_score_i,same.interval_score_s)
    temporal=pd.read_csv(OUT/'temporal_results.csv');assert len(temporal)==288
    assert temporal.invalid_intervals.eq(0).all()
    grouped=temporal.groupby(['backbone','dataset','horizon','method'])
    for _,s in grouped:
        whole=s[s.block=='whole'].iloc[0];blocks=s[s.block!='whole']
        assert blocks.n.sum()==whole.n and blocks.missing.sum()==whole.missing
        assert ((blocks.stop-blocks.start)==592).all()
        for metric in ['coverage','interval_score','width']:
            np.testing.assert_allclose(np.average(blocks[metric],weights=blocks.n),whole[metric],atol=1e-11)
    at1=temporal[temporal.horizon==1];keys=['backbone','dataset','block']
    identity=at1[at1.method=='Shared'].merge(at1[at1.method=='Issued'],on=keys,suffixes=('_s','_i'),validate='one_to_one')
    for metric in ['gamma','window','interval_score','coverage','local_error']:np.testing.assert_array_equal(identity[metric+'_s'],identity[metric+'_i'])
    local=pd.read_csv(OUT/'temporal_pairs.csv');w=local[(local.family=='weather')&(local.horizon==1)&(local.method=='Issued')]
    wb=w[w.block!='whole'];positive=wb.groupby(['backbone','dataset']).local_improvement_pp.apply(lambda x:bool((x>0).all()));assert positive.sum()==3
    hist=pd.read_csv(OUT/'history_grid.csv');assert hist.invalid_intervals.eq(0).all()
    selected=pd.read_csv(OUT/'history_selected.csv')
    for row in selected.itertuples():
        pool=hist[(hist.backbone==row.backbone)&(hist.dataset==row.dataset)&(hist.horizon==row.horizon)&(hist.method==row.method)]
        if row.search!='selected':pool=pool[pool.window==int(row.search[-3:])]
        best=pool.sort_values(['validation','rate','window']).iloc[0]
        assert row.window==best.window and row.rate==best.rate
    source=json.loads((OUT/'history_source_checks.json').read_text())
    assert source['all_checks_passed'] and len(source['cases'])==192 and source['all_evaluated_scorecasts_finite']
    for rec in source['inputs']:
        assert sha(ROOT/rec['job']/'scorecaster.csv')==rec['scorecaster_sha256']
        complete=json.loads((ROOT/rec['job']/'complete.json').read_text())
        if 'acmcp_n450' in rec['job']:assert complete['protocol_sha256']==sha(ROOT/'docs/FOCUSED_FINISH_PROTOCOL_V10.md')
    refits=pd.read_csv(OUT/'acmcp_n450/scorecast_refits.csv');assert len(refits)==60 and refits.absolute_difference.max()<1e-8
    # Match abstract and core tables to current full-precision outputs.
    ps=pd.read_csv(OUT/'positive_summary.csv');x=ps[(ps.family=='weather')&(ps.horizon==24)&(ps.score=='cqr')].set_index('search')
    whole=w[w.block=='whole']
    pv=pd.read_csv(ROOT/'results/strengthening_v9/pv/summary.csv');pv=pv[(pv.settings=='tuned')&(pv.projection=='nonnegative')&(pv.baseline=='signed')].set_index('group')
    headline=json.loads((ROOT/'results/strengthening_v9/main_replay/audit.json').read_text())['mean_paired_gain']
    values={'main_native_gain':headline,'weather_positive_fixed_gain':float(x.loc['positive450','gain']),
        'weather_positive_window_gain':float(x.loc['positive_window','gain']),
        'weather_h1_score_gain':float(whole.gain.mean()),'weather_h1_local_rolling':float(100*whole.local_error_rolling.mean()),
        'weather_h1_local_aci':float(100*whole.local_error.mean()),'pv_positive_gain':float(pv.loc['positive','gain']),
        'pv_positive_coverage':float(100*pv.loc['positive','coverage_cqr']),
        'pv_forecast_high_gain':float(pv.loc['forecast_high','gain']),'pv_forecast_high_coverage':float(100*pv.loc['forecast_high','coverage_cqr'])}
    paper=ROOT/'paper/neurocomputing_focused';abstract=(paper/'manuscript.tex').read_text(encoding='utf-8')
    for k,v in values.items():
        if k.startswith('pv_forecast'):continue
        digits=3 if k.startswith('weather_positive') else 2
        assert f'{v:.{digits}f}' in abstract,(k,v)
    body=(paper/'body.tex').read_text(encoding='utf-8')
    assert 'Appendix Appendix' not in body and 'History computation in progress' not in body
    assert '0.052' in (paper/'table_positive_rates.tex').read_text()
    assert 'Same rolling' in (paper/'table_final_choices.tex').read_text()
    for p in paper.glob('*.tex'):assert 'History computation in progress' not in p.read_text(encoding='utf-8')
    report=dict(all_checks_passed=True,positive_selected_rows=len(pos),temporal_rows=len(temporal),same_rolling_pairs=6,
        one_step_state_identity_cases=len(identity),all_block_local_improvement_pairs=int(positive.sum()),
        common_history_grid_rows=len(hist),original_R_trajectory_cases=192,new_R_trajectory_cases=96,
        new_saved_refit_cases=len(refits),max_new_refit_difference=float(refits.absolute_difference.max()),
        abstract_values=values,scope='Post-review exploratory analyses; saved R scorecasts and refit witness are inputs to the default replay.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
