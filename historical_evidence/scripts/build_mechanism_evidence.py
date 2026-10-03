"""Generate manuscript evidence from saved results; does not select or rerun methods."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/neurocomputing_mechanisms'
OUT = ROOT / 'output/research'


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / 'figures').mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    folder = ROOT / 'results/confirmation_v7/calibration'
    family = pd.read_csv(folder / 'family_summary.csv')
    contrasts = pd.read_csv(folder / 'contrasts.csv')
    selected = pd.read_csv(folder / 'selected.csv')
    assert len(selected) == 864
    assert not selected.duplicated(['backbone','dataset','horizon','score','mode']).any()
    comparisons = [
        ('native_vs_residual','cop','Native / residual, COP'),
        ('native_vs_residual','issued','Native / residual, issued ACI'),
        ('issued_vs_shared','cqr','Issued / shared, CQR'),
    ]
    table = [r'\begin{tabular}{llrrrrr}',r'\toprule',
        r'Comparison & Family & Gain & Median & Wins & Cov. A & Cov. B \\',r'\midrule']
    for contrast, controller, label in comparisons:
        rows = family[(family.contrast == contrast) & (family.controller == controller) & (family.horizon == 24)]
        assert len(rows) == 2
        for _, r in rows.iterrows():
            fam = {'exchange_rate':'FX','traffic':'Traffic'}[r.family]
            table.append(f'{label} & {fam} & {r.mean_gain:.2f} & {r.median_gain:.2f} & {int(r.wins)}/8 & {100*r.coverage_new:.2f} & {100*r.coverage_base:.2f} '+r'\\')
    table += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'table_confirmation.tex').write_text('\n'.join(table)+'\n', encoding='utf-8')

    table = [r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Score & Controller & FX cov. & Traffic cov. & Local & Singleton \\',r'\midrule']
    names = {'rolling':'Rolling','shared':'Shared ACI','issued':'Issued ACI','pi_log':'PI-log',
             'sfogd':'SF-OGD','cop':'COP','grouped':'Grouped ACI','ogd':'Fixed-scale OGD','implicit_delay':'Implicit-delay'}
    for score in ['residual','cqr']:
        for mode, label in names.items():
            rows = selected[(selected.horizon==24)&(selected.score==score)&(selected['mode']==mode)]
            assert len(rows)==16
            group = rows.groupby('family').mean(numeric_only=True)
            table.append(f'{score.upper() if score=="cqr" else "Residual"} & {label} & {100*group.loc["exchange_rate","coverage"]:.2f} & {100*group.loc["traffic","coverage"]:.2f} & {100*group.local_error.mean():.2f} & {100*group.singleton_rate.mean():.3f} '+r'\\')
        if score=='residual':table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'table_confirmation_controls.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(6.7,2.75),layout='constrained')
    plots = [("native_vs_residual","cop","Native vs residual\nCOP"),
             ("issued_vs_shared","cqr","Issued vs shared\nCQR"),
             ("implicit_vs_ogd","cqr","Implicit vs OGD\nCQR")]
    for panel,(ax,(contrast,controller,title)) in enumerate(zip(axes,plots)):
        sub = contrasts[(contrasts.contrast==contrast)&(contrasts.controller==controller)&(contrasts.horizon==24)]
        assert len(sub)==16
        for y,(fam,color) in enumerate([('exchange_rate','#305f91'),('traffic','#c57530')]):
            group=sub[sub.family==fam].sort_values(['backbone','dataset'])
            offsets=np.linspace(-.15,.15,len(group))
            ax.scatter(group.gain,y+offsets,c=color,s=30,alpha=.75,edgecolor='white',linewidth=.4)
            ax.scatter([group.gain.mean()],[y],c=color,marker='|',s=430,linewidth=2.8,zorder=4)
        ax.axvline(0,color='#6b7280',linestyle='--',linewidth=.9)
        ax.set_yticks([0,1],['FX','Traffic'] if panel==0 else ['',''])
        ax.set_ylim(-.45,1.45);ax.set_title(title,fontsize=10,pad=12)
        ax.set_xlabel('Gain (%)')
        ax.grid(axis='x',alpha=.16)
    fig.savefig(PAPER/'figures/confirmation.png',dpi=220)
    fig.savefig(OUT/'confirmation_mechanisms.svg')
    fig.savefig(OUT/'confirmation_mechanisms.png',dpi=220)
    plt.close(fig)

    sources=[folder/name for name in ['family_summary.csv','family_balanced_summary.csv','contrasts.csv','selected.csv']]
    sources += [ROOT/'results/innovation_v4/solar_regimes/summary.csv',ROOT/'results/innovation_v5/grouped/selected.csv',
                ROOT/'results/innovation_v6/implicit_cop/summary.csv']
    audit = {'purpose':'Manuscript summary provenance; no re-selection or hypothesis tests',
             'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             'confirmation_pairs':16,'confirmation_families':2,'selected_rows':len(selected)}
    dest=ROOT/'results/manuscript_mechanisms';dest.mkdir(exist_ok=True)
    (dest/'evidence_sources.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
