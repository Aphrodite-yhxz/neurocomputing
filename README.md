# Predictive information and delayed adaptation

Research artifact for **Separating predictive information from delayed adaptation
gains in time-series foundation models**, by Haoxuanzi Yin and Fan Zhang.
Prepared 3 October 2026 for manuscript review; publication or acceptance is not implied.

The controlled study distinguishes native predictive information from state, rate
and memory effects. On prospectively specified new sources, CQR state gains shrink
after broader calibration searches; weather requires window selection as well as
small rates. PV benefits remain after common nonnegative projection and asymmetric
residual controls, while relevant groups still under-cover. The manuscript reports
the scope and dependence of these comparisons and does not propose a new algorithm.
The focused revision separates positive-rate window effects from exact zero-rate
selection, reports every pair and three consecutive test blocks, and adds common-history
AcMCP comparisons. These post-review analyses revisit inspected data and are exploratory.

## CPU reproduction

Use Python 3.13 and the tested package versions:

```text
python -m pip install -r requirements-v8.txt
python -X utf8 reproduce.py
```

Run from the repository root. The default command needs no network, neural weights,
GPU or R. It recomputes calibration from the supplied forecasts, including the
124-pair 9.98% headline and its original nine-setting validation selection for each
score. It also executes the earlier sensitivity replay, all 4,896 new-source grid
configurations, 1,152 PV trajectories with raw/projected evaluation, and 200 AcMCP/PID
rate configurations using supplied full R scorecasts. Selected rows are references
to computed settings, not extra experiments. It also recomputes 480 positive-rate/grid
selections, 288 selected-trajectory block summaries, and 1,240 common-history candidate
rows. Initial Numba compilation takes time.

Sixty-one result CSVs are compared to reference outputs at absolute/relative tolerance
1e-10. `results/focused_v10/reproduction_report.json` records the outcome;
`verification/` contains the prepared handoff's actual clean-directory replay.
These are local checks, not replication by an independent research group.

**Default AcMCP replay uses fitted scorecasts as inputs.** It reproduces 192
original R interval trajectories across history lengths 450 and 600 before tuning
rates. The two saved 60-refit witnesses are verified, not recomputed by the CPU
entry point. To refit the complete configured
algorithm from forecasts, install R and its `forecast` package, then run:

```text
python -X utf8 experiments/strengthening_acmcp_v9.py --rscript Rscript --force
python -X utf8 scripts/summarize_acmcp_v9.py
Rscript --vanilla scripts/check_acmcp_strengthening_scorecasts.R .
python -X utf8 scripts/build_strengthening_v9.py
python -X utf8 experiments/focused_acmcp_v10.py --rscript Rscript --force
Rscript --vanilla scripts/check_acmcp_focused_scorecasts.R .
python -X utf8 scripts/summarize_focused_acmcp_v10.py
python -X utf8 scripts/build_focused_v10.py
```

`--force` explicitly replaces the corresponding local R outputs. Recorded runs used
R 4.6.1; the history-600 run took roughly one hour with two workers, while the added
history-450 run took about 37 minutes with four workers. Per-job records remain.
This is full MA plus cross-horizon scorecasting with the author's supported **CSS**
option, not the default CSS-ML configuration or an exhaustive search of AcMCP settings.
Three sampled history-450 refits emit a non-invertible MA warning but return the same
finite predictions as the saved source run; affected cases and warning text are retained.
No trajectory was discarded. The extra history comparison uses a common later validation
block to avoid selecting during scorecaster startup; it leaves test origins unchanged.

## Scope and files

- `paper/neurocomputing_focused/`: current PDF, source, tables and figures.
- `src/`, `experiments/`, `scripts/`, `tests/`: implementation and numerical checks.
- `results/`: 164 forecast archives and required comparison/source-check inputs.
- `data/strengthening_v9/`: exact new-source snapshots and four processed series.
- `reference_results/`: outputs checked against freshly recomputed results.
- `historical_evidence/`: earlier results outside the default replay scope.
- `docs/STRENGTHENING_PROTOCOL_V9.md`: unchanged, locally locked protocol.
- `docs/FOCUSED_FINISH_PROTOCOL_V10.md`: bounded exploratory post-review analyses.
- `forecast_manifest.json`, `strengthening_forecast_manifest.json`: archive provenance.
- `MANIFEST.json`: all handoff file hashes.

The earlier paper source is retained because earlier replay scripts regenerate its
displays. **The current manuscript is `paper/neurocomputing_focused/manuscript.pdf`.**
Read [ARTIFACT_MAP.md](ARTIFACT_MAP.md) for exact reproduction boundaries. Full historical
neural inference and every earlier benchmark controller are not rerun by this command.

Optional new-source neural generation is recorded in
`experiments/generate_strengthening_v9.py`. It expects a CUDA environment and the exact
two Hugging Face snapshots already cached locally; checkpoint IDs, file hashes,
context, dtype and package versions accompany each new forecast archive. The CPU
requirements do not install that GPU environment. Pretraining exclusion is not claimed.

Compile the manuscript using pdfLaTeX, BibTeX and two further pdfLaTeX passes. CPU replay
regenerates the mapped tables and figures but does not compile the PDF.

See [DATA_SOURCES.md](DATA_SOURCES.md) and [LICENSING.md](LICENSING.md) for component
terms. Evaluation targets are included. No private author document, credentials,
model weights or installed runtimes are bundled. The repository URL will be added
after author upload; `Supplementary_Software_S1.zip` is the same prepared software.
