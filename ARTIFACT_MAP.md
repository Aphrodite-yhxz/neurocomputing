# Manuscript to artifact map

Current manuscript: `paper/neurocomputing_focused/manuscript.pdf`. The earlier
strengthened source remains for its replay checks; it is not the current paper.

| Focused finding/display | Executable source | Reproduction scope |
|---|---|---|
| Positive-only versus zero-admitting state contrasts; final eight-pair choice table | `experiments/focused_contrasts_v10.py` | Selects from the freshly rebuilt original 4,896-configuration grid; retains 480 selector rows. |
| Every pair across three test blocks and the whole test; one-step local benefits | Same script | Recomputes selected trajectories from forecast arrays and 288 block/whole summaries. Does not retune on test blocks. |
| Common-history 450/600 comparisons | `scripts/summarize_focused_acmcp_v10.py` | Checks 192 original R trajectories; recomputes 1,240 candidate rows with a common post-startup validation segment and unchanged tests. |
| Additional history-450 full source fitting | `experiments/focused_acmcp_v10.py` | Optional R run; default replay uses its supplied complete scorecasts. |
| Additional 60 sampled fit checks and warnings | `scripts/check_acmcp_focused_scorecasts.R` | Optional independent R refitting; default audit reads the saved witness. |
| Final abstract and core displays | `scripts/build_focused_v10.py`, `scripts/audit_focused_v10.py` | Generates tables/figures and checks every numerical abstract claim against recalculated results. |

All focused analyses are exploratory on inspected data. Fixed/common histories
do not equalize predictive information or separate AcMCP startup from its history
parameter. Small positive rates can behave nearly like rolling calibration.

Earlier evidence, retained in the 50-table replay:

| Finding | Executable source | Default replay and boundaries |
|---|---|---|
| 9.98% matched main-panel gain, 124 pairs | `experiments/replay_main_headline_v9.py` | Recomputes all 2,232 score/rate/window configurations and 248 selected references from predictions. Neural inference and other original benchmark rules remain outside this replay. |
| New-source score–state–rate–memory confirmation | `experiments/strengthening_grid_v9.py` | All 4,896 configurations; original, wide, zero/small and window grids; early/full/late selectors; separate rolling controls. |
| New-source core figure, state table and paired rolling comparisons | `scripts/build_strengthening_v9.py` | Rebuilt from fresh grid results, with pair-specific score denominators. |
| Full AcMCP-CSS weather comparison | `scripts/summarize_acmcp_v9.py` | Uses supplied R MA/regression scorecasts, checks all 96 original interval trajectories, then recomputes 200 rate configurations across AcMCP and the matched no-scorecast control. |
| Independent sampled R fit check | `scripts/check_acmcp_strengthening_scorecasts.R` | Optional R execution recomputes 60 fits; default audit reads the saved witness. Full R refitting command is in README. |
| PV nonnegative support, signed-residual and observable-group controls | `experiments/strengthening_pv_v9.py` | 1,152 trajectories, 2,304 raw/projected candidate rows, subgroup scores and components; 32 original pairs. Exploratory old-data analysis. |
| Original/scaled/wide state sensitivity, clipping accounting, simulations, one-step PV decomposition and configured PID | `scripts/reproduce_submission.py` and `scripts/run_review_v8_reproduction.py` | Earlier 30-CSV replay retained. The configured PID and full new-weather AcMCP are different comparisons. |
| Earlier original-grid, family, renewable, benchmark rule and nominal-level results | `historical_evidence/` and supplied original result CSVs | Historical inputs; not all original benchmark configurations are recomputed. The main 9.98% headline is now separately recomputed above. |
| Earlier implicit/grouped controls | `historical_evidence/results/innovation_v5/`, `innovation_v6/` | Saved evidence, not newly rerun grids. |

The default entry point checks 30 earlier, 20 strengthening and 11 focused result CSVs. Saved R
inputs and the refit witness are not counted as recomputed output tables.

Forecast inventory: 124 original one-step archives, 16 development multi-horizon
archives, 16 traffic/exchange-rate archives and eight new weather/air-quality archives.
The four processed new-source series are data inputs, separate from these 164 forecasts.
The earlier 64-archive manifest remains valid; the extension manifest records added
archives, source hashes, retained fields and exact equality after minimal repacking.

`scripts/audit_strengthening_v9.py` checks source and protocol hashes, target clocks,
missing masks and causal context filling, every validation-selected setting, one-step
and zero-rate state identities, log-contrast identities, subgroup recombination and
interval-score components. Deterministic tests also check delayed causality with
missing labels and nonnegative projection. These checks do not establish a stochastic
coverage theorem or broad external generalization.

The new-source protocol was timestamped and hashed locally before acquisition. It
is not an external preregistration. Data sources are new to this calibration study;
eight dependent pairs do not constitute eight independent datasets. New analysis
directories preserve historical identifiers rather than implying publication versions.

`verification/` reports the actual local clean-directory replay and PDF image review.
Neural generation, cached-scorecast calibration replay, full R refitting, historical
input tables and independent external replication are distinct stages.
