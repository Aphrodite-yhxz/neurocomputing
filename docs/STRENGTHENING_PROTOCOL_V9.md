# Locked strengthening protocol, 30 September 2026

This file is fixed before downloading the new numeric data, generating predictions,
or inspecting calibration results. A local timestamp/hash record accompanies it.
It is not an external preregistration. Old results remain exploratory development.

## Scientific question and falsifiable contrasts

The contribution under test is the interaction of score construction, delayed update
state and tuning support, not invention of CQR, ACI, interleaving, or support projection.
For each matched series/model/horizon, measure the issued/shared log score ratio
separately for residual and CQR scores. Report their difference and its change from
the original grid to expanded grids. No claim of universal state superiority is planned.
If attenuation fails on either new source, report that boundary rather than redefining
the hypothesis. Also report direct matched comparisons with fixed-level rolling CQR.
Small differences are not statistical equivalence. Series/model pairs are dependent.

## New data, chosen from metadata only

Two sources not used in any project calibration analysis:

1. MPI-BGC Jena Beutenberg roof weather, first 3,600 hourly observations starting
   2026-01-01. Temperature and air pressure, identified by their documented column
   names (T in degrees C and p in mbar). Download the current 2026 `mpi_roof.zip` from
   https://weather.bgc-jena.mpg.de/weather_data.html and archive its exact hash.
   Aggregate ten-minute records by hour using only records in that hour, label by
   the hour's end; a forecast context includes only completed hours. Sentinel -9999
   is missing. Do not choose another variable on the basis of a result.
2. UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G, PM2.5 at Aotizhongxin and
   Dingling. Use first 3,600 hours starting 2013-03-01 from the published hourly files.
   Download https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip.

Keep the hourly grid. Context-only missing observations are forward filled using
past values (initial missing contexts set to zero until the first observation).
Missing targets are never scored, calibrated or used as controller error feedback.
Quantile windows contain the last W revealed **observed** scores. Weather hours
with any missing ten-minute reading are treated as missing hourly targets. Report
counts. If schema or access fails, stop that source and record the failure; do not
select a replacement based on model performance.

The four series and two models produce eight pairs, not eight independent datasets.
The weather dates provide a temporal separation from the original benchmarks; any
stronger claim about training exclusion requires separately verified checkpoint dates.

## Frozen predictions and chronology

Use the same cached amazon/chronos-bolt-base and amazon/chronos-2 checkpoint snapshots
as the original confirmation. Context 512, origins 600 through N-25 inclusive,
all horizons 1..24, batch 128, bfloat16, stored quantiles .1,.3,.5,.7,.9.
Retain raw targets, filled contexts, observation masks, issue and target indices,
checkpoint file hashes, environment and generator hash.
Primary comparisons use h=1,6,24; alpha=.1; prefix=600 forecast rows.
Validation: forecast rows [600+h-1,1200-h+1); test: [1200,N_forecasts).
Only target outcomes revealed by the cutoff can select a configuration.
Early/late validation sensitivity uses [600+h-1,900-h+1) and [900,1200-h+1).
All three selectors use the same test segment. The full validation interval is primary.

## Nested tuning controls

Shared and issued ACI, each with residual and native-band CQR scores, use identical
causal inputs and clipping [0.001,0.999]. Maintain each issued state's phase even
when its delayed outcome is missing; missing feedback produces no increment.

- Original: gamma {.005,.02,.05}, W=450.
- Wide: {.0001,.0002,.0005,.001,.002,.005,.01,.02,.05,.1} plus original/h, W=450.
- Zero/small: Wide plus {0,.000001,.000005,.00001,.00005}, W=450.
- Window: Zero/small crossed with W={200,450,900}.

Select minimum validation interval score; ties prefer smaller gamma then smaller W.
Report boundary selections, zero selections, selected windows, early/late selection
agreement and all configurations. Rolling uses gamma=0; W=450 and validation-selected
W are distinct controls. No rate/window extension after test inspection.
Report interval score, coverage, width, lower/upper misses and local absolute coverage
error in 150 calendar-origin windows (observed targets only). Display per-source and
per-series/model results without treating dependent pairs as independent replications.

## Faithful multi-step comparator

Run the pinned author AcMCP R implementation on the **two complete weather series**
and both models, at all horizons 1..24, if the weather targets are complete. This
predefined subset avoids substituting an undocumented missing-data modification.
If either series has missing targets, report that AcMCP subset as ineligible rather
than filling evaluation labels or silently changing the algorithm.
Use full MA(h-1) plus cross-horizon regression scorecasting, ncal=600, rolling=TRUE,
Csat=.5 and KI from the first 600 revealed target-clock error rows. Use the author's
supported CSS fitting option for this full high-horizon benchmark; identify it as
AcMCP-CSS, not default CSS-ML. Select lr from {.001,.005,.01,.05,.1} and these/h on
the same validation segment. The scorecaster is independent of lr; it may be cached
once and replayed only after endpoints agree with the unmodified author's recursion
at lr=.01. Retain successful/failed scorecast counts and all failures. This is a full
configured algorithm on a declared subset, not a claim of optimizing every AcMCP
setting or implementing O2CP. No-scorecast PID is an additional matched ablation.

## PV explanation controls on old data (explicitly exploratory)

Use all original 32 one-step PV archives. Compare residual, native-band CQR and signed
residual asymmetric intervals under the same shared-state level controller. The signed
construction uses mu+Q(a/2) and mu+Q(1-a/2) of revealed signed errors, with one total
miscoverage state and the same clipping/window/timing; it is an established construction.
For each construction evaluate raw intervals and known-support projection
[max(0,lower),max(0,upper)]. Projection is output-only; keep the original feedback
trajectory to isolate geometric effects. Report any changed coverage at y=0, upper<0.
Select each construction/projection with the same original gamma x W grid
{.005,.02,.05} x {200,450,900}, with gamma=0 rolling also available as a separate control.
Report both fixed default (.02,450) and individually validation-selected results.
Subgroups: all, actual zero/positive (diagnostic only), and low/high point forecast
using the median of the first 600 point forecasts as a fixed observable threshold.
For each group report sample share, score, width and tail-penalty components, coverage.
Quantify how much apparent native-band benefit remains after both methods use support
projection and after a signed residual comparator is admitted. Do not assume it remains.

## Reproduction and reporting

New code gets deterministic arithmetic, delayed causality, gamma=0 identity, missing
target exclusion and support-projection checks. Preserve old manuscripts/packages.
Provide new protocol, input hashes, complete grids, paired summaries and figure data.
Recompute the old 9.98% headline from original forecast archives and validation tuning,
not merely by copying a saved summary. Separate neural generation from CPU calibration.
Fix the abstract's ambiguous panel phrase and update Adler et al. to ICLR 2026.
Interpret any new conceptual decomposition as an evaluation contribution; establish
novelty from explicit comparisons with prior papers, not from an invented method name.
