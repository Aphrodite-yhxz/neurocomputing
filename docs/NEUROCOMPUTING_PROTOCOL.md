# Revised evaluation protocol v1 — 2026-09-22

Written before running the revised methods. This is a local analysis plan, not an
external preregistration. Existing archives and exploratory results have already been
examined; all reused series are retrospective evidence, not untouched test datasets.

## Research questions

1. With a shared causal protocol, what does the native band contribute beyond residuals?
2. How do score choice and update rule interact? Avoid an unqualified universal ranking.
3. Does a prefix-validated choice of additive versus scale-normalized calibration avoid
   losses when native bands collapse or have poor shape? This is an empirical recipe,
   not a claim to invent conformal score selection.

## Archives and provenance

Main: all 124 archived pairs (31 series x 4 backbones). Geographic sensitivity: all 39
GEFCom pairs. Dense: all 93 available files, checking actual quantile levels; do not
pretend .05/.95 exist for a model exposing only .1/.9. Multi-horizon: all 16 files with
h=1,6,24. Report horizons in steps and distinguish 10-minute solar from hourly load.
No reruns selected according to the sign of a test result.

## Common causal evaluation

First 600 forecast/outcome pairs initialize calibration; next 600 are a validation
prefix; evaluate every method on exactly index 1200 onward. Reuse outcomes only when
observable (delay h for h-step forecasts). Static split methods fit the first 600;
adaptive methods start from the same 600 and may update thereafter. The validation
prefix selects hyperparameters without test outcomes. Preserve method states across
the validation/test boundary. No full-test normalization, error clipping, or silently
dropped failures. Dense/main forecasts are separate panels unless numerical equality
of shared quantiles is established. Check target/time alignment across backbones.

## Scores and interval construction

- residual: absolute error around the archived point prediction.
- native-scale residual: absolute error divided by native interval width with a
  past-only relative floor; no score cap.
- CQR: signed native-band violation, allowing negative margins.
- regularized normalized CQR: CQR score divided by a native width regularized by
  recent absolute forecast errors. Regularization grid {0, .01, .1, .5, 1} selected
  on validation only. All values and failures retained.
- residual-scale residual: a strong control using past residual scale only; checks
  whether native outputs add value beyond ordinary local error normalization.

Sort quantiles per forecast row to repair crossing, recording how often this changes
endpoints; report a no-rearrangement sensitivity. If inversion gives lower > upper,
record an empty-set event and collapse to the center for interval-score evaluation;
use the actual emitted closed interval for feedback, and disclose that this is a
practical repair rather than the exact signed-score inversion. Report the frequency.

## Update rules

Split fixed, rolling fixed-level, clipped ACI, scale-free OGD, and reference PID/SAOCP
where verified against author code. Approximations in the historical module cannot
be labelled as reference methods. Rule sweeps use score-compatible units and a common
validation loss. Report clipped ACI as empirical: do not assign unclipped guarantees.
Full support/infinite intervals, if encountered in reference variants, must be counted,
not silently truncated to improve interval scores.

## Selection and controls

Default ACI gamma=.02, window=450; sensitivity gamma {.005,.02,.05}, window {200,450,900}.
Primary comparison uses fixed defaults; validation-selected versions reported separately.
Selector chooses the lowest validation mean interval score among the declared score
recipes; no test-label access. Compare against each fixed recipe and a retrospectively
best recipe labelled oracle (never used for deployment). If selector does not help,
report its failure rather than expand the grid after seeing test results.

## Metrics and inference

Coverage, tail-specific miss rates, mean width, standard interval score (proper score),
rolling coverage error (150 steps), zero/empty/infinite interval frequency and runtime.
Winkler is compared at the same nominal target; coverage is reported alongside it,
not used to dismiss inconvenient score results. Main target 90%; 80% and 95% sensitivity.
Paired moving-block bootstrap with blocks 200, 1000 resamples; block 100/400 sensitivity.
Compute confidence intervals on per-step differences using the common test mask.
Aggregate relative improvements first by series across backbones, then across series.
Report dataset-family averages and leave-family-out sensitivity; series bootstrap does
not establish geographic independence of correlated PV plants or nested ERCOT loads.

## Deliberately excluded strong claims

No universal distribution-free guarantee for clipped implementation. No proof of
conditional coverage. No claim all models under-cover. No universal absence of useful
internal signals. No proven causal role for heteroscedasticity. No settlement-grade
reserve savings from the old price proxy. No claim of untouched holdout/preregistration.
