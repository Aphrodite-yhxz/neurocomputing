# Amendments and implementation fidelity

Date: 2026-09-22. The original local protocol remains unchanged so result hashes retain
their meaning. It was not an external preregistration. All work uses historical archives.

## Completed declared sensitivities

The main panel was evaluated at miscoverage 0.20, 0.10 and 0.05. Main, GEFCom, dense and
multi-horizon panels use 64 configurations each, including the 3 x 3 ACI grid, an
unrearranged control, regularized scores and validation-selected variants. Block lengths
100/200/400 were all retained for the primary CQR-versus-residual comparison.

## Exploratory additions, after corrected results were available

1. **Fixed-prediction delay intervention.** The multi-horizon findings confound changing
   forecast difficulty with feedback delay. All 16 archived one-step traces from that
   panel are therefore held fixed, and only calibration feedback is withheld by
   1/6/24 steps. This is an intervention on the calibrator's available feedback, not on
   base-model context, and is not a real deployment-delay trial. No hyperparameters were
   selected using its test outcomes.
2. **Midpoint control.** CQR changes centering as well as width allocation. On all 124 main
   pairs, residual and native-scale ACI are also evaluated around the native-band midpoint.
   This was added to avoid attributing a center improvement to heteroscedastic scaling.
3. **No retrospective oracle headline.** The protocol mentioned an oracle comparison.
   Because the selector did not establish a useful new method, no oracle result is used
   as empirical evidence of deployable performance. Complete fixed-recipe results permit
   independent inspection; the absence of an oracle does not affect the primary contrast.

## Fidelity and scope of rule names

- ACI is a clipped practical variant using linearly interpolated empirical quantiles.
- SF-OGD uses a coverage gradient, prefix maximum score scale and signed unprojected
  thresholds. Tie, initialization and projection conventions are explicit adaptations.
- PI-log uses a prefix scale and log-tangent integral correction, without scorecasting.
  It is not full reference conformal PID and is never labelled as such in the new paper.
- SAOCP uses the archived author's expert update and coin-betting weights. The compiled
  positive-score implementation matches an independent driver of the original expert
  class on 1,500 observations to numerical precision. Signed CQR removes projection to
  nonnegative radii, and delayed updates operate in reveal order. Original guarantees
  are not transferred to these changes.
- FACI-R uses author aggregation equations but a rolling 450-score empirical survival
  function, common excluded prefix, signed-score option and delayed feedback. It must
  not be called an exact reproduction of the original full-history forecasting wrapper.
- Singleton repairs are included in the emitted intervals and in empirical metrics.
  Coverage-driven rules use their actual coverage; expert-loss rules retain their
  specified score-space losses. This distinction matters when inversion is empty.

Source URLs, SHA256 hashes and original licenses are in `docs/references/`. Tests verify
the SAOCP expert recursion, not the full theorem or the upstream neural forecaster.

## Unresolved provenance, not hidden failures

The old archives do not include all checkpoint commits, original package versions or
hidden-state arrays. Raw targets were independently validated on all 304 archive/horizon
instances. Model forecasting provenance is described as historical rather than newly
reproduced. Offline inference spot checks subsequently matched 21 origins across three
series for each of the four models, including all stored quantiles. TimesFM matched the
local 2.0 Hugging Face path. Both Chronos models required matching the nominal original
batch grouping (256 contexts); seven-context batches gave some different floating-point
outputs. Moirai matched after retaining its original sorting step. Current checkpoint
hashes and package versions are now recorded. These 84 checks strengthen provenance but
do not replace original run logs or constitute a full reforecast of every archived target.

The dense Chronos-Bolt 0.05/0.95 values coincide with 0.1/0.9 in roughly 99.86% of rows,
and TimesFM has no stored tail levels. Neither supports an exact full-distribution CRPS
claim. The earlier hidden-state, volatility-law and reserve-savings claims are removed.
