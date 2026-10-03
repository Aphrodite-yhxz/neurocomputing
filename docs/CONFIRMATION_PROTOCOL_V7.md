# V7: new-family confirmation of calibration mechanisms

Fixed 2026-09-23 before downloading or inspecting test metrics for this panel.
This confirms or limits empirical mechanisms. V6 failed its continuation gate;
there is no validated new algorithm to announce.

## Data and independence

Author benchmark repository: https://github.com/laiguokun/multivariate-time-series-data .
Use traffic columns [0,20,40,60] and exchange_rate columns [0,2,4,6], zero-based.
Use the first min(8000,available_length) observations, without outcome-dependent
selection. Abort on nonfinite values rather than introducing unrecorded imputation.
Do not interpret zero road occupancy as missingness without source evidence.
The README identifies hourly road occupancy and daily exchange-rate observations;
it contains inconsistent traffic duration wording, so do not infer precise dates.
Archive source commit and blob/file hashes. No repository-wide license is listed;
retain the author's research-use description and do not add raw files to public ZIPs.

These eight series and two data families were absent from our development panels.
Columns within a family are dependent; two model runs are not independent datasets.
The data may have appeared in TSFM pretraining. This is new to our calibration
development, not a guarantee of neural pretraining exclusion.

## Frozen forecast and calibration settings

Frozen Chronos-Bolt and Chronos-2, context512, batch128, bfloat16, all h=1..24.
Origins begin at index600; no global rescaling, no fine-tuning. Save full trajectories
and all target indices. Evaluate calibration at h=1,6,24, nominal coverage90%.
The first600 forecast targets form the initial calibration prefix, and issue row1200
begins testing. Validation uses only labels revealed by the first test issue time,
exactly as V4. No test result may change the panel, method grid or scoring rule.

For both absolute residual and native CQR scores, compare:

1. Rolling quantile, window450.
2. Shared and issued ACI, gamma in {0.005,0.02,0.05}, window450.
3. PI-log and SF-OGD with the existing audited fixed defaults (identified as defaults).
4. COP with window300 and union {0.05,0.1,0.5,1} plus those rates divided by h.
5. Grouped ACI with both memory and state partitioned by the prefix-median native
   width, gamma in {0.005,0.02,0.05}, window450.
6. Fixed-prefix-scale OGD and delay-scaled implicit COP, gamma union
   {0.005,0.02,0.05,0.1} plus those rates divided by h, window300.

Select a configuration within each method by validation interval score, deterministic
method-name tie breaking. Retain the full grid. AcMCP is still an execution/cost
study; do not label these methods a complete state-of-the-art comparison.

## Locked contrasts and reporting

A. Native CQR versus absolute residual under each matched controller: paired relative
interval-score gain, coverage gap, and family-specific results.
B. Issued versus shared ACI at each horizon: test whether the long-delay effect seen
in energy data generalizes; do not assume that it does.
C. Grouped versus ordinary shared ACI: report proper score and both forecast-width
group coverages. These groups are observable at issue time; no true-target routing.
D. Implicit-delay versus matched fixed-scale OGD: check whether the tiny incremental
gain on the development panel persists. A positive result does not erase V6's failure.

Primary summaries weight each of the two data families equally, then series equally,
then models equally. Report pair-level outputs, medians and sign counts. Do not
assign independent-dataset confidence to 16 model--series pairs, or tune around
negative findings. Additional analyses after viewing this test are exploratory.
