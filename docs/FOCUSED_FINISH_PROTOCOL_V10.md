# Focused post-review analysis, 2 October 2026

All analyses here revisit observed sources and are exploratory. They do not extend
the earlier locked new-source confirmation. No new dataset or algorithm is added.
The original protocol, manuscript and handoff remain unchanged.

1. Reuse the full new-source grid. For each score and state compare positive rates
   at W=450; positive rates with W in {200,450,900}; and the previously locked grid
   also admitting zero. Use the original full validation selector, with rate then
   window tie-breaking. Retain fixed W=200 and W=900 positive-rate results too.
   Report paired state gains, selected rates/windows and test scores. Mark pairs
   that select the same zero-rate/window method separately from distinct settings.
   No new test-driven search extension. Very small positive rates need not imply
   practically substantial adaptation; report them, do not redefine them as active.

2. Recompute trajectories for the previously selected CQR shared, issued and rolling
   configurations at h=1,6,24, all eight new-source pairs. Report whole-test and three
   consecutive 592-origin blocks: [1200,1792), [1792,2384), [2384,2976).
   Missing outcomes remain excluded. Local errors use 150-origin windows entirely
   within the reported block. Use paired score gains and local-error differences,
   not ratios of averages. Show every pair and each block, with no selection from
   test results. At h=1 shared and issued are identical: any advantage over rolling
   is an adaptive-level effect, not a state-topology effect. These diagnostics do
   not establish a cause of drift or statistical equivalence.

3. Add full author AcMCP-CSS with history length 450 for the same four complete
   weather archives, all horizons 1..24. Retain unmodified pinned R implementation,
   the same KI from the first 600 revealed error rows, Csat=.5 and original rate
   grid {0.001,0.005,0.01,0.05,0.1} plus these/h. The previously completed history
   600 run is the other candidate. This is a finite configuration sensitivity,
   not exhaustive optimization. The author's ncal couples history and startup;
   there is no undocumented split of that parameter.

   To avoid selecting one configuration while its long-horizon scorecaster is still
   starting up, use common validation [900+h-1,1200-h+1) and unchanged test [1200,N).
   Check that scorecasts are finite throughout this validation and test at h=1,6,24.
   This changes validation for all methods in this additional analysis only.
   CQR receives common windows {450,600}, the earlier zero/small positive rates,
   and the same common validation block. Compare fixed history 600 as well as
   jointly selected history. Rolling selects within the same two histories.
   The no-scorecast PID control uses the same histories and rate grid. Native CQR
   and signed-error methods still differ in information and construction.
   Cached interval replay must match all 96 new original R trajectories at .01;
   independently refit 60 deterministic sampled scorecasts as in the prior audit.
   No additional window or tuning expansion after observing these results.

4. Organize the main paper around native information, incremental adaptation, and
   remaining reliability deficits. Target 16--18 main-text/declaration pages with
   the existing type size; move grids, stage history, derivations and auxiliary
   results to explicit appendices. Keep all negative findings and limits accessible.
   Replace repeated caveats with one clear comparison scope and limitations section.
   No acceptance-probability claim follows from length reduction.

5. Preserve the 50-table forecast-based replay, add the new analyses and direct
   numeric checks for the final abstract/core tables, and replay a fresh package.
   Clearly distinguish supplied neural forecasts, cached R scorecasts and optional
   full R refitting. Prepare a new ZIP; do not upload or submit on the author's behalf.
