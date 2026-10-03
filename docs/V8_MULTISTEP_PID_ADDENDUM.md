# V8 nearest multi-step PID control (fixed before computing its performance)

Use the asymmetric multi-step PID recursion in the pinned Wang/Hyndman author
repository, with every forecast/target time conversion checked against unmodified R
`pid()`. This is the supplied multi-step PID family, not full AcMCP. A Python
implementation may accelerate only the deterministic algebra; source equivalence
must be checked before computing benchmark performance.

Input: signed point-forecast errors, all 16 development and all 16 V7-revisit pairs,
h={1,6,24}. Use all test origins, the existing validation mask, alpha=.1,
rolling ncal=600, Csat=.5, KI=max absolute errors revealed in the first600 target-clock
rows across horizons1..24. KI is fixed from past data, not the full trace.
Preserve the author's zero internal initialization, pre-emission warmup updates,
separate alpha/2 tail feedback, error-range step scale (including newest error),
and unbounded tangent integrator. Do not silently replace the integrator with PI-log.

Rate grid: union {0.005,0.01,0.05,0.1} and these values/h. Scorecasters:
none (PI control), rolling mean, naive last error (author's documented example),
and an explicitly specified least-squares AR(1). The last fits x_t=c+phi*x_(t-1)
on the same600-error window, clips phi to[-.995,.995], recomputes c=ybar-phi*xbar,
and iterates h steps. This custom scorecaster is not claimed to be the author's
default or their best configuration. Validation selects rate and scorecaster jointly;
also report the no-scorecaster ablation. Save each candidate and selected configuration.

The exact author recursion tracks its two tail scores even if endpoints cross.
Preserve that feedback internally for source fidelity. For evaluation only, collapse
crossed endpoints to their midpoint and report their frequency. These are declared
practical repaired intervals, not a claim of validity for the author theorem.
Infinities are retained and reported, never removed from a favorable average.

Report paired score ratios, coverage, local error and interval repairs beside COP
and both extended-grid ACI states on the same test origins. This completes a scoped
multi-step PID comparison, not the missing full AcMCP scorecaster comparison.
