# V8: direct challenges to the mechanism manuscript

Fixed locally before the following additional computations, 23 September 2026.
Motivation: the user's review of the 31-page mechanism draft. These are post-review
sensitivity analyses on already inspected panels, NOT new locked confirmation tests.
Preserve the original V7 protocol and results, and the previous manuscript/package.

## 1. State topology and learning-rate adequacy (first priority)

Run both shared and issued ACI with residual and native CQR scores. Keep prefix600,
window450, alpha0.1, clipping[0.001,0.999], target timing and validation/test masks
unchanged. Run original development archives first, then revisit V7 without changing
the grid in response to either panel's test results.

For h in {1,6,24}, compute the union of:
- G0={0.005,0.02,0.05};
- G0/h;
- {0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1}.

Report three separately validation-selected comparisons: original G0, scaled union
G0 + G0/h, and the complete wide grid. Ties use the existing method-name ordering.
Both state structures get every candidate. Retain every validation and test result,
selected rate, boundary selections, coverage, local error and interval score.
Do not expand the grid again after viewing test performance. For h=1 the two
structures and their selections must coincide. No test-optimal selector is reported.

For errors issued during the test segment and subsequently revealed before the
archive ends, decompose excess miscoverage exactly using the potential identity:
clip term=sum(projection corrections)/(gamma*K);
endpoint term=-(V_end-V_before)/(gamma*K).
Also report full issued-test miscoverage, pending count, and the pending difference.
The sum must match observed revealed-test excess miscoverage within numerical tolerance.

Primary question: how much of the original h24 CQR issued/shared gain survives
adequate smaller-rate controls? Report reduction or reversal without redefining success.
Show learning-rate/horizon performance surfaces with a common within-pair denominator;
these describe sensitivity and must not be mistaken for test-based tuning.

## 2. Controlled dynamical illustrations

Use a known response M_t(a)=clip(alpha+g*(a-a_star(t)),0,1), g in {1,4},
h in {1,6,24}. Show deterministic mean feedback and Bernoulli feedback separately.
For stationary runs compare gain/boundary ratios {0.5,1.2}, starting a little away
from equilibrium. For tracking runs use a_star=.1 then .2 then .06, with fixed
gamma in {.005,.02,.05} and their /h versions. Use 12,000 steps; changes at4000,8000;
20 Bernoulli seeds0..19. These are controlled controller simulations with a prescribed
response, not a forecast benchmark or proof that real-data loss is caused by instability.
Report oscillation and tracking error. Wider stability range need not mean faster tracking.

## 3. Original photovoltaic efficiency decomposition

Use all eight original main-panel PV series and all four models (32 pairs), h1,
default residual-ACI versus default native-CQR-ACI, reproducing the original 35.25%
headline before adding diagnostics. Also report each method's previously validation-
selected ACI settings from the original tuned comparison, without re-selecting.
For all / y==0 / y>0, report fraction, coverage, interval score, width and lower/upper
miss-distance penalties at alpha=.1. Outcomes define evaluation-only groups.
Verify weighted component reconstruction separately for every pair. Decompose each
pair's overall relative gain into zero/positive contributions using that pair's
overall residual score as denominator; conditional percentage gains are separate.
Do not compare this h1 panel numerically as though it were the h24 two-series audit.

## 4. Nearest multi-step control and manuscript consolidation

Preserve the full AcMCP default CSS-ML fitting semantics if that baseline is accelerated.
Its scorecasts are independent of controller rate, so caching may avoid repeated fits;
verify cached output against unmodified author execution before using it as evidence.
An author-source multi-step PID comparison is also relevant. Any chosen scorecaster,
rate grid, prefix normalization and interval repair must be written in a separate
addendum before its performance is inspected. Do not call a no-scorecast control full
AcMCP or full PID. Full O2CP is not required to claim the scoped empirical mechanisms.

Add interval-score ratios to the unified controller reliability table, normalized per
pair by an explicitly named common baseline to avoid pooling physical units.
Refocus the manuscript on native information/state interaction and deployment reliability.
Move the implicit candidate and wrapper-error/timing history into supplementary technical
audit material. Reproduction artifacts remain local until an explicitly authorized release.
