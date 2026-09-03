# Pre-freeze synthetic sensitivity report

**Status:** synthetic design analysis only; exploratory, not mixer-performance
evidence, not a benchmark freeze, and not authorization to begin Stage 2.

## Question and methods

This checkpoint asks whether candidate family totals and analysis procedures could
support the claim that structure improves the feasible-range normalized expectation
gap against both equal-budget baselines in at least two of three structured families.
The unit is one independent hypothetical QUBO instance, represented only by a
synthetic paired difference. No QUBO, manifest record, blind/transfer outcome, or
mixer experiment was generated or accessed.

The run used seed `20260903`, 100 Monte Carlo repetitions per grid point, 199
stratified bootstrap resamples, a target Monte Carlo standard error of `0.05`,
random-baseline standard deviation `0.003`, and a 300-second deterministic cap.
All 96 grid points completed in 66.9 seconds on the development container. The
largest reported binomial Monte Carlo standard error is `0.05`; differences of a
few percentage points are therefore not resolved and require a higher-precision run.

“Sign simultaneous” applies exact sign inference and `alpha/6` one-sided order-
statistic bounds to all six comparisons. “Sign hierarchical” takes the larger
p-value within each family, applies Holm across the three family hypotheses, and
also requires conservative `alpha/3` family bounds. Bootstrap analogues use the
same logic with resampling within four cells. Wilcoxon uses a tie-adjusted normal
signed-rank approximation only as a deliberately secondary sensitivity analysis;
it is not a test of the median under skewness.

## Calibration summary

The table reports the worst observed value over sizes and distribution scenarios.
“Coverage” is the lowest observed simultaneous family lower-bound coverage. These
are Monte Carlo estimates, not guarantees.

| Procedure | Maximum global-null FWER | Maximum mixed-config false-family rate | Minimum lower-bound coverage | Checkpoint interpretation |
| --- | ---: | ---: | ---: | --- |
| Sign, six-comparison simultaneous | 0.04 | 0.02 | 0.93 | Best-calibrated candidate, but worst observed coverage is below 0.95 at this precision |
| Sign, hierarchical IUT-Holm | 0.10 | 0.03 | 0.96 | Coverage promising; observed global-null maximum requires higher-precision diagnosis |
| Stratified bootstrap, simultaneous | 0.05 | 0.01 | 0.87 | Coverage failure; do not approve as implemented |
| Stratified bootstrap, hierarchical IUT-Holm | 0.12 | 0.05 | 0.80 | FWER/coverage failure; do not approve as implemented |
| Wilcoxon hierarchical IUT-Holm, secondary | 0.60 | 0.46 | Not applicable | Invalid in tie/skew stress cases; confirms it must not be primary |

The global-null maximum is selected across 24 null cells, so a single 100-run cell
can look high by Monte Carlo noise. Nevertheless, pre-freeze approval requires
positive evidence of calibration, not an assumption that an unfavorable result is
noise. The bootstrap particularly under-covered in discrete/failure regimes; its
nominal p-values and quantile bounds are therefore not approved. Tests and interval
criteria were evaluated jointly: a family counted only when both corrected tests
rejected and compatible lower bounds exceeded zero.

## Power and sample-size sensitivity

Worst-case power across the six distributions is intentionally stringent. It is
the probability that at least two families, each against both baselines, pass both
the test and confidence-bound criteria while a third family remains null.

| Effect | Total 32 | Total 48 | Total 64 | Total 96 | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.005 | 0.00 | 0.00 | 0.00 | 0.00 | No candidate total is robust to ties/failures at this small effect |
| 0.010 | 0.02 | 0.07 | 0.14 | 0.39 | Simultaneous sign procedure remains underpowered in the worst case |
| 0.015 | 0.21 | 0.58 | 0.86 | 1.00 | Totals 64–96 are promising only for this larger effect |

These values use the best-calibrated six-comparison sign candidate. For Gaussian
differences alone its power at effects `0.010`/`0.015` was respectively `0.15/0.80`
at 32, `0.52/0.98` at 48, `0.80/1.00` at 64, and `0.98/1.00` at 96. Distributional
assumptions therefore dominate any single count recommendation. The hierarchical
sign procedure was generally more powerful, but its global-null audit is unresolved.

## Null negative-control diagnostic

The simulator evaluates the null separately against a candidate practical margin
of `0.005`, requiring both baseline comparisons to clear corrected sign bounds.
This diagnostic cannot count toward the two-family gate. The margin and the
optimizer-failure assignment remain candidates, not frozen decisions. Rows named
`null_sign_candidate_margin_0.005` in the JSON expose the full scenario-by-size
operating characteristics for human review.

## Plain-language decision table

| Human decision | Evidence from this run | Recommendation now |
| --- | --- | --- |
| Freeze a family total? | No total combines demonstrated calibration and robust power over all stress scenarios | **No—do not write final counts** |
| Use sign/order-statistic inference? | Strongest calibration; exact and distribution-free, with conservative ties | Retain as leading candidate; rerun calibration at much smaller MCSE |
| Use stratified percentile bootstrap? | Material under-coverage, despite preserving four-cell counts | Reject this implementation or replace it and revalidate coverage |
| Use Wilcoxon-Holm as primary? | Severe false-positive behavior in asymmetric/tie cases; it targets a signed-rank functional, not a general median | No; secondary only if symmetry is prospectively justified |
| Prefer hierarchical IUT-Holm? | More power, but a global-null cell reached FWER 0.10 | Do not approve until a higher-precision audit resolves calibration |
| Freeze null margin `0.005`? | Tooling supports it, but there is no scientific utility elicitation yet | Keep provisional |
| Begin Checkpoint 2 or Stage 2? | Checkpoint 1 has unresolved method/count choices | Stop for human scientific review |

## Limitations and next decision gate

- Synthetic scales and failure mechanisms are design stress tests, not estimates
  from benchmark performance. They cannot establish that any real effect exists.
- One hundred repetitions give coarse error estimates and maxima across many cells
  are noisy. A reviewed procedure should be rerun with enough repetitions for an
  MCSE near `0.005` before freezing.
- Percentile bootstrap intervals are not repaired merely by adding resamples; an
  alternative calibrated stratified construction may be required.
- Baseline errors share an instance component in the simulator, but the dependence
  and random-baseline variance are assumptions.
- The ITT assignment of zero improvement is conservative for a favorable claim but
  must match an operational optimizer-failure definition before use.

The next decision is human: choose the procedures worth recalibrating, define
acceptable FWER/coverage/power thresholds and the null margin, and authorize a
higher-precision synthetic run. No benchmark count is selected by this report.
