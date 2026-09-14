# Pre-freeze synthetic sensitivity report

**Status:** corrected synthetic smoke test only; not mixer-performance evidence,
not a benchmark freeze, and not authorization to begin Stage 2.

## Scope and smoke configuration

This checkpoint tests statistical-analysis plumbing using synthetic paired
normalized-gap differences. It generated no QUBO, loaded no benchmark or blind/
transfer record, and ran no mixer optimization. One hypothetical independent
instance remains the experimental unit. Positive differences favor structure.

The checked-in smoke result used seed `20260903`, eight Monte Carlo repetitions per
grid point, 199 stratified bootstrap resamples, target MCSE `0.20`, expected-random
estimator-error SD `0.003`, and a 180-second wall-clock safety cap. All 336 points
completed in 18.3 seconds in the development container. The 336 points cross four
candidate totals, four effects, seven scenarios, and low/medium/high dispersion.
With only eight repetitions, the achieved worst-case precision is approximately `0.177`;
the smoke estimates must **not** select sample sizes or analysis procedures.
Here `0.177` is the conservative worst-case Bernoulli precision bound
`0.5/sqrt(8)`, not a plug-in estimate based on an observed rate.

## Corrected data model

For a structured family, the latent per-instance improvement is the requested
effect plus instance-level noise. The ring observation equals that latent value.
Only the expected-random observation receives additional independent Gaussian
error, representing uncertainty from finitely many random-connected baseline
replicates. That measurement error has configured SD `random_baseline_sd`; it is
not part of the latent expected-random median estimand. Both target medians are the
requested effect before a failure intervention. Under skewness, the simulator
still audits inference for the declared latent-median target and exposes any
distortion introduced by baseline-estimation error.

The null control is a fourth, independently seeded effect-zero synthetic family.
It cannot satisfy the two-of-three structured gate. Its two comparisons alone are
tested against the provisional practical margin `0.005`; changing structured
effects does not change its `(0, 0)` population target. It uses the same underlying
distribution and dispersion as the associated grid cell, but structured-method
failure replacements are disabled for the null. Thus even the
`structure_failure` stress retains null targets `(0, 0)`.

The `optimizer_failure` scenario replaces both comparisons with zero on failed
instances. This is a **neutral-value sensitivity convention**, not intention-to-
treat and not inherently conservative. The additional `structure_failure`
scenario assigns a prespecified negative improvement (`-0.40`) and is the adverse,
conservative stress test for a favorable structure claim. Neither convention is
approved for real evaluation.

## Procedures and compatible success criteria

- The exact sign/order-statistic candidate counts ties as non-positive.
- Six-comparison simultaneous inference applies `alpha/6` to all three-family by
  two-baseline comparisons.
- Hierarchical inference forms an intersection-union family p-value from the worse
  baseline comparison and then applies Holm across three structured families;
  conservative `alpha/3` family bounds must also clear zero.
- Percentile-bootstrap p-values and lower quantiles must agree before rejection.
  The bootstrap remains **unapproved** because percentile median inference has not
  demonstrated joint coverage in discrete, skewed, or failure cases.
- Wilcoxon-Holm remains secondary and is not interpreted as a general median test.

Coverage is now a repetition-level event. The simultaneous procedure counts a
repetition covered only if all six lower bounds cover their six targets. The
hierarchical procedure first takes each family's worse (minimum) lower bound and
then requires all three family bounds to cover. Coverage is no longer an average
of marginal family indicators.

Adaptive stopping monitors every decision-relevant Bernoulli outcome: overall
success, false rejection/FWER, joint coverage, and the independent null-margin
diagnostic. It uses `0.5/sqrt(repetitions)` for all of them, so an observed rate of
zero or one cannot imply zero Monte Carlo uncertainty. Each machine-readable row
records this achieved precision and whether stopping occurred because of the
precision target, maximum repetitions, or runtime cap. Per-estimate MCSE fields
also report this conservative bound rather than a zero-at-endpoints plug-in value.

## Smoke diagnostics—not calibration results

Across the 336 extremely small smoke cells, observed maxima/minima were:

| Procedure | Max global-null false-family rate | Max mixed false-family rate | Min joint coverage | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Sign simultaneous | 0.125 | 0.125 | 0.750 | Plumbing exercised; precision is inadequate |
| Sign hierarchical IUT-Holm | 0.250 | 0.125 | 0.750 | Plumbing exercised; precision is inadequate |
| Bootstrap simultaneous | 0.125 | 0.125 | 0.625 | Unapproved; prior under-coverage concern remains |
| Bootstrap hierarchical IUT-Holm | 0.250 | 0.250 | 0.750 | Unapproved; prior under-coverage concern remains |
| Wilcoxon hierarchical, secondary | 0.875 | 0.625 | Not applicable | Sensitivity only; not a median procedure |

Values move in increments of `0.125`, and selecting extrema across hundreds of
cells compounds Monte Carlo noise. These numbers verify serialization and branch
logic only; they neither establish nor refute nominal FWER or coverage. The null
margin diagnostic produced no positive declarations in this smoke run, also with
insufficient precision for a scientific conclusion.

## Plain-language decision table

| Human decision | Corrected evidence | Decision now |
| --- | --- | --- |
| Freeze a family total? | Smoke run has only eight repetitions per point | **No** |
| Approve sign/order-statistic inference? | Logic and joint coverage accounting are corrected | Retain for high-precision validation |
| Approve percentile bootstrap? | Test/bound compatibility is enforced, but coverage remains unvalidated | **No; unapproved candidate** |
| Use Wilcoxon-Holm as primary? | It is not a general median test under skewness | **No; secondary only** |
| Freeze null margin `0.005`? | Independent diagnostic exists, but no utility elicitation exists | Keep provisional |
| Freeze a failure rule? | Neutral and adverse conventions are stress tests only | **No** |
| Begin Checkpoint 2 or Stage 2? | Calibration and scientific thresholds remain unresolved | **No; stop for review** |

## Unresolved statistical decisions

1. Select acceptable FWER, joint-coverage, power, and MCSE thresholds before a
   higher-precision synthetic run.
2. Decide whether expected-random replicate counts make SD `0.003` plausible, or
   specify a variance model tied to the number of random graphs.
3. Replace or reject percentile bootstrap inference after adequate joint-coverage
   validation; increasing resample count alone does not repair poor coverage.
4. Define optimizer failure operationally and freeze a real-data retention/
   imputation rule; neither synthetic replacement is automatically appropriate.
5. Decide whether the candidate null margin `0.005` represents meaningful utility.

No candidate total or final procedure is selected. Benchmark v1 remains
`DRAFT_UNFROZEN`, and a human must authorize any higher-precision calibration.
