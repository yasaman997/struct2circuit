# Pre-freeze design process

**Status:** Checkpoint 1 implemented for synthetic sensitivity review; no benchmark
counts or analysis procedure are approved or frozen.

## Why there are two freezes

The project separates decisions that must be made without performance data:

1. **Design freeze:** lock the estimand, decision rules, multiplicity control,
   failure handling, candidate sample-size rationale, compute budgets, and the
   benchmark-generating distributions. Synthetic sensitivity may inform this
   freeze, but mixer-performance records may not.
2. **Pipeline freeze:** after train/validation-only Stage 2 and later development,
   lock all code, hyperparameters, exclusions, random-baseline replication,
   analysis code, and a cryptographic commitment to externally held blind records.
   Only then may the one-time locked evaluation be authorized.

The draft manifest remains `DRAFT_UNFROZEN`. Neither freeze authorizes opening
blind or transfer records, and changing a frozen item requires a versioned amendment.

## Checkpoint sequence

| Checkpoint | Scientific decision enabled | Permitted evidence | Current state |
| --- | --- | --- | --- |
| 1. Statistical specification and sensitivity | Which candidate counts and procedures have acceptable error, coverage, power, and runtime? | Synthetic paired differences only | **Implemented; awaiting human review** |
| 2. Compute-budget feasibility | Which approved design can be afforded without unequal optimization budgets? | Timing/infrastructure checks on non-blind development inputs | Not started |
| 3. Design freeze review | Lock counts, regimes, estimand, tests, margins, and custody plan | Checkpoints 1–2, with signed amendments | Not started |
| 4. Stage 2 non-AI heuristic study | Does transparent structure contain validation signal under approved controls? | Train/validation only | Not started |
| 5. Pipeline freeze | Lock implementation and the final analysis before release | Development audit, no blind outcomes | Not started |
| 6. One-time evaluation | Evaluate the frozen claim | Externally released blind records | Not authorized |

## Checkpoint 1 statistical contract

- One independently generated QUBO instance is the experimental unit. The primary
  outcome is its feasible-range normalized expectation-gap improvement, baseline
  minus structure, so positive is favorable.
- The estimand is the family-level median paired improvement. A structured family
  succeeds only if structure beats **both** the fixed ring and expected-random
  connected baseline. No validation-selected “strongest” baseline is substituted.
- The overall scientific gate requires success in at least two of three structured
  families. The weak-structure null is separate and can never satisfy that gate.
- The null diagnostic asks whether both comparisons exceed a candidate practical
  margin of `0.005`. This margin is explicitly provisional.
- Primary candidates are an exact conservative sign/order-statistic procedure and
  a four-cell stratified percentile bootstrap whose coverage must be demonstrated.
  Wilcoxon-Holm is secondary unless symmetry is justified before the design freeze.
- Multiplicity candidates are (a) Bonferroni simultaneous bounds/tests across all
  six comparisons and (b) intersection-union family p-values (the worse of two
  comparison p-values), followed by Holm across three structured families.
- A rejection counts only when its corresponding corrected one-sided lower bound
  is also above zero. Hierarchical confidence bounds use the conservative `alpha/3`
  family allocation, so they cannot make a Holm decision more permissive.
- Exact sign inference treats equality as non-positive. The simulator labels a
  zero replacement as a **neutral-value failure sensitivity**, not intention-to-
  treat. A separate conservative stress test assigns failed structure runs a
  negative improvement. An operational failure rule remains unfrozen.

## Sensitivity grid and safeguards

The simulator uses totals `32`, `48`, `64`, and `96`, each balanced over four
size/regime cells; location effects `0`, `0.005`, `0.010`, and `0.015`; and Gaussian,
heavy-tailed, skewed, heterogeneous-regime, rounded-tie, and optimizer-failure
distributions, including neutral and adverse structure-failure variants. Every
case is crossed with explicit low (`0.67`), medium (`1.0`), and high (`1.5`)
dispersion multipliers. Random-baseline uncertainty is a configurable measurement
component applied only to the expected-random comparison.
Positive-effect runs use two alternative families and one null family, directly
auditing mixed configurations as well as the global-null runs.

For each structured family, a latent paired difference equals the requested median
effect plus instance noise. The ring observation is that latent difference. The
expected-random observation adds independent zero-mean estimator error representing
finite random-graph replication; that error is not part of the expected-random
estimand. A fourth, independently seeded effect-zero family supplies the null
diagnostic and never enters the structured success gate.

Monte Carlo work proceeds in batches and stops adaptively after the requested
Monte Carlo standard error or maximum repetitions. A monotonic wall-clock safety
cap is mandatory but is not a deterministic stopping rule across machines. A run
that reaches it is explicitly marked partial, including a valid zero-row result if
the first repetition never starts. Bootstrap draws are vectorized within strata; the implementation does
not nest 10,000 simulations inside 20,000 resamples. Every public stochastic entry
point requires a seed. Completed seeded runs are byte reproducible; serialized
output omits observed wall time and the CLI prints it separately.

## Decision gate

Checkpoint 1 does **not** select counts. A human reviewer must decide which error,
coverage, power, and Monte Carlo-precision thresholds are acceptable, reject or
revise inadequately calibrated methods, decide whether `0.005` is an appropriate
null margin, and authorize a higher-precision run if needed. Until then, the prior
draft counts are placeholders and Checkpoint 2 must not begin.
