# Research protocol: verified structure-conditioned mixers

## Current status

Benchmark v1 remains **`DRAFT_UNFROZEN`**. Stage 1 benchmark foundations and
Checkpoint 1 synthetic sensitivity tooling are implemented. The checked-in
eight-repetition sensitivity run is a smoke test, not mixer-performance evidence
or high-precision calibration. No final sample counts or analysis procedure have
been approved; Stage 2 and blind evaluation have not begun.

The statistical specification below follows the implemented candidates in
[the pre-freeze design](../tasks/PREFREEZE_DESIGN.md) and
[the sensitivity module](../src/struct2circuit/sensitivity.py). See
[the sensitivity report](prefreeze_sensitivity_report.md) for diagnostics and
unresolved decisions. This synchronization does not freeze the benchmark or
authorize further experiments.

## Single central claim

> A structure-conditioned, feasibility-preserving mixer can improve the solution-quality/resource Pareto frontier over fixed and expected-random equal-budget mixers for cardinality-constrained QUBOs.

This is the claim under investigation, not an established result.

This is deliberately narrower than “AI discovers quantum algorithms.” It can be falsified with paired experiments, has a formal invariant, and creates a clean path from a transparent heuristic to a learned circuit generator.

## Scientific object

We study

\[
\min_{x\in\{0,1\}^n} C_Q(x)=x^TQx+c^Tx
\quad\text{subject to}\quad \mathbf{1}^Tx=k.
\]

The Hamming-weight operator is

\[
N=\sum_i (I-Z_i)/2.
\]

Every permitted mixer unitary must satisfy \([U_M,N]=0\). In the pilot, this is guaranteed by representing the state only in the weight-\(k\) basis and using XY exchanges on a connected graph.

## Contribution ladder

1. **Verification contribution:** a typed mixer representation that rejects disconnected or cardinality-violating constructions.
2. **Algorithmic contribution:** a structure-conditioned generator that selects a sparse, connected mixer graph from the QUBO interaction matrix.
3. **Learning contribution:** replace the transparent edge score with a learned policy while retaining the same verifier and edge budget.
4. **Scientific contribution:** discover stable relationships between instance structure and useful mixer motifs that transfer across sizes, families and hardware.
5. **Utility contribution:** establish Pareto improvement after including search amortization, shots, transpilation and strong classical baselines.

Only levels actually supported by evidence may appear in the title or abstract.

## Pilot design

- Family: synthetic block-correlated cardinality QUBOs motivated by index tracking.
- Size: `n=8`, `k=3`, exact feasible dimension `56`.
- Depth: `p=1`.
- Initial state: uniform Dicke state over all feasible decisions.
- Equal-budget comparison: ring mixer versus structure-conditioned connected mixer, each with `n` edges.
- Resource reference: complete mixer with `n(n-1)/2` edges.
- Optimization: identical deterministic grid plus bounded local refinement.
- Primary pilot metric: paired normalized expectation gap, lower is better.
- Secondary: probability of sampling an exact optimum.
- Pilot decision rule: advance only if the median paired gap reduction is positive and its exploratory bootstrap interval excludes zero.

The pilot is used for debugging and effect-size estimation. It is not confirmatory.
Its exploratory bootstrap interval and Wilcoxon result do not approve either
method for confirmatory inference. Passing the pilot rule supports pre-freeze
design and calibration work; it does not bypass the approval gates below.

## Confirmatory design to lock before execution

### Instance families

The three structured families are:

1. Block-correlated portfolio/index-tracking proxy.
2. Weighted densest-`k`-subgraph.
3. Weighted maximum-`k`-vertex-cover.

A fourth, exchangeable weak-structure/null ensemble is a separate negative
control. It never counts toward the two-of-three structured-family success gate.
It probes the limits of a favorable structure claim; it does not prove that all
exploitable instance-level information is absent.

### Splits

- Training: non-AI method development, followed by mixer-policy learning at a later stage.
- Validation: architecture/hyperparameter selection.
- Blind test: locked seeds and distributions; opened once.
- Transfer test: unseen `n`, unseen `k/n`, and unseen hardware topology.

Stage 2 is restricted to train/validation after design approval. Neither freeze
described below authorizes opening blind or transfer records. The current public
draft seeds are provisional and must not become the definitive blind set; final
blind definitions require external custody and a public cryptographic commitment
before the separately authorized one-time evaluation.

### Baselines

The primary equal-budget comparisons are against **both**:

- fixed ring XY-QAOA;
- the expected-random connected XY mixer baseline, estimated from independently
  seeded random-graph replicates at the same edge budget.

Each comparison must match the mixer-edge count and parameter-optimization
budget. The expected-random baseline is an expectation over the declared graph
distribution, not the best sampled random graph. Replicate counts and the
uncertainty model remain subject to design and pipeline review. No
validation-selected "strongest" baseline replaces either primary comparison.

The broader study also plans the following resource references, ablations, and
extended baselines; listing them here does not imply completed evaluations:

- complete XY-QAOA as a higher-resource reference;
- strongest-edge graph without spanning-tree enforcement;
- warm-start QAOA;
- counterdiabatic/commutator-informed constrained QAOA where reproducible;
- random/evolutionary QAS and a predictor-based QAS baseline;
- exact MIQP/CP-SAT on tractable sizes;
- tuned greedy, local search, tabu and simulated annealing.

### Metrics

The primary outcome is the feasible-range normalized expectation-gap improvement
for a paired instance: `baseline gap - structure gap`, so positive favors
structure. The family-level median of these paired improvements is the primary
estimand for each baseline; it is not a difference of separately computed medians.

Additional metrics for the broader study are:

- approximation ratio;
- probability of an optimum and probability within `epsilon` of optimum;
- feasibility probability;
- CVaR of sampled costs;
- circuit evaluations, shots and classical search time;
- two-qubit gate count/depth and routing SWAPs;
- noise robustness and calibration sensitivity;
- zero-shot transfer loss;
- amortized time-to-target including architecture search.

### Statistics

#### Experimental unit and expected-random uncertainty

One independently generated QUBO instance is the experimental unit. Random-graph
replicates estimate that instance's expected-random baseline; they are not
additional independent instances. A structured family succeeds only when both
of its median paired improvements clear the corrected decision rule.

The synthetic model separates latent instance variation from finite-replication
uncertainty. The ring observation equals the latent paired improvement; only the
expected-random observation receives independent, zero-mean estimator error with
SD `random_baseline_sd`. This error is not part of the latent expected-random
median estimand. Under skewness, adding zero-mean error need not preserve the
observed median, so coverage is audited against the declared latent target.

#### Candidate inference and multiplicity

- The exact conservative sign test with an order-statistic lower bound is a
  primary candidate. Equality at the tested threshold counts as non-positive.
- The four-cell stratified percentile bootstrap preserves the size/regime cell
  counts. It remains **unapproved** pending adequate joint-coverage validation,
  including discrete, skewed, and failure scenarios. More resamples alone do not
  establish calibration.
- Wilcoxon-Holm is **secondary only** under the current design; it is not a
  general median test. Any change to that role requires a symmetry justification
  and scientific approval before design freeze.

The implemented primary candidates are assessed under two multiplicity strategies
at configured family-wise level `alpha` (currently `0.05` in the sensitivity code):

| Strategy | Test and lower-bound rule |
| --- | --- |
| Six-comparison simultaneous | Apply `alpha/6` to each of the three-family by two-baseline tests and one-sided lower bounds. Both comparisons must reject and both lower bounds must exceed zero for a family to succeed. |
| Hierarchical intersection-union plus Holm | Use the larger of the two baseline p-values as each family's p-value, then apply Holm across the three structured families. Also require the family's minimum baseline lower bound, computed with `alpha/3` per baseline, to exceed zero. These conservative bounds cannot make a Holm decision more permissive. |

For either primary candidate, a test rejection counts only when its corresponding
corrected lower-bound condition also holds. Bootstrap p-values and lower
quantiles must therefore agree before a success is recorded. Neither candidate
procedure nor multiplicity strategy has been selected for final evaluation.

Coverage is a **joint event per Monte Carlo repetition**. Simultaneous inference
requires all six lower bounds to cover their respective targets. Hierarchical
inference takes each family's minimum baseline lower bound and requires all three
family bounds to cover their respective minimum-of-two-median targets. Averaging
marginal coverage indicators is not the joint-coverage criterion.

#### Separate null diagnostic and failure sensitivities

Each synthetic repetition has a fourth, independently seeded effect-zero family.
Its draw and declared latent targets `(0, 0)` are independent of the structured
effects. It shares the scenario's distributional and dispersion stress but never
receives either structured-method failure replacement. It cannot contribute to
the structured success gate.

The separate null diagnostic asks whether both comparisons exceed the provisional
practical-superiority margin `0.005`, using sign tests and compatible lower bounds
at `alpha/2` per comparison. This margin is not the threshold for G2 and is not
frozen. The synthetic zero targets do not assert zero mixer-performance effects
for the actual weak-structure QUBO ensemble.

The `optimizer_failure` sensitivity replaces both paired improvements on failed
instances with zero. This is a **neutral-value sensitivity convention**, not
intention-to-treat or an inherently conservative rule. The separate
`structure_failure` stress assigns a prespecified negative improvement (default
`-0.40`); the structured latent-median targets account for the failure mixture.
Neither synthetic convention is an approved operational failure policy for real
evaluation.

#### Sensitivity scope and reporting

The synthetic grid evaluates candidate family totals `32`, `48`, `64`, and `96`,
balanced over four size/regime cells; effects `0`, `0.005`, `0.010`, and `0.015`;
and Gaussian, heavy-tailed, skewed, heterogeneous-regime, rounded-tie, neutral
failure, and adverse failure scenarios. Explicit low (`0.67`), medium (`1.0`), and
high (`1.5`) dispersion multipliers are serialized. Global-null runs and mixed
runs with two alternative structured families and one effect-zero structured
family are assessed in addition to the independent negative control.

Report success probability, family-wise false-rejection rate, joint coverage, and
the separate null-margin diagnostic. Monte Carlo precision uses the conservative
Bernoulli bound `0.5 / sqrt(repetitions)` for all four outcomes, including rates
observed at zero or one. Record achieved precision and stopping reason. The
wall-clock safety cap can produce a partial run and is not a deterministic
stopping rule across machines. Completed seeded runs are byte reproducible;
observed wall time is reported separately from serialized results.

The eight-repetition smoke artifact cannot select counts, establish calibration,
or approve an analysis procedure. Continue reporting effect sizes and full
distributions, learning/scaling curves with uncertainty when available, and all
seeds, failed optimizations, exclusions, and resource costs.

### Design and pipeline approval gates

1. **Design freeze:** approve the estimand, decision rules, multiplicity strategy,
   failure handling, counts and sample-size rationale, compute budgets, and
   generating distributions without mixer-performance data. Acceptable error,
   joint-coverage, power, and Monte Carlo-precision thresholds, bootstrap approval,
   the null margin, and expected-random replication assumptions remain open.
   Human review must authorize any higher-precision calibration and progression
   to Checkpoint 2; current draft counts remain placeholders.
2. **Pipeline freeze:** after authorized train/validation development, lock code,
   hyperparameters, exclusions, random-baseline replication, analysis, and the
   commitment to externally held blind definitions. Authorize the one-time locked
   evaluation separately. Changes to frozen items require a versioned amendment.

## Confirmatory success gates

These are prospective gates, not claims established by the pilot or synthetic
smoke test. G2 uses the statistical specification above; finite-shot, hardware,
transfer, and utility claims require their own supporting evaluations.

- **G1 invariant:** feasibility at least `0.99` under finite shots.
- **G2 equal-budget effect:** positive family-level median paired gap improvement against **both** fixed ring and expected-random connected equal-budget baselines in **at least two of the three structured blind families**, with the selected multiplicity-corrected tests rejecting and the corresponding one-sided lower bounds above zero. The separate weak-structure/null control never counts toward this gate.
- **G3 resource effect:** the discovered mixer is non-dominated in gap versus transpiled two-qubit depth.
- **G4 transfer:** positive effect survives an unseen size and hardware graph without architecture retraining.
- **G5 utility:** any end-to-end utility claim requires amortized time-to-target superiority over tuned classical solvers at a shared target quality.

## Key ablations

- remove QUBO structure and retain only graph size;
- remove the connectivity verifier;
- remove hardware features;
- replace learned edge scores with `|Q_ij|`, random scores and spectral scores;
- equalize edge count, depth and total circuit-evaluation budget separately;
- permute variable labels to test equivariance;
- test the exchangeable weak-structure/null ensemble without assuming that all instance-level information is uninformative.

## Closest-work pressure points

Reviewers will correctly note that constraint-preserving mixers, quantum architecture search, structure-aware circuit diagnostics and hardware-aware search already exist. Therefore the paper cannot claim novelty from any one of those phrases. Its defensible novelty must be the verified joint formulation, cross-instance/size transfer, controlled resource comparison, and an interpretable structure-to-mixer law.

The DQI project should remain separate unless a mathematical reduction or codeability criterion is established. DQI is an algorithmic decoding framework, not simply a gate motif to append to QAOA.

## Representative references

1. Farhi, Goldstone and Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028 (2014).
2. Hadfield et al., *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12, 34 (2019).
3. Zhang et al., *Neural Predictor based Quantum Architecture Search*, arXiv:2103.06524 (2021).
4. Duong et al., *Quantum Neural Architecture Search with Quantum Circuits Metric and Bayesian Optimization*, arXiv:2206.14115 (2022).
5. Bucher et al., *Towards Robust Benchmarking of Quantum Optimization Algorithms*, arXiv:2405.07624 (2024).
6. Lipardi et al., *Magic-Informed Quantum Architecture Search*, arXiv:2605.03932 (2026).
7. Assis et al., *A SWAP-free Framework for QAOA*, arXiv:2604.25058 (2026).
8. Falla and Safro, *Constrained Counterdiabatic Quantum Approximate Optimization Algorithm for Portfolio Optimization*, arXiv:2605.06858 (2026).
9. Niu et al., *HamQASBench: A Hamiltonian-Informed Diagnostic Benchmark for Evaluating Quantum Architecture Search*, arXiv:2607.04845 (2026).
10. Leonidas et al., *Quantum Portfolio Optimization: An Extensive Benchmark*, arXiv:2509.17876 (2026 revision).

