# Research protocol: structure-conditioned mixers for cardinality-constrained quantum optimization

## 1. Scope and central hypothesis

### Current study

> **At a fixed number of mixer edges, does a transparent QUBO-conditioned, feasibility-preserving XY mixer improve variational optimization outcomes over structure-agnostic connected mixers on unseen cardinality-constrained QUBOs?**

The present method is **not** a learned policy. It is a deterministic structure-conditioned rule based only on off-diagonal interaction magnitudes `|Q_ij|`.

### Central hypothesis

> **Some cardinality-constrained QUBOs contain interaction structure that can be used as side information to choose a sparse feasible mixer topology with better optimization outcomes than structure-agnostic topologies at the same edge budget.**

This is falsifiable. It can fail on some families, sizes, or structure regimes.

The pilot is exploratory and does not establish this hypothesis beyond its tested synthetic family.

### Future mechanistic question

A separate future question is whether any observed performance difference is mediated by variational optimization behavior—such as gradients, parameter sensitivity, local traps, or concentration. The current pilot does not measure those quantities, so “trainability” is not a present-tense contribution.

## 2. Scientific object and invariant

We study:

**Minimize:** `C_Q(x) = xᵀ Q x + cᵀ x`

**Subject to:** `Σᵢ xᵢ = k`, with `x ∈ {0,1}ⁿ`.

Define the Hamming-weight operator:

`N = Σᵢ (I - Zᵢ) / 2`.

The intended mixer Hamiltonian is an XY exchange Hamiltonian over a graph `G=(V,E)`:

`H_M = Σ_(i,j)∈E (XᵢXⱼ + YᵢYⱼ)/2`.

Each exchange preserves Hamming weight, so the corresponding continuous-time mixer evolution preserves the fixed-weight subspace. In the repository's exact simulator, the stronger implementation guarantee is obtained by representing the state directly in the weight-`k` basis.

Thus:

- **mathematical invariant:** XY exchange preserves Hamming weight;
- **software invariant:** the simulator never allocates amplitudes outside the weight-`k` basis;
- **hardware claim:** none is made until a hardware decomposition is explicitly audited.

## 3. Exact definition of the current structure signal

For a problem matrix Q, define:

`score(i,j) = |Q_ij|`, for `i < j`.

The current generator:

1. ignores diagonal entries and the linear vector c;
2. ranks candidate edges by `score(i,j)`, with deterministic index tie-breaking;
3. constructs a maximum-weight spanning tree using those scores;
4. adds the highest-scoring remaining edges until `edge_budget` is reached.

The generator therefore uses exactly the interaction magnitudes supplied by Q. It does not use optimum values, solver results, benchmark outcomes, planted labels, or any post-optimization quantity.

A common positive rescaling of all off-diagonal Q values leaves the topology unchanged; signs are deliberately ignored in this first study.

If a later method uses signs, diagonal terms, c, spectral features, or another structural descriptor, it is a **new method** and must be evaluated separately.

## 4. What is and is not matched

### Pilot control

The pilot matches **mixer edge count** between the structure-conditioned and ring mixers.

An equal number of graph edges is a construction-level control. It does **not** imply equal two-qubit gate count, circuit depth, routing overhead, shot cost, or hardware execution cost.

The complete mixer is a higher-edge-count reference and is not a matched baseline.

### Planned resource accounting

Resource claims are separated into:

1. **construction budget:** mixer edge count;
2. **quantum execution budget:** transpiled two-qubit gates, depth, routing/SWAP count, shots, repetitions;
3. **end-to-end budget:** architecture construction/search time plus quantum execution and classical optimization.

A “resource Pareto” claim is permitted only after the relevant quantities are measured.

## 5. Primary estimand and outcome hierarchy

For instance s, let `g_structure(s)` and `g_baseline(s)` be normalized feasible-range expectation gaps, with lower better.

Define paired improvement:

`Δ_s = g_baseline(s) - g_structure(s)`.

Positive values favor the structure-conditioned method.

### Primary estimand

For each predeclared family f:

> **the median paired improvement `median(Δ_s)` over independently generated test instances from that family.**

The overall scientific question is evaluated through the predeclared family-level analysis. No post-hoc family, comparator, or metric selection is permitted.

### Outcome hierarchy

1. **Primary:** normalized feasible-range expectation-gap improvement.
2. **Secondary:** optimum-sampling probability and epsilon-optimal sampling probability.
3. **Efficiency:** objective evaluations, shots, classical search time.
4. **Circuit resources:** transpiled two-qubit gates, depth, routing/SWAP count.
5. **Mechanistic diagnostics:** gradients, optimization trajectories, parameter sensitivity, and landscape diagnostics, if a later stage is approved.

A claim must be tied to the outcome actually measured.

### Normalized gap definition

For an instance with feasible costs C_min and C_max:

`gap = (E[C] - C_min) / (C_max - C_min)` when C_max > C_min.

If C_max = C_min, the normalized gap is undefined. Such instances require a predeclared handling rule before confirmatory evaluation; they must not be excluded after results are seen.

## 6. Experimental unit and independence

The independent experimental unit is the generated QUBO instance.

Random mixer graphs are repeated within an instance only to estimate the expected random comparator; those graph draws are not treated as independent QUBO instances.

Repeated optimizer initializations are computational replicates, not new scientific instances.

Seeds and instance IDs must be disjoint across train, validation, blind test, and transfer sets.

## 7. Planned benchmark families

1. block-correlated portfolio/index-tracking proxy;
2. weighted densest-k-subgraph;
3. weighted maximum-k-vertex-cover;
4. weak-structure negative control.

The first three are structured families; the fourth is a negative control.

The negative control tests whether the proposed rule creates an apparent advantage without the particular planted structure used to motivate it. It **cannot prove that no exploitable instance-level signal exists**.

Structural regimes are design factors, not knobs to tune after observing mixer performance.

## 8. Generalization dimensions

“Transfer” is split into separate scientific claims:

- **instance transfer:** unseen instances from the same distribution;
- **family transfer:** unseen problem family;
- **scale transfer:** unseen problem size;
- **constraint transfer:** unseen k/n;
- **hardware transfer:** unseen connectivity/decomposition.

A positive result in one dimension does not imply the others.

## 9. Baselines

Baseline categories will be included only when implementations and resource accounting can be verified.

### Structure-agnostic constrained mixers

- fixed ring XY-QAOA;
- complete XY-QAOA as a higher-resource reference;
- random connected XY mixers at the same edge budget.

### Stronger algorithmic references

- warm-start QAOA;
- constrained counterdiabatic/commutator-informed variants where an exact, reproducible implementation is available;
- architecture-search baselines only if their search space, compute budget, and objective can be made comparable.

### Classical references

- exact MIQP/CP-SAT on sizes where exact solution is computationally credible;
- tuned greedy/local search/tabu/simulated annealing.

No baseline will be included merely because its name is impressive. It must be implemented or reproduced with documented settings and a fair comparison budget.

## 10. Architecture search and learning: future work only

The current method should be called a **structure-conditioned mixer generator**, not a general “architecture-search framework.”

A future learned extension could replace `|Q_ij|` with a learned edge-scoring or architecture-selection policy. Before that experiment:

- the feature interface must be frozen;
- every feature must be computable from the target instance before optimization;
- optimum values, solver runtimes, observed mixer performance, and other outcome variables must be excluded;
- permutation behavior/equivariance must be specified;
- training/validation/test separation must be enforced;
- search/construction cost must be counted in end-to-end utility.

The model class is not precommitted. “Learning-based architecture selection” is a future method label, not a present contribution.

## 11. Mechanistic analysis: only if performance differences survive

If a robust performance effect survives the family-level controls, the next question is **why**.

Candidate analyses include:

- gradient distributions at controlled parameter points;
- convergence trajectories;
- sensitivity to initialization;
- local-optimum/trap diagnostics;
- cost-landscape summaries;
- relationships between selected edges and measurable QUBO structure.

These measurements can support a trainability/mechanism claim only if the data directly support it.

## 12. Statistical analysis

The confirmatory analysis should remain proportional to the scientific question: one primary estimand, a small predeclared comparator set, and transparent uncertainty.

Planned ingredients:

- paired effect estimates with confidence intervals;
- an exact or simulation-validated sign/order-statistic procedure when appropriate for the declared estimand;
- Wilcoxon as a secondary robustness analysis, not as a generic test of a population median under arbitrary skewness;
- bootstrap only if pre-freeze simulation demonstrates adequate coverage for the declared estimand and design;
- effect sizes and distributions, not p-values alone;
- explicit reporting of failed optimizations and exclusions.

The exploratory pilot's bootstrap interval and Wilcoxon statistic are descriptive outputs only.

### Multiple comparisons

The primary family-level claims and comparator hierarchy must be frozen before blind evaluation. The analysis must state exactly which family-level claims are primary and how multiplicity is controlled.

A method that succeeds against one comparator but not the other will not be described as a robust family-level win.

### Failure handling

Optimizer failure must be defined operationally before the confirmatory run.

Preferred reporting:

- retain the instance;
- record a failure indicator and cause;
- report failure rates by method;
- do not silently impute a favorable value.

If a failure cannot be represented by the primary outcome, the analysis should use a prespecified sensitivity analysis or a clearly defined estimand-specific handling rule. No exclusion or imputation rule may be chosen after blind outcomes are observed.

## 13. Feasibility and verification

In the exact simulator, feasibility is structurally guaranteed because the state vector is defined only over weight-k basis states.

Therefore a finite-shot feasibility threshold such as 0.99 is **not** a meaningful current software gate.

Current software gates instead verify:

- valid mixer graph;
- exact edge budget;
- connectivity;
- Hermiticity of the constructed XY Hamiltonian;
- fixed-weight basis construction.

Future noisy/hardware experiments must define feasibility operationally from sampled bitstrings and report leakage/readout errors separately.

## 14. Nulls, leakage, and confounding

A null ensemble is a negative control, not a proof of no signal.

Problem structure and problem difficulty are not interchangeable. A future predictor must distinguish:

- structural descriptors available before solving;
- outcome variables such as optimal cost;
- solver-runtime or optimization-failure variables;
- post-hoc circuit performance.

If “difficulty” is used as a feature, its construction must be audited for target leakage and circularity.

The current transparent generator avoids this issue because its only input is Q.

## 15. Required ablations

When a confirmatory study is approved:

- remove QUBO structure and retain only graph size;
- replace `|Q_ij|` with random scores;
- replace it with spectral scores;
- test sign-aware versus magnitude-only scoring as a separately declared method;
- equalize edge count first, then separately compare transpiled depth/gates;
- permute variable labels to test permutation equivariance;
- test null and weak-structure regimes;
- measure construction cost.

Ablations used for confirmatory claims must be frozen before the locked evaluation.

## 16. Decision gates

The gates answer scientific questions rather than requiring a positive result:

- **G1 — software correctness:** graph, basis, Hermiticity, and manifest invariants pass.
- **G2 — design validity:** structure signal, estimand, failure rule, family set, comparison budgets, and analysis are frozen before blind evaluation.
- **G3 — family-level effect:** the primary effect against both predeclared structure-agnostic comparators is evaluated in the required number of structured families under corrected inference.
- **G4 — robustness/generalization:** any transfer claim is supported only by the specific transfer dimension tested.
- **G5 — end-to-end utility:** any resource/utility claim includes construction/search cost and measured quantum execution resources.

A null or mixed result is scientifically valid.

## 17. Closest-work pressure points

The relevant prior-work categories are:

- constrained/alternating-operator mixers;
- problem-informed or custom mixer design;
- quantum architecture search;
- structure-aware quantum-algorithm diagnostics;
- hardware-aware circuit design and compilation.

The project should not claim novelty from any category alone.

The eventual paper should include a direct comparison table:

| Prior-work capability | Present Struct2Circuit study |
| --- | --- |
| Feasibility-preserving constrained mixer | Yes, XY fixed-weight construction |
| Problem-conditioned topology | Yes, `|Q_ij|`-conditioned graph |
| Learned policy | **No, future extension** |
| General architecture search | **No** |
| Matched edge-budget study | **Current pilot / planned benchmark control** |
| Transpiled resource study | **Future** |
| Cross-family evaluation | **Planned** |
| Mechanistic trainability study | **Future** |

The scientific differentiation must come from the controlled experiment, not from labels.

## 18. Reproducibility, integrity, and status

The repository separates:

- exploratory results;
- provisional benchmark design;
- pre-freeze statistical checks;
- future confirmatory evaluation.

Blind records should not be treated as secure merely because a local flag is absent. The current repository guard is a procedural safeguard, not an access-control boundary.

Before a definitive blind evaluation, blind instance definitions should be generated under external custody inaccessible to model developers and routine CI, with a public cryptographic commitment and a documented release procedure.

The current benchmark status is `DRAFT_UNFROZEN`.

## 19. Literature and novelty discipline

The literature list is a reading map, not a novelty claim. Before any manuscript submission, every citation must be checked against the exact method and a direct comparison must be made with the closest constrained-mixer, problem-informed, architecture-search, and benchmarking work.

Do not state that the current method is novel merely because its components are combined.

## 20. Current research boundary

The present scientific contribution, if supported by future experiments, is:

> **A controlled study of whether off-diagonal QUBO interaction structure can serve as side information for sparse, feasibility-preserving mixer design under a fixed edge budget.**

Learned policies, trainability mechanisms, hardware performance, broad architecture search, and quantum advantage remain separate research questions.
