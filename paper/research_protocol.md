# Research protocol: structure-conditioned mixers for cardinality-constrained quantum optimization

## 1. Scope and central hypothesis

### Current study

> **At a fixed number of mixer edges, does a transparent QUBO-conditioned, feasibility-preserving XY mixer improve variational optimization outcomes over structure-agnostic connected mixers on unseen cardinality-constrained QUBOs?**

The historical pilot used a deterministic rule based on off-diagonal `|Q_ij|`, not a learned policy. That heuristic failed to generalize in the first independent weighted DKS test. The current mechanism candidate is the exact fixed-`k` conditional RMS exchange cost derived from `Q`, `c`, and `k`, with mean and variance retained separately. Low/high RMS are competing hypotheses; these code repairs supply no new performance evidence.

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

## 3. Historical structure signal and current mechanism candidate

For a problem matrix Q, define:

`score(i,j) = |Q_ij|`, for `i < j`.

The historical strong-interaction generator:

1. ignores diagonal entries and the linear vector c;
2. ranks candidate edges by `score(i,j)`, with deterministic index tie-breaking;
3. constructs a maximum-weight spanning tree using those scores;
4. adds the highest-scoring remaining edges until `edge_budget` is reached.

The generator therefore uses exactly the interaction magnitudes supplied by Q. It does not use optimum values, solver results, benchmark outcomes, planted labels, or any post-optimization quantity.

A common positive rescaling of all off-diagonal Q values leaves the topology unchanged; signs are deliberately ignored in this first study.

The conditional exchange descriptor is a **new method** and must be evaluated separately from that historical rule. For an occupied-to-unoccupied exchange `i -> j`, write `d0=Q_jj-Q_ii+c_j-c_i`, `a_l=Q_jl-Q_il` for `l != i,j`, `N=n-2`, and `m=k-1`. Under a uniform feasible prior conditioned on `x_i=1,x_j=0`, define:

- `mu_ij = d0 + 2*m*mean(a)`;
- `var_ij = 4*m*(N-m)/(N-1)*mean((a-mean(a))**2)` for `N>1`;
- `rms_ij = sqrt(mu_ij**2 + var_ij)`.

For `N=0`, the mean is `d0`; for `N=1`, it is `d0+2*m*a_1`. Variance is zero for both cases, for `k=1` or `k=n-1`, and when all coefficients are equal. The mean is directed and antisymmetric. RMS is the primary symmetric score, with mean and variance separately available for mechanism interpretation. These are actual conditional transition moments, unlike the former coefficient norm. They use no solutions or optimization outcomes and are mathematically invariant under feasible-equivalent objective encodings.

Numerically, reduce `2*m/N` to the integer ratio `p/d` and evaluate the mean as `(d*d0+p*sum(a_l))/d`. Feed `d` copies of each original diagonal/linear term and `p` copies of each original interaction term into one compensated sum, then divide by `d`. Thus no rounded row difference, weighted product, or centered aggregate precedes the final cancellation. For `m=N`, this is the direct compensated expression `d0+2*sum(a_l)`; for `m=0` (including `N=0`), sum only the original `d0` terms. Variance retains the reference-site calculation: form `z_l=a_l-a_r` using a compensated sum of `Q_jl,-Q_il,-Q_jr,Q_ir`, choosing the minimum reference by compensated comparisons, and use `Var(z)=Var(a)`. These are the original conditional moments without a transformed matrix. Input information already rounded away cannot be recovered; the compensated numerator and variance arithmetic must remain within floating-point range, and final rounding precludes universal bitwise invariance.

Low/high RMS feed the existing minimum/maximum spanning-tree-first construction followed by ranked unused edges. Connectivity and exact edge count are preserved. Exact score ties use deterministic label-based ordering and can break permutation equivariance. Neither ranking direction is claimed to be superior before data.

## 3A. Initialization is an experimental factor

Initialization is part of the algorithmic configuration, not a nuisance variable
that can be silently "controlled away."

There are three conditions: `uniform`, `mixer_low`, and `mixer_high`. The
implemented positive-sign XY adjacency has a unique positive-amplitude
Perron--Frobenius highest state for connected mixers and `0<k<n`.
`mixer_high` is therefore the ground state of `-H_M`; `mixer_low` is the
opposite-extremum sensitivity condition. Neither guarantees better performance.
The spectral-projector fallback at zero uniform overlap is deterministic but
can depend on labels; it is not a canonical permutation-equivariant low state.

Comparing a topology under different initializations measures sensitivity to the
topology/initialization pairing. It does **not** identify a pure causal topology
effect. Uniform-state fidelity with each extremal eigenspace is reported
separately as an alignment diagnostic.

## 3B. Cost scale and parameter domain

For new mechanism-stage experiments, use `S=C_max-C_min` and optimize the
dimensionless coordinate `u=gamma*S` on `[0,2*pi]`. The grid, L-BFGS-B, and
Powell share the same declared bounds and use centered normalized costs
`(C-C_min)/S` for phases and the numerical objective. Returned physical gamma
is `u/S`. This avoids dependence of solver tolerances on raw objective units.

This is a characteristic-scale convention, not a fundamental gamma period.
The shared beta box `[0,pi]` is also a comparison convention, not a universal
mixer period. Domain sensitivity and optimization-reference calibration remain
future prerequisites to interpreting new topology rankings; no calibration is
part of this repair pass. The historical pilot and independent DKS entry points
explicitly use `legacy`, retaining their original raw gamma `[0,2*pi]` grid,
raw expectation objective, solver settings, and candidate selection.

`finite_objective` means only that the final expectation is finite. Record the
selected source (`grid`, `lbfgsb`, `powell`, or `not_run`) and its nullable
success status separately from the two solver statuses. A grid result has no
local-convergence status. Retain evaluation count, normalized improvement over
the grid, declared coordinate bounds, and gamma/beta boundary indicators.
Successful local termination does not certify global accuracy.

## 4. What is and is not matched

### Pilot control

The pilot matches **mixer edge count** between the structure-conditioned and ring mixers.

An equal number of graph edges is a construction-level control. It does **not** imply equal two-qubit gate count, circuit depth, routing overhead, shot cost, or hardware execution cost.

The complete mixer is a higher-edge-count reference and is not a matched baseline.

The initialization diagnostic requires its declared edge budget to equal the
actual fixed-ring edge count and records actual counts for every topology.
It compares ring, historical strong-`|Q|`, and random topologies across three
initializations; it is not yet the planned conditional-RMS mechanism study.

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

For normalization, evaluate each feasible cost with `math.fsum` over individual
occupied `Q_ij` and `c_i` coefficients. Binary occupation selects these terms
exactly; summing them together avoids the cancellation error from separately
rounded quadratic and linear totals. Derive `C_min`, `C_max`, and the span from
these accurately summed costs, with no absolute raw-unit threshold. Raw costs,
phases, and expectation arithmetic remain unchanged for legacy search.
The gap uses normalized costs directly. For resolved, nonconstant costs,
`probability_optimum` sums probability over states whose compensated float64 cost
equals `C_min` exactly, before division by the span. No proximity tolerance is
applied. Comparing normalized costs to zero would be insufficient: normalization
can underflow a strictly positive gap to zero. The gap calculation itself is
unchanged, so a zero normalized gap at extreme scales need not imply optimum
probability one.

This is minimum equality for the declared compensated-float64 objective, not
exact rational minimization of the coefficient polynomial. Final cost rounding
can merge distinct coefficient-level values into numerical ties, even when the
overall span remains resolved. Recomputing historical probability metrics may
therefore change values obtained under the former `1e-10` near-optimum mask;
stored historical results and their reported conclusions are preserved. See
[the historical-results status note](../results/README.md).

If accurately summed costs still collapse to a zero span, retain the conservative
`cost_status="constant_or_unresolved"`. This covers true constancy and variation
lost in input representation or final floating-point rounding. Normalized gap
is undefined for constants; optimum probability is mathematically one for a
proven constant, but is withheld because this policy intentionally combines
constant and unresolved cases. Both metrics serialize as `None`, JSON `null`,
or blank CSV. The normalized optimizer skips search with
`selected_source="not_run"`. Nonfinite costs, compensated-summation overflow,
or an unrepresentable span/returned physical gamma raise an explicit error.

The current aggregation rule retains every raw row and validates the expected
instance IDs supplied by the study design, never inferring completeness from
observed IDs. Pilot summaries use the declared configuration count and zero-based
IDs, and require every summarized outcome (normalized gap and optimum probability)
for every comparator. Gap plots require every expected instance/comparator gap.
Constant/unresolved rows, missing instances/comparators, and undefined/nonfinite
outcomes cause refusal with affected/expected and completely missing instance
counts. A future study may adopt a different predeclared policy before observing
outcomes.

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

A future learned extension could replace the declared deterministic descriptor with a learned edge-scoring or architecture-selection policy. Before that experiment:

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

The historical transparent generator uses only `Q`; the current conditional
exchange descriptor uses `Q`, `c`, and `k`. Neither uses optimum labels or
optimization outcomes. Exact feasible-span normalization does require
enumerating feasible costs in this small-system simulator; its computational
cost must be acknowledged in any later scalability claim.

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
| Problem-conditioned topology | Historical strong-`\|Q_ij\|` interaction-magnitude heuristic; current candidate: exact fixed-cardinality conditional exchange RMS from `(Q, c, k)`. Low-RMS and high-RMS topologies are competing hypotheses; neither has demonstrated a performance advantage. |
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

> **A controlled study of whether exact fixed-cardinality conditional exchange RMS computed from `(Q, c, k)` can serve as side information for sparse, feasibility-preserving mixer design under a fixed edge budget, with the historical strong-`|Q_ij|` interaction-magnitude heuristic retained as a comparator. Low-RMS and high-RMS topologies are competing hypotheses; neither has demonstrated a performance advantage.**

Learned policies, trainability mechanisms, hardware performance, broad architecture search, and quantum advantage remain separate research questions.
