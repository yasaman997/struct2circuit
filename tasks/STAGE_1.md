# Stage 1 task: benchmark foundations and essential controls

> **Historical infrastructure plan.** This document preserves the Stage 1 tasks and completion record from the strong-`|Q_ij|` phase. It does not authorize a new experiment or freeze a benchmark. The current candidate is fixed-cardinality conditional exchange RMS, and numerical calibration comes next; see the [current pre-freeze status](PREFREEZE_DESIGN.md).

## Stage objective

Build the infrastructure needed to test whether QUBO interaction structure contains useful signal for constrained mixer design beyond fixed or random equal-edge mixer graphs.

Stage 1 builds infrastructure only. It does not train a learned policy or run the final blind comparison.

## Scientific terminology used in this stage

- **equal-edge:** same number of mixer graph edges;
- **quantum-resource matched:** only used after gates, depth, routing, and shots have actually been measured;
- **structure-conditioned:** the historical method in this Stage 1 plan used only off-diagonal |Q_ij| scores;
- **learned policy:** future work, not Stage 1;
- **trainability:** future mechanistic analysis, not a Stage 1 outcome.

## Status

- [x] Stage 1A: random connected equal-edge mixer baseline
- [x] Stage 1B: additional structured and weak-structure problem generators
- [x] Stage 1C: deterministic split/manifest tooling and pre-freeze report
- [ ] Scientific approval to freeze Benchmark v1

## Stage 1A: random connected equal-edge baseline

### Scientific question

Does the structure-conditioned mixer beat an instance-independent connected mixer with the same number of edges, rather than merely beating one particular ring topology?

### Implementation

Maintain:

random_connected_mixer(n, edge_budget, seed, max_attempts=10_000)

Requirements:

- validate n - 1 <= edge_budget <= n(n-1)/2;
- enumerate all simple undirected candidate edges;
- use a local numpy.random.default_rng(seed);
- sample exactly edge_budget edges without replacement;
- reject disconnected samples and retry;
- return exactly one connected mixer;
- preserve seed and sampling method in construction metadata;
- never inspect Q, c, objective values, optima, labels, or results.

Rejection sampling is acceptable for the pilot-scale regime. Its scalability limitation must remain documented.

### Tests

- connectivity;
- exact edge count;
- uniqueness and no self-loops;
- deterministic replay;
- variation across seeds;
- boundary budgets;
- invalid arguments;
- public signature contains no problem-dependent input;
- all pre-existing tests pass.

## Stage 1B: problem generators

Implement and test:

1. weighted densest-k-subgraph;
2. weighted maximum-k-vertex-cover;
3. weak-structure negative-control QUBO.

Generator tests must verify objective equivalence by exhaustive enumeration on small instances, deterministic replay, parameter validation, and correct permutation metadata.

Do not expose planted labels to any future mixer-policy feature interface.

The negative control is a weak-structure negative control, not a proof that no exploitable information exists.

## Stage 1C: split and manifest tooling

### Scientific question

Can future learning/evaluation be performed without tuning on final test cases?

Manifest records must include:

- benchmark version and provisional/frozen status;
- stable instance ID;
- split and family;
- n, k, generator seed and parameters;
- generator implementation/version;
- random-baseline seed namespace;
- canonical record checksum.

Requirements:

- disjoint seeds and IDs across splits;
- deterministic canonical ordering;
- byte-identical regeneration from the same configuration;
- SHA-256 manifest checksum;
- default commands exclude blind-test instances;
- blind access requires explicit authorization;
- no blind performance during Stage 1;
- frozen changes require a new benchmark version.

### Sample-size and freeze gate

Do not call a count “powered” or “frozen” without an approved rationale.

Before Benchmark v1 is frozen:

1. use the pilot only as a sensitivity anchor;
2. validate candidate analysis procedures with synthetic simulation;
3. estimate compute cost on non-blind development inputs;
4. produce a plain-language design table;
5. obtain scientific approval for the estimand, families, regimes, counts, failure rule, and analysis.

Only after approval may the definitive blind set be created under external custody.

## Out of scope for Stage 1

- learned mixer-policy training;
- architecture-search claims;
- trainability claims;
- tuning generator regimes to favor the proposed mixer;
- confirmatory or blind performance experiments;
- noise/hardware experiments;
- strong classical comparisons;
- quantum-advantage claims.

## Completion report

At each checkpoint report:

1. scientific purpose;
2. files changed;
3. exact commands;
4. tests passed/failed/skipped;
5. assumptions and limitations;
6. confirmation that blind performance was not inspected;
7. checklist status;
8. precise next scientific decision.
