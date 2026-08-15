# Stage 1 task: benchmark foundations and essential controls

## Stage objective

Build the infrastructure needed to test whether QUBO-conditioned mixer topology contains useful signal beyond fixed or random equal-resource mixer graphs.

Stage 1 contains three sequential checkpoints. Complete and verify each checkpoint before proceeding. This stage builds infrastructure only; it does not train an AI policy or run the final blind comparison.

## Status

- [ ] Stage 1A: random connected equal-edge mixer baseline
- [ ] Stage 1B: additional structured and null problem generators
- [ ] Stage 1C: deterministic split/manifest tooling and pre-freeze report
- [ ] Scientific approval to freeze Benchmark v1

## Before editing

1. Read `AGENTS.md`, `START_HERE.md`, `paper/research_protocol.md`, `README.md`, the relevant source modules, and the existing tests.
2. Confirm that the working directory is the Struct2Circuit project root.
3. Check version-control status. The current distributed scaffold may not yet be a Git repository. If Git is absent, initialize a local repository and attempt a baseline commit named `baseline: exploratory pilot v0.1`. Do not change global Git identity settings; if a commit cannot be created, report the blocker and preserve the unmodified baseline by another explicit, non-destructive checkpoint.
4. Run the existing test suite and record the baseline result.
5. Present the complete Stage 1 plan, then implement the checkpoints in order.

## Stage 1A: random connected equal-edge baseline

### Scientific question

Does the structure-conditioned mixer beat an instance-independent connected mixer with the same number of edges, rather than merely beating a particular ring topology?

### Implementation

Add a public mixer constructor compatible with the existing `MixerSpec` interface:

```python
random_connected_mixer(n, edge_budget, seed, *, max_attempts=10_000)
```

Requirements:

- Validate `n - 1 <= edge_budget <= n * (n - 1) / 2`.
- Enumerate all simple undirected candidate edges `(i, j)` with `i < j`.
- Use `numpy.random.default_rng(seed)` locally.
- Sample exactly `edge_budget` edges uniformly without replacement.
- Reject disconnected samples and retry up to `max_attempts`.
- Return exactly one connected `MixerSpec` with unique edges and no self-loops.
- Raise informative errors for invalid arguments or exhausted attempts.
- Do not accept or inspect `Q`, `c`, objective values, optimal solutions, labels, or experiment results.
- Preserve the seed and sampling method in construction metadata available through the existing representation.

For the pilot-scale regime, rejection sampling of uniformly selected fixed-size edge sets is acceptable and produces a uniform draw conditional on connectivity. Document the scalability limitation rather than silently switching to a biased procedure.

### Tests

- connectivity;
- exact edge count;
- uniqueness and no self-loops;
- deterministic replay for the same seed;
- evidence that a set of different seeds is not accidentally constant;
- boundary cases: spanning-tree budget and complete-graph budget;
- invalid `n`, edge budget, and `max_attempts`;
- public signature contains no problem-dependent information;
- all pre-existing tests continue to pass.

### Gate 1A

Run the complete tests. If any fail, stop and diagnose. If they pass, mark Stage 1A complete and summarize why this baseline is scientifically necessary. Do not run performance comparisons yet.

## Stage 1B: problem generators

Implement three deterministic generators compatible with `CardinalityQUBO`.

### 1. Weighted densest-k-subgraph

Original objective:

```text
maximize sum_{i<j} w_ij x_i x_j, subject to sum_i x_i = k
```

Under the repository's minimization convention with symmetric `Q`:

```text
Q_ij = Q_ji = -w_ij / 2 for i != j
c_i = 0
```

Support configurable `n`, `k`, density, weight distribution, planted community strength, and seed. Apply a reproducible random vertex permutation. Return complete generator metadata, but do not expose planted labels to a mixer policy.

### 2. Weighted maximum-k-vertex-cover

Original objective:

```text
maximize sum_{(i,j) in E} w_ij * (x_i + x_j - x_i*x_j),
subject to sum_i x_i = k
```

Under the minimization convention:

```text
c_i = -sum_{j:(i,j) in E} w_ij
Q_ij = Q_ji = w_ij / 2 for i != j
```

Use a documented zero-diagonal convention. Support configurable `n`, `k`, density, weights, community or hub strength, and seed. Apply reproducible random vertex permutations.

### 3. Weak-structure/null ensemble

Generate an exchangeable negative-control QUBO with independently sampled, zero-mean off-diagonal coefficients and independent linear coefficients. Support configurable density, coefficient scale, `n`, `k`, and seed. Normalize scales using a documented rule. Add no community, geometric locality, or planted solution. Describe this as a weak-structure negative control, not proof that no exploitable instance-level information exists.

### Generator tests

- valid symmetric finite `Q`, finite `c`, and `0 < k < n`;
- deterministic replay;
- correct permutation/inverse-permutation metadata;
- parameter validation;
- for small `n`, enumerate every feasible bitstring and prove numerically that QUBO energy agrees with the original densest-k-subgraph objective up to the documented sign/additive constant;
- perform the equivalent exhaustive test for maximum-k-vertex-cover;
- null coefficients and metadata satisfy their documented construction.

### Gate 1B

Run the complete tests. If any objective-equivalence test fails, do not proceed. If all pass, mark Stage 1B complete. Do not compare mixer performance or select favorable generator parameters.

## Stage 1C: split and manifest tooling

### Scientific question

Can the future learning and evaluation process be reproduced without tuning on final test cases?

### Implementation

Create configuration-driven tooling for train, validation, blind-test, and transfer splits covering:

- existing block-correlated problems;
- densest-k-subgraph;
- maximum-k-vertex-cover;
- the null ensemble.

The configuration must control sizes, cardinalities, instance counts, structure-strength regimes, distributions, master seed, random-baseline replicate count, normalization, and exact-solver limits.

Manifest records must include:

- benchmark version and provisional/frozen status;
- stable instance ID;
- split and family;
- `n`, `k`, generator seed, and generator parameters;
- generator implementation/version identifier;
- random-baseline seed namespace;
- checksum of the canonical record.

Requirements:

- All seeds and instance IDs are disjoint across splits.
- Generation is deterministic and canonically ordered.
- Two runs from the same configuration are byte-identical.
- Produce a SHA-256 checksum for the complete manifest.
- Default commands exclude blind-test instances.
- Loading or evaluating blind-test instances requires an explicit `--allow-blind` flag.
- No blind-test performance may be generated, inspected, or summarized during Stage 1.
- Changing a frozen configuration requires a new benchmark version.

### Sample-size and freeze gate

Do not silently invent a confirmatory sample size and call it frozen. Before finalizing Benchmark v1:

1. Use the pilot only for conservative effect-size/variance sensitivity analysis.
2. Produce a short pre-freeze report describing candidate instance counts, detectable effects, computational cost, and assumptions.
3. Generate a provisional manifest labeled `DRAFT_UNFROZEN`.
4. Stop for scientific approval of the counts, size ranges, structural regimes, and primary analysis.

Only after explicit approval may the status be changed to `FROZEN`, the definitive checksum recorded, and future tuning restricted to train/validation.

### Manifest tests

- no duplicate IDs or seeds;
- split disjointness;
- deterministic byte-identical regeneration;
- stable record and complete-manifest checksums;
- default exclusion of blind data;
- explicit blind-access guard;
- schema and parameter validation;
- all previous tests continue to pass.

### Gate 1C

Complete the code, tests, and provisional pre-freeze report. Stop before declaring Benchmark v1 frozen. Request scientific approval with a plain-language table of the proposed design choices.

## Out of scope for Stage 1

- AI/ML mixer policy training;
- tuning generator regimes to favor the proposed mixer;
- a confirmatory or blind performance experiment;
- noise or hardware experiments;
- warm-start, counterdiabatic, QAS, or classical solver comparisons;
- publication or quantum-advantage claims.

## Completion report

At each checkpoint and at the end of Stage 1C, report:

1. Scientific purpose of the completed work.
2. Files changed.
3. Exact commands run.
4. Tests passed, failed, or skipped.
5. Assumptions and limitations.
6. Confirmation that blind-test performance was not inspected.
7. Current checklist status.
8. The precise decision required before further work.

