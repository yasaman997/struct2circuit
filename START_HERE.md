# Struct2Circuit: start here

This is the non-technical guide to the project. Read this file before reading source code.

## 1. The research question

For a cardinality-constrained optimization problem, we must select exactly `k` items from `n` candidates. Examples include selecting assets for a portfolio or vertices from a graph.

QAOA needs a **mixer**: the part of the quantum circuit that moves probability between candidate solutions. The usual approach uses the same hand-designed mixer for every problem instance.

Our question is:

> If we build the mixer using the structure of each optimization instance, can we obtain better solutions than fixed or random mixers at the same circuit-resource budget?

This is not yet a claim of quantum advantage. It is a narrower and testable question about quantum-circuit design.

## 2. The proposed solution

Represent the optimization instance by its QUBO matrix `Q`. Interpret strong pairwise entries in `Q` as structural information. Use that information to construct a sparse connected graph of XY mixer interactions.

The proposed mixer must satisfy three rules:

1. **Feasible:** it preserves the requirement that exactly `k` variables equal one.
2. **Connected:** it can move between all feasible choices rather than trapping the state in disconnected regions.
3. **Fairly budgeted:** it uses the same number of mixer edges as the comparison mixer.

The current transparent method selects strong `|Q[i,j]|` edges while enforcing connectivity. Later, an AI policy may learn which edges to select. The same feasibility verifier and resource budget will remain in place.

## 3. The hypothesis

Primary hypothesis:

> On unseen structured cardinality-constrained QUBOs, a structure-conditioned feasible mixer reduces the normalized optimality gap relative to the strongest fixed or random mixer with the same edge budget.

Lower normalized gap is better.

Important negative control:

> On null problems with no stable exploitable structure, the method should not show a reliable advantage.

This negative result would strengthen the interpretation: the method works when relevant structure exists, not universally.

## 4. What already exists: Step 0

The repository contains a small exploratory pilot:

- 24 synthetic block-correlated problems;
- `n=8`, `k=3`, and QAOA depth `p=1`;
- fixed ring mixer versus structure-conditioned mixer, both with eight edges;
- complete mixer as a more expensive reference;
- exact noiseless simulation within the feasible subspace.

The structure mixer beat the ring on 17 of 24 instances and improved the median normalized gap. This is encouraging, but it is not publishable confirmation because the study is small, uses one synthetic family, and lacks a random equal-budget control and blind test.

## 5. The roadmap

| Stage | Question answered | Implementation | Evidence needed | Status |
| --- | --- | --- | --- | --- |
| 0. Exploratory pilot | Is the idea executable and potentially nontrivial? | Existing simulator, ring/structure/complete mixers, 24-instance pilot | Feasibility, determinism, preliminary effect | Complete |
| 1A. Random baseline | Does structure beat an instance-independent mixer with the same edge count? | Random connected mixer ensemble | Connectivity, exact edge count, determinism, independence from `Q` | **Next** |
| 1B. Problem families | Does the idea extend beyond the portfolio-like generator? | Densest-`k`-subgraph, maximum-`k`-vertex-cover, and null generators | Exhaustive objective-equivalence tests | Not started |
| 1C. Frozen benchmark | Can we prevent tuning on the final test cases? | Train, validation, blind-test, and transfer seed manifest | Disjoint seeds, stable checksum, blind-test guard | Not started |
| 2. Non-AI heuristic study | Does the transparent `|Q[i,j]|` rule survive stronger controls? | Run matched experiments on train/validation only | Paired effects, failure regimes, null results | Not started |
| 3. AI mixer policy | Can learning improve on the transparent heuristic? | Graph/edge-scoring policy with verifier | Validation gain over heuristic and random search | Not started |
| 4. Transfer and resources | Does the result survive new sizes and realistic circuit costs? | Depth 1–3, noise, hardware graphs, transpilation | Quality-versus-depth Pareto frontier | Not started |
| 5. Locked blind test | Does the final method generalize without further tuning? | Run the frozen pipeline once | Corrected confidence intervals and all failed runs | Not started |
| 6. Paper | Which claims are actually supported? | Figures, tables, methods, limitations, open code | Reproducible manuscript with no overclaiming | Not started |

Do not begin the AI policy at Stage 3 until Stages 1A–2 show that instance structure contains a reproducible signal. Otherwise, AI would add complexity without answering the scientific question.

## 6. The first coding milestone: Stage 1A

### Goal

Add a random connected mixer using exactly the same number of edges as the structure-conditioned mixer.

This is the most important missing control. Comparing only against a ring does not tell us whether the benefit comes from `Q` structure or merely from choosing a different graph.

### Algorithm

Inputs:

- `n`: number of variables/qubits;
- `edge_budget`: required number of mixer edges;
- `seed`: reproducibility seed.

Procedure:

1. Validate `n - 1 <= edge_budget <= n(n - 1)/2`.
2. List all possible undirected edges `(i, j)` with `i < j`.
3. Use a local seeded random-number generator.
4. Sample exactly `edge_budget` edges without replacement.
5. Check whether the sampled graph is connected.
6. If disconnected, resample; otherwise return a `MixerSpec`.
7. Stop with a clear error if a configured maximum number of attempts is exhausted.

Sampling fixed-size edge sets uniformly and rejecting disconnected draws produces a uniform sample conditional on connectivity.

### Pseudocode

```python
def random_connected_mixer(n, edge_budget, seed, max_attempts=10_000):
    validate_edge_budget(n, edge_budget)
    rng = np.random.default_rng(seed)
    possible_edges = list(combinations(range(n), 2))

    for _ in range(max_attempts):
        indices = rng.choice(
            len(possible_edges), size=edge_budget, replace=False
        )
        edges = tuple(sorted(possible_edges[i] for i in indices))
        if graph_connected(n, list(edges)):
            return MixerSpec(
                name="random_connected",
                n=n,
                edges=edges,
                construction=f"random_connected_seed_{seed}",
            )

    raise RuntimeError("Could not sample a connected mixer")
```

The actual implementation should reuse existing validation and type conventions in `mixers.py`.

### Tests required

1. The graph is connected.
2. It has exactly `edge_budget` unique edges.
3. It contains no self-loops.
4. The same seed returns the same graph.
5. Different seeds can produce different graphs.
6. Invalid budgets raise clear errors.
7. The function accepts only `n`, `edge_budget`, and random-sampling controls—it must not read `Q`, costs, or solutions.
8. All existing tests still pass.

### Definition of done

Stage 1A is complete when the implementation and tests pass and a small demonstration shows several reproducible random mixers for `n=8`, `edge_budget=8`. Do not run or interpret the blind experiment at this stage.

## 7. How Codex will be used

Use one Codex task per roadmap milestone.

For every task, Codex should:

1. explain its plan before editing;
2. identify the files it will change;
3. implement only the current milestone;
4. write or update tests;
5. run the tests;
6. summarize the changes in plain language;
7. report assumptions and limitations;
8. stop for scientific review before moving to the next milestone.

Codex is responsible for code implementation, repetitive tests, experiment scripts, result files, and technical debugging. It is not responsible for deciding whether a scientific claim is justified.

The human review does not require reading every line. At each milestone, review this checklist:

- What question does this code answer?
- Is the comparison scientifically fair?
- What data or split did it use?
- Which tests passed?
- What conclusion is permitted, and what conclusion is not permitted?

## 8. Exact Codex task for Stage 1A

Open Codex in Plan mode at the repository root and provide this task:

> Inspect the current Struct2Circuit repository and propose a plan for Stage 1A only. Implement a reproducible `random_connected_mixer(n, edge_budget, seed)` in `src/struct2circuit/mixers.py`, reusing the existing `MixerSpec` and `graph_connected` conventions. It must sample exactly `edge_budget` unique undirected edges uniformly without replacement and reject draws until the graph is connected. It must not accept or inspect `Q`, costs, solutions, or experiment results. Validate `n - 1 <= edge_budget <= n(n - 1)/2`, use a local NumPy RNG, support a bounded `max_attempts`, and raise informative errors. Add tests for connectivity, exact edge count, uniqueness, lack of self-loops, deterministic replay, invalid budgets, and preservation of all existing behavior. Run the complete test suite. Do not modify the pilot experiment, add new problem generators, train an AI model, or run new scientific comparisons. Stop after reporting the files changed, exact commands run, test results, assumptions, and any concerns.

After Codex finishes, review its summary and test output before authorizing Stage 1B.

## 9. File map in plain language

You do not need to read all of these now.

- `START_HERE.md`: project question, roadmap, and current task. Read this first.
- `README.md`: technical summary and commands.
- `src/struct2circuit/problems.py`: defines optimization problems and generates test instances.
- `src/struct2circuit/mixers.py`: defines how the quantum circuit moves between feasible solutions. Stage 1A changes this file.
- `src/struct2circuit/simulator.py`: simulates QAOA exactly on small problems.
- `src/struct2circuit/optimize.py`: searches for good QAOA angles.
- `src/struct2circuit/analysis.py`: calculates statistics and creates results.
- `experiments/run_pilot.py`: runs the existing exploratory experiment.
- `tests/test_invariants.py`: checks that key mathematical and software properties hold. Stage 1A adds tests here or in a focused new test file.
- `paper/research_protocol.md`: detailed future experimental rules; consult later.
- `paper/manuscript_outline.md`: possible paper organization; not an active manuscript.
- `results/pilot_report.md`: preliminary result; not confirmatory evidence.

For now, read only Sections 1–8 of this file. The next decision point comes after Stage 1A tests pass.
