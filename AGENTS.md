# Struct2Circuit repository instructions

## Purpose

This repository investigates one falsifiable claim:

> A structure-conditioned, feasibility-preserving mixer improves the solution-quality/resource Pareto frontier over fixed or random equal-budget mixers for cardinality-constrained QUBOs.

Do not broaden this into a claim of quantum advantage or generic "AI discovery of quantum algorithms."

## Source-of-truth order

1. `paper/research_protocol.md` controls scientific definitions, integrity rules, and permitted claims.
2. The active file in `tasks/` controls the current implementation scope and acceptance criteria.
3. `START_HERE.md` explains the project and roadmap in plain language.
4. Existing source code and tests define current software interfaces.

If these sources conflict materially, stop and report the conflict instead of silently choosing one.

## Mathematical contract

- Problems have the form `min x.T @ Q @ x + c.T @ x` over binary `x`, subject to `sum(x) == k`.
- `Q` is stored symmetrically. Off-diagonal terms in `x.T @ Q @ x` are therefore counted twice.
- Every permitted XY mixer must preserve Hamming weight `k`.
- Mixer graphs used for general feasible-state reachability must be connected.
- Equal-budget comparisons must use the same mixer-edge count and the same parameter-optimization budget.

## Research-integrity rules

- Label pilot results exploratory.
- Never inspect, optimize on, summarize, or report blind-test performance unless the active task explicitly authorizes the final locked evaluation.
- Training data are for learning; validation data are for architecture and hyperparameter choices; blind data are opened once after the pipeline is frozen.
- Report all seeds, failed optimizations, exclusions, and resource costs.
- Null or negative results must be retained.
- Do not implement the AI policy until the benchmark, controls, and non-AI heuristic study are complete.
- Do not claim publication readiness, generalization, hardware advantage, scaling advantage, or quantum advantage from infrastructure tests or the pilot.

## Engineering rules

- Before editing, inspect the relevant files and run the existing tests.
- Preserve public behavior unless the active task requires a change.
- Prefer small, typed, documented functions and local NumPy random generators.
- Every stochastic public interface must accept an explicit seed.
- Add no new production dependency without explaining why it is necessary.
- Do not rewrite unrelated files.
- Add exhaustive small-instance tests when converting a combinatorial objective into a QUBO.
- Run the complete test suite after every checkpoint:

```bash
python3 -m unittest discover -s tests -v
```

## Working protocol

For each checkpoint:

1. State the scientific question it enables.
2. Present a short implementation plan before editing.
3. Identify files to be changed.
4. Implement only the active checkpoint.
5. Add and run tests.
6. Review the diff for leakage, unfair comparisons, mathematical errors, and regressions.
7. Update the active task's status without marking unverified work complete.
8. Report changes, commands, test results, assumptions, limitations, and the next decision gate in plain language.

If a test or mathematical check fails, stop progression to the next checkpoint and diagnose it.

