# Scientific scope and claim boundary

Struct2Circuit currently studies a mechanistic candidate:

> Use the exact fixed-cardinality conditional RMS of feasible swap costs, derived from `Q`, `c`, and `k`, as side information when constructing a sparse, connected XY mixer under a fixed edge budget.

Conditional mean and variance are retained separately. Low-RMS and high-RMS edge selection are competing hypotheses; neither has yet demonstrated a QAOA performance improvement. The descriptor and topology construction are deterministic, and no learned policy is implemented.

## Historical evidence

The original strong-`|Q_ij|` heuristic beat the ring on 17/24 block-correlated pilot instances at the same 8-edge budget. The 28-edge complete mixer was a higher-edge-count reference. This was exploratory, family-specific evidence.

On the first independent weighted densest-k-subgraph family, the same heuristic lost to the ring on 17/24 instances. That negative result is retained. Strong `|Q_ij|` is a historical comparator, not the primary current mechanism candidate or a generally validated method.

The [historical-results status note](../results/README.md) explains why the original pilot report's advancement decision is preserved but is not current authorization for a larger blind benchmark.

## Established mathematical and software results

- The mixer preserves the fixed-Hamming-weight feasible subspace in the exact simulator.
- The conditional exchange moments follow from uniform sampling of the remaining occupied variables at fixed `k`; exact enumeration provides a software check of that derivation. Floating-point range and rounding limitations remain explicit in the [mechanism document](MECHANISM_HYPOTHESIS.md).
- The graph constructor preserves connectivity and its declared edge budget. Equal edge counts do not establish equal circuit or hardware resources.
- The simulator exposes three initialization conditions: `uniform`, `mixer_low`, and `mixer_high`. With positive-sign XY exchange amplitudes, `mixer_high` has the Perron–Frobenius positive-amplitude interpretation for a connected mixer. The low extremum is a sensitivity control. Neither extremum guarantees better QAOA performance or efficient hardware preparation, and these comparisons do not isolate topology from initialization.

These are mathematical and software properties, not performance evidence for the conditional-RMS candidate.

## Untested hypotheses and next phase

Numerical calibration is the next scientific phase. It must precede a mechanism-performance study; passing regression tests does not substitute for calibration. No large confirmatory benchmark has been authorized or frozen.

The current questions are whether either RMS direction predicts useful mixer transitions, whether mean and variance help interpret any observed effect, and whether that effect survives all three initialization conditions and declared comparison controls.

## Later questions

- If an effect is established, does it survive independent problem families?
- Does it survive unseen sizes and cardinality ratios?
- Does it remain after measured transpilation and execution costs?
- What mechanism explains any effect?
- Can a learned policy improve upon the transparent rule?
- Is there any useful relationship between instance structure and variational optimization behavior?

These questions are not presented as completed results.

The project has not demonstrated quantum advantage, hardware advantage, learned architecture discovery, or improved trainability.

## Falsification

The hypothesis is weakened or rejected if the proposed structure signal fails to provide reproducible benefit against predeclared structure-agnostic baselines under the approved analysis, or if apparent benefit disappears after appropriate resource and failure controls.

A mixed result is scientifically informative: the project should identify the structural regimes in which the method helps and those in which it does not.
