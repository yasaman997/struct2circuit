# Scientific scope and claim boundary

Struct2Circuit currently studies one concrete intervention:

> Use off-diagonal QUBO interaction magnitude |Q_ij| as side information when constructing a sparse, connected XY mixer for a cardinality-constrained QUBO.

The current method is deterministic and transparent.

## Present-tense claims

- The mixer preserves the fixed-Hamming-weight feasible subspace in the exact simulator.
- The pilot compares an 8-edge structure-conditioned mixer with an 8-edge fixed ring.
- The pilot also reports a 28-edge complete mixer as a higher-edge-count reference.
- The pilot result is exploratory and family-specific.

## Future-tense questions

- Does the effect survive independent problem families?
- Does it survive unseen sizes and cardinality ratios?
- Does it remain after measured transpilation and execution costs?
- What mechanism explains any effect?
- Can a learned policy improve upon the transparent rule?
- Is there any useful relationship between instance structure and variational optimization behavior?

These questions are not presented as completed results.

## Falsification

The hypothesis is weakened or rejected if the proposed structure signal fails to provide reproducible benefit against predeclared structure-agnostic baselines under the approved analysis, or if apparent benefit disappears after appropriate resource and failure controls.

A mixed result is scientifically informative: the project should identify the structural regimes in which the method helps and those in which it does not.
