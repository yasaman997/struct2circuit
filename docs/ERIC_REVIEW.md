# Scientific review: what a prospective PI should be able to verify

This document is an internal-facing audit of the repository's scientific boundary. It is not a claim of external endorsement.

## Current method

The implemented structure-conditioned mixer is a deterministic graph generator using only off-diagonal QUBO interaction magnitudes |Q_ij|. It builds a maximum-weight spanning tree and adds the strongest remaining interactions to reach a declared edge budget.

## Current evidence

The only performance result currently reported is the exploratory 24-instance, n=8, k=3, p=1 pilot on a synthetic block-correlated family.

## Explicit non-claims

The repository does not currently claim:

- learned architecture search;
- trainability improvement;
- hardware advantage;
- quantum advantage;
- cross-family generalization;
- scalability;
- superiority to classical optimization.

## Comparison discipline

“Equal-edge” means equal graph-edge count. It does not mean equal circuit depth, two-qubit gates, routing, shots, or end-to-end runtime.

The complete mixer is a higher-resource reference, not a matched baseline.

## Scientific next step

The next meaningful contribution is not more README polish. It is an independently controlled evaluation that determines whether the |Q_ij|-conditioned rule carries reproducible signal beyond fixed and random connected baselines, across predeclared problem families and unseen instances.

## Questions a reviewer should be able to answer from the repository

1. What information does the proposed method use? — |Q_ij| only.
2. What does it preserve? — fixed Hamming weight in the exact simulator.
3. What is the current experimental unit? — one generated QUBO instance.
4. What is the primary outcome? — paired normalized feasible-range expectation-gap improvement.
5. What is matched now? — mixer edge count.
6. What is not matched yet? — hardware execution resources and end-to-end construction/search cost.
7. Is the current result confirmatory? — no.
8. Is the method learned? — no.
9. Is trainability measured? — no.
10. What would falsify the idea? — failure to reproduce a positive effect against both predeclared structure-agnostic comparators across the required structured families, or evidence that any apparent gain disappears after resource/failure controls.
