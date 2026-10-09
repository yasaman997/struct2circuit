# Working manuscript outline

> **Status: prospective outline.** This is a future-paper planning document. It does not represent completed results, a frozen protocol, or a publication claim.

## Working title

**Struct2Circuit: Structure-Conditioned Mixers for Cardinality-Constrained Quantum Optimization**

## One-sentence contribution

We investigate whether exact fixed-cardinality conditional exchange RMS, derived from `Q`, `c`, and `k`, can select a useful sparse XY topology at a matched edge budget, and whether any observed effect survives initialization controls and later independent evaluation.

The strong-`|Q_ij|` heuristic is a historical comparator: it beat the ring on 17/24 block-correlated pilot instances and lost on 17/24 independent weighted densest-k-subgraph instances. No QAOA performance result yet establishes superiority of either low-RMS or high-RMS selection. Numerical calibration is the next phase; the outline does not authorize or freeze a large confirmatory study.

## Abstract skeleton

1. **Problem:** constrained variational algorithms require a mixer that respects the feasible subspace, and generic structure-agnostic choices may ignore useful information in the optimization instance.
2. **Method:** compare low/high conditional exchange RMS as competing topology-selection hypotheses under a declared edge budget; retain conditional mean and variance separately and include historical magnitude-based comparators.
3. **Evaluation:** after numerical calibration, compare against the declared controls under `uniform`, `mixer_low`, and `mixer_high` initialization. Any later independent-family or hardware-resource evaluation must be reported only if performed under an approved protocol.
4. **Result:** leave this item unfilled until the relevant experiments and analysis exist; report positive, negative, mixed, or null outcomes with their uncertainty and scope.
5. **Meaning:** determine whether optimization-instance structure provides useful side information for constrained circuit design, and identify regimes in which it does not.

## Main sections

1. Introduction and falsifiable hypothesis.
2. Related work: constrained mixers, problem-informed circuit design, architecture search, and hardware-aware resource accounting.
3. Cardinality-constrained problem and fixed-weight XY invariant.
4. Conditional exchange moments, RMS descriptor, and fixed-budget mixer construction; historical magnitude heuristic as a comparator.
5. Numerical calibration, three initialization conditions, experimental protocol, and comparison budgets.
6. Results: quality, uncertainty, failures, and measured resources.
7. Generalization by instance, family, scale, and constraint ratio.
8. Mechanistic analysis relating exchange moments and topology to observed outcomes; any gradient/trajectory/landscape interpretation requires corresponding measurements.
9. Limitations: noiseless simulation scale, construction cost, resource accounting, and no demonstrated quantum advantage, hardware advantage, learned architecture discovery, or trainability improvement.
10. Reproducibility and data/code release.

## Figures to build only after the corresponding experiments exist

1. Method schematic: `(Q, c, k)` → conditional exchange moments → low/high RMS graph → constrained QAOA, with the initialization condition identified.
2. Paired primary outcome on held-out instances.
3. Quality versus measured transpiled two-qubit depth/gates.
4. Separate transfer plots for family, scale, and constraint ratio.
5. Null-control and ablation results.
6. Structure-to-selected-edge analysis, with uncertainty.

## Scientific decision rule

The paper should not be written around a positive result in advance.

A credible outcome includes:

- positive effect that survives the declared controls;
- a negative result showing that a descriptor direction worsens outcomes under the tested conditions;
- a mixed result identifying where structure helps and where it does not;
- or a null result showing that the proposed structural signal does not reliably improve the chosen architecture under the tested budget.

The scientific contribution is the controlled answer to the question, not a predetermined success story.

For the implemented positive-sign XY Hamiltonian, `mixer_high` is the Perron–Frobenius positive-amplitude reference for a connected mixer; `mixer_low` is the opposite-extremum sensitivity control. These conditions do not guarantee improvement, isolate a pure topology effect, or demonstrate efficient hardware preparation. Any eventual contribution or novelty claim must follow the evidence and related-work assessment.
