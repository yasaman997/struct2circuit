# Working manuscript outline

> **Status: prospective outline.** This is a future-paper planning document. It does not represent completed results, a frozen protocol, or a publication claim.

## Working title

**Struct2Circuit: Structure-Conditioned Mixers for Cardinality-Constrained Quantum Optimization**

## One-sentence contribution

We investigate whether a transparent |Q_ij|-conditioned sparse XY-mixer generator can improve variational optimization outcomes at a matched edge budget, and whether any observed effect survives independent problem families, unseen instances, and measured circuit-resource accounting.

## Abstract skeleton

1. **Problem:** constrained variational algorithms require a mixer that respects the feasible subspace, and generic structure-agnostic choices may ignore useful information in the optimization instance.
2. **Method:** construct a connected sparse XY mixer from off-diagonal QUBO interaction magnitudes under a declared edge budget.
3. **Evaluation:** compare against fixed and random connected mixers across independent problem families with paired instance-level analysis and explicit resource accounting.
4. **Result:** report the effect actually supported by the locked analysis, including null or negative results.
5. **Meaning:** determine whether optimization-instance structure provides useful side information for constrained circuit design, and identify regimes in which it does not.

## Main sections

1. Introduction and falsifiable hypothesis.
2. Related work: constrained mixers, problem-informed circuit design, architecture search, and hardware-aware resource accounting.
3. Cardinality-constrained problem and fixed-weight XY invariant.
4. Transparent structure-conditioned mixer generator.
5. Experimental protocol and comparison budgets.
6. Results: quality, uncertainty, failures, and measured resources.
7. Generalization by instance, family, scale, and constraint ratio.
8. Mechanistic analysis, only if directly supported by gradient/trajectory/landscape measurements.
9. Limitations: noiseless simulation scale, construction cost, resource accounting, and absence of a quantum-advantage claim.
10. Reproducibility and data/code release.

## Figures to build only after the corresponding experiments exist

1. Method schematic: QUBO interaction graph → verified mixer generator → constrained QAOA.
2. Paired primary outcome on held-out instances.
3. Quality versus measured transpiled two-qubit depth/gates.
4. Separate transfer plots for family, scale, and constraint ratio.
5. Null-control and ablation results.
6. Structure-to-selected-edge analysis, with uncertainty.

## Scientific decision rule

The paper should not be written around a positive result in advance.

A credible outcome includes:

- positive effect that survives the declared controls;
- a mixed result identifying where structure helps and where it does not;
- or a null result showing that the proposed structural signal does not reliably improve the chosen architecture under the tested budget.

The scientific contribution is the controlled answer to the question, not a predetermined success story.
