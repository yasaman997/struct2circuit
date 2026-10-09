# Literature differentiation

This document defines the scientific neighborhood of Struct2Circuit. It is a positioning map, not a novelty claim.

## Closest work

| Work | Established result | Relationship to Struct2Circuit | Remaining distinction |
| --- | --- | --- | --- |
| Hadfield et al., *From the Quantum Approximate Optimization Algorithm to the Quantum Alternating Operator Ansatz* (2019) | Feasibility-preserving/custom mixing operators for constrained optimization. | Struct2Circuit uses a fixed-weight XY mixer. | The present question is about selecting a sparse XY topology from instance information under a fixed edge budget, not inventing constrained mixers. |
| He et al., *Alignment between initial state and mixer improves QAOA performance for constrained optimization* (2023) | Initial-state/mixer alignment can materially affect constrained-QAOA performance. | Directly relevant to the interpretation of topology comparisons. | The current project initially fixed the uniform feasible state; alignment is now an explicit control rather than something assumed away. |
| Kordonowy & Leipold, *The Lie algebra of XY-mixer topologies and warm starting QAOA for constrained optimization* (2026) | XY topology changes dynamical structure and can affect optimization behavior. | Establishes that mixer topology is a substantive algorithmic variable. | Struct2Circuit does not claim a Lie-algebra or trainability theorem. It asks whether pre-optimization instance information can select a sparse topology. |
| Perlin et al., *Q-CHOP: Quantum constrained Hamiltonian optimization* (2025) | A different constrained quantum optimization framework with explicit feasibility preservation; Eric Anschuetz is a coauthor. | Same broad constrained-optimization setting. | Q-CHOP is not an instance-conditioned sparse XY-QAOA topology-selection method. |
| Du et al., *Quantum circuit architecture search for variational quantum algorithms* (2020) | Circuit architecture can be searched/optimized as an algorithmic resource. | Motivates treating circuit design as a scientific variable. | Struct2Circuit is not architecture search and does not learn a circuit policy. Its current rule is deterministic and transparent. |
| Ni et al., *An Adaptive Mixer Allocation Strategy for the Quantum Alternating Operator Ansatz* (2026) | Mixer allocation can be adapted to problem structure during an algorithmic procedure. | Close in the broad idea that mixer structure can be problem-dependent. | Struct2Circuit selects a sparse XY topology before parameter optimization using a declared instance descriptor; it does not claim the broader adaptive-mixer result. |

## What the project can legitimately ask

The current defensible question is:

> **Given a cardinality-constrained QUBO and a fixed mixer edge budget, can a transparent pre-optimization instance descriptor select a sparse feasibility-preserving topology that improves shallow variational optimization over structure-agnostic connected topologies?**

This is an empirical question about **instance-conditioned constrained circuit design**.

It is not a claim to have invented constrained mixers, custom mixers, architecture search, or topology-aware QAOA.

## Why the initialization issue matters

The literature makes it unsafe to interpret every topology comparison as a pure topology effect. Different XY graphs can have different mixer ground spaces and dynamical structure, and alignment between an initial state and the mixer can affect low-depth performance.

The original Struct2Circuit pilot used a uniform feasible state for every topology. That remains a valid fixed-initialization comparison, but it does not isolate topology from alignment. The simulator now exposes three explicit conditions: `uniform`, `mixer_low`, and `mixer_high`. The positive-sign XY Hamiltonian gives the highest mixer state the Perron–Frobenius positive-amplitude interpretation for connected mixers; `mixer_low` is an opposite-extremum sensitivity control. The spectral construction is deterministic, but its fallback in a degenerate eigenspace can depend on variable labels.

These conditions diagnose sensitivity to the topology/initialization pairing. Neither extremum guarantees better optimization or supplies a hardware state-preparation claim, and reporting all three does not by itself isolate a pure topology effect. Numerical calibration comes next; no new performance result from these controls is established here.

This is a correction to the experimental design, not a claim that the previous pilot was invalid.

## What the current evidence says

The strong-`|Q_ij|` rule beat the ring on 17/24 block-correlated pilot instances, then lost to the ring on 17/24 instances in the first independent weighted densest-k-subgraph benchmark. This makes that historical rule a **failed-to-generalize candidate heuristic**, not a validated algorithm. Both exploratory results remain preserved with their [historical status](../results/README.md).

The useful scientific distinction is now:

> high objective-interaction alignment is not equivalent to a useful shallow-QAOA mixer.

The current candidate is the exact fixed-cardinality conditional exchange RMS derived from `Q`, `c`, and `k`, with conditional mean and variance retained separately. Low-RMS and high-RMS edge selection are competing hypotheses. The question is whether this descriptor captures useful information missed by raw pairwise magnitude, and whether any resulting effect survives the three initialization controls. Its mathematical derivation and software checks establish neither performance superiority nor novelty relative to the literature.

## Claim boundary

Until the next experiment is completed, the repository should use language such as:

- “tests”;
- “candidate structural descriptor”;
- “exploratory evidence”;
- “failed to generalize on the first independent family.”

It should avoid language such as:

- “new mixer methodology”;
- “validated structure-conditioned mixer”;
- “architecture search”;
- “trainability improvement”;
- “quantum advantage.”

## Primary references

1. Hadfield et al. (2019), *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12, 34. https://doi.org/10.3390/a12020034
2. He et al. (2023), *Alignment between initial state and mixer improves QAOA performance for constrained optimization*, npj Quantum Information 9, 121. https://doi.org/10.1038/s41534-023-00787-5
3. Kordonowy & Leipold (2026), *The Lie algebra of XY-mixer topologies and warm starting QAOA for constrained optimization*, npj Quantum Information 12, 61. https://doi.org/10.1038/s41534-026-01192-4
4. Perlin et al. (2025), *Q-CHOP: Quantum constrained Hamiltonian optimization*, ACM Transactions on Quantum Computing. https://doi.org/10.1145/3778864
5. Du et al. (2020), *Quantum circuit architecture search for variational quantum algorithms*. https://arxiv.org/abs/2010.10217
6. Ni et al. (2026), *An Adaptive Mixer Allocation Strategy for the Quantum Alternating Operator Ansatz*. https://doi.org/10.1002/qute.202500487
