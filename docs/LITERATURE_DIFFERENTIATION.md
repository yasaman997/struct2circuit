# Literature differentiation

This document is a direct comparison map, not a novelty claim. The purpose is to identify what Struct2Circuit shares with prior work, what it does not yet do, and where the actual empirical question begins.

## 1. Closest work

| Work | What it establishes | Overlap with Struct2Circuit | What is different here |
| --- | --- | --- | --- |
| Hadfield et al., *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz* (2019) | Generalizes QAOA to feasibility-preserving/custom mixing operators for hard constraints. | Struct2Circuit uses a fixed-weight XY mixer, exactly the kind of constrained mixer enabled by this framework. | The present question is not whether a constrained mixer is useful. It asks whether the topology of a sparse XY mixer should be conditioned on the target QUBO under a fixed edge budget. |
| He et al., *Alignment between initial state and mixer improves QAOA performance for constrained optimization* (2023) | Shows that initial-state/mixer alignment matters for constrained QAOA, including ring, complete, and other XY topologies on portfolio instances. | It directly demonstrates that XY connectivity can affect low-depth QAOA outcomes. | The alignment studied is between the initial state and mixer ground state. Struct2Circuit instead asks whether the mixer topology itself can be selected from instance information before optimization, while holding the mixer edge count fixed. |
| Kordonowy & Leipold, *The Lie algebra of XY-mixer topologies and warm starting QAOA for constrained optimization* (2026) | Shows that XY topology changes the accessible Lie algebra and can affect trainability, simulability, and optimization; studies path, cycle, clique and richer generator sets. | This is a major pressure point because it shows that mixer topology is scientifically substantive, not merely an implementation detail. | The present study does not claim a Lie-algebra or trainability result. Its first question is whether an instance-conditioned sparse topology gives better shallow-QAOA outcomes at a controlled edge budget. A future mechanism study must distinguish topology-selection effects from generic DLA effects. |
| Perlin et al., *Q-CHOP: Quantum constrained Hamiltonian optimization* (2025 ACM TQC; arXiv 2024) | Introduces a different constrained quantum optimization algorithm that keeps evolution in the feasible subspace and benchmarks constrained Hamiltonian optimization broadly. | Same broad problem class: constrained combinatorial optimization and explicit feasibility handling; Eric Anschuetz is a coauthor. | Q-CHOP is an adiabatic/continuous-time Hamiltonian interpolation method, not an instance-conditioned sparse XY-QAOA topology rule. Struct2Circuit should not position itself as a new constrained-optimization framework. |
| Du et al., *Quantum circuit architecture search for variational quantum algorithms* (2020) | Shows that automated circuit-architecture search can improve VQA performance under resource/noise considerations. | It establishes architecture choice as an optimization problem and motivates comparing ansatz design strategies. | Struct2Circuit currently performs no architecture search and learns nothing. Its intervention is a deterministic, transparent, per-instance graph rule; calling it “architecture search” would overstate the work. |
| Ni et al., *An Adaptive Mixer Allocation Strategy for the Quantum Alternating Operator Ansatz* (2026) | Adapts which mixer operations are applied using an evaluation function and reports solution-quality/resource benefits for MIS. | Very close in the broad idea of allocating constrained-mixer structure rather than using one fixed mixer. | AMA-QAOA+ selects mixer allocation during the algorithmic procedure for a different problem class. Struct2Circuit selects a sparse XY topology directly from QUBO interaction magnitudes before parameter optimization and tests it under a fixed edge budget. |

## 2. What the project is not claiming

The literature makes the following distinctions important:

- **Constrained mixer:** established prior art; not the contribution.
- **Custom/problem-tailored mixer:** established prior art; not enough by itself.
- **Mixer topology affects performance:** already established; not new by itself.
- **Architecture search:** established and much broader than the current method.
- **Trainability/topology mechanisms:** active recent literature; current work does not establish one.
- **Instance-conditioned sparse XY topology under a matched edge budget:** this is the narrow empirical niche being tested.

Even that niche should not be described as novel until a more systematic literature search confirms that no prior work has already performed essentially the same comparison.

## 3. The direct scientific differentiation

The cleanest current formulation is:

> Given a cardinality-constrained QUBO and a fixed mixer edge budget, does a transparent topology rule based only on the QUBO interaction matrix outperform structure-agnostic connected XY topologies on unseen instances?

This is a **controlled empirical question about side information for ansatz design**, not a claim that Struct2Circuit invented constrained mixers, custom mixers, QAOA architecture search, or trainability theory.

## 4. Current evidence changes the positioning

The first independent benchmark on weighted densest-k-subgraph instances does not support the current strong-|Qij| rule as a generally useful topology heuristic. Therefore the present project should now be framed as a test of a specific structural hypothesis, including its failure, rather than as a search for evidence that structure conditioning “works.”

That negative result is scientifically useful because it forces the mechanism question into the open: if the objective interaction graph matters, which graph property should determine a mixer topology, and under what regimes?

## Primary references

1. Hadfield et al. (2019), Algorithms 12, 34. https://doi.org/10.3390/a12020034
2. He et al. (2023), npj Quantum Information 9, 121. https://doi.org/10.1038/s41534-023-00787-5
3. Kordonowy & Leipold (2026), npj Quantum Information 12, 61. https://doi.org/10.1038/s41534-026-01192-4
4. Perlin et al. (2025), ACM Transactions on Quantum Computing, Q-CHOP. https://doi.org/10.1145/3778864
5. Du et al. (2020), Quantum circuit architecture search for variational quantum algorithms. https://arxiv.org/abs/2010.10217
6. Ni et al. (2026), An Adaptive Mixer Allocation Strategy for the Quantum Alternating Operator Ansatz. https://doi.org/10.1002/qute.202500487
