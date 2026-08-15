# Research protocol: verified structure-conditioned mixers

## Single central claim

> A structure-conditioned, feasibility-preserving mixer improves the solution-quality/resource Pareto frontier over fixed mixers for cardinality-constrained QUBOs.

This is deliberately narrower than “AI discovers quantum algorithms.” It can be falsified with paired experiments, has a formal invariant, and creates a clean path from a transparent heuristic to a learned circuit generator.

## Scientific object

We study

\[
\min_{x\in\{0,1\}^n} C_Q(x)=x^TQx+c^Tx
\quad\text{subject to}\quad \mathbf{1}^Tx=k.
\]

The Hamming-weight operator is

\[
N=\sum_i (I-Z_i)/2.
\]

Every permitted mixer unitary must satisfy \([U_M,N]=0\). In the pilot, this is guaranteed by representing the state only in the weight-\(k\) basis and using XY exchanges on a connected graph.

## Contribution ladder

1. **Verification contribution:** a typed mixer representation that rejects disconnected or cardinality-violating constructions.
2. **Algorithmic contribution:** a structure-conditioned generator that selects a sparse, connected mixer graph from the QUBO interaction matrix.
3. **Learning contribution:** replace the transparent edge score with a learned policy while retaining the same verifier and edge budget.
4. **Scientific contribution:** discover stable relationships between instance structure and useful mixer motifs that transfer across sizes, families and hardware.
5. **Utility contribution:** establish Pareto improvement after including search amortization, shots, transpilation and strong classical baselines.

Only levels actually supported by evidence may appear in the title or abstract.

## Pilot design

- Family: synthetic block-correlated cardinality QUBOs motivated by index tracking.
- Size: `n=8`, `k=3`, exact feasible dimension `56`.
- Depth: `p=1`.
- Initial state: uniform Dicke state over all feasible decisions.
- Equal-budget comparison: ring mixer versus structure-conditioned connected mixer, each with `n` edges.
- Resource reference: complete mixer with `n(n-1)/2` edges.
- Optimization: identical deterministic grid plus bounded local refinement.
- Primary pilot metric: paired normalized expectation gap, lower is better.
- Secondary: probability of sampling an exact optimum.
- Pilot decision rule: advance only if the median paired gap reduction is positive and its exploratory bootstrap interval excludes zero.

The pilot is used for debugging and effect-size estimation. It is not confirmatory.

## Confirmatory design to lock before execution

### Instance families

1. Block-correlated portfolio/index-tracking proxy.
2. Densest-`k`-subgraph or Max-`k`-vertex-cover.
3. A deliberately weak-structure/null ensemble.

The null ensemble is essential: the method should not claim universal improvement when no exploitable matrix structure exists.

### Splits

- Training: mixer-policy learning only.
- Validation: architecture/hyperparameter selection.
- Blind test: locked seeds and distributions; opened once.
- Transfer test: unseen `n`, unseen `k/n`, and unseen hardware topology.

### Baselines

- fixed ring XY-QAOA;
- complete XY-QAOA;
- random connected graphs at the same edge budget;
- strongest-edge graph without spanning-tree enforcement;
- warm-start QAOA;
- counterdiabatic/commutator-informed constrained QAOA where reproducible;
- random/evolutionary QAS and a predictor-based QAS baseline;
- exact MIQP/CP-SAT on tractable sizes;
- tuned greedy, local search, tabu and simulated annealing.

### Metrics

- normalized optimality gap and approximation ratio;
- probability of an optimum and probability within `epsilon` of optimum;
- feasibility probability;
- CVaR of sampled costs;
- circuit evaluations, shots and classical search time;
- two-qubit gate count/depth and routing SWAPs;
- noise robustness and calibration sensitivity;
- zero-shot transfer loss;
- amortized time-to-target including architecture search.

### Statistics

- paired bootstrap confidence intervals;
- paired Wilcoxon tests with Holm correction across primary comparisons;
- effect sizes and full distributions, not p-values alone;
- learning/scaling curves with uncertainty;
- all seeds, failed optimizations and excluded runs reported.

## Confirmatory success gates

- **G1 invariant:** feasibility at least `0.99` under finite shots.
- **G2 equal-budget effect:** positive paired median gap reduction against the strongest equal-edge fixed/random baseline on at least two blind families, with corrected confidence interval excluding zero.
- **G3 resource effect:** the discovered mixer is non-dominated in gap versus transpiled two-qubit depth.
- **G4 transfer:** positive effect survives an unseen size and hardware graph without architecture retraining.
- **G5 utility:** any end-to-end utility claim requires amortized time-to-target superiority over tuned classical solvers at a shared target quality.

## Key ablations

- remove QUBO structure and retain only graph size;
- remove the connectivity verifier;
- remove hardware features;
- replace learned edge scores with `|Q_ij|`, random scores and spectral scores;
- equalize edge count, depth and total circuit-evaluation budget separately;
- permute variable labels to test equivariance;
- test a null ensemble in which `Q` structure is uninformative.

## Closest-work pressure points

Reviewers will correctly note that constraint-preserving mixers, quantum architecture search, structure-aware circuit diagnostics and hardware-aware search already exist. Therefore the paper cannot claim novelty from any one of those phrases. Its defensible novelty must be the verified joint formulation, cross-instance/size transfer, controlled resource comparison, and an interpretable structure-to-mixer law.

The DQI project should remain separate unless a mathematical reduction or codeability criterion is established. DQI is an algorithmic decoding framework, not simply a gate motif to append to QAOA.

## Representative references

1. Farhi, Goldstone and Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028 (2014).
2. Hadfield et al., *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12, 34 (2019).
3. Zhang et al., *Neural Predictor based Quantum Architecture Search*, arXiv:2103.06524 (2021).
4. Duong et al., *Quantum Neural Architecture Search with Quantum Circuits Metric and Bayesian Optimization*, arXiv:2206.14115 (2022).
5. Bucher et al., *Towards Robust Benchmarking of Quantum Optimization Algorithms*, arXiv:2405.07624 (2024).
6. Lipardi et al., *Magic-Informed Quantum Architecture Search*, arXiv:2605.03932 (2026).
7. Assis et al., *A SWAP-free Framework for QAOA*, arXiv:2604.25058 (2026).
8. Falla and Safro, *Constrained Counterdiabatic Quantum Approximate Optimization Algorithm for Portfolio Optimization*, arXiv:2605.06858 (2026).
9. Niu et al., *HamQASBench: A Hamiltonian-Informed Diagnostic Benchmark for Evaluating Quantum Architecture Search*, arXiv:2607.04845 (2026).
10. Leonidas et al., *Quantum Portfolio Optimization: An Extensive Benchmark*, arXiv:2509.17876 (2026 revision).

