# Working manuscript outline

## Working title

**Struct2Circuit: Verified Structure-Conditioned Mixers for Cardinality-Constrained Quantum Optimization**

## One-sentence contribution

We introduce a verified architecture-search framework that conditions sparse XY-mixer topology on optimization-instance structure and test whether it improves the quality/resource frontier with transfer to unseen instances, sizes and hardware graphs.

## Abstract skeleton

1. **Problem:** constrained QAOA performance depends strongly on a hand-designed mixer, while generic QAS can waste search on infeasible or hardware-expensive circuits.
2. **Method:** a typed graph grammar guarantees Hamming-weight preservation and connectivity; a structure-conditioned policy proposes sparse mixers under an explicit edge/depth budget.
3. **Evaluation:** blind paired benchmarks across portfolio-like and graph problems; equal search, shot and transpilation budgets; exact and noisy simulation plus small-device validation.
4. **Result:** report only the effect that survives the locked analysis, including null or negative regimes.
5. **Meaning:** identify when QUBO structure predicts useful quantum mixing topology and when it does not.

## Figures required for a submission

1. Method diagram: instance graph + hardware graph -> verified mixer generator -> QAOA evaluation -> active-learning loop.
2. Equal-budget paired performance on blind families.
3. Approximation-quality versus transpiled two-qubit depth Pareto fronts.
4. Transfer heat map across train/test size, density and hardware.
5. Ablation and null-ensemble results.
6. Interpretable motif analysis connecting matrix structure to selected edges.

## Main sections

1. Introduction and falsifiable contribution.
2. Related work and exact differentiation from QAS, custom mixers and hardware-aware compilation.
3. Cardinality-constrained problem and invariant-preserving grammar.
4. Structure-conditioned generator and verifier.
5. Experimental protocol and matched-budget baselines.
6. Results: quality, resources, transfer and statistical uncertainty.
7. Mechanistic interpretation and failure regimes.
8. Limitations: simulation scale, search amortization, noise drift and absence of broad quantum advantage.
9. Reproducibility statement.

## Journal decision rule

- **PRX Quantum:** only if there is a theorem or a strong cross-family/hardware milestone.
- **Quantum:** primary target for a significant, rigorous algorithmic result with open code and convincing transfer.
- **npj Quantum Information:** strong option if real-hardware and broad quantum-information evidence dominate.
- **Quantum Science and Technology / Physical Review Research:** appropriate if the contribution remains primarily numerical or application-centered.

