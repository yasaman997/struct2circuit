# Struct2Circuit

**Independent research in constrained quantum optimization.** Struct2Circuit investigates whether information in a cardinality-constrained quadratic binary optimization problem (QUBO) can inform the choice of a sparse, feasibility-preserving XY mixer for shallow QAOA.

> **Research question:** Can a transparent, pre-optimization descriptor of a problem instance identify a mixer assignment that improves constrained variational optimization—and under what conditions does the relationship fail?

**Status (October 2026):** Exploratory mechanism research. A controlled C6 mixer-alignment experiment has been designed, reviewed, implemented, and merged into `main`, **but has not been executed**. There is **no established performance advantage** for conditional-exchange-RMS selection, no certified global-optimum comparison, and no quantum-advantage claim.

## Scientific progress and evidence

| Step | Evidence | Interpretation |
| --- | --- | --- |
| Exploratory pilot | Structure-conditioned mixer beat an eight-edge ring on **17/24** block-correlated QUBOs (`n=8, k=3, p=1`); median normalized gap **0.101715 vs. 0.112558**. | Hypothesis-generating; one synthetic family. |
| Independent family check | On **24** weighted densest-k-subgraph QUBOs (`n=9, k=4, p=1`), the original strong-`\|Q_ij\|` rule **lost to ring on 17/24**; median gap **0.488890 vs. 0.476476**. | The original heuristic did **not generalize**. High interaction-weight alignment was insufficient. |
| Numerical calibration | The frozen P33 development qualification **FAILED**: **2/192** landscape-level objective discrepancies and **9** dependent graph-contrast discrepancies; **19/192** empirical references remained unresolved. | Optimizer/reference reliability is a separate issue from mixer quality. Historical failed evidence is preserved, not relabeled as a pass. |
| Exploratory mechanism analysis | On the eight development QUBOs, high-RMS beat shuffled-high-RMS in **7/7 eligible** uniform-initialization comparisons; some equal-spectrum mixer pairs showed different observed empirical references. Initialization comparisons also showed phase-sensitive leads. | Post hoc, small-sample, incomplete-reference observations; neither causal isolation nor independent validation. |
| Controlled follow-up | A **32-QUBO**, four-family, paired C6 relabeling experiment is implemented with independent numerical-reference routes, explicit missingness, and bounded evidence accounting. | **Implemented and tested, not run.** Its directional hypothesis remains unanswered. |

The historical pilot and independent-family result are described in [the saved-results status note](results/README.md) and [independent benchmark report](results/independent_dks_benchmark_v1.md). The Stage 7N exploratory report and P33 development evidence are maintained in separate workspace archives; they are **not included in this GitHub repository**, so the brief exploratory numbers above are not a substitute for access to those records. See [the controlled experiment specification](paper/controlled_mixer_alignment_v1.md) for the prospective design. The previously published calibration protocols are unchanged: [v1](paper/numerical_calibration_protocol_v1.md) and [v2](paper/numerical_calibration_protocol_v2.md).

## Why the descriptor changed

The first heuristic selected graph edges with large off-diagonal `|Q_ij|`. But for a feasible exchange moving one occupied variable `i` to one unoccupied variable `j`, the direct interaction term `Q_ij x_i x_j` vanishes on both sides of the swap. It therefore does **not** alone describe the energy change of that exchange.

For symmetric `Q` and `C(x) = xᵀQx + cᵀx`, the feasible swap has cost difference

```text
ΔC_ij(x) = (Q_jj − Q_ii) + (c_j − c_i)
           + 2 Σ_(ℓ ≠ i,j) x_ℓ (Q_jℓ − Q_iℓ).
```

The current descriptor is the **conditional exchange root-mean-square cost difference**,

```text
ρ_ij = sqrt( E[(ΔC_ij)^2 | x_i=1, x_j=0, Σx=k] ),
```

where the expectation is over the *uniform feasible prior*, not a QAOA output distribution. Its conditional mean and variance are derived algebraically from the QUBO coefficients. It is a physically better motivated **candidate descriptor**, not a proven predictor of QAOA performance. See [the mechanism derivation](docs/MECHANISM_HYPOTHESIS.md) and [literature differentiation](docs/LITERATURE_DIFFERENTIATION.md).

The working mechanism question is:

`instance descriptor → mixer assignment → feasible-state transition graph → QAOA interference and landscape → observed optimization outcome`

Each implication remains to be tested; a high RMS score is not assumed to be intrinsically beneficial.

## Current prospective experiment: matched C6 alignment

The approved study changes **which QUBO variables occupy the vertices of a six-cycle XY mixer**, while holding the cost function fixed. A permutation relabels the graph as `G' = PGPᵀ`. The two arms therefore share the same **unlabeled topology, feasible-space mixer spectrum, uniform initial state, circuit depth, parameter domain, and QUBO**. Only the relative cost–mixer assignment changes.

- **Panel:** 32 fresh QUBOs, eight from each of four prespecified families; `n=6, k=3, p=1`.
- **Treatment:** enumerate 720 vertex permutations yielding 60 distinct labeled C6 graphs; remove exact cost-symmetry equivalences; use a prespecified normalized conditional-exchange-RMS score to select the highest- and lowest-alignment assignments **before evaluating QAOA**.
- **Outcome:** paired complete empirical B00 references, `Δ = r_high − r_low` (negative favors higher alignment). A *material win* requires both arms complete and `Δ < −0.00012`.
- **Decision:** directional support requires **at least 24 material wins out of all 32 QUBOs**; contradiction of the specified majority prediction requires `W + U ≤ 8`; otherwise **inconclusive**. Unresolved comparisons remain in the denominator.
- **Numerical safeguards:** independent grid/local and DE/polish routes, mandatory refinement and boundary checks, candidate/control provenance, read-only evidence verification, one-use execution authorization, and hard limits of **20,000 actual evaluations per arm**, **1.28 million total**, and **3,600 cumulative CPU-seconds**.

Matching topology and spectrum make this a controlled test of the **high-versus-low alignment selection rule within C6**. Even a positive result would not prove that RMS is the unique physical mechanism or that the obtained empirical references are global optima. Numerical budget exhaustion may yield incomplete comparisons, and the sample size is intentionally conservative.

**Implementation:** [experiment guide](src/struct2circuit/alignment_experiment/README.md) · [frozen design](paper/controlled_mixer_alignment_v1.md) · [entry point](experiments/run_controlled_alignment.py).

**Execution status:** The official 32-QUBO panel has **not** been generated or evaluated. No protected historical validation holdout has been accessed. Running the study requires a **separate explicit authorization** following operational preflight; code merge and green tests do not authorize execution.

## Implementation and reproducibility

The repository contains:

- Exact fixed-Hamming-weight feasible-subspace QAOA simulation and XY mixer Hamiltonians.
- Cardinality-constrained QUBO generators; ring, random-connected and structure-conditioned mixer constructions.
- Uniform initialization and `mixer_low`/`mixer_high` spectral sensitivity diagnostics. Graph-dependent initial states are **not** treated as pure topology effects.
- Dimensionless cost-angle conventions, compensated feasible-cost arithmetic and explicit unresolved-cost handling.
- Independent benchmark and exploratory analysis code, numerical-calibration history, prospective alignment experiment machinery, and synthetic/integrity regressions.

Run the test suite without starting the official study:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 tests/run_calibration_regressions.py
```

The alignment experiment's default entry point **prints metadata only**:

```bash
python3 experiments/run_controlled_alignment.py
```

The last merged PR passed **440 automated tests** (including **2 skipped** in GitHub Actions), covering scientific invariants, numerical reference bookkeeping and adversarial evidence-integrity cases. Passing tests support implementation correctness; they do not provide research results. The officially authorized execution environment is separately pinned; consult the [experiment guide](src/struct2circuit/alignment_experiment/README.md), rather than assuming that a local Python environment qualifies for execution.

## Scope and scientific claims

The project is about **instance-conditioned constrained mixer design**, not inventing XY mixers, circuit architecture search, demonstrating hardware advantage, or proving asymptotic quantum speedup. See [related-work positioning](docs/LITERATURE_DIFFERENTIATION.md) and the [prospective manuscript outline](paper/manuscript_outline.md).

The previous structure-conditioned heuristic failed its first independent-family generalization check. The new alignment hypothesis is **prospective and untested**. Positive, negative, mixed and unresolved outcomes are all legitimate research findings when reported with their numerical and sampling limitations.

> **Research principle:** The objective is not to make a structure-conditioned mixer win. It is to determine when—and why—available problem structure does or does not improve constrained quantum optimization.
