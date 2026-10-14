# Struct2Circuit

**A controlled test of cost–mixer alignment in constrained QAOA.** Does assigning the same sparse XY mixer graph to different variables of a fixed QUBO change shallow variational performance in a way a pre-optimization structural score can predict?

The starting point was a falsification: a mixer built from strong `|Q_ij|` interactions performed well in one pilot family but **lost to a ring in 17/24 instances** of the first independent family. The resulting research question is not whether every problem-informed mixer wins; it is **which properties of feasible cost changes, if any, help select useful constrained quantum dynamics**.

**Current status — October 2026:** The prospective **32-QUBO, matched-C6 alignment experiment is implemented and merged, but NOT EXECUTED**. There is no established conditional-exchange-RMS performance advantage, certified optimum ordering, scalability claim, or quantum advantage.

## Evidence: observation → falsification → mechanism question

| Stage | What was observed | What it establishes |
| --- | --- | --- |
| **Pilot** | On 24 block-correlated QUBOs (`n=8, k=3, p=1`), the original eight-edge strong-`|Q_ij|` mixer beat the ring on **17/24**; median normalized gap **0.101715 vs. 0.112558**. | A hypothesis-generating signal on one synthetic family. |
| **Independent family** | On 24 weighted densest-k-subgraph instances (`n=9, k=4, p=1`), the same heuristic **lost on 17/24**; median gap **0.488890 vs. 0.476476** for ring. | **Failed generalization.** High correspondence with interaction magnitudes did not imply better depth-one outcomes. |
| **Mechanistic correction** | A feasible XY exchange changes the cost according to diagonal and linear terms and *differences* between interaction rows. The direct `Q_ij x_i x_j` contribution cancels. | Raw `|Q_ij|` is not itself the energy difference induced by the exchange. An exact conditional exchange statistic is the next candidate descriptor. |

Primary navigable evidence: [pilot report](results/pilot_report.md), [independent DKS report](results/independent_dks_benchmark_v1.md), and [historical result qualifications](results/README.md).

Two additional **development-only** findings inform the new study but do not validate it. The frozen P33 numerical calibration **FAILED** (2/192 landscape discrepancies, 19/192 unresolved references; nine dependent contrast discrepancies). A later **post hoc eight-QUBO** mechanism analysis observed a high-RMS advantage over its shuffled comparator in 7/7 eligible uniform-initialization comparisons, with one comparison unresolved. Neither result is an independent prospective confirmation. The underlying P33/Stage 7N records are archived outside GitHub; the [development-evidence audit](docs/DEVELOPMENT_EVIDENCE_AUDIT.md) gives the disclosed counts, archive/hash pointers, and **explicit limits on third-party verification**. It is an evidence inventory, not a reproduction of unavailable raw data.

## Mathematical question: a meaningful exchange statistic is not a performance theorem

For symmetric `Q`, cardinality constraint `Σ_i x_i=k`, and `C(x)=xᵀQx+cᵀx`, exchanging occupied `i` with unoccupied `j` gives

```text
ΔC_ij(x) = Q_jj − Q_ii + c_j − c_i
           + 2 Σ_(ℓ ≠ i,j) x_ℓ (Q_jℓ − Q_iℓ).
```

The present score starts with the conditional exchange RMS

```text
ρ_ij = sqrt(E[(ΔC_ij)^2 | x_i=1, x_j=0, Σ_i x_i=k]),
```

where expectation is taken over the **uniform feasible prior**. Its conditional mean and variance are calculable directly from `Q`, `c`, and `k`, **before** simulating or optimizing a quantum circuit. Unlike `|Q_ij|`, it explicitly reflects feasible exchange-cost differences.

**Established mathematically:** the exchange identity and conditional moments; C6 graph relabelings preserve unlabeled topology, feasible-space mixer spectrum, and the uniform feasible state.

**Not established:** that a large RMS produces a better optimized p=1 objective, that RMS is the sole mediator, or that high-versus-low improvement would generalize. RMS is not a quantum transition probability. In fact, short-angle expansions of the positive-sign XY mixer can connect **squared** exchange-cost differences to an initially *uphill* objective slope at small positive cost/mixer angles. That is a reason to investigate finite-angle interference and optimizer behavior, **not** a proof for either score direction. The frozen prospective experiment tests a declared directional prediction; its design is not revised on the basis of this observation.

The motivating chain remains a **hypothesis**:

`cost descriptor → relative graph assignment → feasible-state quantum dynamics → finite-angle objective landscape → observed empirical reference`

Read the [mechanism derivation](docs/MECHANISM_HYPOTHESIS.md) and [literature positioning](docs/LITERATURE_DIFFERENTIATION.md). The scientific focus is more specific than proposing another generic custom mixer.

## Prospective experiment: why a six-vertex ring?

At `n=6, k=3`, the constrained Hilbert space has `C(6,3)=20` basis states. This deliberately **small, classically tractable** case supports exhaustive graph-orbit construction, transported-Hamiltonian identities, exact state-vector controls, and tight numerical-provenance checks. It is a **mechanism laboratory**, not a demonstration that quantum hardware solves hard instances.

The experiment keeps the QUBO fixed and selects two assignments of the **same six-cycle** mixer: high versus low conditional exchange-RMS alignment. Relabeling the graph changes its edges relative to the cost variables, while preserving topology, spectrum, XY edge weights, uniform initial state, `p=1`, and the B00 parameter domain.

| Frozen design element | Prespecified choice |
| --- | --- |
| Independent problem draws | **32 fresh QUBOs**, eight from each of four families |
| Candidate graph assignments | **720** permutations → **60** labeled C6 graphs; exact cost symmetries quotient equivalent assignments |
| Pairing | Highest versus lowest rounded alignment scores, selected **before any QAOA evaluations** |
| Primary contrast | `Δ = r_high − r_low`; a material high-alignment win requires both empirical references complete and `Δ < −0.00012` |
| Inference | **W ≥ 24** of 32 for directional support; **W + U ≤ 8** for contradiction of the prespecified majority prediction; otherwise inconclusive |
| Measurement safeguards | Two independent reference routes, mandatory fine/boundary/controls, conserved 32-unit denominator, missingness, linked numerical admissions and evidence; 20,000 evaluations/arm and 3,600 cumulative CPU-seconds |

This comparison isolates the **selection-rule contrast within fixed C6**; it does **not** establish superiority over a random C6 assignment or a structure-agnostic algorithm. Even a positive finding would be about **finite numerical references**, not globally certified B00 minima. Incomplete arms remain unknown rather than excluded or rerun with expanded budgets.

**[Frozen scientific design](paper/controlled_mixer_alignment_v1.md)** · [experiment implementation/authorization guide](src/struct2circuit/alignment_experiment/README.md) · [entry point](experiments/run_controlled_alignment.py).

**Execution boundary:** No official panel generated, no experiment executed, and no protected validation-holdout data accessed. Running it requires a separate operational preflight and explicit persisted authorization. A merged implementation or successful test suite is **not** that authorization.

## Software, ownership, and reproducibility

This independent project includes exact feasible-subspace QAOA, constrained QUBO generators, ring/random/instance-conditioned XY mixers, three initialization diagnostics (`uniform`, `mixer_low`, `mixer_high`), dimensionless cost-angle treatment, and numerical-integrity/provenance tests. Changing a topology-dependent initial state is **not** interpreted as a pure topology intervention.

The scientific contribution being developed is the **question and experimental test** of a coefficient-only cost–mixer alignment rule under controlled dynamics. This repository does not establish a new optimal mixer, universal algorithm, hardware-effective circuit, or asymptotic speedup. Repository history records code and decisions; it does not by itself verify the division of intellectual contributions among any outside assistance, tools, or collaborators. Contributions and external assistance should be described accurately in any application or manuscript.

Set up a development environment and run regression tests (**no official experiments**):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 tests/run_calibration_regressions.py
```

The controlled experiment's default command prints metadata only, without generating QUBOs:

```bash
python3 experiments/run_controlled_alignment.py
```

At the merged implementation revision, CI reported **440 tests run, 2 skipped, 0 failures**. These tests verify software behavior, not experimental qualification or the proposed hypothesis. Official numerical execution has separately pinned runtime and authorization conditions: consult the [experiment guide](src/struct2circuit/alignment_experiment/README.md).

## Claim boundary

The first strong-`|Q_ij|` heuristic **failed its independent-family check**. The conditional-RMS mechanism hypothesis remains **prospective and untested**. A negative, mixed, or inconclusive controlled outcome must be reported as faithfully as a positive one. Beyond the present study, evaluation of practical selection value would require appropriate structure-agnostic comparators and tests across larger systems and other graph families; these are **not** retroactive additions to the frozen design.

> **Research principle:** The objective is not to make a structure-conditioned mixer win. It is to find out when—and why—problem information does or does not improve constrained quantum dynamics.
