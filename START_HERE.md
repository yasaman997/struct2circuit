# Struct2Circuit: start here

> **Project entry point:** start with [README.md](README.md). This page is a compact orientation guide to the current scientific question and status.

## The current question

Struct2Circuit asks:

> **Can a transparent pre-optimization instance descriptor select a sparse, feasibility-preserving mixer topology that improves shallow variational optimization over structure-agnostic connected topologies, and under what conditions does that relationship fail?**

The historical first descriptor was off-diagonal interaction magnitude `|Q_ij|`. The first independent family did not support that rule as a generally useful heuristic. The current candidate is exact fixed-cardinality conditional exchange RMS, with low/high rankings treated as competing, unvalidated hypotheses.

## What the evidence says

The original 24-instance block-correlated pilot favored the structure-conditioned mixer over the fixed ring on 17/24 instances.

The first independent weighted densest-k-subgraph test reversed that pattern: the strong-`|Q_ij|` mixer lost to the ring on 17/24 instances and had the worst median normalized gap among the main three comparators.

Read the [historical-results status note](results/README.md) before interpreting the preserved pilot report's original advancement decision. That decision is historical provenance, not current approval for a larger benchmark.

The important result is therefore not “structure works.” It is:

> **High alignment with QUBO interaction magnitude was not sufficient to improve p=1 QAOA on the independent family.**

## The main experimental correction

The original comparison fixed the same uniform feasible initial state for every mixer. That is useful for a fixed-initialization comparison, but it does not isolate topology from mixer/initial-state alignment.

The simulator supports three explicit initialization conditions:

- `uniform` — uniform feasible superposition;
- `mixer_low` — opposite-extremum sensitivity condition in the lowest eigenspace;
- `mixer_high` — positive-amplitude Perron–Frobenius reference for connected mixers, equivalently the ground state of `-H_M`.

The implemented XY Hamiltonian has positive exchange amplitudes. Neither extremum guarantees better optimization; the deterministic spectral fallback when uniform overlap is zero can depend on labels. This is a topology × initialization sensitivity diagnostic, not a claim that topology has been causally isolated. It is implemented in:

```text
experiments/run_alignment_control.py
```

No alignment-control result is currently treated as evidence for the main claim.

## Mechanism status

The original local `|Q_ij|` intuition was corrected: for a single feasible exchange, the direct `Q_ij x_i x_j` contribution cancels.

The remaining hypothesis is graph-level:

```text
instance descriptor
        ↓
  mixer topology
        ↓
feasible-state transition graph
        ↓
variational dynamics
        ↓
optimization outcome
```

The missing link is the transition-graph mechanism. See [docs/MECHANISM_HYPOTHESIS.md](docs/MECHANISM_HYPOTHESIS.md).

## Current next step

Do **not** start another large confirmatory benchmark yet.

The next scientific phase is a small, predeclared gamma/beta-domain and optimizer calibration. Controlled mechanism comparisons follow it; no large confirmatory benchmark is authorized or frozen.

The planned study compares historical strong/weak `|Q_ij|` with low/high exact conditional exchange RMS at fixed `k`, plus shuffled and random controls. Conditional mean and variance are retained separately; the former coefficient norm is no longer the primary descriptor. Low/high are competing hypotheses. All three initialization conditions are reported.

New runs optimize `u = gamma * (C_max-C_min)` on `[0,2*pi]`, with centered normalized costs and identical bounds for the grid and both local solvers. Returned gamma remains in physical units. This gamma interval and the shared beta interval `[0,pi]` are declared comparison conventions, not universal periods. Both historical entry points explicitly retain `legacy` gamma semantics. No scientific domain calibration or new performance claim accompanies these repairs.

Normalized costs use compensated coefficient summation without an absolute unit cutoff. For resolved objectives, optimum probability uses exact minimum equality of those float64 costs before normalization, with no near-optimum tolerance. Final cost rounding can still produce numerical ties. If no distinct costs remain, the combined constant/unresolved status withholds normalized gap and optimum probability as JSON `null` or blank CSV. Optimum probability would be one for a proven constant; the combined policy intentionally does not make that distinction. Raw legacy arithmetic is preserved. Recomputed probability metrics may differ from historical values; stored results are not rewritten. All raw rows are retained, and summaries validate declared instance IDs, comparators, and every summarized outcome, including optimum probability; refusals report affected/expected counts. Alignment runs enforce the actual ring edge budget and record actual edge counts.

The repository already contains statistical sensitivity and pre-freeze infrastructure. Those tools remain available for a later confirmatory study; they are not the current scientific bottleneck.

## Repository map

- `README.md` — project overview, current evidence, and claim boundary.
- `docs/MECHANISM_HYPOTHESIS.md` — mechanism, controls, and falsification boundary.
- `docs/LITERATURE_DIFFERENTIATION.md` — closest literature and precise scope.
- `results/README.md` — current interpretation and preservation policy for historical results.
- `results/independent_dks_benchmark_v1.md` — first independent benchmark result.
- `experiments/run_alignment_control.py` — initialization/alignment control.
- `src/struct2circuit/simulator.py` — exact feasible-subspace simulation and initialization modes.
- `tests/` — mathematical and implementation invariants.

## Reproduce the checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

An optional reproduction of the historical pilot should write to a separate directory:

```bash
python3 experiments/run_pilot.py --instances 24 --n 8 --k 3 --output /tmp/struct2circuit-pilot-reproduction
```

The alignment-control experiment is exploratory and uses a separate seed namespace:

```bash
python3 experiments/run_alignment_control.py --instances 24 --n 9 --k 4
```
