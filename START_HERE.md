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

## Current status and next decision (November 2026)

**The P33 numerical calibration FAILED**, with 2 failed and 19 unresolved references among 192 landscapes; see the [development-evidence audit](docs/DEVELOPMENT_EVIDENCE_AUDIT.md). Its reference schedules are not qualified. The later Stage 7N high-RMS-versus-shuffled signal (7/7 eligible comparisons, one unresolved) was **post hoc and exploratory**, with raw records outside this repository; it does not validate the descriptor.

A subsequent **32-QUBO fixed-C6 high/low alignment experiment** has an approved, [frozen two-arm design](paper/controlled_mixer_alignment_v1.md) and [implemented code](src/struct2circuit/alignment_experiment/README.md), but it is **NOT EXECUTED**. It compares high versus low conditional-RMS assignments of the same six-cycle graph at `n=6, k=3`; this controls graph topology and spectrum but does **not** test against a random assignment. No official panel has been generated and no authorization to execute is implied by the implementation or passing tests.

An alternative H/L/R design and broader comparisons remain scientific proposals, **not changes to the frozen protocol**. They require independent review and a separately approved study. Do not turn the original calibration plan, earlier five-comparator list, or any exploratory result into a retrospective modification of the C6 design.

For implementation semantics (legacy versus normalized gamma, compensated cost metrics, withheld unresolved gaps, and initialization diagnostics), consult the [scientific scope](docs/SCIENTIFIC_SCOPE.md) and [experiment guide](src/struct2circuit/alignment_experiment/README.md). Future tests must preserve historical data and require fresh output paths. The [DKS CSV schema erratum](results/independent_dks_schema_erratum.md) explains the three misleading historical sign-column names; future regenerated summaries use corrected headers.

## Repository map

- `README.md` — project overview, current evidence, and claim boundary.
- `docs/MECHANISM_HYPOTHESIS.md` — mechanism, controls, and falsification boundary.
- `docs/LITERATURE_DIFFERENTIATION.md` — closest literature and precise scope.
- `results/README.md` — current interpretation and preservation policy for historical results.
- `results/independent_dks_benchmark_v1.md` — first independent benchmark result.
- `results/independent_dks_schema_erratum.md` — historical DKS paired-difference header/sign correction.
- `docs/DEVELOPMENT_EVIDENCE_AUDIT.md` — disclosed P33 and Stage 7N observations and verification limits.
- `paper/controlled_mixer_alignment_v1.md` — frozen prospective C6 design (not executed).
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
