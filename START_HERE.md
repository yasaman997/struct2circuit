# Struct2Circuit: start here

> **Project entry point:** start with [README.md](README.md). This page is a compact orientation guide to the current scientific question and status.

## The current question

Struct2Circuit asks:

> **Can a transparent pre-optimization instance descriptor select a sparse, feasibility-preserving mixer topology that improves shallow variational optimization over structure-agnostic connected topologies, and under what conditions does that relationship fail?**

The first descriptor tested is off-diagonal interaction magnitude `|Q_ij|`. The first independent family did not support that rule as a generally useful heuristic.

## What the evidence says

The original 24-instance block-correlated pilot favored the structure-conditioned mixer over the fixed ring on 17/24 instances.

The first independent weighted densest-k-subgraph test reversed that pattern: the strong-`|Q_ij|` mixer lost to the ring on 17/24 instances and had the worst median normalized gap among the main three comparators.

The important result is therefore not “structure works.” It is:

> **High alignment with QUBO interaction magnitude was not sufficient to improve p=1 QAOA on the independent family.**

## The main experimental correction

The original comparison fixed the same uniform feasible initial state for every mixer. That is useful for a fixed-initialization comparison, but it does not isolate topology from mixer/initial-state alignment.

The simulator now supports two explicit initialization conditions:

- `uniform` — uniform feasible superposition;
- `mixer_low` — deterministic lowest-eigenspace reference;
- `mixer_high` — deterministic highest-eigenspace reference.

Both extrema are kept because the sign convention matters. This is an initialization-sensitivity diagnostic, not a claim that topology has been causally isolated. It is implemented in:

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

The next study compares strong/weak `|Q_ij|` with low/high exchange-profile scores derived directly from the exact feasible swap-cost formula, plus shuffled and random controls. It reports all three initialization conditions and uses cost-span-normalized gamma search for new runs.

The repository already contains statistical sensitivity and pre-freeze infrastructure. Those tools remain available for a later confirmatory study; they are not the current scientific bottleneck.

## Repository map

- `README.md` — project overview, current evidence, and claim boundary.
- `docs/MECHANISM_HYPOTHESIS.md` — mechanism, controls, and falsification boundary.
- `docs/LITERATURE_DIFFERENTIATION.md` — closest literature and precise scope.
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

The existing pilot can be regenerated with:

```bash
python3 experiments/run_pilot.py --instances 24 --n 8 --k 3
```

The alignment-control experiment is exploratory and uses a separate seed namespace:

```bash
python3 experiments/run_alignment_control.py --instances 24 --n 9 --k 4
```
