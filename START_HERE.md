# Struct2Circuit: start here

> **Project entry point:** start with [README.md](README.md). This page is a compact orientation guide for the repository's current scientific scope and status.

## The current question

Struct2Circuit studies one narrow question:

> **At a fixed number of mixer edges, does a transparent QUBO-conditioned, feasibility-preserving mixer improve variational optimization outcomes over structure-agnostic connected mixers on unseen cardinality-constrained QUBOs?**

The current structure signal is exactly the off-diagonal interaction magnitude |Q_ij|. The generator builds a maximum-weight spanning tree under that score and then adds the strongest remaining edges until the requested edge budget is reached.

It does **not** use the linear term, optimal solutions, objective values, solver results, benchmark outcomes, or labels.

This is a study of **constrained circuit design**, not a learned architecture-search system and not a quantum-advantage claim.

## What is implemented

### Exploratory pilot

The deterministic, noiseless pilot uses 24 synthetic block-correlated instances with n=8, k=3, and QAOA depth p=1.

The structure-conditioned and ring mixers each use eight edges. The structure-conditioned mixer wins 17 of 24 instances, with median normalized gap changing from 0.112558 to 0.101715.

A 28-edge complete mixer is shown only as a higher-edge-count reference.

The result is exploratory and family-specific. It does not establish generalization, scaling, hardware performance, trainability, or quantum advantage.

### Stage 1 foundations

Stage 1 implements and tests:

- an instance-independent random connected mixer with an exact edge budget;
- additional structured and weak-structure problem generators;
- deterministic provisional train/validation/blind/transfer manifest tooling;
- procedural safeguards against accidental blind-record access.

Benchmark v1 remains DRAFT_UNFROZEN. No blind performance has been evaluated.

### Pre-freeze checkpoint

The repository also contains synthetic-only tooling for statistical sensitivity and inference plumbing.

The checked-in smoke result uses eight Monte Carlo repetitions per cell and is **not** sufficient to select a sample size, certify power, or approve a final analysis. It exists to exercise the machinery and expose unresolved decisions.

## What is deliberately not claimed

The repository does not currently claim:

- a learned quantum architecture;
- a trainability improvement;
- a hardware advantage;
- a quantum advantage;
- superiority over classical optimization;
- cross-family or cross-size generalization.

These are future evaluation targets.

## Current decision gate

The next scientific step is review of the pre-freeze statistical specification and compute budget. Until those decisions are approved, do not freeze Benchmark v1 or run a final blind comparison.

## Where to look

- README.md — project overview and current scope.
- paper/research_protocol.md — scientific definitions, estimands, controls, and decision rules.
- paper/stage_1_prefreeze_report.md — provisional benchmark rationale and cost.
- tasks/PREFREEZE_DESIGN.md — freeze sequence and approval gates.
- paper/prefreeze_sensitivity_report.md — smoke-level statistical diagnostics.
- src/struct2circuit/ — implementation.
- tests/ — invariant and integrity tests.

## Reproduce the verified checks

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -e .
    python3 -m unittest discover -s tests -v

The exploratory pilot can be regenerated with:

    python3 experiments/run_pilot.py --instances 24 --n 8 --k 3

Treat regenerated pilot output as exploratory unless a future benchmark version explicitly says otherwise.
