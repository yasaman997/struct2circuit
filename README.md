# Struct2Circuit

**Reproducible exploratory study of structure-conditioned mixers for cardinality-constrained quantum optimization.**

This project studies whether exploitable combinatorial structure can inform
feasibility-preserving circuit design and improve constrained quantum search
relative to equal-resource fixed and expected-random mixer baselines.

The paper-level claim under investigation is:

> A structure-conditioned, feasibility-preserving mixer can improve the
> solution-quality/resource Pareto frontier over fixed and expected-random
> equal-budget mixers for cardinality-constrained QUBOs.

The current release is deliberately narrow. It provides an exact feasible-subspace simulator and a deterministic pilot experiment comparing:

- a fixed ring XY mixer;
- a structure-conditioned XY mixer built from the strongest entries of the QUBO matrix while preserving graph connectivity; and
- a complete XY mixer as a higher-resource reference.

The initial application is a block-correlated selection model motivated by cardinality-constrained bond index tracking. The code is independent of Qiskit and operates directly in the Hamming-weight-`k` subspace, whose dimension is `binomial(n, k)` rather than `2**n`.

## Current exploratory pilot

The repository already contains the first deterministic run: 24 instances with `n=8`, `k=3` and `p=1`. The structure-conditioned mixer and fixed ring both used eight mixer edges per layer.

- Structure won / tied / lost against ring on **17 / 0 / 7** paired instances.
- Median normalized gap decreased from **0.112558** to **0.101715**.
- Paired median gap reduction was **0.014678**.
- Exploratory bootstrap 95% interval: **[0.001460, 0.016576]**.
- One-sided paired Wilcoxon `p = 0.001752`.
- The 28-edge complete mixer reached a median gap of **0.096086**.

The pilot motivates proceeding to pre-freeze benchmark design and statistical
calibration. It does **not** establish generalization, hardware performance,
scaling or quantum advantage.

## Scientific status

This is a **pilot and research scaffold**, not evidence of quantum advantage. A positive pilot result only justifies advancing to deeper circuits, broader instance families, stronger classical/quantum baselines, noise models, blind splits, and hardware-aware resource accounting. The preregistered claim ladder is defined in `paper/research_protocol.md`.

Stage 1 benchmark foundations are implemented. The candidate benchmark remains
`DRAFT_UNFROZEN` pending scientific approval; blind records are excluded by
default and no blind performance has been run. See
`paper/stage_1_prefreeze_report.md` for the proposed counts and decision gate.

Checkpoint 1 of the subsequent pre-freeze design process provides synthetic-only
statistical sensitivity tooling. The checked-in analysis is an eight-repetition
smoke artifact, not high-precision calibration; it does not generate QUBOs or
select benchmark counts. Reproduce it without overwriting the checked-in result:

```bash
python3 tools/prefreeze_sensitivity.py \
  --output /tmp/prefreeze_sensitivity_smoke.json \
  --seed 20260903 \
  --runtime-cap 180 \
  --min-repetitions 8 \
  --max-repetitions 8 \
  --bootstrap-samples 199 \
  --target-mcse 0.20 \
  --random-baseline-sd 0.003
cmp /tmp/prefreeze_sensitivity_smoke.json results/prefreeze_sensitivity.json
```

See `tasks/PREFREEZE_DESIGN.md` and `paper/prefreeze_sensitivity_report.md` before
interpreting the machine-readable output. Bootstrap approval, the optimizer-
failure policy, the null practical-superiority margin, and final sample counts
all remain open decisions for human scientific review. Stage 2 and blind
evaluation have not begun.

Generate or inspect the provisional manifest without exposing blind records:

```bash
python3 tools/benchmark_manifest.py generate \
  --config configs/benchmark_v1_draft.json \
  --output manifests/benchmark_v1_draft.json
python3 tools/benchmark_manifest.py inspect \
  --manifest manifests/benchmark_v1_draft.json
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 experiments/run_pilot.py --instances 24 --n 8 --k 3
```

Results are written to `results/`:

- `pilot_results.csv`: per-instance/per-mixer measurements;
- `pilot_summary.json`: paired statistics and configuration;
- `pilot_report.md`: automatically generated interpretation;
- `pilot_quality_resource.png`: quality/resource diagnostic figure.

## Repository map

```text
src/struct2circuit/
  problems.py       Cardinality-QUBO definitions and generators
  mixers.py         Fixed and structure-conditioned mixer graphs
  simulator.py      Exact feasible-subspace QAOA simulation
  optimize.py       Deterministic parameter optimization
  analysis.py       Paired statistics and pilot report generation
  benchmark.py      Provisional manifest construction and blind-access guards
  sensitivity.py    Synthetic paired-inference sensitivity analysis
experiments/
  run_pilot.py      Reproducible pilot entry point
tools/
  benchmark_manifest.py      Generate or inspect the provisional manifest
  prefreeze_sensitivity.py   Run the bounded synthetic sensitivity grid
tests/
  test_invariants.py          Feasibility, Hermiticity and determinism tests
  test_benchmark_manifest.py Benchmark integrity and access-guard tests
  test_sensitivity.py         Statistical-tooling regression tests
tasks/
  PREFREEZE_DESIGN.md         Checkpoint sequence and statistical specification
paper/
  research_protocol.md
  stage_1_prefreeze_report.md
  prefreeze_sensitivity_report.md
results/
  prefreeze_sensitivity.json  Checked-in eight-repetition smoke artifact
```

## Reproducibility contract

- Every random source is seeded.
- Every generated mixer is validated as connected.
- Every simulated state remains in the exact feasible basis.
- Search evaluations and edge counts are recorded.
- The structure-conditioned and ring mixers use the same mixer-edge budget.
- Pilot results are labeled exploratory; confirmatory thresholds are evaluated only on locked blind instances.

## Closest literature to beat

The project is positioned against constraint-preserving alternating operators, predictor/training-free quantum architecture search, hardware-aware QAS, and recent rigorous quantum-optimization benchmarking. The manuscript protocol contains the working comparison table and representative references.
