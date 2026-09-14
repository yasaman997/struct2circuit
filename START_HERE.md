# Struct2Circuit: start here

This is the non-technical guide to the project. The repository investigates one
narrow, falsifiable question:

> Can structure-conditioned, feasibility-preserving mixers improve constrained
> quantum search over fixed and expected-random equal-resource mixers?

The project concerns cardinality-constrained QUBOs, where exactly `k` of `n`
binary variables must be selected. Its XY mixers preserve that constraint, and
its exact simulator works directly in the fixed-Hamming-weight feasible subspace.
This is a study of circuit design—not a claim of quantum advantage.

## What is implemented

### Exploratory pilot

The repository contains a deterministic, noiseless pilot on 24 synthetic
block-correlated instances with `n=8`, `k=3`, and QAOA depth `p=1`. The fixed
ring and structure-conditioned mixers each use eight edges. Structure won, tied,
and lost on 17, 0, and 7 instances, respectively; median normalized gap changed
from approximately `0.1126` to `0.1017`.

This result is exploratory. It motivates benchmark design and statistical
calibration but does not establish generalization, scaling, hardware performance,
or quantum advantage.

### Stage 1 foundations

Stage 1 implements and tests:

- an instance-independent random connected mixer with an exact edge budget;
- three structured problem families plus a weak-structure negative control;
- a deterministic provisional train/validation/blind/transfer manifest; and
- procedural guards against accidental blind-record access.

Benchmark v1 remains `DRAFT_UNFROZEN`. Its visible blind metadata are provisional,
no blind performance has been evaluated, and definitive blind definitions require
external custody before a future freeze.

### Pre-freeze Checkpoint 1

The current checkpoint adds synthetic-only tooling for paired median inference,
multiplicity control, joint coverage, an independent null diagnostic,
expected-random estimator uncertainty, and optimizer-failure sensitivity. The
checked-in 336-cell result uses only eight Monte Carlo repetitions per cell and is
a reproducibility smoke test—not high-precision calibration or mixer-performance
evidence.

No final sample count has been selected. The bootstrap procedure is unapproved,
the optimizer-failure policy is unfrozen, and the null practical-superiority
margin remains provisional. A higher-precision calibration requires explicit
human scientific approval.

## Current decision gate

Stage 2 has not begun. Do not run train/validation mixer comparisons, open the
blind evaluation, train an AI policy, or freeze Benchmark v1 until the pre-freeze
statistical decisions have been reviewed and approved.

## Where to look

- `README.md` — technical overview, pilot results, and reproducibility commands.
- `paper/research_protocol.md` — scientific definitions, integrity rules, and
  permitted claims.
- `paper/stage_1_prefreeze_report.md` — provisional benchmark rationale and
  limitations.
- `tasks/PREFREEZE_DESIGN.md` — the two-freeze model and checkpoint sequence.
- `paper/prefreeze_sensitivity_report.md` — smoke-level statistical findings,
  limitations, and unresolved decisions.
- `results/prefreeze_sensitivity.json` — deterministic machine-readable smoke
  output.
- `src/struct2circuit/` — problem, mixer, simulator, benchmark, and sensitivity
  implementations.
- `tests/` — mathematical invariants, benchmark safeguards, and statistical
  regression tests.

## Reproduce the verified checks

Install the package and run the complete test suite:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

The exact command for regenerating the pre-freeze smoke artifact is documented in
`README.md` and `paper/prefreeze_sensitivity_report.md`. Write regenerated output
to a temporary path and compare it with the checked-in JSON rather than
overwriting the reviewed artifact.
