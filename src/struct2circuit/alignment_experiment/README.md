# Controlled C6 mixer-alignment experiment

The byte-identical Stage 7O design is `paper/controlled_mixer_alignment_v1.md`,
SHA-256 `032d4d997c5aefcde3bdef8ae64a206b744701073bf46f1974bf68793d0cac41`.
Stage 7P approved implementation; Stage 7Q implements it. The copied document's
original proposal wording is preserved intentionally. **No execution approval
is supplied by the code, design approval or this README.**

The panel consists of 32 independent fresh QUBOs (eight per family), n=6, k=3.
Every QUBO's 720 C6 permutations reduce to 60 labeled cycles before exact cost
symmetry reduction. Conditional exchange RMS selects the high and low assignments
before any quantum objective is evaluated. Cost, topology, native spectrum,
uniform initialization, depth one and B00 remain fixed.

## Read-only commands

From the repository, the default entry point prints metadata and the exact
implementation identity without creating authorization files, instances or outputs:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/run_controlled_alignment.py
```

After a separately authorized run, a different reviewer can inspect persisted
coefficients, selection, scalar references, controls, hashes and resource receipts:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/run_controlled_alignment.py --verify /absolute/path/to/existing-study
```

Verification uses existing coefficients and numerical records. It generates no
QUBOs and evaluates no quantum objectives. It checks structural Hamiltonians and
their spectra to reconstruct control identities. `CONSISTENT` means internal
evidence consistency; it is not global-optimum certification or scientific
publication approval. Independent researcher approval remains required.

## Later execution authorization

Execution requires a clean committed checkout, the frozen runtime
(Python 3.12.14 / NumPy 2.5.3 / SciPy 1.18.1), one numerical thread, unchanged
inherited scientific source and documents, sufficient memory/disk, and a separate
persisted authorization. **Do not execute this example during implementation.**

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python experiments/run_controlled_alignment.py --execute \
  --authorization /workspace/mixer-mechanism-authorizations/approved-study.json \
  --output /workspace/mixer-mechanism-runs/approved-fresh-study
```

The authorization JSON requires exactly these fields:

| Field | Required value |
|---|---|
| execution_authorized | Boolean true supplied only after separate execution approval |
| scope | `stage7o_alignment_c6_v1` |
| design_sha256 | Exact design hash above |
| implementation | Exact object emitted by the metadata command, including revision and source/runtime hashes |
| manifest_sha256 | `canonical_hash(manifest())` from this package's frozen metadata |
| output | Exact absolute fresh output directory |
| approval_identity | Nonempty identity of the separate execution approval |
| historical_archive | `/workspace/calibration-archives/stage7b-development-e8dea4d-v1.tar.gz` |
| historical_archive_sha256 | `d7043055769685c8ed1858cb2cd4da6d6ba81372c703a1ca8126109007a578c5` |
| development_records | The two exact path/hash pairs below |

```json
{
  "/workspace/calibration-runs/stage7b-development-e8dea4d-v1/instances.jsonl": "351ae59c04ad9cb4219b283ac440af65909c004ec21bad3928e9de72057da103",
  "/workspace/calibration-runs/stage7k-development-2042e4e-v2-p33-v1/instances.jsonl": "be56f03d4e4e8f87fc6fc2ad6a8d386ca3af790689d6dbba4baeb04d5f38de13"
}
```

No phase or validation options exist. Outputs under the repository or historical
calibration/archive directories, existing paths, traversal and symlink aliases
are rejected before generation. Public seed metadata is an explicit allowlist;
protected benchmark manifests and validation coefficients are never opened.
Historical development hashes prevent coefficient reuse. Duplicate panel QUBOs
stop the run before the first simulator, without replacements.

The authorization's adjacent `.used` marker is published durably before generation.
The output and evaluation journal are created exclusively. A failed or interrupted
run consumes its capability; there is no restart, resumption or reset mechanism.
Copying the same authorization does not provide another fresh destination.

## Reference, accounting and interpretation

The B00-only reference specialization reuses existing candidate recording,
dyadic grids, analytic gradients and bounded solver primitives. Both arms always
receive coarse and fine grids, four/eight separated local starts, independent
DE0/DE1 with separate polishes, four faces at each level and two Powell starts.
The existing calibration boundary helper is deliberately not called because it
adds B11 traces. No P9/P17/P33 production search or selected point is used.

Numerical controls bind the source, coefficient, graph, state and objective
identities. The verifier reconstructs grid coverage, required local/Powell starts,
DE seeds, face brackets, control coverage, route minima and coarse-to-fine evidence.
R<=1e-5 and 0<=F<=1e-5 are empirical completion criteria, not certified errors.
Solver flags and booleans cannot substitute for the required recorded work.

The maximum is 20,000 admitted actual numerical computations per arm and
1,280,000 total. Fresh and repeated native evaluations, direct oracles and
derivative mixer evolutions are charged; exact cache hits are separately counted.
The journal records an admission before its backend call, so failed work remains
charged. This is conservative accounting when a backend fails after admission.
Immutable wave receipts replay against that journal and cannot reset cumulative
arm/domain/evaluation/CPU counters. Coarse cache files bind the exact objective and
are hash-checked before fine work, preserving grid reuse while streaming numerical
states one arm at a time.

Numerical admission stops at 3,300 cumulative CPU-seconds. The 300-second reserve
remains within a 3,600-second total ceiling; the wall limit is 7,200 seconds.
The worker and its evidence-writing coordinator are measured through OS process
exit, including imports, preparation, sealing and finalization. The outer receipt
observer follows the existing v2 accounting boundary. Its tiny observation/write
overhead is outside the settled coordinator/worker measurement. A settlement over
the hard ceiling makes the independent decision inconclusive. No scientific
publication is performed by either process.

The Stage 7P review established that full completion is not guaranteed: worst-case
solver allowances plus derivative evolutions and controls can exceed an arm cap.
Such arms remain unresolved; budgets, seeds and methods do not escalate.

All 32 paired rows and 64 arms remain present. Delta is high minus low reference.
Wins require two verified complete references and Delta < -0.00012; losses require
Delta > +0.00012; equality and intervening values are neutral. Other pairs are
unknown. W>=24 supports the direction; W+U<=8 contradicts the stated majority
prediction; otherwise the result is inconclusive. Any integrity failure prevents
directional publication. Complete-case family medians/ranges, paired route deltas,
plot-ready score separations and missingness are descriptive. The unadjusted
Pearson association is recorded without a p-value; it is undefined when there is
insufficient variation. Graph arms, starts and seeds are not extra QUBO replicates.

## Synthetic validation

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p test_alignment_experiment.py -v
```

New tests use explicitly constructed unrelated toy coefficients, structural
matrices, scalar records and solver stubs. They never call official generators or
run QAOA optimization. Run the existing full deterministic regression suite with
the same environment before publishing implementation changes. Tests passing
establish software regression evidence, not numerical qualification or execution
authorization.
