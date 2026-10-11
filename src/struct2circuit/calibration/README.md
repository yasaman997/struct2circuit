# Numerical calibration driver v1

Implementation of `paper/numerical_calibration_protocol_v1.md`, pinned to SHA-256
`6c9945bf4c6d5c3d8c2b71e6f80a78d6dd7b671f02a12c70aa2bf4c1a734e9a1`.
No approved calibration QUBOs were generated during implementation. The benchmark
remains `DRAFT_UNFROZEN`; numerical qualification establishes no RMS benefit.

| Module | Responsibility |
| --- | --- |
| `config.py` | Metadata-only manifest, exact seeds/settings, domains/grids, public seed-use audit, graph labels/canonical aliases |
| `objective.py` | Native compensated normalized objective, retained state, exact coordinates/cache and analytic derivatives; adapter for unchanged production optimizer |
| `reference.py` | Independently seeded grid/local/boundary and DE/polish routes, candidate pooling, one finer-grid/bundle allowance |
| `diagnostics.py` | Direct exponential and derivative oracles, spectral records/coordinates, cost-unit and transported-label controls, optional Bv helper |
| `budget.py` | Actual/request/unique/cache/oracle/gradient accounting and cumulative hard ceilings |
| `decisions.py` | Oriented statistics, guarded thresholds, reference stopping exception, missingness and failure precedence |
| `storage.py` | Exclusive outputs, provenance, immutable versioned lock and one-time validation generation barrier |
| `driver.py` | Routine coverage, mandatory audit/sentinel wave, triggered work, reserve/policy and locked-validation orchestration |

`experiments/run_numerical_calibration.py` prints **metadata only** by default.
It sets numerical threads to one before importing NumPy/SciPy. Official execution
needs a fresh output directory and a separate persisted authorization JSON with
`calibration_execution_authorized: true`, the matching `protocol_sha256`, and
explicit `phases`. No such study execution is authorized by this implementation.
Invoke a future authorized official run through the entry point in a new process;
changing thread environment variables after importing a numerical library does
not reconfigure an existing BLAS pool. NumPy/SciPy/Python versions are locked.

Before generation, the driver audits only the declared public benchmark
configuration and historical seed-use CSV columns. Benchmark seeds are derived
from public configuration metadata without building or opening blind manifest
records. Source hashes and the scoped calibration identifier namespace are
recorded. Missing public metadata or a collision stops execution; no redraw is
allowed. An additional external seed inventory, if one exists, must be checked
before separately authorizing execution; the local check covers its explicit
public allowlist.

Future development writes a persisted validation lock only after its required
numerical checks resolve. Validation needs that lock before generation, restores
the cumulative budget, and uses the locked policy and scope. Byte changes,
in-memory payload changes, source/software changes, policy substitution, repeated
validation generation, and a second run with that lock are rejected. The lock is
a procedural safeguard, not secure blind custody or a tamper-proof signature.

Locks bind all eight finalized development artifacts by content hash. A complete
decision table, resolved coverage and a finalized manifest are required before
atomic lock publication and are checked again before validation generation. The
recorded development directory must remain available and unchanged. Older locks
without this evidence binding are rejected.

Development phase status is recorded separately from whole-study qualification.
Without completed validation the latter remains `INCONCLUSIVE`, unless a known
mandatory failure already requires `FAILED`. Missing/unresolved references never
pass, including after budget interruption. Optional Bv is explicitly omitted by
the standard workflow; its helper requires the optional shared account after all
mandatory work, and supports no expanded production qualification. If used later,
it is restricted to the preselected 12 development audit cases in the protocol.

The eight deterministic artifacts are exclusively created in a new directory
outside historical results. The owned manifest is finalized with actual phase
status; other existing artifacts cannot be replaced. Candidate records preserve
u/beta, physical gamma, separate raw expectation, route/settings/seeds, attempts
and accounting. Landscape/decision records preserve independent route values,
R/F/E/K, challenge gains, face distances, missingness and separate claim counts.
Descriptive statistics and near-best graph sets cannot override failures.

Implementation conventions frozen before execution: seeded uniform interior
sampling uses the finite-difference margins inside B11, and cyclic relabeling uses
`permutation[new] = old = np.roll(np.arange(n), 1)`. Transported-state arithmetic
and rebuilt low-fallback sensitivity are reported separately.

Verification uses n=4, k=2 synthetic fixtures with unrelated IDs/seeds and reduced
budgets, writing only temporary directories. It includes numerical negative
controls, independent-route pooling, traversal/symlink/hardlink sentinels, lock
corruption and holdout-isolation tests. Run the new tests with:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_calibration*.py'
```

These tests do not establish study accuracy or completion within its budgets.
For example, a complement-symmetric toy objective has nearly flat non-analytic
faces with floating-point sampled minima. Its prescribed scalar trace/refinement
caps can leave the reference unresolved and exhaust a reduced shared account;
the workflow correctly reports `INCONCLUSIVE` without loosening a tolerance.
Runtime and memory at official panel size remain unmeasured. Exact cached states,
retained candidates and persisted unique keys can be substantial. Independent
Stage 5B review should challenge route independence, face/cut completeness,
refinement/F accounting, scheduling/accounts, reserve activation, lock/policy
barriers, output truthfulness and memory/runtime practicality before execution.
