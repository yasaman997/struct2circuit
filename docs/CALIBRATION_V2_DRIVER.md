# Numerical calibration v2 implementation

The Stage 7H scientific review approved implementation against the exact protocol
SHA-256 `abcb7cb1b7fc7b2f2a114b6c5a4328acc18a78aebca00544a20c383bd719548d`.
The reviewed document's bytes and status marking are preserved. This implementation
does not authorize official development, validation, or diagnostic execution.

`experiments/run_numerical_calibration_v2.py` prints metadata by default. That path
constructs no QUBOs. Numerical phases require a separately persisted authorization.
The `calibration_v2` package is outside v1's source-hash glob and reuses v1's native
objective, production adapter, reference algorithms, controls, and resource sampler.
The v1 protocol, driver, lock verifier, optimizer, physics and thresholds are unchanged.

P33 is fixed. Development includes IDs 0–7, 192 labeled conditions and 672 pairs;
validation includes IDs 8–11, 96 conditions and 336 pairs. Routine B00 work precedes
audits and round-robin triggered actions. Required controls, boundary work and the
single refinement/bundle allowance remain compulsory. Decisions recalculate scalar
evidence: demonstrated failures precede missing work; incomplete work never qualifies.
Raw counterexamples survive interruption of subsequent refinement.

V2 accounting replaces the v1 account partition with a 4,500,000 development cap,
8,000,000 study cap, protected 3,500,000 validation allocation and unchanged 120,000
landscape / 28,800 CPU-second ceilings. Fresh repeats and oracles are charged;
cache reads and gradient work reusing a state have separate counters. Restoring a
unique-key inventory never restores evaluated values or states.

The entry point supervises numerical workers with `wait4`. Immutable resource
cutoff records are followed by process-exit receipts covering worker imports,
construction, computation, reporting and shutdown. Subsequent lock-publication
overhead is deducted in a linked publication record, retaining its measurement
cutoff. Validation imports and verification are charged from process startup.
Unsealed receipts, depleted resources and accounting disagreement reject a lock.
CPU receipt/publication cutoffs remain explicit; external execution telemetry should
preserve the final supervisor exit accounting when an official run is authorized.

The new schema-2 verifier reconstructs expected coverage, grids, boundary nodes,
independent route identities/settings, raw R/F/E/K, controls, aliases, pairs and
artifact hashes. It never accepts a producer's `QUALIFIED` flag alone. Only exact
official limits and complete development evidence can publish a v2 lock. V1 locks
cannot authorize v2. Consumption is atomic and fsynced before validation generation;
the marker derives from the bound resource receipt, so copying a lock does not
create another capability. Interruption leaves it consumed.

Every official authorization must contain `calibration_execution_authorized=true`,
the exact `protocol_sha256`, current `driver` identity, `approval_identity`, a single
`phase`, absolute fresh `output`, and `historical_archive` pointing to the preserved
FAILED archive with the protocol's recorded hash. Validation additionally binds
`lock_sha256`. Official admission enforces Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1,
all four numerical thread variables equal to 1, and actual loaded one-thread pools.
There are no authorized execution files or official commands in this change.

Diagnostics require a separately authorized, protocol-fixed design. Development
permission must prospectively set `diagnostics_authorized_before_validation=true`
for that optional study; the lock binds this choice. Diagnostic permission also
binds `core_artifact_hashes` and `diagnostic_design` from the fixed manifest. Diagnostics
can begin only after qualified validation and consume remaining cumulative resources.
They restore retained coefficients, edges and states, without calling generators.
Unavailable/unalignable evidence is rejected; unadmitted work is `NOT_RUN` and unresolved.
Broader-domain gains remain separate. Reliable B00 discoveries and shared defects
produce a hashed addendum and can withdraw effective qualification without editing
core artifacts, changing policy or rerunning validation.

Run synthetic checks with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p test_calibration_v2.py
```

The regression runner is `tests/run_calibration_regressions.py`. It discovers the
whole existing suite and explicitly skips the two tests that open the protected
repository benchmark manifest. GitHub CI uses this runner as well. These exclusions
are reported as skips, never passes; no existing test or production guard is weakened.
Positive lock verification uses actual unrelated n=4 toy calculations with a
substituted toy metadata contract. It never generates prescribed study instances.

Unit tests establish engineering contracts. They do not establish P33 reliability,
official convergence, dimension-126 CPU/memory feasibility, wider-domain adequacy,
or RMS superiority. Independent implementation review and separate phase authorization
remain necessary before official execution.
