# Historical results and current scientific status

> **Read this status note before interpreting the saved reports.** These are preserved exploratory artifacts. The pilot report's original advancement decision records the historical state at that time; it is not current approval for a larger blind benchmark. No large confirmatory benchmark has been authorized or frozen.

## What the saved reports establish

- [Original pilot report](pilot_report.md): the deterministic strong-`|Q_ij|` sparse XY mixer beat the ring on 17/24 block-correlated instances at the same edge budget. The report, CSV, JSON summary, and figure preserve the original statistics and provenance.
- [First independent weighted densest-k-subgraph benchmark](independent_dks_benchmark_v1.md): the same heuristic lost to the ring on 17/24 instances. The later independent family did not reproduce the pilot's positive effect; this negative result and its instance records are retained.

The original pilot's “larger blind study is justified” decision and “learned mixer” wording are historical text, not the current scientific position or evidence of an implemented learned model. Neither report establishes general superiority. Their proposed next steps are historical plans, not authorization to run them now.

## Current direction

Strong `|Q_ij|` is a historical comparator, not the primary mechanism candidate. The current candidate is exact fixed-cardinality conditional exchange RMS derived from `Q`, `c`, and `k`, retaining conditional mean and variance separately. Low-RMS and high-RMS selection are competing hypotheses, and neither has an established QAOA performance advantage.

The simulator supports `uniform`, `mixer_low`, and `mixer_high`. With positive-sign XY exchange amplitudes, `mixer_high` has the Perron–Frobenius positive-amplitude interpretation for connected mixers; the low extremum is a sensitivity control. These conditions diagnose initialization sensitivity without guaranteeing an improvement or isolating a pure topology effect.

**The later P33 development calibration FAILED** (2 failed and 19 unresolved references of 192); Stage 7N's high-RMS exploratory comparison is nonconfirmatory and not independently reproducible from this GitHub checkout. The fixed-C6 32-QUBO high-versus-low assignment experiment is **implemented but NOT EXECUTED**. Regression validation is not performance evidence. See the [development-evidence audit](../docs/DEVELOPMENT_EVIDENCE_AUDIT.md), [frozen C6 design](../paper/controlled_mixer_alignment_v1.md), and [scientific scope](../docs/SCIENTIFIC_SCOPE.md). There is no demonstrated quantum or hardware advantage, learned architecture discovery, or improved trainability.

## Metric provenance and preservation

**Historical DKS CSV erratum:** In the preserved `independent_dks_benchmark_v1_instances.csv`, the three columns called `structure_minus_*` contain **baseline minus structure**, contrary to their names. Read the [schema/sign erratum](independent_dks_schema_erratum.md) before analyzing those columns. The generator now uses correctly named `*_minus_structure` headers for *new* output only; the archived CSV is not changed.

All previously stored reports, CSV/JSON records, and the PNG figure are left unchanged. In particular, this semantics repair does not regenerate the pilot or independent DKS results, rewrite their statistics, or change their experimental conclusions.

The pilot and independent DKS entry points preflight every output destination and refuse existing artifacts before simulation. To reproduce them, explicitly select a fresh directory with `--output`; unrelated files in that directory are preserved. The alignment entry point applies the same check to its output filename. There is no overwrite option, automatic random path, or dependency on Git metadata.

The result field `expectation` retains legacy raw NumPy cost arithmetic even for dimensionless runs. Normalized gap, extrema, span, and optimum classification use compensated float64 costs instead; a gap reconstructed from the raw expectation can disagree through cancellation or rounding. Historical expectation values retain their original meaning.

Historical optimum-probability values reflect the earlier implementation's tolerance-based treatment of near-minimal states. The corrected `probability_optimum` sums probability only for states whose compensated float64 cost equals the minimum, before normalization or division. Distinct represented costs are not merged by a near-optimal tolerance; ties introduced by final cost rounding remain numerical ties, so this is not a claim of exact rational minimization. Recomputing probability metrics under the corrected semantics may therefore change reported values; any future recomputation must be identified separately rather than silently replacing these artifacts. The conservative `constant_or_unresolved` policy continues to withhold normalized gap and optimum probability when costs cannot be resolved.

The retained [pre-freeze sensitivity output](prefreeze_sensitivity.json) concerns analysis infrastructure, not QAOA performance or approval of a confirmatory benchmark.
