# Independent DKS instance-summary CSV: column-sign erratum

**Scope:** Archived exploratory result `results/independent_dks_benchmark_v1_instances.csv` (24 instances). This is a **schema/name defect**, not a reversal of the numeric results, an experiment rerun, or a correction to the published win/loss interpretation. The archived CSV and raw data remain unchanged.

## The defect

The generator wrote **baseline normalized gap minus structure normalized gap**, but labeled each column as if it held **structure minus baseline**. All three historical columns are affected:

| Historical CSV header (misleading) | Stored arithmetic, unchanged | Correct header used by the generator for future output |
| --- | --- | --- |
| `structure_minus_ring` | `ring_gap - structure_gap` | `ring_minus_structure` |
| `structure_minus_random_mean` | `random_mean_gap - structure_gap` | `random_mean_minus_structure` |
| `structure_minus_inverse` | `inverse_structure_gap - structure_gap` | `inverse_minus_structure` |

The metric is **normalized expectation gap** (smaller is better). A **positive** stored paired difference therefore favors the structure-conditioned mixer; a **negative** stored difference favors the baseline. For the first saved instance (ID 0), `structure_gap=0.40623712`, `ring_gap=0.37308987`, so the correctly named `ring_minus_structure` value is **−0.03314725**. The historical column `structure_minus_ring` also stores **−0.03314725**, even though literal structure-minus-ring would instead be **+0.03314725**.

## Scientific interpretation

The [independent benchmark report](independent_dks_benchmark_v1.md) already defines paired improvement as **baseline minus structure** and correctly reports the structure-conditioned mixer losing to the ring in **17 of 24** instances. Its signed median differences and conclusion are **not** overturned by this header defect. In particular, do not negate historical stored values again when reproducing the report's baseline-minus-structure analysis.

## Forward compatibility and provenance

- **Preserved:** `independent_dks_benchmark_v1_instances.csv`, the archived raw CSV, original report/figures, seeds, objective values, and all historical analyses. They are not regenerated or rewritten.
- **Corrected for future fresh runs:** `experiments/run_independent_dks_benchmark.py` now names all three output columns `*_minus_structure` while keeping the original baseline-minus-structure arithmetic. This is an intentional **header-only output-schema change**; it does not modify model, optimizer, instances, scores, or the paired sign convention.
- **Compatibility warning:** Code consuming archived CSVs must explicitly map the three historical names to their actual `*_minus_structure` meaning. Code consuming new summaries should use the corrected names. Do not combine old and new CSVs by header without an explicit schema adapter.
- **Regression checks:** `tests/test_experiment_smoke.py` checks both a freshly generated temporary summary's corrected headers/arithmetic and the preserved 24-row historical file's old headers/arithmetic. The synthetic smoke run is not the controlled C6 experiment.

This correction neither changes the frozen C6 scientific design nor licenses its execution. The [research-status overview](../README.md) and [historical-results status note](README.md) define the current evidence boundary.
