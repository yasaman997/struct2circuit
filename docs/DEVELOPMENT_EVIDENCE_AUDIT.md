# Development-evidence audit: numerical calibration and exploratory mechanisms

**Status:** Read-only, non-qualifying research summary (October 2026). This document inventories disclosed development findings; it is **not** a rerun, a certified analysis, a validation-holdout release, or an independently reproduced audit of off-repository files.

## Why this exists

The [top-level research overview](../README.md) summarizes two post-benchmark stages whose underlying raw records were retained in a separate research workspace rather than committed to this repository. A faculty reader needs to know exactly what was observed, what failed, how the numbers were grouped, and which supporting material is available. **The off-repository result counts and archival hashes below are reported in the approved research record and prior development reports; they have not been recomputed from archive bytes in this GitHub checkout.**

For public, directly navigable exploratory data, consult [the independent DKS benchmark](../results/independent_dks_benchmark_v1.md), [pilot report](../results/pilot_report.md), and [saved-results status](../results/README.md).

## P33: official development numerical calibration — FAILED

The frozen Stage 7L P33 calibration compared finite empirical numerical references and solver outputs for a development panel. Its objective was **numerical reliability**, not testing the structural-mixer hypothesis.

| Evidence item | Reported observation |
| --- | --- |
| Landscape-level cases | 192 total: **171 pass, 2 fail, 19 unresolved** |
| Dependent graph contrasts | 672 total: **546 pass, 9 fail, 117 unresolved** |
| Numerical controls | 20,148 reportedly passed |
| Counted numerical objectives | 1,404,045 |
| Settled CPU time | 568.14324 seconds |

The two explicit landscape discrepancies were reported for instance 0 with graph indices 4 and 5. The development postmortem implicated search routes repeatedly occupying inferior observed parameter basins. This is an empirical optimizer/reference failure, **not** a mathematical certificate of local or global minima. Dependent graph contrasts must not be interpreted as 672 independent problem draws.

**Qualification outcome: FAILED.** The historical P9/P17/P33 schedules are not qualified numerical reference substitutes. The 19 unresolved cases remain unresolved; neither changing optimizer settings nor hiding failed cases is an authorized repair to these historical records.

Read-only points of orientation in this repository: [v1 protocol](../paper/numerical_calibration_protocol_v1.md), [v2 protocol](../paper/numerical_calibration_protocol_v2.md), and the calibration source under [`src/struct2circuit/calibration_v2/`](../src/struct2circuit/calibration_v2/). These materials specify relevant software and numerical conventions, but **do not contain the archived P33 outcome rows**.

## Stage 7N: exploratory mechanism analysis — NOT A CONFIRMATORY RESULT

Stage 7N was a post hoc, read-only analysis of a small development dataset, not a fresh prospective test.

| Scope or comparison | Reported observation |
| --- | --- |
| Independent QUBOs | **8**, comprising two per each of four families; `n=6, k=3` |
| Landscape records | **192**, of which **19** were unresolved |
| Equal-spectrum exploratory graph pairs | 17 reported; 16 complete |
| High exchange RMS versus its shuffled control under uniform initialization | High RMS favored in **7/7 eligible comparisons**; one additional comparison was unresolved |
| Low exchange RMS versus its shuffled control under uniform initialization | Low RMS favored in **1/6 eligible comparisons** |

**Interpretation limits:** comparisons within one QUBO are dependent; the eight QUBOs, not landscape rows, are the independent sampling units. Graph shuffling altered more than the isolated cost-alignment treatment and therefore did not identify RMS as a unique causal mediator. The exploratory numerical references may share finite-search blind spots. Observed direction, missingness and source-selection rules prevent any claim of independent validation or generality. The high/low directions were hypothesis-generating, not a frozen success claim.

The subsequent [controlled C6 experiment](../paper/controlled_mixer_alignment_v1.md) is a prospective response to these ambiguities. It fixes topology and feasible-subspace mixer spectrum while changing graph assignment relative to a held-fixed cost function. **Its 32 official QUBOs have not been generated or evaluated.**

## Archive pointers and disclosed content hashes

Stage 7N materials were reportedly retained separately at:

```text
/workspace/research-archives/stage7n-exploratory-2042e4e-v1/
```

| Named archived item | Reported SHA-256 |
| --- | --- |
| `report.md` | `7d7d00b2ecc785a15fa4c79c0eb8dc443b0242be63aea4ea80dbfbd44379c2aa` |
| `landscapes.csv` | `98bf6028b913f67185fec16041af8c696da0a1f69db5a6cc6a74b0cb80d9bb9f` |
| `stage7n-analysis.zip` | `68d33e872b86211ab0777cc60ea9bd241e41983a730b64582e096ba47b8c2d8b` |

The [frozen Stage 7O design, §1](../paper/controlled_mixer_alignment_v1.md) records additional hash/inventory statements and explains how Stage 7N motivated the prospective hypothesis. These are provenance pointers, **not links to downloadable files**. An archive path inside a previous computing workspace does not imply that a public GitHub visitor can read it. P33 development evidence and the failed-calibration archive are also outside this repository; no P33 raw-case archive is claimed to be attached here.

### Read-only checks for a researcher who already possesses the archived files

From a local environment containing the Stage 7N directory:

```bash
ARCHIVE=/workspace/research-archives/stage7n-exploratory-2042e4e-v1
sha256sum "$ARCHIVE/report.md" "$ARCHIVE/landscapes.csv" "$ARCHIVE/stage7n-analysis.zip"
python3 -m zipfile -t "$ARCHIVE/stage7n-analysis.zip"
```

Compare the three displayed digests with the inventory above. The integrity check **does not** recompute the scientific tables. Full numerical reproduction or independent audit of the quoted comparisons requires the underlying landscape rows, selection/group definitions, numerical-reference completeness criteria and P33 outcome records. They are not distributed in this GitHub snapshot. Any later release should consist of separately reviewed **development-only** evidence; no protected validation/holdout material should be accessed or made public.

## What can and cannot be independently verified here

- **Can inspect:** the repository's pilot and independent DKS results, stated experimental methods, frozen prospective C6 design, simulator/selection code, and synthetic numerical-integrity regressions.
- **Can check with authorized archive access:** hashes, archived Stage 7N landscape-level data, eligibility/missingness, control composition and reported descriptive comparisons.
- **Cannot establish from this repository alone:** a raw-case independent reproduction of P33 or Stage 7N, correctness of any asserted global optimum, a causal sole-RMS mechanism, or a prospective improvement for C6 high-RMS assignment.

**Scientific bottom line:** P33 failed its qualification. Stage 7N is exploratory and non-independent across graph observations. The only appropriate next claim about the controlled experiment is that it is implemented but **not executed**.