# Stage 1 pre-freeze report

**Status:** provisional infrastructure proposal (`DRAFT_UNFROZEN`), not a frozen benchmark and not confirmatory evidence.

## Scientific purpose

The proposed manifest separates learning, model selection, locked evaluation, and size/cardinality transfer before any Stage 2 experiment. It covers the pilot-like block family, two graph objectives, and a weak-structure negative control. Stage 1 generated no QAOA or mixer-performance result and did not inspect blind-instance performance. The repository guard provides **procedural blindness only**: it prevents accidental default access but is not a cryptographic or access-control boundary for a contributor who can read repository files.

## Proposed design for scientific approval

| Choice | Train | Validation | Blind test | Transfer |
| --- | ---: | ---: | ---: | ---: |
| Instances per family | 32 | 16 | 32 | 12 |
| Sizes/cardinalities | `(8,3)`, `(10,4)` | `(8,3)`, `(10,4)` | `(8,3)`, `(10,4)` | `(12,5)`, `(12,4)` |
| Total over four families | 128 | 64 | 128 | 48 |
| Permitted use before freeze/final evaluation | learning studies | architecture and parameter choices | none | infrastructure only |

Each in-distribution split has two predeclared structural regimes per family. The transfer split changes `n` and includes two cardinality ratios. Proposed controls are eight independently seeded random connected mixers per instance, per-instance feasible-range normalized optimality gap, and exact solving only through `n=12`.

## Sensitivity and cost assumptions

The exploratory 24-instance pilot reported a paired median gap reduction of about `0.0147` and a bootstrap interval of roughly `[0.0015, 0.0166]`. Those values are used only as a rough sensitivity anchor, not as confirmatory evidence or a power guarantee. Because multi-family effects and random-baseline variance are unknown, the proposal deliberately uses 32 blind instances per family—more than the pilot—but does **not** claim that 32 is a formally powered sample size. Before freezing, the scientist should run a simulation-based paired-test sensitivity analysis over effects smaller than the pilot estimate (for example `0.005`, `0.010`, and `0.015`) using train/validation variance only.

The draft contains 368 instances. With eight random controls plus ring and structure controls, one depth/configuration pass would require approximately `368 * 10 = 3,680` mixer-instance optimizations; complete-mixer references would add 368. This excludes depths, optimizer repeats, shots, and architecture search, so the eventual cost can multiply substantially. Exact feasible dimensions at the largest draft size are `C(12,5)=792` and `C(12,4)=495`; dense exact simulation and repeated diagonalization may dominate runtime and memory. These costs must be benchmarked on train/validation before counts are approved.

## Assumptions and limitations

- Regimes are scientifically plausible placeholders, not empirically selected favorable settings.
- The null ensemble is exchangeable and weak-structure; it cannot prove the absence of all exploitable instance-level information.
- The draft does not yet establish statistical power, affordable compute, or representative application distributions.
- Generator audit metadata must remain outside any mixer-policy feature interface.
- Blind records exist only as guarded manifest metadata; their problems have not been generated, solved, evaluated, or summarized. Because these provisional seeds are visible in the repository, they must not be promoted as the definitive blind set.
- Before freezing, definitive blind seeds and generator definitions must be placed in external custody inaccessible to model developers and routine CI. The public repository should retain only a cryptographic commitment until the one-time locked evaluation is authorized.

## Required approval gate

Before changing the status to `FROZEN`, approve or revise: (1) 32/16/32/12 instances per family, (2) the `(n,k)` ranges, (3) both declared regimes for every family, (4) eight random-baseline replicates, (5) normalized gap as the primary metric, (6) exact-solver limit `n=12`, (7) the planned train/validation-only sensitivity and compute study, and (8) an external custodian and commitment/release procedure for definitive blind definitions. Any later change to a frozen configuration must use a new benchmark version.
