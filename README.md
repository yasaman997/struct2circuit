# Struct2Circuit

**Project overview:** Struct2Circuit studies whether information in a cardinality-constrained QUBO can be used as side information when choosing a sparse, feasibility-preserving XY mixer topology under a fixed edge budget.

## Current research question

> **Can a transparent pre-optimization instance descriptor select a sparse constrained mixer topology that improves shallow variational optimization over structure-agnostic connected topologies, and under what conditions does that relationship fail?**

The historical first descriptor was the magnitude of off-diagonal QUBO interactions, `|Q_ij|`. The current mechanism candidate is exact fixed-cardinality conditional exchange RMS, with mean and variance retained separately. Neither low-RMS nor high-RMS selection has an established QAOA performance advantage.

For a cardinality-constrained QUBO:

**Minimize:** `C_Q(x)`  
**Subject to:** `Σᵢ xᵢ = k`

with `x ∈ {0, 1}ⁿ`. The mixer uses XY exchanges, so the simulation remains exactly inside the fixed-weight feasible subspace.

## What has been learned

### 1. Exploratory pilot

The original pilot used 24 synthetic block-correlated QUBOs with `n=8`, `k=3`, `p=1`, and an eight-edge structure-conditioned mixer versus an eight-edge ring mixer.

The structure-conditioned rule won 17/24 instances and reduced the median normalized gap from `0.112558` to `0.101715`.

This was a useful hypothesis-generating result, but it was one synthetic family and did not establish generalization.

### 2. First independent test

A fresh weighted densest-k-subgraph family was then evaluated with 24 instances, `n=9`, `k=4`, `p=1`, a nine-edge budget, eight random connected mixer replicates per instance, and an inverse-`|Q_ij|` control.

The original strong-`|Q_ij|` rule did **not** reproduce the pilot effect:

| Mixer | Median normalized gap |
| --- | ---: |
| Inverse-`|Q_ij|` control | **0.472628** |
| Ring | **0.476476** |
| Random connected mean | **0.485262** |
| Strong-`|Q_ij|` structure | **0.488890** |

The structure rule lost to the ring on 17/24 instances. Its interaction-weight alignment was nevertheless very high, showing that high correspondence with the QUBO interaction graph was not sufficient for better depth-one QAOA performance on this family.

This is treated as a **negative generalization result**, not as something to tune away.

Start with the [historical-results status note](results/README.md), which places the preserved pilot advancement decision in context. See [results/independent_dks_benchmark_v1.md](results/independent_dks_benchmark_v1.md) for the full exploratory analysis. No larger confirmatory benchmark is currently authorized or frozen.

## The important experimental correction

The original comparisons used the same uniform feasible initial state for every mixer. That is a clean fixed-initialization comparison, but it does not isolate topology from mixer/initial-state alignment.

Recent constrained-QAOA work makes this a scientifically relevant control. The simulator now supports:

- `uniform`: uniform superposition over feasible weight-`k` states;
- `mixer_low`: deterministic opposite-extremum sensitivity condition in the lowest mixer eigenspace;
- `mixer_high`: positive-amplitude Perron–Frobenius reference in the highest mixer eigenspace for connected mixers.

The implemented XY Hamiltonian has positive exchange amplitudes, so `mixer_high` is equivalently the ground state of `-H_M`. The spectral projector construction is deterministic; its fallback when uniform overlap vanishes can depend on variable labels. Neither extremum guarantees better QAOA performance. Their overlap with the uniform state is measured separately. These modes diagnose sensitivity to the topology/initialization pairing; they do **not** by themselves isolate a pure topology effect.

`experiments/run_alignment_control.py` implements this diagnostic. **No initialization-diagnostic result has yet been folded into the main claim.**

## Mechanism hypothesis

The first local intuition—“large `|Q_ij|` makes the corresponding exchange favorable”—is not correct. For a single feasible exchange, the direct `Q_ij x_i x_j` contribution cancels.

The remaining hypothesis is graph-level:

`instance descriptor → mixer topology → feasible-state transition graph → variational dynamics → optimization outcome`

The current candidate is the exact conditional RMS exchange cost under uniformly sampled feasible states with `x_i=1`, `x_j=0`, and weight `k`. It uses `Q`, `c`, and `k`; conditional mean and variance remain separately available for interpretation. The former norm of exchange coefficients is not this fixed-`k` statistic and is no longer the primary score. Low-RMS and high-RMS topologies are competing hypotheses, with neither direction validated by these repairs.

See [docs/MECHANISM_HYPOTHESIS.md](docs/MECHANISM_HYPOTHESIS.md).

## Literature position

The project sits at the intersection of constrained QAOA, XY-mixer topology, instance-informed circuit design, and problem-structure/algorithm interaction.

It does **not** claim novelty for constrained mixers, custom mixers, architecture search, adaptive mixer allocation, or trainability theory. The narrow empirical question is whether a transparent pre-optimization instance descriptor can select a sparse XY topology under a matched construction budget.

See [docs/LITERATURE_DIFFERENTIATION.md](docs/LITERATURE_DIFFERENTIATION.md).

## Implementation

Implemented:

- cardinality-constrained QUBO definitions and generators;
- exact fixed-weight feasible-subspace simulation;
- fixed, random-connected, and structure-conditioned mixer graphs;
- explicit uniform/low-spectrum/high-spectrum initialization diagnostics;
- deterministic p=1 parameter optimization with feasible-cost-span scaling for new studies and a legacy mode for historical reproduction;
- paired exploratory analysis;
- reproducible independent benchmark scripts;
- invariant and mechanism tests;
- benchmark manifests and integrity safeguards.

## Current status

**Exploratory research; mechanism-control stage.**

The strong-`|Q_ij|` rule has failed to generalize to the first independent family. The next experiment is therefore **not** another large confirmatory benchmark.

The planned study compares historical strong/weak `|Q_ij|`, low/high conditional exchange RMS, shuffled controls, and random topology under all three initialization conditions. New mechanism-stage runs use the dimensionless coordinate `u = gamma * (C_max-C_min)` on `[0,2*pi]` for the grid and both local solvers; returned gamma is `u/(C_max-C_min)`. Centered, normalized costs remove objective-unit dependence from solver tolerances. This interval is a predeclared scale convention, not a fundamental period; the common beta interval `[0,pi]` is also a comparison convention. Historical pilot and independent DKS scripts explicitly retain `gamma_scale="legacy"`.

Normalized costs use compensated summation of the represented QUBO coefficients, preventing cancellation noise from becoming an artificial landscape. Tiny resolved spans remain nonconstant without an absolute unit cutoff. For a resolved objective, `probability_optimum` sums probability only on states whose compensated float64 cost equals the minimum exactly, before normalization; it applies no near-optimum tolerance. Final cost rounding can still create numerical ties. If all accurately summed costs are indistinguishable, `cost_status="constant_or_unresolved"` withholds normalized gap and optimum probability (`None`, JSON `null`, or blank CSV). A proven constant objective has optimum probability one; it is withheld here because constant and unresolved cases are intentionally combined. Raw legacy phases and expectations are unchanged. Historical summaries retain all raw rows and validate declared instance IDs, required comparators, and every summarized outcome, including optimum probability. Incomplete comparisons fail with affected/expected counts. The optimizer reports the selected candidate's source and status separately from each solver's status; `finite_objective` means only finite final expectation.

QUBO construction and exchange descriptors share an absolute symmetry check: `rtol=0`, `atol=1e-12`, without symmetrizing inputs. Exact descriptor identities assume symmetric `Q`; accepted small asymmetry is input tolerance, not a relative-accuracy guarantee for tiny objectives. Stored historical results are preserved; recomputing their probability metrics under minimum equality may change values previously obtained with the near-optimum mask.

This repair makes no new performance claim. The next scientific phase is a small, predeclared gamma/beta-domain and optimizer calibration. Controlled mechanism comparisons follow calibration.

The statistical sensitivity and pre-freeze machinery already in the repository is retained for reproducibility and later use. It is not the current scientific bottleneck, and no final sample-size or confirmatory-analysis claim is being made from the existing smoke runs.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Optional historical pilot reproduction, writing outside the preserved results:

```bash
python3 experiments/run_pilot.py --instances 24 --n 8 --k 3 --output /tmp/struct2circuit-pilot-reproduction
```

Alignment control:

```bash
python3 experiments/run_alignment_control.py --instances 24 --n 9 --k 4
```

The alignment-control command generates new exploratory records; it does not modify or overwrite the existing independent benchmark results. It requires `--edge-budget` to match the actual ring edge count and records actual counts. It compares ring, historical strong-`|Q|`, and random topologies across initializations; it is not yet the planned conditional-RMS mechanism study.

## Repository map

```text
src/struct2circuit/
    problems.py       Cardinality-QUBO definitions and generators
    mixers.py         Fixed, random, and structure-conditioned mixer graphs
    simulator.py      Exact feasible-subspace QAOA + initialization controls
    optimize.py       Deterministic parameter optimization
    analysis.py       Exploratory statistics and reporting
    benchmark.py      Benchmark manifest construction and blind guards
    sensitivity.py    Statistical sensitivity tooling
experiments/
    run_pilot.py
    run_independent_dks_benchmark.py
    run_alignment_control.py
results/
docs/
paper/
tasks/
tests/
```

## Research principle

The repository distinguishes an interesting observation from a validated method.

> **The goal is not to prove that structure-conditioned mixers work. The goal is to determine whether problem structure contains useful information for constrained circuit design, identify the mechanism if it does, and record clearly where a proposed mapping fails.**
