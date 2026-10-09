# Pre-freeze design process

**Status:** confirmatory design remains paused; numerical calibration is the next scientific phase. No confirmatory benchmark counts or analysis procedure are frozen, and a large blind benchmark has not been authorized.

## Why the freeze is paused

The original design infrastructure was built around a planned cross-family benchmark. The first independent family then showed that the historical strong-`|Q_ij|` rule did not reproduce its pilot benefit: it beat the ring on 17/24 pilot instances and lost on 17/24 independent weighted densest-k-subgraph instances. The original reports and advancement decision remain preserved as [historical artifacts](../results/README.md).

A more basic experimental question now has priority: **does either conditional-exchange RMS direction yield a useful topology, and how sensitive is any apparent effect to mixer/initial-state alignment?** Numerical calibration precedes this performance question; regression tests alone do not answer it.

Until that question is addressed, freezing a larger confirmatory benchmark would give false precision to an unresolved mechanism.

## Current sequence

| Stage | Purpose | Status |
| --- | --- | --- |
| Exploratory pilot | Establish implementation and generate the first hypothesis | Complete |
| Independent family | Test whether the first heuristic generalizes | Complete; negative for strong-`|Q_ij|` |
| Numerical calibration | Check numerical behavior before interpreting mechanism-performance comparisons | **Next** |
| Initialization/alignment control | Diagnose sensitivity to the topology/initialization pairing | Implemented; scientific evaluation pending calibration |
| Mechanistic score comparison | Compare a small predeclared set of structural descriptors | Not started |
| Design freeze | Lock estimand, counts, controls, failures, and resources | Paused |
| Pipeline freeze | Lock implementation and analysis before blind release | Not started |
| Confirmatory evaluation | Evaluate the frozen claim once | Not authorized |

## Statistical tooling status

The repository contains synthetic sensitivity, bootstrap, multiplicity, coverage, and Monte Carlo tooling from the earlier pre-freeze work. That machinery is retained because it records the project's intended statistical discipline and can be reused later.

It is **not** being expanded during the current mechanism stage. Existing smoke runs do not select a final sample size, establish power, or authorize a confirmatory claim.

The scientific bottleneck is currently mechanism identification, not another layer of statistical infrastructure.

## Current experimental control

The simulator exposes three initialization conditions:

- `uniform`: uniform feasible superposition;
- `mixer_low`: deterministic reference in the lowest mixer eigenspace;
- `mixer_high`: deterministic reference in the highest mixer eigenspace.

The comparison is implemented in `experiments/run_alignment_control.py`. The purpose is diagnostic: if topology rankings change substantially under initialization control, the earlier result cannot be interpreted as a pure topology effect.

For the positive-sign XY Hamiltonian, `mixer_high` is the Perron–Frobenius positive-amplitude reference for a connected mixer. The low extremum is a sensitivity control, and a spectral-projection fallback can depend on variable labels. Neither reference guarantees improved optimization, isolates a pure topology effect, or establishes practical hardware state preparation. No new performance result from these controls is claimed.

## Mechanism-stage design principle

After calibration, the mechanism study should remain small and predeclared. Its current conditional exchange RMS candidate is derived from `Q`, `c`, and fixed `k`, with mean and variance retained separately. The planned comparison retains:

1. historical strong `|Q_ij|`;
2. historical weak `|Q_ij|`;
3. low/high conditional exchange RMS as competing hypotheses, neither with an established performance advantage;
4. shuffled-score control;
5. random connected topology.

Each candidate should be evaluated under the same edge budget, optimizer, problem generator, and initialization conditions. No candidate should be selected because it performs best on the same records used to introduce it.

Only after this stage identifies a defensible estimand and mechanism should the existing design-freeze machinery be resumed.
