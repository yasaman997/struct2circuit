# Pre-freeze design process

**Status:** paused at the exploratory mechanism-control stage. No confirmatory benchmark counts or analysis procedure are frozen.

## Why the freeze is paused

The original design infrastructure was built around a planned cross-family benchmark. The first independent family then showed that the current strong-`|Q_ij|` rule does not generalize from the pilot.

A more basic experimental question now has priority: **can any apparent topology effect be separated from mixer/initial-state alignment and from the particular structural score?**

Until that question is addressed, freezing a larger confirmatory benchmark would give false precision to an unresolved mechanism.

## Current sequence

| Stage | Purpose | Status |
| --- | --- | --- |
| Exploratory pilot | Establish implementation and generate the first hypothesis | Complete |
| Independent family | Test whether the first heuristic generalizes | Complete; negative for strong-`|Q_ij|` |
| Initialization/alignment control | Separate topology from mixer/initial-state alignment | **Next** |
| Mechanistic score comparison | Compare a small predeclared set of structural descriptors | Not started |
| Design freeze | Lock estimand, counts, controls, failures, and resources | Paused |
| Pipeline freeze | Lock implementation and analysis before blind release | Not started |
| Confirmatory evaluation | Evaluate the frozen claim once | Not authorized |

## Statistical tooling status

The repository contains synthetic sensitivity, bootstrap, multiplicity, coverage, and Monte Carlo tooling from the earlier pre-freeze work. That machinery is retained because it records the project's intended statistical discipline and can be reused later.

It is **not** being expanded during the current mechanism stage. Existing smoke runs do not select a final sample size, establish power, or authorize a confirmatory claim.

The scientific bottleneck is currently mechanism identification, not another layer of statistical infrastructure.

## Current experimental control

The simulator exposes two initialization conditions:

- `uniform`: uniform feasible superposition;
- `mixer_ground`: normalized projection of the uniform feasible state onto the mixer ground-state eigenspace.

The comparison is implemented in `experiments/run_alignment_control.py`. The purpose is diagnostic: if topology rankings change substantially under initialization control, the earlier result cannot be interpreted as a pure topology effect.

The aligned state is an exact small-system control. It is not being presented as a practical hardware state-preparation method.

## Mechanism-stage design principle

The next study should remain small and predeclared. Candidate structural signals are:

1. strong `|Q_ij|`;
2. weak `|Q_ij|`;
3. interaction-profile similarity;
4. shuffled-score control;
5. random connected topology.

Each candidate should be evaluated under the same edge budget, optimizer, problem generator, and initialization conditions. No candidate should be selected because it performs best on the same records used to introduce it.

Only after this stage identifies a defensible estimand and mechanism should the existing design-freeze machinery be resumed.
