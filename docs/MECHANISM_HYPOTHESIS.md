# Mechanism hypothesis

## Current question

The project asks whether information from a cardinality-constrained QUBO can be used as side information when choosing a sparse, feasibility-preserving XY mixer topology under a fixed edge budget.

The first score tested was strong `|Q_ij|`. It is now treated as a **candidate heuristic that failed to generalize in the first independent test**, not as an established mechanism.

## 1. What the first hypothesis got wrong

For a feasible bitstring with `x_i = 1` and `x_j = 0`, exchanging the occupations of `i` and `j` does not change the `Q_ij x_i x_j` term: that term is zero before and after the exchange. The single-exchange objective difference instead depends on diagonal, linear, and differences in the two variables' interactions with the occupied variables.

Therefore:

> A large `|Q_ij|` is not, by itself, a local-energy-gradient justification for placing an XY edge between `i` and `j`.

This correction is central to the project.

## 2. The narrower remaining hypothesis

A defensible hypothesis is graph-level rather than edge-local:

> Under a limited number of XY exchange edges, some pre-optimization structural descriptor of the QUBO may select a feasible-state transition graph whose shallow variational dynamics are more compatible with the objective landscape than a structure-agnostic topology.

The current causal chain is therefore:

`instance descriptor → mixer topology → feasible-state transition graph → variational dynamics → optimization outcome`

The middle of this chain is still unproven.

## 3. A critical confound: initialization and mixer alignment

Different XY topologies can have different mixer ground spaces and different spectral structure. The original benchmark used the same uniform feasible initial state for every topology. That makes the comparison clean as a **fixed-initialization benchmark**, but it does not isolate topology from initial-state/mixer alignment.

This matters because recent constrained-QAOA work has shown that initial-state/mixer alignment can affect low-depth performance. The literature map therefore treats alignment as a required control rather than a nuisance detail.

The simulator now exposes two explicit initialization modes:

- `uniform`: uniform superposition over all feasible weight-`k` states;
- `mixer_ground`: the normalized projection of the uniform feasible state onto the mixer Hamiltonian's ground-state eigenspace.

The second mode is an exact diagnostic/control in the small-system simulator. It is **not** being presented as a hardware-efficient state-preparation prescription.

The new `experiments/run_alignment_control.py` compares these initializations while holding the mixer graph and optimization procedure fixed. No alignment-control result is currently used as evidence for the main hypothesis.

## 4. Testable predictions

A useful mechanism must make predictions beyond “the selected graph has high score.”

1. **Structural score separation:** the proposed rule should measurably differ from ring/random controls on the declared structural descriptor. This is a construction sanity check, not evidence of QAOA benefit.
2. **Outcome association:** if the descriptor is useful, its relationship to paired QAOA improvement should persist after initialization is controlled.
3. **Alignment robustness:** a topology effect that disappears or reverses when initialization is matched/aligned should not be attributed to topology alone.
4. **Negative controls:** weak/inverse and shuffled descriptors should remove the claimed correspondence if that correspondence is causal.
5. **Mechanistic specificity:** any surviving advantage should be related to a concrete transition-graph or mixer-spectral property, not merely to generic graph density or degree.

## 5. What the independent benchmark established

The first independent weighted densest-k-subgraph benchmark did not reproduce the pilot's positive effect. With the uniform feasible initialization, the strong-`|Q_ij|` topology lost to the ring on 17/24 instances and had a higher median normalized gap than both the ring and random-mean comparators.

The selected topology nevertheless had very high interaction-weight alignment. Thus high alignment with the QUBO interaction graph was **not sufficient** for better depth-one QAOA performance on that family.

This weakens the original strong-`|Q_ij|` hypothesis. It does not show that all instance-conditioned mixer design is ineffective.

## 6. Current falsification boundary

The current strong-`|Q_ij|` rule should not be promoted to a general method unless it survives:

- independent problem families;
- initialization/alignment controls;
- matched optimization and circuit-resource accounting;
- weak/inverse/shuffled controls;
- and a mechanistic analysis that identifies what transition-graph property is responsible.

If those tests fail, the project should treat the strong-`|Q_ij|` mapping as a failed candidate and move to a different structural descriptor rather than tuning the same rule until it succeeds.

## 7. Next scientific experiment

The next experiment is deliberately small and mechanistic, not another large benchmark. Compare a predeclared set of structural signals under the same mixer edge budget and both initialization controls:

1. strong `|Q_ij|`;
2. weak `|Q_ij|`;
3. interaction-profile similarity between variables;
4. shuffled-score control;
5. random connected topology.

The objective is to determine whether any pre-optimization structural signal predicts a useful transition graph **after the initialization confound is exposed**.

Only after that question is answered should a larger confirmatory cross-family benchmark be frozen.
