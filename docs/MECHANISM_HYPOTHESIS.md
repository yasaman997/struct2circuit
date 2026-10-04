# Mechanism hypothesis

## 1. Revised hypothesis

The original intuition was too close to a local-transition story: large absolute QUBO coefficients might somehow make the corresponding XY exchange more favorable.

That is not the right mechanism.

For a feasible bitstring with xi = 1 and xj = 0, let x' be the bitstring obtained by exchanging those occupations. For symmetric Q,

ΔC = C(x') - C(x)
   = (Qjj - Qii)
     + 2 Σ_{l not equal to i,j} xl (Qjl - Qil)
     + cj - ci.

The Qij term cancels because xi xj = 0 both before and after the swap.

Therefore:

> A large |Qij| is **not** a direct local-energy-gradient justification for placing an XY edge between i and j.

This is an important limitation of the present method and should be visible to a serious reader.

## 2. The mechanism that remains plausible

The defensible hypothesis is instead a **graph-level topology hypothesis**:

> The off-diagonal QUBO matrix defines an interaction graph. Under a limited number of XY exchange edges, a sparse mixer whose topology preserves strong interaction structure may shape the feasible-state transition graph in a way that is more compatible with the global objective landscape than a structure-agnostic topology.

This is deliberately weaker than saying that a selected edge produces a favorable swap.

The proposed rule is therefore a graph-approximation heuristic:

QUBO interaction graph
→ sparse mixer topology
→ feasible-state transition graph
→ shallow variational dynamics
→ optimization outcome.

The causal link in the middle is the part that remains to be demonstrated.

## 3. Concrete predictions

The hypothesis makes four experimentally testable predictions.

### P1 — structural alignment

At the same edge budget, the strong-|Qij| mixer should have much higher interaction-weight alignment with the target QUBO than ring or random connected mixers.

This is almost guaranteed by construction and is therefore a **sanity check**, not evidence of usefulness.

### P2 — performance should track structural usefulness, not alignment alone

If QUBO-interaction alignment is actually useful for QAOA topology selection, instances or structural regimes with stronger exploitable interaction organization should show larger positive paired improvement.

A high alignment score by itself is not enough. The benchmark must connect that alignment to the measured QAOA outcome.

### P3 — inverse-score control should remove the proposed advantage

Construct a deliberately opposite topology by using the weakest |Qij| interactions first, while keeping the same connectivity requirement and edge budget.

If strong interactions are genuinely useful side information, the strong-score topology should outperform this inverse-score control on the regimes where the hypothesis is supposed to apply.

If the inverse topology does as well or better, the current score is not supported as a useful mechanism.

### P4 — shuffled-score control should collapse toward random

A future controlled experiment should randomly permute the interaction scores before topology construction. This preserves the distribution of scores and the same deterministic construction procedure while destroying correspondence between scores and variable pairs.

If the pairwise correspondence is causal, performance should move toward the structure-agnostic random comparator.

## 4. What would falsify the mechanism

The mechanism should be regarded as unsupported if:

1. strong-|Qij| topology has no reproducible outcome benefit across independent structured families;
2. inverse-score or shuffled-score controls perform comparably or better;
3. any apparent advantage disappears after matching optimization budget or circuit resources;
4. performance correlates only with generic graph properties such as degree distribution, not with correspondence to the QUBO interaction structure.

A failure is not a failure of “quantum optimization.” It is a failure of this particular structure-to-mixer map.

## 5. First independent test

The first independent benchmark intentionally changes the problem family from the original block-correlated portfolio proxy to weighted densest-k-subgraph.

Settings:

- 24 fresh instances;
- n = 9;
- k = 4;
- edge density = 0.55;
- planted-community strength = 1.8;
- QAOA depth p = 1;
- mixer edge budget = 9;
- exact fixed-weight simulation;
- same deterministic p=1 optimizer for every mixer;
- 8 independent random connected mixer graphs per instance;
- an inverse-|Qij| topology as a mechanism control;
- seeds are outside the pilot seed namespace;
- benchmark is exploratory, not frozen or confirmatory.

The primary comparison remains normalized feasible-range expectation gap, with lower better.

## 6. Result

The strong-|Qij| rule did **not** reproduce the pilot effect.

| Comparison | Median paired improvement favoring structure | Wins / ties / losses |
| --- | ---: | ---: |
| Structure vs fixed ring | -0.011456 | 7 / 0 / 17 |
| Structure vs mean of 8 random mixers | -0.007976 | 10 / 0 / 14 |
| Structure vs inverse-|Qij| control | +0.013549 | 15 / 0 / 9 |

The 95% bootstrap interval for structure vs ring was [-0.017384, -0.000430]. For structure vs random mean it was [-0.015608, 0.003493]. These are exploratory intervals only.

The median normalized gaps were:

- ring: 0.476476
- structure: 0.488890
- random mean: 0.485262
- inverse structure: 0.472628

The strong-|Qij| mixer had very high interaction-weight alignment by construction (median top-budget alignment 0.9855), while ring was 0.4097 and the inverse control was 0.0 in almost all instances.

This is a useful negative result: **high alignment with the QUBO interaction graph did not translate into better p=1 optimization on this independent family.**

The inverse-vs-structure Wilcoxon comparison gives p = 0.0212, but the corresponding sign-test p-value is 0.154. That disagreement means the inverse result should be treated as a diagnostic signal, not as a confirmatory claim.

## 7. Current scientific interpretation

The independent result weakens the broad hypothesis that “use strong |Qij| interactions as sparse XY edges” is a generally useful topology rule.

It does not establish that QUBO structure is irrelevant to mixer design.

The more interesting question is now:

> Which structural information, if any, should determine the **transition graph** of a fixed-weight mixer?

The current |Qij| score may be measuring an objective interaction graph while the relevant quantity for shallow XY dynamics may instead involve:

- node-conditioned interaction profiles;
- exchange-cost similarity;
- pairwise differences in interaction neighborhoods;
- community-level structure;
- spectral structure;
- multi-edge patterns rather than individual coefficients.

Those are candidate mechanisms, not results.

## 8. Next experiment before any blind benchmark

Do not freeze the existing benchmark around the current strong-|Qij| rule yet.

The next scientific experiment should compare a small, predeclared family of mechanistic scores:

1. strong |Qij|;
2. weak |Qij|;
3. exchange-profile similarity, based only on pre-optimization Q and c;
4. shuffled-score control;
5. random connected topology.

The experiment should ask which structural signal, if any, changes the mixer transition graph in a way that predicts QAOA improvement.

Only after that score-selection question is settled should a larger multi-family blind benchmark be frozen.
