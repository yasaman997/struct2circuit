# First independent benchmark: weighted densest-k-subgraph

**Status:** exploratory independent benchmark; not frozen and not confirmatory.

## Purpose

The original pilot used one block-correlated portfolio/index-tracking proxy family. This benchmark changes the problem generator to weighted densest-k-subgraph, uses fresh seeds, and adds an inverse-|Qij| control specifically to test the proposed mechanism.

## Fixed settings

| Setting | Value |
| --- | --- |
| Instances | 24 |
| Problem family | weighted densest-k-subgraph |
| n | 9 |
| k | 4 |
| edge density | 0.55 |
| planted community strength | 1.8 |
| QAOA depth | p = 1 |
| mixer edge budget | 9 |
| random connected replicates | 8 per instance |
| parameter grid | 9 × 9, followed by the repository's two local refinements |
| primary metric | normalized feasible-range expectation gap, lower is better |
| simulator | exact fixed-weight subspace |
| problem seed namespace | 900000000 + 10007 × instance |
| random mixer seed namespace | 1700000000 + 997 × instance + replicate |

The seed namespace is separate from the original pilot and from the current draft benchmark manifest.

## Primary result

Paired improvement is defined as baseline gap minus structure-conditioned gap, so positive values favor the proposed structure rule.

| Comparison | Median paired improvement | 95% bootstrap interval | Wins / ties / losses |
| --- | ---: | ---: | ---: |
| Structure vs fixed ring | -0.011456 | [-0.017384, -0.000430] | 7 / 0 / 17 |
| Structure vs mean of 8 random mixers | -0.007976 | [-0.015608, 0.003493] | 10 / 0 / 14 |
| Structure vs inverse-|Qij| | +0.013549 | [-0.006952, 0.025825] | 15 / 0 / 9 |

Median normalized gaps:

| Mixer | Median gap |
| --- | ---: |
| Fixed ring | 0.476476 |
| Structure-conditioned | 0.488890 |
| Mean of 8 random connected mixers | 0.485262 |
| Inverse-|Qij| control | 0.472628 |

The structure-conditioned mixer therefore performs worse than the ring by about 0.0115 normalized-gap units at the median and worse than the random-mean comparator by about 0.0080.

The inverse-|Qij| control has the best median gap among these four methods. This is a mechanism diagnostic, not evidence that the inverse rule is a better general algorithm.

## Mechanism diagnostics

The interaction-alignment ratio is:

selected |Qij| mass / maximum |Qij| mass achievable by the same number of edges.

Median alignment:

| Mixer | Median alignment |
| --- | ---: |
| Structure-conditioned | 0.985517 |
| Fixed ring | 0.409685 |
| Inverse-|Qij| | approximately 0 in almost all instances |

Thus the benchmark creates a strong separation in the intended structural signal. The negative performance result is therefore not explained by a failure to implement the score.

The critical scientific observation is:

> High alignment with the QUBO interaction graph did not produce better depth-one QAOA outcomes on this independent family.

This weakens the current hypothesis that strong |Qij| interactions are the right side information for choosing sparse XY edges.

## Statistical caution

The structure-vs-ring Wilcoxon one-sided p-value is 0.9606 because the observed effect points in the opposite direction to the proposed hypothesis. The sign-test p-value is 0.9887.

For structure versus the random-mean comparator, the Wilcoxon p-value is 0.9737 and the sign-test p-value is 0.8463.

For inverse structure versus structure, the Wilcoxon comparison gives p = 0.0212, while the sign-test p-value is 0.1537. The disagreement is a warning against turning the inverse result into a confirmatory claim.

All intervals and tests in this report are exploratory.

## What changed scientifically

The first pilot suggested that structure-conditioned mixers might help on one block-correlated family.

The independent benchmark shows that the same deterministic score and topology rule can lose on another structured family even when the selected mixer is highly aligned with the QUBO interaction graph.

The correct conclusion is therefore not “structure does not matter.” It is:

> The current mapping from QUBO interaction magnitude to sparse XY topology is not a generally validated mechanism.

That is a useful narrowing of the research question.

## Next experiment

Before freezing a large blind benchmark, compare a small mechanistic family of scores under the same construction budget:

1. strong |Qij|;
2. weak |Qij|;
3. a score based on similarity of each variable's interaction profile;
4. shuffled interaction scores;
5. random connected topology.

The target is to determine whether any pre-optimization structural signal predicts a useful mixer transition graph. Only after that mechanism-selection question is resolved should the project return to a confirmatory cross-family benchmark.
