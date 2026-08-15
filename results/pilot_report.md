# Struct2Circuit pilot report

**Status:** exploratory pilot; not confirmatory evidence and not a quantum-advantage claim.

## Central claim under test

A structure-conditioned, feasibility-preserving mixer improves the solution-quality/resource Pareto frontier over fixed mixers for cardinality-constrained QUBOs.

## Pilot result

- Paired median normalized-gap reduction, structure versus ring: **0.014678**.
- Exploratory paired-bootstrap 95% interval: **[0.001460, 0.016576]**.
- Structure wins / ties / losses: **17 / 0 / 7**.
- One-sided paired Wilcoxon p-value: **0.00175214**.
- Median optimal-solution probability increase: **0.005985**.

At the same mixer-edge budget, the ring median gap was **0.112558** and the structure-conditioned median gap was **0.101715**. The complete mixer used **28** edges, compared with **8** for both equal-budget mixers.

## Decision

The pilot clears the pre-specified advancement rule. A larger blind study is justified.

## What this does not establish

This small, noiseless, depth-one study does not establish generalization, hardware advantage, scaling advantage or superiority to tuned classical optimization. It is a software/invariant check and an effect-size estimate for the confirmatory design.

## Next confirmatory milestone

Lock generator seeds before evaluation; include multiple structural regimes and at least one non-financial cardinality family; compare against XY-QAOA, warm-start and counterdiabatic baselines; incorporate cost-layer and transpilation resources; evaluate depths 1-3; and report the search cost required to construct each learned mixer.
