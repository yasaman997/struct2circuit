# Struct2Circuit

> **Start here:** This README is the project overview. It states exactly what is implemented, what the pilot shows, what is not yet tested, and what the next scientific decision is. The project deliberately separates the **current study** from later research extensions.

## Current study

**Struct2Circuit** is an ongoing computational study of whether information already present in a cardinality-constrained QUBO can be used to construct a **sparse, feasibility-preserving XY mixer** that is useful under a matched **edge budget**.

The current, falsifiable question is:

> **At a fixed number of mixer edges, does a transparent QUBO-conditioned mixer produce better variational optimization outcomes than structure-agnostic connected mixers on unseen cardinality-constrained QUBOs?**

The current method is deliberately narrow. It is **not** a learned policy, a general quantum architecture-search framework, or a claim of quantum advantage.

### What “structure-conditioned” means here

For a QUBO matrix Q, the current generator uses only the off-diagonal interaction magnitudes |Q_ij|. It:

1. ranks candidate interactions by |Q_ij|;
2. builds a maximum-weight spanning tree under those scores to guarantee connectivity;
3. adds the strongest remaining interactions until the edge budget is reached.

The linear term c, optimal solution, objective value, solver output, labels, and benchmark results are not supplied to the mixer constructor. The rule is deterministic given Q and the requested edge budget. Because only relative interaction magnitudes are used, multiplying all off-diagonal Q entries by a common positive factor does not change the selected topology; signs are deliberately ignored in this first study.

This makes the scientific intervention explicit: **use pairwise objective structure as side information for constrained circuit topology.**

## Problem and feasibility

We study:

**Minimize:** C_Q(x)

**Subject to:** Σᵢ xᵢ = k

with x ∈ {0, 1}ⁿ.

The constraint fixes Hamming weight to k. The mixer is therefore constructed from XY exchanges, which preserve the fixed-weight feasible subspace.

The exact simulator represents only the feasible basis. This is a computational convenience and an exact noiseless invariant—not a claim that a hardware implementation would have zero leakage.

The current study follows:

**QUBO structure**  
↓  
**feasible mixer topology**  
↓  
**variational circuit**  
↓  
**classical parameter optimization**  
↓  
**solution quality**

## What has actually been tested

### Exploratory pilot

The initial pilot contains **24 synthetic block-correlated QUBO instances** with:

- n = 8;
- k = 3;
- QAOA depth p = 1;
- 8-edge structure-conditioned mixer;
- 8-edge fixed-ring mixer;
- 28-edge complete mixer as a **higher-edge-count reference**, not a matched-resource baseline.

The paired exploratory result is:

| Metric | Structure-conditioned | Fixed ring |
|---|---:|---:|
| Wins / ties / losses | **17 / 0 / 7** | — |
| Median normalized gap | **0.101715** | **0.112558** |

The one-sided paired Wilcoxon result is p = 0.001752. This is reported descriptively for the exploratory pilot; it is **not** a confirmatory significance claim and is not a basis for claiming generalization.

The complete 28-edge reference has median normalized gap 0.096086. It uses a substantially larger edge budget and therefore should not be interpreted as evidence for or against the 8-edge comparison.

### What the pilot does not establish

It does not establish:

- quantum advantage;
- superiority over strong classical optimization;
- scalability;
- hardware-level performance;
- noise robustness;
- transfer across problem families, sizes, or hardware graphs;
- universal usefulness of structure-conditioned mixers;
- a causal explanation in terms of trainability, gradients, local minima, or loss landscapes.

Those are **future questions**, not current results.

## Primary scientific target

The next stage is to test whether the observed effect, if real, is tied to identifiable instance structure rather than one synthetic family or one hand-chosen topology.

The primary current target is therefore:

> **Estimate the family-level paired effect of the transparent structure-conditioned mixer versus predeclared structure-agnostic connected baselines, under an explicitly matched construction budget, and test whether that relationship survives independent problem families and unseen instances.**

The current primary outcome is normalized feasible-range optimality-gap improvement. Secondary outcomes include optimum-sampling probability and, later, measured circuit cost.

“Better” must therefore be defined by the predeclared outcome for each claim; the project will not switch to whichever metric favors the proposed method after seeing results.

## Fair comparison: edge budget versus quantum resources

The pilot matches **mixer edge count**, not total quantum resources.

An equal number of graph edges is a construction-level control. It does **not** imply equal two-qubit gate count, circuit depth, routing overhead, or hardware execution cost.

For this reason:

- **pilot:** equal-edge comparison;
- **planned resource study:** transpiled two-qubit gates, depth, routing/SWAP count, shots, and classical search cost;
- **resource Pareto claim:** only after those quantities are actually measured.

The complete mixer is a higher-edge-count reference and is not a matched baseline.

## Controls and generalization

The planned benchmark separates:

- fixed ring;
- random connected graphs at the same edge budget;
- transparent structure-conditioned graphs;
- multiple independent problem families;
- a weak-structure negative control.

The random baseline is an **expected-random comparator** estimated from multiple independent random graphs, not a single lucky draw.

The negative control tests whether the proposed rule can create an apparent advantage when the generator removes obvious planted structure. It cannot prove that no exploitable information exists.

Generalization is reported by dimension:

- **instance transfer:** unseen instances from the same distribution;
- **family transfer:** unseen problem family;
- **scale/cardinality transfer:** unseen n or k/n;
- **hardware transfer:** unseen hardware connectivity.

These are separate claims and will not be collapsed into a generic “transfer” label.

## Trainability is a future mechanistic question

The current pilot does not measure trainability.

If a later study shows a performance difference, a separate mechanistic analysis may test:

- optimization trajectories;
- gradient statistics;
- parameter sensitivity;
- local minima or traps;
- concentration/landscape diagnostics;
- robustness to initialization and optimizer choice.

Only evidence from those measurements will support a trainability explanation.

## Learning is a future extension

The present mixer is a transparent deterministic heuristic, not an AI system.

A later study may replace the fixed |Q_ij| score with a **learned mixer-selection policy**. Any such model would use only features available before solving the target instance and would be evaluated with explicit leakage controls. It would not be described as “AI discovering quantum algorithms” unless the evidence justified that much stronger claim.

Possible future models include interpretable edge scorers or graph-based predictors. The model class is intentionally not fixed before the scientific question and feature interface are settled.

## Experimental philosophy

The repository emphasizes:

- exact fixed-weight feasibility in noiseless simulation;
- explicit definitions of what information reaches the mixer;
- matched edge budgets as the pilot control;
- independent random baselines;
- paired instance-level analysis;
- train/validation/test separation for future learning;
- blind evaluation only after a design and pipeline freeze;
- explicit treatment of optimizer failures;
- resource accounting rather than edge count alone;
- reproducible seeds, manifests, and checks;
- separation of exploratory observations from confirmatory claims.

## Current status

**Exploratory pilot completed; benchmark design and pre-freeze infrastructure in development.**

Implemented:

- cardinality-constrained QUBO definitions and generators;
- fixed-weight feasible-subspace simulation;
- fixed, random-connected, and transparent structure-conditioned mixers;
- deterministic pilot optimization;
- paired pilot statistics;
- provisional benchmark manifests and blind-access guards;
- synthetic statistical sensitivity tooling;
- automated tests for mathematical invariants and benchmark integrity.

Not completed:

- confirmatory blind evaluation;
- learned mixer policy;
- trainability/landscape study;
- hardware/transpilation study;
- strong classical benchmark study;
- claims of generalization or quantum advantage.

The benchmark remains DRAFT_UNFROZEN. Blind performance has not been evaluated.

## Reproducibility

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -e .
    python3 -m unittest discover -s tests -v
    python3 experiments/run_pilot.py --instances 24 --n 8 --k 3

Generated pilot outputs should be treated as reproducible exploratory artifacts, not frozen benchmark results.

## Repository map

    src/struct2circuit/
        problems.py       Cardinality-QUBO definitions and generators
        mixers.py         Fixed, random, and structure-conditioned mixer graphs
        simulator.py      Exact feasible-subspace QAOA simulation
        optimize.py       Deterministic parameter optimization
        analysis.py       Pilot statistics and reporting
        benchmark.py      Provisional manifest construction and blind guards
        sensitivity.py    Synthetic statistical sensitivity analysis
    experiments/
        run_pilot.py      Reproducible exploratory pilot entry point
    tools/
        benchmark_manifest.py
    tests/
    paper/
    tasks/
    results/

## Research protocol

The full protocol defines the current estimand, structure signal, controls, resource accounting, generalization dimensions, statistical analysis, and decision gates.

See [paper/research_protocol.md](paper/research_protocol.md).

## Research direction beyond the current study

The longer-term program is:

**problem structure → constrained circuit design → measured optimization behavior → interpretable prediction → generalization**

A learned policy, trainability analysis, architecture selection, hardware evaluation, and broader problem families are possible extensions. They are not being presented as completed contributions.

The core scientific idea remains deliberately simple:

> **Can structure in an optimization problem be used as side information to design a constrained variational circuit more effectively, and under what conditions does that stop working?**
