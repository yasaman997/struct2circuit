# Struct2Circuit

> **Start here:** This README is the primary entry point. It summarizes the research question, completed exploratory work, current status, and how to reproduce the verified checks. Detailed protocol and planning documents are linked below.

### Can problem structure inform the design of variational quantum algorithms?

**Struct2Circuit** is an ongoing computational research project investigating whether exploitable structure in optimization problems can be translated into better **feasibility-preserving quantum circuit design**.

The project begins with cardinality-constrained QUBOs and asks:

> **Under a fixed quantum-resource budget, can information about the optimization instance be used to construct a more effective variational quantum architecture than a structure-agnostic baseline?**

The longer-term question is:

> **Which properties of an optimization problem determine when a particular quantum architecture is useful or trainable?**

---

## Research idea

A variational quantum algorithm does not operate independently of the problem it is solving. The problem Hamiltonian, feasible subspace, mixer structure, and classical optimization interact.

Struct2Circuit studies this interaction by comparing **problem-informed** and **structure-agnostic** feasible mixers.

For cardinality-constrained QUBOs, we consider the optimization problem:

**Minimize:** `C_Q(x)`

**Subject to:** `Σᵢ xᵢ = k`

with `x ∈ {0, 1}ⁿ`. The constraint fixes the Hamming weight to `k`, so the mixer must preserve this feasible subspace.

The current study therefore follows:

**QUBO structure**  
↓  
**feasible mixer topology**  
↓  
**variational quantum circuit**  
↓  
**classical parameter optimization**  
↓  
**solution quality**

The central hypothesis is that **problem structure may contain information that can be exploited when designing the circuit itself**, rather than only when constructing the objective Hamiltonian.

---

# Exploratory pilot

### Matched-resource comparison

The initial pilot uses **24 synthetic cardinality-constrained QUBO instances** with:

- `n = 8` variables;
- `k = 3`;
- QAOA depth `p = 1`;
- an **8-edge structure-conditioned mixer**;
- an **8-edge fixed-ring mixer** as the primary matched-resource baseline;
- a **28-edge complete mixer** as a higher-resource reference.

### Preliminary result

| Metric | Structure-conditioned | Fixed ring |
|---|---:|---:|
| Instances won | **17 / 24** | 7 / 24 |
| Median normalized gap | **0.101715** | 0.112558 |

The paired pilot comparison gives an exploratory Wilcoxon signed-rank **p = 0.001752**.

The complete 28-edge mixer reached a median normalized gap of **0.096086** and serves as a higher-resource reference.

### What the pilot does *not* establish

This experiment does **not** establish:

- quantum advantage;
- hardware-level performance;
- scalability;
- superiority over strong classical optimization methods;
- generalization beyond the tested instance family;
- that structure-conditioned mixers are universally preferable.

The pilot is exploratory: its purpose is to test whether the proposed experimental question is sufficiently promising to justify a more rigorous controlled study.

---

# The research question I'm pursuing

The most interesting question is not simply whether one mixer can outperform another.

It is whether **performance varies systematically with properties of the underlying optimization instance**.

This leads to the next research question:

> **What structural properties of an optimization instance explain when a problem-informed variational architecture helps?**

Candidate features include:

- interaction-graph density;
- degree and community structure;
- coefficient variation;
- constraint tightness;
- degeneracy;
- spectral characteristics;
- other measures of combinatorial structure and instance difficulty.

The goal is to move from:

**“Which circuit wins?”**

toward:

**“Can we understand and predict which circuit should work for a given problem?”**

---

# From benchmarking to explanation

The planned research direction is:

### 1. Problem structure
Identify measurable structural features of optimization instances.

↓

### 2. Circuit structure
Determine how those features can inform feasible mixer construction.

↓

### 3. Optimization behavior
Study whether architectural differences are associated with differences in convergence, parameter sensitivity, gradients, or optimization traps.

↓

### 4. Prediction
Test whether classical statistical or machine-learning models can predict which architecture is likely to perform best from the instance structure.

↓

### 5. Generalization
Test whether the resulting relationships persist across new instances, problem sizes, constraints, and circuit/resource regimes.

The aim is **not** to use machine learning merely as another benchmark. The goal is to ask whether learning can reveal interpretable relationships between **problem structure and quantum-algorithm behavior**.

---

# Experimental philosophy

The project emphasizes controlled and reproducible computational experiments.

The methodology includes:

- matched mixer-resource budgets;
- exact feasible-subspace simulation;
- deterministic and seeded experiments;
- explicit feasibility and Hermiticity validation;
- paired statistical comparisons;
- blind evaluation for confirmatory experiments;
- null controls;
- sensitivity analysis;
- explicit resource accounting;
- separation of exploratory and confirmatory results.

A promising pilot is treated as a **hypothesis to investigate**, rather than as evidence for a broad quantum-advantage claim.

---

# Current status

**Exploratory pilot completed; confirmatory benchmark in development.**

The current research codebase includes:

- cardinality-constrained QUBO generation;
- feasible-subspace simulation;
- fixed and structure-conditioned mixer construction;
- deterministic parameter optimization;
- paired statistical analysis;
- reproducible experiment manifests;
- blind-access safeguards;
- synthetic sensitivity analysis;
- automated tests for core invariants and benchmark integrity.

The confirmatory benchmark has **not** been presented as completed. The candidate benchmark remains `DRAFT_UNFROZEN`, and blind evaluation has not begun.

---

# Why this problem?

Cardinality-constrained QUBOs provide a controlled setting in which several questions can be studied together:

- How much problem structure should a circuit exploit?
- What is the cost of encoding that structure?
- Does a structure-aware architecture remain competitive at the same resource budget?
- Which instance properties predict when it helps?
- Are improvements associated with changes in the optimization behavior of the variational circuit?

This makes the project a bridge between **combinatorial optimization, quantum algorithms, machine learning, and computational studies of variational optimization**.

---

# Reproducibility

The project is designed so that experimental claims can be traced back to explicit configurations and reproducible computations.

Random sources are controlled, generated mixers are validated, simulations remain within the required feasible subspace, and resource usage is explicitly recorded.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 experiments/run_pilot.py --instances 24 --n 8 --k 3
```

---

# Repository map

```text
src/struct2circuit/
    problems.py       Cardinality-QUBO definitions and generators
    mixers.py         Fixed and structure-conditioned mixer graphs
    simulator.py      Exact feasible-subspace QAOA simulation
    optimize.py       Deterministic parameter optimization
    analysis.py       Paired statistics and pilot report generation
    benchmark.py      Provisional manifest construction and blind-access guards
    sensitivity.py    Synthetic paired-inference sensitivity analysis
experiments/
    run_pilot.py      Reproducible pilot entry point
tools/
    benchmark_manifest.py
    prefreeze_sensitivity.py
tests/
    test_invariants.py
    test_benchmark_manifest.py
    test_sensitivity.py
tasks/
    PREFREEZE_DESIGN.md
paper/
    research_protocol.md
    stage_1_prefreeze_report.md
    prefreeze_sensitivity_report.md
results/
    prefreeze_sensitivity.json
```

---

# Research protocol

The detailed protocol documents the contribution ladder, instance families, controls, baselines, metrics, statistical analysis, transfer tests, ablations, and success criteria.

**For the full experimental design, see [`paper/research_protocol.md`](paper/research_protocol.md).**

---

## Current research direction

The project started with a narrow question:

> **Can exploiting optimization-instance structure improve feasible mixer design under a fixed resource budget?**

The next question is more general:

> **Can structural information about a computational problem be used to design, select, or learn quantum circuits that are more effective or resource-efficient—and can the resulting behavior be understood in terms of the structure and complexity of the underlying problem?**

That is the direction I am currently developing.
