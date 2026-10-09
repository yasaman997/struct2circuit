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

The simulator now exposes three explicit initialization modes:

- `uniform`: uniform superposition over all feasible weight-`k` states;
- `mixer_low`: deterministic opposite-extremum sensitivity condition in the lowest mixer eigenspace;
- `mixer_high`: positive-amplitude Perron–Frobenius reference in the highest mixer eigenspace for connected mixers.

The positive-sign XY Hamiltonian is the adjacency matrix of the feasible exchange graph. For a connected mixer and `0 < k < n`, that graph is connected: its highest state is the unique positive-amplitude Perron–Frobenius reference, equivalently the ground state of `-H_M`. The lowest state is an opposite-extremum sensitivity condition. Projecting a fixed reference onto an extremal eigenspace removes arbitrary eigenvector-basis choices. When uniform overlap vanishes, the deterministic computational-basis fallback can depend on labels; determinism does not imply permutation equivariance. Phase fixing removes a global phase only. Neither condition guarantees better QAOA performance. Uniform-state fidelity with each extremal eigenspace is reported separately.

These are exact small-system diagnostics, not hardware-efficient state-preparation prescriptions. Comparing them measures sensitivity to the **topology/initialization pairing**; it does not identify a pure causal topology effect.

## 4. Testable predictions

A useful mechanism must make predictions beyond “the selected graph has high score.”

1. **Structural score separation:** the proposed rule should measurably differ from ring/random controls on the declared structural descriptor. This is a construction sanity check, not evidence of QAOA benefit.
2. **Outcome association:** evaluate the descriptor's relationship to paired QAOA improvement within each declared initialization condition.
3. **Initialization sensitivity:** uniform-versus-uniform compares topologies from a shared state; uniform-versus-spectral within one topology measures initialization sensitivity; cross-topology spectral comparisons change both topology and initial state. A reversal across these conditions is evidence of interaction, not successful causal isolation.
4. **Controls:** low/high rankings are competing directional hypotheses. Shuffling scores through the same graph-construction pipeline tests whether their correspondence with the instance matters.
5. **Mechanistic specificity:** any surviving advantage should be related to a concrete transition-graph or mixer-spectral property, not merely to generic graph density or degree.

## 5. What the independent benchmark established

The first independent weighted densest-k-subgraph benchmark did not reproduce the pilot's positive effect. With the uniform feasible initialization, the strong-`|Q_ij|` topology lost to the ring on 17/24 instances and had a higher median normalized gap than both the ring and random-mean comparators.

The selected topology nevertheless had very high interaction-weight alignment. Thus high alignment with the QUBO interaction graph was **not sufficient** for better depth-one QAOA performance on that family.

This weakens the original strong-`|Q_ij|` hypothesis. It does not show that all instance-conditioned mixer design is ineffective.

## 6. Current falsification boundary

The historical strong-`|Q_ij|` rule failed its first independent generalization check and remains a comparator. The current conditional-RMS candidate likewise must not be promoted to a general method without:

- independent problem families;
- initialization/alignment controls;
- matched optimization and circuit-resource accounting;
- weak/inverse/shuffled controls;
- and a mechanistic analysis that identifies what transition-graph property is responsible.

The independent negative result is retained. Low- and high-RMS are untested competing hypotheses; null, mixed, or negative results must also be retained rather than choosing a direction after observing performance. Numerical calibration precedes these mechanism comparisons.

## 6A. Concrete transition-level mechanism candidate

The current descriptor is derived from the exact feasible exchange-cost formula,
rather than generic interaction magnitude. For exchanging occupied i with
unoccupied j,

`Delta C_ij(x) = (Q_jj-Q_ii) + (c_j-c_i) + 2 sum_(l != i,j) x_l (Q_jl-Q_il)`.

This identity assumes symmetric `Q`. Problem construction and descriptor evaluation
both validate with `rtol=0`, `atol=1e-12` and leave coefficients unchanged. For
accepted asymmetry `epsilon=max|Q-Q.T|`, the displayed expression can differ from
the actual quadratic-cost swap by at most `2*(k-1)*epsilon`; mean and RMS inherit
that absolute bound, apart from floating-point evaluation. The input tolerance
does not promise relative accuracy for tiny objectives. All supplied generators
produce symmetric matrices.

Condition on uniformly sampled feasible states with `x_i=1`, `x_j=0`, and
`sum(x)=k`. Write `N=n-2`, `m=k-1`, `d0=Q_jj-Q_ii+c_j-c_i`, and
`a_l=Q_jl-Q_il` for the remaining sites. Exactly `m` of these `N` sites are
occupied. For `N>1`, define the exact conditional moments:

`mean_a = sum(a_l)/N`

`variance_a = sum((a_l-mean_a)^2)/N`

`mu_ij = E[Delta C_ij] = d0 + 2*m*mean_a`

`var_ij = Var(Delta C_ij) = 4*m*(N-m)/(N-1)*variance_a`

`rms_ij = sqrt(mu_ij^2 + var_ij)`.

For `N=0`, `mu=d0` and `var=0`; for `N=1`, `mu=d0+2*m*a_1`
and `var=0`. At `k=1`, the shift is exactly `d0`; at `k=n-1`, it is
`d0+2*sum(a_l)`. Equal coefficients also give zero conditional variance.
Reversing `i,j` negates the conditional mean and preserves variance and RMS.

Conditional RMS is the primary symmetric edge score. Mean and variance remain
available separately to distinguish systematic shifts from occupancy-dependent
variation. The former norm of exchange coefficients omits fixed-`k` occupancy
and cross terms; it is not this statistic and is no longer the primary score.
These moments use only `Q`, `c`, and `k`, without solutions or optimization
outcomes. Mathematically they respect variable permutations and feasible-equivalent
encodings; positive scaling by `a` multiplies mean/RMS by `a` and variance by `a^2`.
Additive feasible objective constants cancel from all exchange differences.

For the mean, the implementation reduces `2*m/N` to the integer ratio `p/d`
and evaluates `(d*d0 + p*sum(a_l))/d`. One `fsum` receives `d` copies of
each original diagonal/linear term and `p` copies of each original interaction
term `Q_jl, -Q_il`; division follows the complete cancellation. Repetition avoids
rounding weighted products or a centered aggregate before that cancellation.
At `m=N`, `p=2,d=1`, so this is the direct deterministic exchange sum. At
`m=0` (including `N=0`), only the original `d0` terms enter the sum.

Variance still uses a reference remaining site `r`, forming `z_l = a_l-a_r`
as one compensated sum of `Q_jl, -Q_il, -Q_jr, Q_ir`. Compensated comparisons
select a minimum `a_r` without first rounding large row differences. Since
`Var(a)=Var(z)`, this computes the same conditional variance. No transformed
matrix is materialized. Floating-point arithmetic cannot recover information
already rounded out of the inputs. The compensated numerator and variance
calculation must remain within floating-point range, and final sum/division
rounding still prevents a universal bitwise invariance guarantee.

Two opposite hypotheses are deliberately distinguishable:

- low-RMS edges have smaller conditional mean-square exchange changes and may support shallow mixing;
- high-RMS edges have larger conditional mean-square exchange changes and may improve cost discrimination.

Neither direction is privileged before data are inspected. Both must be treated
as predeclared competing mechanisms rather than tuning the sign after seeing
performance.

The graph constructor remains spanning-tree-first followed by ranked remaining
edges, using exactly the declared edge budget and preserving connectivity.
Unique edge scores give permutation-equivariant selection; deterministic index
tie-breaking can break equivariance at exact ties. Low/high choices can also
change degrees, bottlenecks, and spectra. Those are properties to measure, not
evidence of a transition mechanism on their own.

## 7. Next scientific experiment

The planned next experiment is small and mechanistic. Compare a predeclared set of structural signals under the same mixer edge budget and all three initialization conditions:

1. strong `|Q_ij|`;
2. weak `|Q_ij|`;
3. low/high conditional exchange RMS defined above;
4. shuffled-score control;
5. random connected topology.

The objective is to determine whether pre-optimization exchange statistics predict useful transitions within declared topology/initialization configurations. New optimization uses `u=gamma*(C_max-C_min)` on `[0,2*pi]` with normalized costs. This is a characteristic-scale convention, not a fundamental gamma period. The shared beta box `[0,pi]` is also a convention. Parameter-domain and optimizer-reference calibration remain prerequisites to interpreting new topology rankings; they are not part of this code-repair pass.

The existing alignment script retains the historical strong-`|Q|` comparator and requires the actual ring edge count for all topologies. It is a sensitivity diagnostic, not the complete conditional-RMS mechanism study. No new scientific performance data or conclusion is supplied by these repairs.

Only after that question is answered should a larger confirmatory cross-family benchmark be frozen.
