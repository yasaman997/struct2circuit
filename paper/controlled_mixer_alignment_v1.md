# Stage 7O — Controlled cost–graph alignment experiment, design v1

Status: **PROPOSED FOR REVIEW; IMPLEMENTATION AND NUMERICAL EXECUTION NOT AUTHORIZED.**

Recommend one experiment: 32 fresh n=6, k=3 QUBOs, eight from each of four fixed generators, with the highest- and lowest-alignment relabelings of a single six-cycle topology. Use uniform initialization, depth one, B00, and identical finite empirical-reference work for both arms. There are at most 64 landscapes and exactly 32 planned paired experimental units. This is a new mechanism experiment, not another P33 qualification attempt.

## 1. Evidence preserved and scope

Inspected source: clean main at `2042e4eb60d5eae71b4fbc1500af4ae065adde1f`. The two frozen protocols remain unchanged:

| File | SHA-256 |
|---|---|
| paper/numerical_calibration_protocol_v1.md | `6c9945bf4c6d5c3d8c2b71e6f80a78d6dd7b671f02a12c70aa2bf4c1a734e9a1` |
| paper/numerical_calibration_protocol_v2.md | `abcb7cb1b7fc7b2f2a114b6c5a4328acc18a78aebca00544a20c383bd719548d` |

All 32 Stage 7N files were copied byte-for-byte from `/tmp/struct2circuit-stage7n-p2go5bjm` into the separate durable workspace archive:

`/workspace/research-archives/stage7n-exploratory-2042e4e-v1/`

“Durable” here means outside temporary storage in the workspace; this is not a claim of an external backup. The original paths embedded in the copied provenance and README remain unchanged intentionally.

| Archived deliverable | SHA-256 |
|---|---|
| report.md | `7d7d00b2ecc785a15fa4c79c0eb8dc443b0242be63aea4ea80dbfbd44379c2aa` |
| landscapes.csv | `98bf6028b913f67185fec16041af8c696da0a1f69db5a6cc6a74b0cb80d9bb9f` |
| stage7n-analysis.zip | `68d33e872b86211ab0777cc60ea9bd241e41983a730b64582e096ba47b8c2d8b` |
| derived-hashes.json | `fd4895288f6cd1f67b2e77ed713312e79315b2c3103a61eae5afba431bf98e10` |

Verification checked all 30 entries in the derived hash manifest, ZIP integrity and all 31 ZIP member contents against their source files, and every copied file against its original. The ZIP is 865,383 bytes. All 20 original Stage 7L files bound by Stage 7N provenance also match their recorded sizes and hashes. None was rewritten. The historical P9/P17/P33 failures remain prior development evidence, not measurements in this fresh panel.

Stage 7N motivates the hypothesis: high-RMS had lower complete uniform references than its shuffled counterpart in 7/7 eligible cases, while one case was unresolved; matching-spectrum examples also differed. Those observations involved only eight QUBOs and did not isolate topology. They determine the hypothesis, not its prospective test result. No original development instance enters the new panel.

## 2. One primary, falsifiable hypothesis

**Hypothesis:** for fresh QUBOs from the equally weighted four-family population defined below, selecting a higher conditional exchange-RMS alignment within the relabeling orbit of C6 produces a materially lower, numerically resolved empirical B00 reference more often than not. The target prevalence exceeds one half, under the fixed measurement procedure below.

This operational hypothesis is about observed empirical reference quality. It is not a claim about certified global B00 minima, monotonicity at every intermediate score, or every topology or problem size.

For symmetric Q and cost `C(x)=x^T Q x+c^T x`, with |x|=k, define for edge i–j:

`rho_ij = sqrt(E[(C(x-e_i+e_j)-C(x))^2 | x_i=1, x_j=0, |x|=k])`.

The conditional expectation is uniform over the remaining feasible subsets. An implementation must use the reviewed analytic conditional moments (or exactly equivalent arithmetic), not sampled QAOA probabilities. Explicitly, N=n−2, m=k−1, b_l=2(Q_jl−Q_il), d0=Q_jj−Q_ii+c_j−c_i:

`mu=d0+(m/N) sum_l b_l`

`variance=m(N-m)/(N(N-1)) sum_l (b_l-mean(b))^2`

`rho_ij=sqrt(mu^2+variance)`.

Use exact rational moments of the stored binary64 coefficients, followed by binary64 square root, as in the archived Stage 7N descriptor calculation. With feasible cost span S=C_max−C_min, define the alignment metric

`A(G;Q,c) = (1/(6*S)) sum_{(i,j) in E(G)} rho_ij`.

Use `math.fsum` for the sum. Its division by S makes the descriptor dimensionless; within a QUBO it does not change ordering. Preserve unrounded values; select using A rounded to 12 decimal places with Python's fixed rounding rule. This is a prospective selection convention, not a tolerance retroactively applied to v1/v2. Conditional RMS describes a uniform feasible prior, including both conditional mean and variance; it is not an optimized-state transition rate, gradient, or dynamical explanation by itself.

For each arm, minimize the normalized objective directly:

`g_G(u,beta)=<psi_G|D|psi_G>`, `D=(C-C_min)/S`,

`psi_G=exp(-i beta H_G) exp(-i u D) |uniform_k>`,

`u in [0,2*pi], beta in [0,pi]`, with physical gamma `u/S`.

Let r_H and r_L be the complete empirical references for selected high- and low-alignment graphs. The sole primary paired contrast is

`Delta = r_H-r_L`; **negative favors higher alignment**.

The QUBO draw is the unit of replication. Graphs, grid points, DE seeds and solver starts are repeated measurements within that unit.

## 3. Controlled graph intervention and equivalences

Fix the unweighted base cycle with edges 01,12,23,34,45,05. Enumerate its vertex permutations, form G'=P G P^T, and canonicalize each labeled graph as its sorted list of undirected edges. C6 has 12 automorphisms, so its 720 permutations yield exactly 60 distinct labeled edge sets, not 720 independent treatments.

Before QAOA work, also quotient these edge sets by exact symmetries of the fixed cost. At n=6 this can be checked using permutations of the 20-entry exact feasible cost vector computed from the already generated QUBO: retain permutations that preserve that vector exactly, and keep the lexicographically first edge list in each resulting orbit. This removes known cost-preserving equivalent assignments without using any QAOA outcome. Do not merge merely close cost vectors, spectra or objective values. Record every alias. Additional equivalences not captured by these structural tests are a limitation; only two selected arms are measured and they are never counted as independent QUBOs.

Select L as the lexicographically first representative attaining the minimum rounded A, and H as the lexicographically first attaining its maximum. Record the full 60-graph score/alias table, selected permutations and score separation before search. If the two score extrema coincide after rounding, or fewer than two inequivalent representatives exist, record **NO_ALIGNMENT_CONTRAST**; do not draw another QUBO or search alternative topologies. It remains in the 32-unit denominator as untestable/unknown. Constant or numerically unresolved feasible costs are similarly retained with their explicit status and null normalized metrics; no small-unit heuristic or replacement is permitted.

All arms keep Q, c, n, k, coefficient scale, native unit XY edge weights, topology, depth, initial state, domain and numerical criteria fixed. Q is not permuted during the intervention; only its variables' assignment to cycle vertices changes.

**Spectrum invariance is algebraic.** Vertex permutation P induces a permutation unitary U_P on the fixed-cardinality basis. The native XY Hamiltonians satisfy `H_G'=U_P H_G U_P^dagger`. Hence their full feasible-space spectra, multiplicities and widths coincide exactly. Also `U_P|uniform_k>=|uniform_k>`. The cost diagonal generally does not commute with U_P, which is precisely what makes this a relative-alignment intervention. No new spectrum or objective was evaluated for this design. At later implementation, test the transported Hamiltonian matrix identity; eigenvalue agreement alone is a weaker check.

This intervention identifies the effect of the **high-versus-low score selection rule within C6** on the measured outcome. It does not isolate RMS from every correlated property of cost–graph alignment: eigenvectors relative to cost and other edge descriptors also vary. A positive finding cannot establish that RMS is the unique causal mediator.

## 4. Fresh panel, seeds and controls

Plan 32 QUBOs, indexed `(f,r)` with f=0,1,2,3 and r=0,...,7. All have n=6, k=3, feasible dimension 20. Use the generator implementations from the inspected commit, with these fixed settings:

| f | Generator | Parameters beyond n,k,seed |
|---|---|---|
| 0 | block_correlated_qubo | n_blocks=2; block_strength=.72; cross_strength=.16; linear_scale=.55 |
| 1 | weighted_densest_k_subgraph_qubo | density=.55; weight_range=(.5,1.5); weight_distribution=uniform; planted_community_strength=1.8 |
| 2 | weighted_max_k_vertex_cover_qubo | density=.55; weight_range=(.5,1.5); weight_distribution=uniform; community_strength=1.5; hub_strength=1.5 |
| 3 | weak_structure_null_qubo | density=.55; coefficient_scale=1 |

Freeze the exact seed derivation now. For role in `qubo`, `de0`, `de1`, `controls`, `order`, form the UTF-8 string with no newline:

`struct2circuit/stage7o/alignment-c6/v1|<role>|f=<f>|r=<r>`

Use the unsigned big-endian integer of its complete SHA-256 digest as seed. QUBO generation uses the existing `default_rng(seed)` call; search/control randomness explicitly uses `Generator(PCG64(seed))`, with software pinned. DE0 and DE1 have different role seeds. For each role, instantiate separate generators from the same seed for H and L: this pairs the initial random designs across arms, without sharing outcomes or optimizer populations. The DE algorithm can then follow different trajectories. Use the low bit of the `order` digest to choose the first arm within each QUBO. No seed truncation or outcome-dependent reseeding.

Before separately authorized generation, materialize and collision-check this seed manifest against itself and permitted public seed metadata only. Do not open frozen benchmark records or validation coefficients to do so. Namespace separation and n=6 exclude reusing the existing calibration validation cases (n=9); no calibration ID, coefficient file, or holdout data is an input. Compare future generated coefficient identities with accessible prior development identities and with the new panel itself. An exact duplicate is a provenance exception requiring a stop and disclosure before outcome searches, not permission to select a replacement. A different seed alone is not proof of a different QUBO.

Eight weak-structure-null draws give a prespecified family comparison. This family is **not an exact zero-effect null**: selecting graphs using the realized cost may exploit its random structure. Report its contrasts on the same footing and compare descriptively with structured families. No requirement that its effect vanish is imposed.

A true implementation negative control transports the QUBO, graph and uniform state together by the fixed cyclic permutation `(i -> i+1 mod 6)` at the fixed 5x5 B00 coordinate grid. Normalized objectives and transported probabilities must agree within 1e-10. This covariance identity checks the intervention machinery; it is not a third research arm or an extra independent QUBO.

## 5. Fixed empirical outcome measurement

Do not call P9, P17 or P33, and do not inject their selected points. Use a separate mechanism runner with the following same finite two-route recipe for every valid arm. The old calibration entry points and protocols remain unchanged.

1. Use endpoint-inclusive coarse axes `u=pi*j/16, j=0..32` and `beta=pi*j/64, j=0..64`. Use exactly one nested fine grid `u=pi*j/32, j=0..64`, `beta=pi*j/128, j=0..128`. These are 2,145 and 8,385 points; exact cached coarse nodes are not charged again. This fixes N_beta=128 for the new experiment, avoiding arm-dependent decisions around the historical width≈8 rounding boundary. Independently, each feasible C6 state has at most six unit hopping neighbors, so ||H||<=6 and spectral width<=12; this mesh also satisfies the historical `N_beta>=8W` planning criterion. That criterion is not a convergence proof.
2. Route A: from the coarse grid take its best point, then the lowest points separated from all selected starts by at least two cells in max-axis grid-index distance, ties by value/u/beta, for four starts. At the fine level repeat this selection for eight starts. Use analytic-gradient bounded L-BFGS-B, maxiter=120, ftol=1e-13, gtol=1e-9, at most 250 objective requests per attempt. Record exact required starts and attempts, including interruption status. No production start is included.
3. Route B: DE0 over B00, independent of A's candidates: best1bin, popsize=8, maxiter=63, tol=1e-8, atol=1e-10, mutation=(.5,1), recombination=.7, Latin hypercube initialization, immediate updating, workers=1, polish=False, cap 1,024 objective requests. Polish its best feasible observation with a separate 250-call L-BFGS-B attempt using the same settings. At the fine stage always perform DE1 with maxiter=126 and cap 2,032, then another 250-call polish. Use the frozen paired role seeds above.
4. Scan the **four B00 faces only**, at both mesh levels. Preserve endpoints/corners; at each strict sampled interior minimum run bounded scalar minimization in the neighboring bracket, xatol=1e-10, at most 60 calls per bracket and 150 new scalar-refinement evaluations per face per level. Retain all face candidates in A. Beta=0 is flat by the circuit identity; record sampled checks. Other faces require scans unless both variation<=1e-10 and an analytic eigenstate-residual flatness certificate hold, as in the reviewed helper. If a required face cap prevents its prescribed scan, mark it unresolved. No B11 traces or wider-domain searches are part of this experiment; do not call the old helper that automatically expands primary traces to B11.
5. Always perform one two-start bounded Powell bundle after the fine searches, at the two best distinct stored locations, ties by value/u/beta. Use maxiter=180, ftol=1e-12, xtol=1e-10 and 250 requests per attempt. Preserve each start's route lineage; a shared winner never creates an independent route. No second bundle, fine level, DE replicate, or cross-arm candidate transfer is permitted.
6. Freshly reevaluate selected candidates. Keep independent route minima a and b, pooled best r, and the pre-fine pooled value r_coarse. Set `R=max(|a-b|,|a-r|,|b-r|)` and `F=r_coarse-r`. A complete reference requires finite values, all prescribed attempts and face work, all controls, `R<=1e-5`, and `0<=F<=1e-5`, including endpoints. Preserve coarse candidates in the pool, so F cannot legitimately become negative. Numerical disagreement, not a solver success flag, determines unresolved status. A solver flag alone neither certifies convergence nor vetoes an otherwise fully recorded, passing finite search. Incomplete mandatory work always remains unresolved.

Every arm receives the full schedule irrespective of whether its coarse result supports the hypothesis. If fine work finds a substantially better value and F fails, preserve that value as a best observation but leave the reference unresolved; do not buy another grid to stabilize it. Reuse of a cache requires the same objective identity and exact coordinates; no values cross distinct arms or QUBOs.

Mandatory controls are bound to coefficient, edge, state, domain and source identities. Before local searches, use eight role-seeded interior B00 points with finite-difference margins `h_u=1e-5`, `h_beta=1e-5/max(1,W)`. Require analytic versus central-difference gradient agreement within 1e-6. At those points, the four B00 corners, and each final route winner and selected reference, require native versus direct-matrix-exponential normalized objective and probability agreement, and state/probability normalization, within 1e-10. Fresh native selected-value agreement uses 1e-10. Run the transported negative control above for each arm. No controls need wider domains. Reuse the reviewed formulas, tolerances and identity checking, not the official calibration workflow.

Any oracle mismatch, nonfinite arithmetic, broken permutation identity, cross-objective cache reuse or evidence-accounting defect is an experiment integrity failure. Retain all partial records and withhold the primary scientific decision pending investigation; do not silently delete the affected arm. Unfinished controls caused solely by a resource stop mean unknown, not a passed reference.

## 6. Scheduling, finite resources and expected runtime

Use one process for numerical work, with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 set before imports. Pin the inspected runtime (Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1) and record the implementation hash. Round-robin QUBOs in order r=0..7, then f=0..3; arm order is seed-fixed. Complete a panel-wide coarse/control wave, then a panel-wide fine/bundle/final-control wave in the same order. No optional diagnostics or family-specific extra effort.

| Resource | Proposed hard ceiling for this new experiment |
|---|---:|
| Actual numerical evaluations per arm, controls included | 20,000 |
| Total actual numerical evaluations, all 64 arms | 1,280,000 |
| Cumulative process-tree CPU, including preparation and sealing/finalization | 3,600 seconds |
| Wall time | 7,200 seconds |
| Process-tree resident memory | 4 GiB |
| Prelaunch free disk requirement | 5 GiB; stop new numerical work below 1 GiB |

These are new finite allocations, not increases to or reuse of the old calibration budget. No transfers above an arm's 20,000 cap. Budget every actual numerical backend evaluation, including controls, direct oracle calls, fresh checks and gradient work that requires a new evolution. Count requests, cache hits, actual computations and derivative/oracle categories separately. Exact cache hits do not count as new evolutions; vectorization must not hide multiple evaluations. Preserve monotonically cumulative counters and settled CPU receipts using the repaired accounting contract. Stop admitting new numerical work at 3,300 CPU-seconds to leave 300 seconds within the 3,600 hard limit for evidence publication; if publication cannot finish within the hard cap, report incomplete evidence rather than claiming settled completion. All failed work is charged.

Historical basis: Stage 7L used 1,404,045 actual evaluations and 568.14324 settled CPU-seconds for 192 small landscapes: approximately 7,313 evaluations/landscape and 405 CPU-seconds/million evaluations, with a 2.02-GiB peak process RSS. It included P33 and mixed reference effort, so it is not a direct per-arm prediction. Here every valid arm gets fine work. The 64 fine grids alone contain 536,640 points. A rough upper planning sum per arm is 8,385 unique grid points + 3,000 A local requests + 3,056 DE requests + 500 DE-polish + 500 Powell + 1,200 face refinements = 16,641, before controls and fresh checks. The 20,000 ceiling leaves finite headroom; it is not a proof of completion under all cache, control or solver conditions.

Plan approximately 0.8–1.2 million actual evaluations, about 5–9 CPU minutes by historical linear scaling; the full evaluation cap scales to 518 CPU-seconds. Allow roughly 10–30 minutes wall time for implementation overhead, evidence handling and host variation, with the hard limits above. These are estimates, not measured performance for the new runner. Stream one QUBO pair's numerical state at a time and retain immutable candidate files, rather than holding every trace in RAM.

Reference completion is scientifically uncertain: Stage 7L had 19/192 unresolved references overall and 6/64 under uniform initialization. Those rates involve other graph selections and only eight QUBOs, so do not extrapolate them as a success guarantee. All-reference fine checks may expose additional instability. On any ceiling, preserve completed work, mark all remaining scheduled rows with their exact unexecuted/unresolved reason, and do not extend budgets or rerun unfavorable outcomes.

## 7. Primary decision, missingness and secondary reporting

Predeclare a material empirical improvement of 1e-4 normalized objective, plus a 2e-5 comparison buffer. The buffer is a decision margin relative to the two 1e-5 agreement criteria, **not a certified bound on reference error**.

For two complete references, call a material high-alignment win only if `Delta < -1.2e-4`; call a material loss if `Delta > +1.2e-4`; otherwise record neutral/buffer. Equality is neutral. Also report both route-specific deltas as diagnostics, without creating extra tests. If either reference is unresolved, alignment contrast is absent, or a cost is untestable, the paired outcome is unknown. Never impute its Delta or silently exclude it.

Let W,L,T,U count wins, losses, neutral pairs and unknowns; they always sum to 32. The primary target p is the equal-family average probability of a material win for a fresh QUBO under this fixed empirical measurement contract. The independent QUBO draws, not their arm searches, justify replication. Deterministic seeds are a reproducible pseudorandom sample specification, not proof of universal generalization.

Use one conservative simultaneous 95% Hoeffding interval with arbitrary missingness:

`epsilon = sqrt(log(40)/(2*32)) = 0.24008069782998018`

`lower = max(0, W/32-epsilon)`

`upper = min(1, (W+U)/32+epsilon)`.

This bounds the mean win probability under independent draws even with unequal family probabilities. Unknown cases are allowed every possible win assignment; it does not assume missing at random. A global resource stop can censor measurements but cannot add wins, so the envelope remains conservative. This interval concerns the specified empirical outcome, not accuracy against a global optimum.

- **Directional support:** lower>0.5, equivalently W>=24. Report support only for this generator mixture, C6, n=6, k=3 and measurement contract.
- **Contradiction of the stated majority hypothesis:** upper<0.5, equivalently W+U<=8. This can reflect frequent losses or lack of material benefit; distinguish those in the report. A reliably positive Delta also contradicts any universal superiority assertion, which is not the primary hypothesis.
- **Inconclusive:** all other cases, or an integrity failure. Failure to reach support is not proof that alignment has no effect.

This small first experiment has deliberately conservative power: moderate benefits can remain inconclusive. No sample-size extension, alternative primary metric or topology switch follows from the observed result. Numerical pass/fail and scientific support/contradiction/inconclusive are separate labels.

Publish all 32 paired rows and all planned 64 arm rows with identities, scores, candidate coordinates, raw best observations, R/F, face/attempt/control completion and reasons for missingness. Unknown normalized values serialize as null plus status, never NaN or zero. Raw unresolved minima remain visible but are not ranked as complete references.

Only these descriptive secondary analyses are planned:

1. Within each of the four families, show W/L/T/U out of eight, all paired Deltas, eligible-pair median and range, explicitly conditional on completeness. Do not treat family cells as powered independent tests or pool only favorable families.
2. Plot Delta versus alignment-score separation, marking unknown outcomes without invented coordinates; provide unadjusted descriptive associations on complete pairs with their denominator. This does not estimate a causal effect per unit RMS.
3. Tabulate missingness by family and arm, including route disagreement, fine improvement, face caps and controls/resource interruptions. Compare high/low completion rates to expose differential numerical difficulty. Present the weak-structure family as the predefined specificity comparison, not a known zero-effect control.

No scan over secondary p-values, all graph pairs, alternative effect thresholds, initializers or optimizer settings. A paired effect cannot establish scalable improvement, quantum advantage, an optimum ordering, validation success, or superiority of all structure-conditioned mixers. Shared reference blind spots remain possible even with independent routes.

## 8. Minimal later implementation and decision gate

After design approval, implement a small, separate mechanism configuration/runner and scalar analysis command. Reuse the reviewed QUBO generators, native XY physics, normalized objective, descriptor arithmetic, solver primitives and cumulative resource guards. Add only the C6 orbit/score selection, new seed/panel namespace, B00-only face schedule, fixed full-reference schedule and the paired report. Do not repurpose the official calibration driver, revise its protocols or add a validation-lock pathway.

Required synthetic/structural regression checks before execution approval: 60 C6 labeled graphs; automorphism/cost-symmetry deduplication and deterministic ties; transported Hamiltonian/uniform-state identity; paired seeds and no cross-arm cache identities; required route starts and attempt counts; controls bound to each objective; budget ceilings and interrupted evidence retention; complete/unknown reporting and the fixed statistical decision using synthetic scalar rows. An implementation review must confirm the control-inclusive resource ledger and B00-only calls. No official panel may be generated to test the implementation before separate execution authorization.

Freeze this design's hash, implementation revision, seed manifest, generator settings, output destinations and new-experiment scope in a separately approved execution record. No authorization file or numerical output directory is created now. Keep the historical calibration validation holdout completely unused; a future replication would require its own fresh research panel and authorization, not access to that holdout. Retain P9/P17/P33 FAILED records without modification or replacement.

**Documented limitations:** single small topology/size; finite, conservative panel power; incomplete or jointly misleading empirical references; RMS correlated with other alignment features; simulated classical measurement rather than hardware performance; runtime estimated from a different schedule. These limit interpretation but do not prevent review of this controlled first experiment.

**Execution gates, not unresolved scientific alternatives:** review and freeze this design; implement and test the isolated runner; verify public seed nonreuse, resources and fresh paths; obtain separate execution authorization. No numerical execution is authorized by this document.

STAGE 7O — CONTROLLED MECHANISM DESIGN READY FOR REVIEW
