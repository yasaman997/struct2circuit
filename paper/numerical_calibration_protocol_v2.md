# Prospective numerical calibration protocol v2

**Status: PROPOSED — AWAITING SCIENTIFIC APPROVAL; NOT AUTHORIZED FOR EXECUTION**

Rules version: `numerical_calibration_protocol_v2`.
Scientific scope: `P33_B00_native_XY_finite_panel_v2`.
This document specifies a prospective study. It is neither implementation approval,
execution authorization, a validation lock, nor evidence that P33 is accurate.
The benchmark remains `DRAFT_UNFROZEN`; no RMS performance direction is selected.

## 1. Historical identity, evidence and interpretation

The historical execution/source baseline is
`e8dea4d09f3a40542b361a9e191d3dacf5e295f7` in `yasaman997/struct2circuit`.
The immutable [v1 protocol](numerical_calibration_protocol_v1.md) has SHA-256
`6c9945bf4c6d5c3d8c2b71e6f80a78d6dd7b671f02a12c70aa2bf4c1a734e9a1`.
Stage 7B output is `/workspace/calibration-runs/stage7b-development-e8dea4d-v1`;
provenance is `/workspace/calibration-run-provenance/stage7b-development-e8dea4d-v1`.
Its preserved archive is
`/workspace/calibration-archives/stage7b-development-e8dea4d-v1.tar.gz`, SHA-256
`d7043055769685c8ed1858cb2cd4da6d6ba81372c703a1ca8126109007a578c5`.
These paths describe historical evidence, not v2 destinations.

Stage 7B remains permanently **FAILED**. Independent verification reproduced:

| Landscape (instance, graph, initialization) | P17 E | P9 E |
|---|---:|---:|
| (0,7,0), random connected, uniform | 0.00418422868004828 | 0.004184228680048294 |
| (0,4,1), high RMS, mixer_low | 0.0014596721758214776 | 0.001459672175821422 |

The associated ten P17 K failures are seven comparisons (0,7), (1,7), (2,7),
(3,7), (4,7), (5,7), (6,7) for instance 0/uniform and three comparisons (1,4),
(3,4), (4,5) for instance 0/mixer_low. They share the two underlying production
deficits and are not ten independent discoveries. The references support genuine
observed B00 counterexamples; later unfinished refinement does not erase them.
P33 was not evaluated, no development validation lock was published, and no RMS
superiority was established.

The recorded B00-to-B10/B11 gain reaches 0.1928321056873999, B00-to-B01 reaches
0.14901872459221216, B00-to-B22 reaches 0.17798280497574293, and B11-to-Bsign reaches
0.12876875126448473 on complete participating references. Total B00-to-Bsign gain
reaches 0.2249598717015613; this last quantity is not a pure sign effect. Preserve
the case identities, unresolved records and original interpretation in the v1
artifacts. These observed domain limitations are not repaired by changing v2
acceptance semantics.

Stage 7D established workload infeasibility; Stage 7F specified the amendment
implemented in this document. A qualified v2 result would not make v1 successful.
All v1 files, results, authorization records and lock-verification behavior remain
unchanged. Historical expenditure is not replaced by a new v2 ledger.

P33 is the only qualifying candidate. P9/P17 are historical comparators, with no
repeat optimization required by this minimal study and no fallback eligibility.
No P65, P129, new optimizer, modified initialization or policy-selection ladder
is permitted. A further production recipe would require a new scientific proposal.

## 2. Primary scientific target and numerical definitions

The sole primary target is numerical qualification of unchanged depth-one P33 on
B00 for the complete panel in section 3. Qualification means empirical agreement
with independently checked B00 references and reliable paired graph comparisons;
it is not certified global optimality or unrestricted-angle QAOA accuracy.

The following reproduce v1 sections 1–3. Minimize the cardinality QUBO
`C(x) = x.T @ Q @ x + c.T @ x`, with binary x and `sum(x)=k`.
The feasible dimension is `d=binomial(n,k)`. Let `C_comp` be the simulator's
compensated float64 feasible costs, `C_min=min(C_comp)`, `C_max=max(C_comp)`,
`S=C_max-C_min`, and `D=diag((C_comp-C_min)/S)` for S>0. With native feasible-space
XY Hamiltonian H and fixed initial state s:

`u=gamma*S`,
`psi(u,beta)=exp(-i*beta*H) @ exp(-i*u*D) @ s`,
`g(u,beta)=sum_x |psi_x|^2 * D_x`.

Use the existing `evaluate_dimensionless` pipeline and its direct `normalized_gap`.
Record the actual selected float64 u and beta, physical gamma, raw expectation,
cost status and feasible span. Do not reconstruct g from raw `expectation`, replay
raw-cost phases, or treat `gamma*S` as exact coordinate recovery after rounding.
The existing `expectation` and `finite_objective` meanings are unchanged. Search
and qualification use g; physical gamma remains returned through the existing API.

`cost_status="constant_or_unresolved"` means no normalized optimization: keep
normalized metrics null, preserve the case and mark required evidence unresolved.
Do not redraw, replace, or impute optimum probabilities. Nonfinite/invalid numerical
arithmetic is a separate essential failure. Do not alter cost-resolution rules.

For landscape ell=(instance,graph,initialization), let p be freshly checked P33 g
and r(B) the complete empirical reference in box B. Independent refined route
values are a and b; eligible pooled candidates determine r.

| Quantity | Definition | Rule |
|---|---|---|
| R | `max(abs(a-b),abs(a-r),abs(b-r))` | Reference stopping tolerance 1e-5 |
| E | `p-r(B00)` | Performance threshold 1e-4 |
| F | `r_coarse(B)-r_fine(B)` | Reference stopping tolerance 1e-5 |
| Contrast | `g_j-g_i`, for i<j | Positive favors i |
| K | `abs((p_j-p_i)-(r_j(B00)-r_i(B00)))` | Performance threshold 2e-4 |
| A(B,B') | `r(B)-r(B')` for nested boxes | Diagnostic materiality threshold 1e-3 |
| Clearly separated ordering magnitude | `abs(contrast)` | Threshold 2e-3 |
| eta | Performance ambiguity distance | 1e-5 |

For E/K/A use finite value q and nominal threshold T: q<T-eta is clear pass
(nonmaterial for gains), q>T+eta is clear failure (material for gains), and the
closed interval [T-eta,T+eta] is ambiguous. Compute float64 endpoints once; do not
use rounded subtraction to leave gaps. Thus E passes below 0.00009 and fails above
0.00011; K passes below 0.00019 and fails above 0.00021. Equality at either endpoint
is ambiguous. Undefined/nonfinite evidence is unresolved, never passing.

R and F use their separate nonnegative stopping rule: finite values <=1e-5 pass,
including equality; values >1e-5 or unavailable values leave references unresolved.
The performance guard band does not apply to R/F. F is applicable only after a
prescribed refinement; it is not zero-imputed for routine-only references.
Unexpected negative R/F, E or nested gains are diagnosed according to their
definitions and monotonicity, not silently clipped. Roundoff-sized discrepancies
remain recorded; they do not authorize changing thresholds.

Ordering is clear only above 0.00201 in magnitude; below 0.00199 is a near-tie;
the intervening closed band is ambiguous. A clear reversal requires opposite
contrast signs and both magnitudes >0.00201. Near-ties carry no ranking claim and
do not themselves fail E/K. No average overrides an individual required failure.

## 3. Complete instance, graph and initialization panel

These definitions reproduce v1 sections 4–5. IDs are numerical-calibration IDs,
not benchmark IDs. Public seed metadata are permitted; validation coefficients,
generated instances and outcomes are not accessible before the one-time gate.

| Family f | Generator | Exact fixed parameters |
|---|---|---|
| 0 | `block_correlated_qubo` | `n_blocks=2, block_strength=.72, cross_strength=.16, linear_scale=.55` |
| 1 | `weighted_densest_k_subgraph_qubo` | `density=.55, weight_range=(.5,1.5), weight_distribution="uniform", planted_community_strength=1.8` |
| 2 | `weighted_max_k_vertex_cover_qubo` | `density=.55, weight_range=(.5,1.5), weight_distribution="uniform", community_strength=1.5, hub_strength=1.5` |
| 3 | `weak_structure_null_qubo` | `density=.55, coefficient_scale=1` |

| f | Original development ID / seed | Prospectively active reserve ID / seed | Validation ID / seed |
|---|---|---|---|
| 0 | 0 / 420000000 | 1 / 420000001 | 8 / 420100000 |
| 1 | 2 / 420001000 | 3 / 420001001 | 9 / 420101000 |
| 2 | 4 / 420002000 | 5 / 420002001 | 10 / 420102000 |
| 3 | 6 / 420003000 | 7 / 420003001 | 11 / 420103000 |

All eight development instances use n=6,k=3,d=20; validation uses n=9,k=4,d=126.
All four reserves activate before any new outcome. Keep their original reserve
provenance, but they are mandatory v2 development cases. Neither failure nor
constant costs permits replacement, seed changes, removal or family selection.

For each instance use exactly m=n physical edges and these eight labels:

| Graph ID | Label | Unchanged construction |
|---|---|---|
| 0 | ring | `ring_mixer(n)` |
| 1 | strong_abs_Q | `structure_conditioned_mixer(Q,m)` |
| 2 | weak_abs_Q | `score_conditioned_mixer(abs(Q),m,prefer="low")` |
| 3 | low_RMS | `score_conditioned_mixer(exchange_profile_scores(Q,c,k),m,prefer="low")` |
| 4 | high_RMS | Same RMS matrix, `prefer="high"` |
| 5 | shuffled_low_RMS | Shuffled RMS matrix, `prefer="low"` |
| 6 | shuffled_high_RMS | Same shuffled matrix, `prefer="high"` |
| 7 | random_connected | `random_connected_mixer(n,m,seed=520000000+instance_id)` |

Shuffle upper-triangle RMS entries in lexicographic (i,j) order using
`np.random.default_rng(620000000+instance_id).permutation`; mirror and zero the
diagonal. Record the actual permutation and edges; never redraw graphs.
Construct all graphs for a phase before QAOA optimization. Native feasible-space
H has unit XY exchange amplitudes, with no degree, edge-count or spectral rescaling.

Initialization IDs are 0=uniform, 1=mixer_low, 2=mixer_high. Uniform is normalized
on the feasible basis. Spectral states retain the simulator's extremal eigenspace
test (`atol=1e-10, rtol=0`) and projection threshold 1e-12: project uniform first;
otherwise use the first computational-basis projection above that norm threshold
in lexicographic order, with the existing deterministic phase rule. No tolerance
or label-dependent fallback is repaired here. Do not choose a best initialization.

Canonical graph aliases use identical normalized edge contracts for the same
QUBO; the lowest graph ID is canonical. Reuse only identical objective/model/state
contracts. Record requested and canonical identities/seeds. Keep all labeled rows,
including aliased pairs; they are not independent replicates. Different QUBOs
may share H decompositions, not cost-dependent objective values.

| Phase | Explicit IDs | Landscapes | Unordered pairs within instance/initialization |
|---|---|---:|---:|
| Development | 0–7 | 8*8*3=192 | 8*3*28=672 |
| Validation | 8–11 | 4*8*3=96 | 4*3*28=336 |
| Total | 0–11 | 288 | 1,008 |

Every pair i<j among graph IDs 0–7 is required, without selecting only separated
rankings. Coverage is defined before generation, not from observed rows.

## 4. Production recipe, reference seeds and domains

P33 calls unchanged `optimize_p1(...,grid_size=33,gamma_scale="feasible_span")`.
B00 bounds are u in [0,2*pi], physical beta in [0,pi]. Both 33-point production
axes exclude their upper endpoint. L-BFGS-B and Powell start at the same best
grid point; eligible finite candidates compete with that point. Preserve the
existing production settings (L-BFGS-B maxiter=120, ftol=1e-13, gtol=1e-9;
Powell maxiter=180, ftol=1e-12, xtol=1e-10), selected coordinates and success flags.
Do not substitute analytic reference gradients into the production optimizer.
Every production evaluation, including final reporting, is fresh and charged.

| Box ID | Domain | u bounds | Physical beta bounds | v2 role |
|---|---|---|---|---|
| 0 | B00 | [0,2*pi] | [0,pi] | Sole production and essential reference domain |
| 1 | B10 | [0,4*pi] | [0,pi] | Optional gamma-extent diagnostic |
| 2 | B01 | [0,2*pi] | [0,2*pi] | Optional beta-extent diagnostic |
| 3 | B11 | [0,4*pi] | [0,2*pi] | Optional joint-extent diagnostic; auxiliary control/trace coordinates |
| 4 | B22 | [0,8*pi] | [0,4*pi] | Optional further-envelope diagnostic |
| 5 | Bsign | [0,4*pi] | [-2*pi,2*pi] | Optional signed-sector diagnostic |

No automatic period or sign reduction is allowed. Normalized costs do not imply
a 2*pi u period, and mixer spectra do not imply a pi beta period. Simultaneous
sign reversal for real states does not justify independent sign reversal.
Common-spectral-time Bv optimization remains OMITTED and outside this minimal v2;
its omission excludes common-spectral-time conclusions, not spectral relevance.

For reference box b and replicate r in {0,1}, preserve v1 search seeds:

- b=0–3: `720000000+10000*instance_id+100*canonical_graph_id+10*initialization_id+2*b+r`.
- b=4–5: `730000000+10000*instance_id+100*canonical_graph_id+10*initialization_id+2*(b-4)+r`.

Replicate 0 is routine; replicate 1 is the single allowed escalation replicate.
Record requested-label and actual canonical seeds. P33 has no added random seed.
The eight interior-control points use seed
`820000000+10000*instance_id+100*canonical_graph_id+10*initialization_id`.
Freeze Python 3.12.14, NumPy 2.5.3 and SciPy 1.18.1, with interpreter/BLAS identities
recorded. A dependency change requires review before holdout access.

Public seed checking uses only `configs/benchmark_v1_draft.json`, the problem_seed
column of `results/pilot_results.csv`, and that column of
`results/independent_dks_benchmark_v1_instances.csv`. Do not access
`manifests/benchmark_v1_draft.json` or other protected benchmark records. The v2
metadata namespace is distinct; intentional reuse of the prescribed v1 calibration
seeds is disclosed, not misrepresented as an independent development panel.
Unverifiable external identifier collisions require an external check before
generation, not new outcome-selected seeds.

## 5. Exact grid and boundary rules

Reproduce v1 sections 7–8 and `config.grid_axes`; do not change spacing to make a
grid fit. For each instance, compute `W_max=max_G(lambda_max(H_G)-lambda_min(H_G))`
across its graphs. N_beta is the smallest power of two at least max(64,8*W_max),
using `1 << (ceil(max(64,8*W_max))-1).bit_length()`. All graphs and initializations
of the instance share this mesh. At W_max<=8 use 64; immediately above 8 use 128;
above 16 use 256. No tolerance is added around this computational threshold.

At level L=0 or 1, use u denominator `16*2**L` and beta denominator
`(N_beta/2)*2**L`: form `pi*(integer_index/denominator)`, clip by exact box bounds,
and include both endpoints explicitly. Old dyadic nodes are retained exactly.
The B00 routine grid has `33*(N_beta/2+1)` points; the fine grid has
`65*(N_beta+1)`. Compute only B00's two-dimensional grid for essential work.
Do not require full B10/B01/B11 grids merely to qualify B00.

Preserve the existing shared primary boundary helper: six full B11 traces
u=0,2*pi,4*pi and beta=0,pi,2*pi, with level-matched nodes. Auxiliary coordinates
outside B00 are charged but are not a B11 reference or B11 qualification.
Only eligible B00 points enter its reference pool, and only its required faces
determine B00 face completeness. Split scalar brackets at internal cuts u=2*pi
or beta=pi; scan strict sampled nonflat minima, preserve endpoints and corners.
Use bounded scalar minimization with xatol=1e-10, at most 60 objective calls per
bracket and 150 newly computed values per full trace per level. A capped nonflat
required face remains unresolved. Shared traces are cached once per level.

Beta=0 is probability-flat. For u=0, skip redundant refinement of a mixer
eigenstate only after both sampled variation<=1e-10 and
`2*beta_max*norm(H@s-lambda*s)<=1e-10`; otherwise scan. Record flat flags, all
face distances divided by their box widths, boundary hits at relative distance
<=1e-8, and the distinct near-boundary trigger at <=.01 from a nonflat face.

Additional diagnostic boxes use four box-boundary traces with the same scalar
caps. All local/global/face candidates must be within their declared bounds.
Identical coordinates do not permit sharing values between different models or
initial states. No rounded coordinate buckets, batching undercounts or silently
discarded faces are permitted.

## 6. Independent empirical references and refinement

Route A starts with the eligible endpoint-inclusive grid and face candidates.
Select at most four starts: best grid point, a distinct best eligible production
point, then lowest grid values separated by at least two cells in maximum-axis
grid-index distance. Break ties by value/u/beta; the production start is exempt
from separation. Record fewer available starts rather than substituting seeds.
Use analytic-gradient L-BFGS-B with maxiter=120, ftol=1e-13, gtol=1e-9 and a
250-objective-call attempt cap.

Route B uses independent DE replicate 0, not Route A initialization:
strategy=best1bin, popsize=8, maxiter=63, tol=1e-8, atol=1e-10,
mutation=(.5,1), recombination=.7, init=latinhypercube, updating=immediate,
workers=1, polish=False, at most 1,024 objective calls. Separately polish its best
feasible observed point with the same analytic local settings and 250-call cap.
Record polished route values, interruptions and all useful finite candidates.

Freshly check returned candidates; these calls charge actual evaluations even
when outside a solver's reported objective-call count. Pool feasible checked
route/face/production/contained candidates only after separately establishing
a and b. Shared injected winners never become independent routes. Require
R<=1e-5, completed required faces and appropriate numerical controls. A backend
success flag is not a substitute for agreement, nor alone a veto if all prescribed
checks nevertheless complete. Preserve interrupted useful candidates.

B00 scheduled fine audits use the unchanged family graph and all initializations:

| f | Audit graph | Development IDs | Validation ID |
|---|---:|---|---:|
| 0 | 0 | 0,1 | 8 |
| 1 | 7 | 2,3 | 9 |
| 2 | 3 | 4,5 | 10 |
| 3 | 4 | 6,7 | 11 |

There are 24 labeled development and 12 validation B00 audits. They do not
implicitly schedule four essential primary reference domains in v2.

Trigger numerical B00 escalation for unresolved R, incomplete nonflat required
faces, material/ambiguous E, material/ambiguous K for either member of a pair,
performance guard-band ambiguity, or a selected point within 1% of a nonflat face.
Scheduled audits use the same one-level allowance. Broader-domain gains or
cross-domain rankings are diagnostic triggers, not automatic essential B00 work.
Numerically unresolved K is handled conservatively: resolve/refine affected
missing B00 references within the same allowance; no unresolved pair can pass.

The full bundle halves both B00 mesh spacings once, uses at most eight separated
Route A starts, independent DE replicate 1 with maxiter=126 and at most 2,032
calls, separate 250-call polish, and bounded Powell at the two best distinct
candidate locations (maxiter=180, ftol=1e-12, xtol=1e-10, 250-call cap each).
Preserve route identities even for pooled starts; Powell does not manufacture
independence. Perform the level-1 face rechecks within the same trace allowances.

For an audit without a numerical trigger, fine grid, four fresh-start local
attempts and face rechecks suffice. Retain the independent routine DE route.
If a numerical trigger remains, complete only the remaining up-to-eight starts
and the one replicate-1 DE/Powell bundle; do not repeat already consumed allowances.
Capture r_coarse before refinement and r_fine after it; require F<=1e-5 and R<=1e-5.
An incomplete bundle never becomes complete merely because pooled values agree.

Distinguish a numerically resolved routine reference from completed prescribed
trigger work. An already resolved reference can demonstrate an E/K failure while
later work is unfinished, preserving v1's negative-evidence rule. Qualification
requires every prescribed audit/trigger action complete. Only one finer level
and one replicate-1 bundle per canonical landscape/domain are allowed in the
entire v2 study, including later diagnostics. Remaining ambiguity or reference
disagreement after that allowance is INCONCLUSIVE, not permission for more routes.

## 7. Required controls and spectral reporting

Preserve v1 section 10 and the reviewed native evaluator. The following control
coverage is explicit, so omitted wider-domain references are not mistaken for
omitted numerical safeguards:

| Control | Required cases |
|---|---|
| Direct-exponential pointwise and gradient checks | All 288 core labeled landscapes, with canonical reuse |
| Mapped spectral-coordinate check on 5x5 B00 points | All 288 core labeled landscapes, with canonical reuse |
| Power-of-two cost-unit checks and cyclic transport/reconstruction | Original development IDs 0,2,4,6, all graph/initialization labels, as in v1 |
| Native feasible-space spectral/initialization records | All 288 core labeled landscapes |

For each canonical landscape, generate exactly eight preseeded interior points
uniformly in B11 with margins (1e-5,1e-5/max(1,W_G)), using section 4's interior
seed and the unchanged NumPy sampling procedure. Before reference analytic
gradients, compare `d_u psi=-i*U_beta*D*U_u*s`, `d_beta psi=-i*H*psi`,
`d g=2*Re(<d psi|D psi>)` with central finite differences at those points.
Steps are h_u=1e-5 and h_beta=1e-5/max(1,W_G); tolerance is 1e-6, equality passes.

Compare native spectral evolution with direct matrix exponentials at those eight
points, the nine unique corners of the shared primary grid
{0,2*pi,4*pi} x {0,pi,2*pi}, P33's selected point and the selected B00 reference.
For selected references, also perform fresh native reevaluation. Recheck selected
points after prescribed reference updates; retain completed comparisons before a
terminal resource check. Probability, normalized-gap and state/probability
normalization tolerances are 1e-10, including equality. Out-of-box auxiliary
control points do not constitute wider-domain accuracy evidence. An essential
oracle mismatch or nonfinite arithmetic is FAILED; an unfinished required check
is unresolved. Do not relax tolerances to conceal discrepancies.

On the endpoint-inclusive 5x5 B00 grid, use center
`c_H=(lambda_max+lambda_min)/2`, `H_tilde=(H-c_H*I)/W_G`, `v=beta*W_G`.
Compare matched-coordinate probabilities and g while retaining the original
initial state; do not reconstruct it under a changed eigenspace tolerance.
This is a coordinate identity, not common-spectral-time optimization.

On original development IDs 0,2,4,6, freeze graphs and initial states and compare
coefficient multipliers 2^-20 and 2^20 on the same 5x5 points: normalized costs,
probabilities, g and inverse physical-gamma scaling. Retain the existing 1e-10
relative/zero-absolute gamma-scaling comparison. For cyclic relabeling use
`np.roll(np.arange(n),1)`, transport the QUBO, already constructed graph and state,
and compare probabilities in mapped feasible-basis order. Transport mismatch is
an essential pointwise failure. Separately reconstructed spectral-state sensitivity
is a diagnostic: use the unchanged 1e-3 materiality threshold/1e-5 guard and report
label dependence without silently changing the low fallback. Required records
must complete; material/ambiguous reconstructed-state sensitivity is a labeling
limitation, not by itself an E/K or transported-state failure.

Record actual feasible-space lambda_min/max, W_G, rho=max(abs(lambda)), low/high
multiplicities, next-distinct extremal gaps, neighboring separations, physical and
feasible-graph degree summaries, uniform extremal fidelities, projection/fallback
branches, residuals, beta and v. Record g(0,0) and initial optimum probability.
Physical adjacency spectra do not replace feasible-space spectra. Spectral
initialization changes both topology and initial state across graphs; state
preparation cost, causal topology effects and hardware fairness are not qualified.

## 8. Deterministic scheduling and phase separation

Before generation, require scientific approval, implementation verification and
explicit phase-specific execution authorization tied to the approved v2 hash and
driver. Verify source/environment, single-thread pools, real resource admission,
permitted public identifiers, historical integrity and fresh destinations.
Preview/metadata construction must never generate a QUBO.

Within each phase, use canonical cases in ascending
`(initialization_id,graph_id,instance_id)` order, retaining alias labels. The
mandatory waves are:

1. Generate exactly that phase's prescribed QUBOs and all graphs; record identities.
   Development generates IDs 0–7 only. Validation generation is gated by section 12.
2. Across all canonical cases, complete pre-search controls, P33 production,
   routine B00 references and remaining routine controls/selected-point checks.
3. Across scheduled audit cases, apply the B00 audit work. No broader search starts.
4. Recompute all 28 labeled graph pairs per stratum. Resolve triggered B00 work
   in round-robin passes, at most one pending refinement/bundle action per
   canonical case per pass, with numerical triggers fixed by sections 2 and 6.
   Refresh pairs and selected-point controls after updates until no permissible
   work remains or a hard stop occurs. Existing bounds limit actions, not outcomes.
5. Finalize complete core outputs and classify. Preserve every missing or unfinished
   required row. No favorable graph/initialization receives substituted effort.

Development QUALIFIED permits immutable v2 lock publication only. It is not
whole-study qualification. Development FAILED/INCONCLUSIVE permits no usable lock
and no validation. Lock publication does not authorize validation execution.

After separate authorization, validation runs once with P33 and the frozen rules.
Do not choose policies, domains, tolerances, seeds, allocations or audits from
validation outcomes. After qualified validation, and only under explicitly covered
diagnostic authorization, section 11's optional work may begin. A failed or
inconclusive core phase does not automatically launch diagnostics, retries or
another experiment.

## 9. Evaluation, CPU and operational resource contract

| Limit | Hard v2 allowance |
|---|---:|
| Development actual evaluations | 4,500,000 |
| Whole v2 study, including validation and any optional diagnostics | 8,000,000 |
| Protected evaluations at validation start | At least 3,500,000 |
| Canonical landscape across phases/domains/controls/diagnostics | 120,000 |
| Cumulative official execution CPU | 28,800 seconds |

Validation receives `8,000,000 - actual_spent`, not a reset allowance. Development
cannot borrow the protected validation allocation. After validation, optional
diagnostics use only the actual remaining study/CPU/landscape resources. The old
6,000,000/2,000,000 hard partition does not apply to v2: retain workload/phase
categories as provenance labels, not separate exhaustion gates. This is a new
prospective allocation; v1's 2,971,814 evaluations and 564.031178 full-process CPU
seconds remain historical expenditure, reported separately and in combined research
expenditure. Neither is retroactively relabeled or erased.

Charge every actual objective/state calculation, including fresh repetitions,
independent exponentials, transformed-model work, finite-difference probes,
production final reports and returned-candidate checks. Requests, cache hits,
unique exact float64 coordinate/model keys, actual evaluations, analytic-gradient
work reusing a state, matrix exponentials, H eigensolvers/reuse, CPU and wall time
remain separate counters. Exact-coordinate cache reads do not charge actual
evaluations. Charge the computing domain and canonical landscape that first
needed a fresh value; retain all other memberships and aliases. Batching charges
each computed point. No cache sharing across different objective identities.

Enforce all ceilings before starting additional numerical work. A resource stop
is sticky across processes; no counter reset, account relabeling, cache clearing
or new phase bypasses it. Protect completed essential failures during terminal
reporting, even if a later reporting step is interrupted. Resource exhaustion is
FAILED only when an essential scientific failure is established; otherwise missing
required core work is INCONCLUSIVE. Optional exhaustion leaves its adequacy claim
unresolved, with section 11's contradiction exception.

CPU covers official construction, imports, numerical checks, reporting, lock
publication/verification and diagnostics across all official processes. Use
full-process phase measurements, not only an earlier numerical snapshot; record
wall time separately. Every CPU snapshot has a measurement cutoff. Publication,
verification or finalization overhead after a cutoff must be recorded and deducted
before further numerical work. The immutable lock is not rewritten to account for
later overhead: subsequent phase/resource records carry these charges and link its
hash. All such deductions are monotone and cannot enlarge a locked allowance.
If CPU is depleted or finalization crosses its ceiling, disclose the overrun,
stop further execution and reject lock publication/use; preserve completed failure
evidence. Best-effort terminal evidence is not permission to continue optimization.

Start each numerical process with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1,
MKL_NUM_THREADS=1 and NUMEXPR_NUM_THREADS=1 before importing numerical libraries;
retain PYTHONDONTWRITEBYTECODE=1. Verify actual loaded numerical pool counts rather
than only environment strings. No weakening of the production thread guard.

Retain the reviewed Linux/cgroup resource sampler and fail-closed missing-information
behavior. Operational memory allowance is min(24 GiB, .75*effective_total_RAM).
Reserve 512 MiB+1,100 bytes per retained unique key, plus known extra loading peaks;
disk reserve is 512 MiB+512 bytes/key. Sample at least every one second or 256
objective requests, and force checks before expensive admission, artifact/lock
finalization and publication. These are sampled headroom guards, not guarantees
against every native spike or competing process. Preserve exclusive output
preflight, rejection of traversal/aliases/occupied destinations and historical
output protection. No overwrite option or concurrency guarantee is added.

## 10. Workload justification and honest completion limits

For essential B00 work with the retained shared traces:

| N_beta | Routine B00 | Refined B00 | Refined B00 union all level-1 primary trace nodes |
|---|---:|---:|---:|
| 64 | 1,089 | 4,225 | 4,734 |
| 128 | 2,145 | 8,385 | 9,150 |
| 256 | 4,257 | 16,705 | 17,982 |

The last column is `69*N_beta+318`, with overlapping nodes counted once.
W<=2*m bounds development N_beta by 128 and validation by 256 without inspecting
validation QUBOs. Original IDs 0,6 had N_beta=64 and IDs 2,4 had 128; reserve and
validation widths remain unknown. The instance-wide maximum is used even if an
individual graph's width is smaller.

Routine reference allowances are 4*250+1,024+250=2,274 calls; full refinement adds
8*250+2,032+250+2*250=4,782 calls. Allow up to 1,800 scalar face values over two
levels and 18 fresh returned-candidate checks: 8,874 reference evaluations in a
conservative envelope before cache reductions. These are allowances, not demanded
counts or convergence guarantees. The fixed-control planning allowance of 300
per landscape includes fixed probes/transformed checks and bounded selected-point
rechecks; it is not a new execution cap or numerical tolerance.

| P33-only planning component | Development upper envelope | Validation upper envelope |
|---|---:|---:|
| Grids and trace nodes, refinement on every landscape | 1,756,800 | 1,726,272 |
| Reference-attempt/face/fresh-check envelope | 1,703,808 | 851,904 |
| P33 grids and final reporting | 209,280 | 104,640 |
| Fixed-control planning allowance | 57,600 | 28,800 |
| Subtotal before production local searches | 3,727,488 | 2,711,616 |
| Headroom within development cap / protected validation allocation | 772,512 | 788,384 |

The combined subtotal is 6,439,104; saved original widths reduce it to 6,227,136.
The 4.5/3.5-million allocation freezes calculated headroom rather than promising
completion. The historical P9/P17 local calls are development-only observations,
not a P33 or dimension-126 bound. Production local-search calls, empirical
agreement, CPU, memory, lock loading and resource contention remain uncertain.
If remaining resources cannot finish required work, report INCONCLUSIVE unless
a failure is already established. No automatic larger allocation is permitted.

The wider-domain obstruction remains: at N_beta=128, routine B22 is 129*257=33,153
and fine B22 is 257*513=131,841. On historical 4:3:1, 107,786 native coordinates
lay on that fine grid: 33,153 routine B22 +24,768 reused fine-primary coordinates
+49,865 new fine-B22 coordinates. B22 charged 24,768 new routine-grid +519 routine
solver +43 face +49,865 finer-grid evaluations =75,195, within a 120,000 landscape
total. Another 24,055 coordinates were missing; finishing the grid required at
least 144,055 cumulative evaluations before later reference work. This accounting
is not changed. Moving those searches to separate diagnostics does not make
their full refinement feasible or establish wider adequacy.

## 11. Broader-domain diagnostics and later contradictions

Preserve v1 sections 6, 9–11's domain meanings. Optional here means independent of
B00 qualification, not scientifically unimportant. Their study must be explicitly
authorized and frozen before validation outcomes; outcome-conditioned challenges
are exploratory and never independent confirmation.

The predeclared diagnostic base matrix is B10,B01,B11 for every core labeled
landscape. Scheduled family audits also identify their finer diagnostic work.
Further/signed sentinel identities are fixed, one per instance:

| f | Sentinel graph / initialization | Development IDs | Validation ID |
|---|---|---|---|
| 0 | 0 / 0 | 0,1 | 8 |
| 1 | 4 / 2 | 2,3 | 9 |
| 2 | 3 / 1 | 4,5 | 10 |
| 3 | 7 / 0 | 6,7 | 11 |

These are 12 sentinel landscapes and 24 scheduled B22/Bsign searches. Additional
paired challenges use the preserved rules: B11 optimum within 1% of an upper face,
material/ambiguous B00-to-B11 gain, or clear/possibly ambiguous primary-domain
ranking reversal; challenge both members of each affected pair. Preserve ambiguity
conservatively rather than assuming no reversal. Record trigger identities even
when their work cannot be admitted.

After qualified validation, diagnostic waves use ascending
(initialization_id,graph_id,instance_id), canonical reuse and these priorities:
(1) routine B10,B01,B11 per case in that order; (2) routine sentinel B22,Bsign;
(3) scheduled diagnostic audits; (4) triggered creation/refinement/bundle actions,
one pending case/domain action per round-robin pass, domain order B10,B01,B11,
B22,Bsign. Finish the routine coverage wave before spending on triggered work.
Use the same methods, bounds, seeds, faces and one-level/one-bundle allowances.
No new QUBO, seed or initializer is selected from diagnostic outcomes. Reuse exact
retained/persisted generated models and state identities; absent models do not
authorize redraw or regeneration and are recorded as unavailable.

Before each diagnostic grid/action, calculate its minimum fresh coordinate union
against the actual same-objective cache and remaining landscape/study/CPU headroom.
The historical unique-key inventory is not a resident cache: only complete,
validated retained/restored evaluation entries permit reuse without a fresh charge.
Do not admit a grid whose unavoidable fresh-count floor exceeds remaining
evaluation allowance. Admit only while resource sampling passes. If an action
cannot start, record its explicit NOT_RUN reason with adequacy INCONCLUSIVE;
if it starts but stops, preserve useful candidates and unfinished requirements.
No unrun or unresolved diagnostic passes by omission. Workload admission is not
an empirical convergence or CPU-time certificate.

Use r(B00)-r(B10/B01/B11), A_direct=r(B00)-r(B22), A_step=r(B11)-r(B22),
S_signed=r(B11)-r(Bsign), and total r(B00)-r(Bsign). Evaluate gains only with
resolved participating references. The total signed-box improvement is not a pure
negative-beta effect. Larger/sign boxes are references, not qualified production
algorithms. Report primary cross-domain pair ordering and clear/ambiguous reversals.

Each declared finite-envelope claim is ADEQUATE only if all its declared comparisons
are resolved/nonmaterial and no clear required reversal defeats it; DEFEATED if
a resolved material gain or clear reversal defeats it; otherwise INCONCLUSIVE.
ADEQUATE means only the sampled declared envelope, not unrestricted angles. Known
historical defeated claims remain visible alongside new diagnostic observations.
Neither ADEQUATE nor DEFEATED diagnoses descriptor superiority.

Later diagnostics cannot protect a false B00 qualification. The contradiction
procedure is:

1. Retain exact candidate coordinates, model/state/objective identity, numerical
   checks, source/domain and spent-budget evidence. Only reliable finite
   candidates actually inside B00 may affect its reference pool; never project
   an out-of-box point or assume an unproved period.
   Reliability requires the same objective identity, fresh native reevaluation
   and a passing independent direct-exponential check at that point under the
   unchanged 1e-10 pointwise tolerance. Charge any needed checks to remaining
   study/landscape/CPU allowances; insufficient resources invoke step 4.
2. Assess affected B00 references, E and K with the unchanged definitions and
   remaining previously prescribed allowance. Importing a candidate does not
   create an independent route. No second refinement level/replicate is allowed.
3. A resolved essential E/K or shared arithmetic/oracle failure changes effective
   qualification to FAILED. If essential evidence is invalidated but cannot be
   resolved within the remaining allowance, change it to INCONCLUSIVE. A merely
   better out-of-B00 solution affects broader adequacy, not B00 accuracy.
4. If resources prohibit a needed reliability check, record the concern and
   unresolved affected qualification; do not ignore it to preserve QUALIFIED.
5. Issue a separately hashed append-only addendum identifying the original core
   artifact/lock hashes, the discovery, affected cases, recalculations, resource
   status and effective decision. Never edit original development/validation
   records or lock. Until resolution, no readiness claim may rely on withdrawn
   evidence. A reliable new B00 discovery is development information, not a new
   independent validation sample.

No contradiction permits policy substitution, changed bounds, new thresholds,
validation rerun, replacement lock, favorable record removal or hidden retry.
Diagnostic work after core validation remains exploratory and is identified as
such even when its numerical results are resolved.

## 12. Qualification and separate v2 validation-lock contract

For either core phase, the authoritative expected set is its full section 3
matrix, including aliases, every B00 reference, every required control/audit/trigger
and every graph pair. Classify in this order:

| Outcome | Exact rule |
|---|---|
| FAILED | At least one demonstrated mandatory P33 E/K, arithmetic/oracle or other essential contract failure; retain unfinished cases too |
| INCONCLUSIVE | No failure above, but any required case/check/reference/action is missing, constant/unresolved, ambiguous, interrupted or unverified |
| QUALIFIED | Every required B00 reference, E/K comparison, numerical control, refinement and coverage requirement passes; finalized evidence is intact and resource publication/use gates pass |

Historical P9/P17 failures do not count as P33 failures; they determine those
recipes' ineligibility. Broader-domain incompleteness or reconstructed-state
label sensitivity alone does not change a valid B00 numerical status. Shared
defects and B00 contradictions are governed by section 11. Report all limitations.
No favorable mean or newly selected initialization overrides an essential failure.

Development QUALIFIED authorizes lock eligibility only, not execution or
whole-study qualification. Validation QUALIFIED supports a finite-panel B00
claim with explicit domain, label and holdout limitations; never unqualified
"calibration passed", unrestricted performance, hardware fairness or RMS advantage.

The v2 lock schema and verifier are separate from unchanged v1 creation and
verification. v1's approved hash/rules/required-domain interpretation cannot be
relaxed, and no v1 lock is translated into a v2 capability. A v2 payload uses
rules_version=`numerical_calibration_protocol_v2`, schema_version=2,
scope=`P33_B00_native_XY_finite_panel_v2`, policy=33 and domain=B00. Its protocol
hash is the SHA-256 of the scientifically approved complete v2 document, recorded
externally; this draft has no self-referential approved hash or authorization.

The minimum lock payload binds:

- Approved v2 protocol hash, rules/schema/scope, approval identity, exact P33
  recipe/B00 float64 bounds/objective/initialization contracts.
- Exact driver revision/source identity and interpreter, NumPy, SciPy and BLAS
  identities; before-import environment and actual one-thread pool contract.
- All development IDs 0–7 and original/reserve provenance; exact graph/initialization
  matrix, aliases, edge/state/coefficient identities, seeds, generators/settings,
  audits, triggers, thresholds and consumed refinement allowances.
- Complete essential P33 production, route/face/R/F/E/K/control evidence and
  development QUALIFIED status, with no required pending work or resource stop.
- Finalized immutable core artifact locations and SHA-256s, not just a success flag.
- Historical v1 revision/protocol/artifact identity, permanent FAILED status and
  P9/P17 ineligibility; no assertion of a historical v1 lock that did not exist.
- Cumulative actual/request/unique/cache/domain/phase/landscape counters, CPU/wall
  records and measurement cutoffs, limits and exact spent/remaining allowances;
  development spent<=4,500,000, total remaining>=3,500,000, remaining CPU>0,
  and current resource admission passing.
- Fixed validation IDs 8–11, seeds/n/k/generator metadata, all 96 conditions,
  336 pairs, controls/audits/allowances/procedures and one-time-use rule.
- Frozen diagnostic matrix, sentinels, triggers, ordering, admission, resource
  priority and contradiction/addendum rules. Pending optional diagnostics are
  not incorrectly treated as incomplete essential development evidence.

The independent verifier reconstructs the expected Cartesian sets and pair
identities; checks exact coverage/alias consistency, production/objective identity,
seed/settings/version identity, bounds and finite values; recalculates R/F/E/K and
bands from persisted route/production values; checks required faces, controls,
audit/trigger completion and used allowances; verifies budgets and all artifact
hashes. If persisted data do not establish a requirement, reject rather than
trusting a producer's QUALIFIED or complete flag. Verify historical failure
identity and immutability without reading protected benchmark records.

Publish only finalized, flushed/fsynced evidence and a checksummed immutable lock
using exclusive atomic publication outside historical results. Reject occupied or
aliased destinations, altered/missing evidence, unknown/mismatched versions,
substituted policy, incomplete development, stale resources or depleted allowances.
Publication/verification overhead after the lock's stated cutoff is charged before
any later work, without increasing its original remaining-budget grant. Verification
must check the current monotone resource records as well as the immutable grant.

A separate explicit validation authorization must cover the approved v2 hash,
driver, phase, lock hash and fresh destination. Only after all verification and
resource checks pass, exclusively and durably create its one-time consumption
marker containing that lock hash, BEFORE validation QUBO generation or inspection.
An existing marker rejects another official attempt. Internal deterministic
reference routes are permitted within that one run; they are not holdout retries.
Interruption leaves the capability consumed and required unfinished work
INCONCLUSIVE unless failure is already established. No automatic restart/resume.
Further validation after outcome exposure would require a new reviewed study and
independent validation design, not silently reusing this panel as fresh evidence.

## 13. Artifacts, authorization and scientific independence

Use fresh versioned study/phase identifiers and destinations outside v1/historical
results, preserving exclusive preflight and original paths. No actual path, new
driver SHA or approved v2 hash is invented in this draft. Bind them before execution.
Retain the eight core artifact roles from v1 section 14:

| Artifact | Required v2 content |
|---|---|
| calibration_manifest.json | v2 identity/approval/authorization, explicit expected core matrix, P33/B00, phase/core status, separate diagnostic status, seeds/settings and full provenance |
| instances.jsonl | prescribed ID/family/split/reserve provenance, n/k/d, generator/parameters/seed, coefficient identities and observed/missing/error status |
| graphs.jsonl | labels/canonical aliases, actual edges/shuffle/random identities, state hashes, initial metrics and native spectral records |
| candidates.jsonl | exact u/beta/physical gamma, g/raw expectation, objective identity, route/start/seed/bounds, solver status/interruption and accounting; preserve demonstrated controls before failure |
| landscapes.jsonl | all expected rows, P33 and independent B00 references, R/F/E, faces, controls/audits/trigger work and unresolved reasons; optional domains explicitly separate |
| paired_contrasts.csv | every expected i<j pair, production/reference contrasts, K/band status, aliases and initialization, with no selected favorable subset |
| budget.json | actual/request/cache/unique/oracle/gradient/H counters, study/phase/domain/landscape/cutoff records, spent/remaining grants and required unfinished work |
| decision.md | development versus validation core outcome, all counterexamples, coverage, diagnostic limitations, historical identity and supported finite scope |

Use JSON null/CSV blank for undefined metrics, never favorable imputation or NaN.
Store aggregate counts, maximum/median/90th-percentile E/K descriptively by family,
graph, initialization and size; retain every failed identity. QUBOs are experimental
units; starts, seeds, initializations and labels are not independent QUBO replicates.
If resource-safe full reporting cannot finish, retain the bounded emergency evidence
path and observed failures; reporting-incomplete evidence cannot publish a lock.

Core artifacts are immutable after finalization. Optional diagnostic evidence and
contradiction addenda use separate fresh artifacts linked by hashes, without
rewriting core rows. Subsequent resource records account for actual cross-process
costs without modifying a finalized core ledger or lock. Preserve logs, phase
authorizations, full-process telemetry, artifact hashes and byte-preserving export
archives. A local archive is not a claim of external durable export.

Every execution phase requires separate explicit permission naming v2 identity
and allowed phase/destinations; neither a proposed/approved document nor a lock
is execution permission. Development authorization covers all active reserves but
not validation. Diagnostic permission cannot retroactively retune the frozen design.
No authorization file or lock is created by protocol preparation.

Prior development informed this design and is not independent confirmation.
The unchanged validation panel is a procedural holdout, not cryptographically blind
external custody. Changing both n and k tests combined transfer, not their separate
effects; four validation QUBOs establish no population failure rate. Validation
outcomes cannot tune the design. Later mechanism instances must be new and separately
authorized; no calibration record is confirmatory conditional-RMS benefit evidence.
Retain `DRAFT_UNFROZEN`, no RMS direction selection and the disclosed Stage 7A
out-of-allowlist byte-hash access without repeating protected-record access.

## 14. Exact v1-to-v2 scientific changes

| Contract | Immutable v1 | Prospective v2 |
|---|---|---|
| Target/acceptance dependency | Bounded B00 target with required wider completion gates | Essential P33/B00 qualification; separate exploratory broader adequacy |
| Policy eligibility | P9/P17 routine, conditional P33/preference ladder | P33 only; prior P9/P17 failures permanent; no fallback/repeat optimization required |
| Development reserves | Conditional activation | IDs 1,3,5,7 all active before new outcomes; IDs 0–7 required |
| Essential reference matrix | Four primary domains plus required challenges | B00 only; retain auxiliary traces/controls, not full essential wider grids |
| Audits | Family/all-initialization primary audits | Same identities and B00 allowance; wider audits are diagnostic |
| Reference/production mathematics | Native objective, existing methods/thresholds | Unchanged methods, objective, bounds, settings, tolerances and one-bundle limits |
| Ordering | Wider scheduled/triggered work can precede conditional policy work | B00 development -> immutable v2 lock -> one-time B00 validation -> optional diagnostics |
| Evaluation allocation | 6-million essential/2-million shared hard partition | 4.5-million development cap, 8-million study total, >=3.5-million protected validation; categories are provenance |
| Landscape/CPU/resources | 120,000 per landscape, 28,800 CPU seconds, reviewed guards | Retained; explicit cumulative full-process/cutoff accounting and protected priority |
| Phase statuses | v1 required-domain acceptance semantics | FAILED/INCONCLUSIVE/QUALIFIED for essential core; diagnostic ADEQUATE/DEFEATED/INCONCLUSIVE separate |
| Wider discoveries | v1 integrated evidence | Reliable B00/shared-defect contradictions withdraw effective v2 qualification through immutable-linked addenda |
| Lock | v1 hash/schema/completion checks | Separate v2 schema/version/hash/verifier and durable one-time marker; no translation or weakening of v1 |
| Historical files/interpretation | Permanent v1 evidence | Unchanged; v2 is a new prospective allocation, not retroactive success |

## 15. Minimal implementation handoff and approval boundary

This is an implementation mapping, not permission to implement. Add a versioned
v2 path outside v1's behavior and source-identity inputs; do not change the v1 hash
verifier, constants, default dispatch or existing acceptance/lock interpretation.
Do not put v2 sources into a glob that silently changes v1 `driver_identity()`.
Reuse reviewed numerical helpers only where their contracts remain unchanged.

| Concern | Existing reference module(s) | Minimum separately approved v2 adaptation |
|---|---|---|
| Configuration/protocol identity | calibration/config.py; experiments/run_numerical_calibration.py | Separate v2 configuration/hash/namespace and explicit v2 entry path; same metadata recipes, P33-only panel/roles |
| Budget/phase accounting | calibration/budget.py; calibration/resources.py | v2 study/development grants and monotone cross-process records; reuse unchanged sampled guards and numerical limits, without v1 account partition |
| B00-first scheduling | calibration/driver.py; calibration/reference.py | v2 mandatory/diagnostic waves, B00-only essential references, preserved audits/one-bundle methods and explicit expected pairs |
| Objective and controls | calibration/objective.py; calibration/diagnostics.py | Reuse native objective/P33 adapter and control contracts; select exact required coverage without changing arithmetic |
| Qualification/reporting | calibration/decisions.py; calibration/driver.py; calibration/storage.py | v2 essential versus diagnostic classification, historical eligibility, immutable core records and linked addenda; same formulas/bands |
| Lock creation/verification | calibration/storage.py | Separate schema-2 v2 creator/verifier independently checking completeness, hashes, resources and consumed allowances |
| Validation authorization | calibration/driver.py; calibration/storage.py; entry point | Explicit v2 phase/hash/lock permission and durable consumption before any generation; no fallback/retry |

No change is required to simulator.py, optimize.py, mixers.py, problems.py, numerical
objective arithmetic or scientific thresholds. A v2 implementation must preserve
v1 behavior and demonstrate version rejection, coverage/threshold semantics,
failure preservation, budget/CPU/resource stops, immutable evidence and one-time
validation gating on regression fixtures before any official generation. Passing
those tests is an engineering gate, not numerical qualification or execution permission.

The remaining limitations are uncertain production local costs, R/F convergence,
CPU/memory feasibility at d=126, optional-domain incompleteness, procedural holdout
custody and finite-panel generalization. Missing essential evidence, P33 failure,
invalid provenance, resource depletion or an unverified v2 implementation are
genuine execution/qualification blockers. No new optimizer, statistical framework,
global-optimum certificate or automatic scientific experiment is specified.

This separate repository document is self-contained and remains
**PROPOSED — AWAITING SCIENTIFIC APPROVAL; NOT AUTHORIZED FOR EXECUTION**.
