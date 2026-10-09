# Numerical calibration protocol v1

**Status: APPROVED FOR CALIBRATION-DRIVER IMPLEMENTATION — CALIBRATION NOT EXECUTED**

This document revises the Stage 4A proposal following the Stage 4B independent
review. It is a design, not a frozen protocol, an execution authorization, or a
record of calibration results. Implementation and execution require a separate
instruction after scientific approval. No RMS performance direction is selected.

## 1. Baseline, purpose, and implementation contracts

The scientific baseline is `dc68bbdaefc970d3a14ea3b9baefc8a9fa1e63a2`, the merge
of PR #11 into `main`. Its repair parent is
`2407eb557fd6ca91395741f4dbd70e2b02d39f94`. Sources governing this design are
[the simulator](../src/struct2circuit/simulator.py),
[the optimizer](../src/struct2circuit/optimize.py),
[mixers](../src/struct2circuit/mixers.py),
[problem generators](../src/struct2circuit/problems.py),
[the research protocol](research_protocol.md), and
[the mechanism hypothesis](../docs/MECHANISM_HYPOTHESIS.md).

The purpose is to assess numerical reliability before a later mechanism study:
missed minima inside a declared box, cost-unit dependence, domain truncation,
spectral timescales, boundary effects, and initialization sensitivity. Calibration
does not test whether either conditional-RMS direction is scientifically superior.

Relevant contracts remain unchanged:

- `optimize_p1` fixes bounds to `[0,2*pi] x [0,pi]`. It has no bounds argument.
  Both production grid axes exclude the upper endpoint. L-BFGS-B and Powell start
  at the same best grid point; finite returned candidates compete with that point.
- Its default grid size is 17. The alignment entry point defaults to 9 and uses
  the historical strong-`|Q|` comparator, not a full conditional-RMS comparison.
- `gamma_scale="feasible_span"` searches normalized phases and minimizes
  `normalized_gap`. Physical gamma is returned through the existing API.
- `expectation` averages legacy raw NumPy costs, even after normalized search.
  `finite_objective` checks this raw expectation's finiteness only.
- The feasible-space XY Hamiltonian has unit exchange amplitudes; it is not
  normalized by degree, edge count, or spectrum.
- Spectral references use the existing `1e-10` absolute eigenspace tolerance and
  `1e-12` projection-norm rule. The low-state fallback can depend on labels.
- Conditional-RMS scores describe a uniform fixed-cardinality conditional prior,
  not the transition distribution of an optimized or spectrally initialized state.

Passing unit tests establishes contracts, not numerical calibration accuracy.
Historical reproduction remains in `gamma_scale="legacy"` and outside this study.

## 2. Mathematical definitions and claim scope

For a cardinality-constrained QUBO, minimize

`C(x) = x.T @ Q @ x + c.T @ x`, with `x in {0,1}^n` and `sum(x)=k`.

The feasible dimension is `d=binomial(n,k)`. Let `C_comp(x)` be the compensated
float64 feasible cost used by the simulator, and define

`C_min=min(C_comp)`, `C_max=max(C_comp)`, `S=C_max-C_min`,

`D=diag((C_comp-C_min)/S)`, `u=gamma*S`, for `S>0`.

For graph G and fixed initial state s, the depth-one state and primary objective are

`psi(u,beta) = exp(-i*beta*H_G) @ exp(-i*u*D) @ s`,

`g(u,beta) = <psi|D|psi> = sum_x |psi_x|^2 * D_x`.

Use `evaluate_dimensionless` and its direct `normalized_gap`. Do not reconstruct
g from raw `expectation`, or replay the normalized state through the raw-cost
`evaluate(gamma,beta)` method. Store the actual selected u and beta explicitly,
along with returned physical gamma. A recording wrapper around the unchanged
normalized evaluator can capture the optimizer's final reporting coordinates;
`gamma*S` alone is not an exact coordinate-recovery contract under rounding.

If `cost_status="constant_or_unresolved"`, skip numerical optimization and keep
normalized metrics null. Do not impute an optimum probability or redraw the
instance. Such required conditions cannot count as successful calibration and
leave the declared panel qualification INCONCLUSIVE. Constructor or nonfinite
arithmetic errors are recorded separately as failures.

**Production qualification is restricted to B00.** All larger or signed boxes
are diagnostic reference searches. A better expanded-box reference demonstrates
a limitation of B00 for that broader target; it does not calibrate production in
the expanded box. Adoption of an expanded-domain production recipe needs separate
predeclared development and validation tests. Those tests are not authorized here.

The proposed default target is the deliberately bounded, positive-time B00
algorithm with native Hamiltonians and common physical-beta bounds. Additional
domain-adequacy claims receive separate explicit statuses. Restricted acceptance
may coexist with a recorded failure of a broader adequacy claim; it must be labeled
ACCEPTED WITH EXPLICIT LIMITATIONS. If scientific approval instead makes broader
adequacy an essential target, record that choice before generation; a failed
broader claim then produces overall FAILED, not a retrospective scope reduction.

## 3. Calibration estimands and thresholds

For landscape ell=(instance, graph, initialization), write `p_ell,N` for the freshly
reevaluated production result on B00 with grid size N, and `r_ell(B)` for the
complete independently refined empirical reference in box B. Retain feasible
production and narrower-box candidates in eligible final pools so that empirical
reference values are nested and never worse merely through bookkeeping.

| Quantity | Definition and orientation | Threshold, normalized-gap units |
|---|---|---|
| Reference-route agreement R | `max(|a-b|, |a-r|, |b-r|)`, with a and b independently refined route values and r the final eligible pooled value | `1e-5` |
| Production deficit E | `p_ell,N-r_ell(B00)`; positive means production missed a better candidate | `1e-4` |
| Domain gain A(B,B') | `r_ell(B)-r_ell(B')`, for nested boxes B subset B'; positive means enlargement helped | `1e-3` |
| Clearly separated ordering | `|contrast|`, where `contrast=g_j-g_i`; positive favors i | `2e-3` |
| Paired-contrast distortion K | `|(p_j,N-p_i,N)-(r_j(B00)-r_i(B00))|`; same instance and initialization | `2e-4` |
| Guard band eta | Ambiguity distance from a performance acceptance threshold | `1e-5` |
| Reference refinement F | `r_coarse(B)-r_fine(B)` for complete refined references, not raw grid minima | `1e-5` |

These are engineering choices, not measured noise floors or certified error
intervals. They assume later effects of interest around `1e-2` or larger; smaller
scientific effect targets require tighter numerical calibration first.

Evaluate K for every unordered graph-label pair i<j among IDs 0–7, within
each instance and initialization: 28 pairs per stratum, 672 essential pairs
and 1,008 if all reserves activate. Keep duplicate-label pairs and their canonical
provenance. Compare reference ordering across every pair of primary boxes. A
clearly separated reversal means opposite contrast signs with BOTH magnitudes
above `2e-3+eta`; a near-tie does not demonstrate a reversal. Also describe the
set within `2e-3` of each box's best graph, without selecting a winner or using
membership in that descriptive set as an additional acceptance test.

For upper-limit performance tests E, K, A, and signed gain, use a nominal threshold T:

- `q < T-eta`: clear pass, or nonmaterial gain.
- `q > T+eta`: demonstrated violation, or material gain, provided its references
  are resolved. Negative conclusions do not depend on a mixer winning.
- `|q-T| <= eta`: ambiguous, including equality at T and both band endpoints.
  Apply the one allowed additional reference bundle below; if still ambiguous,
  the affected mandatory acceptance decision is INCONCLUSIVE.

In float64 implementation, compute the two band endpoints once and classify the
remaining closed interval `[T-eta,T+eta]` as ambiguous. This gives every finite
value a branch without introducing a rounded-subtraction gap at an endpoint.

For ordering, `|contrast| > 2e-3+eta` is clearly separated;
`|contrast| < 2e-3-eta` is a near-tie. The intervening band is an ambiguous ordering
magnitude. Do not call either a reliable ranking. If ambiguity affects whether a
domain challenge is required, activate the challenge conservatively for both
members rather than assuming no reversal. A near-tie with resolved E and K is not
itself a failure of numerical calibration; it is the absence of an ordering claim.

Reference R and F have their own stopping rule: `<=1e-5` passes, including equality;
`>1e-5` requires escalation and otherwise leaves the reference unresolved. The
performance guard band is not applied to these reference stopping tolerances.
Applying a `1e-5` guard to a nonnegative `1e-5` upper limit would require R or F
to be negative for a clear pass. This explicit exception resolves that arithmetic
conflict without changing any declared threshold.

Reevaluate and diagnose any violation of expected monotonicity rather than
silently clipping a discrepancy. Report maximum E and K, every offending case,
and median/90th percentile descriptively, stratified by family, initialization,
graph rule, and size. No average can conceal a failed required condition.

## 4. Complete instance and seed matrix

Preserve the Stage 4A generators and settings:

| f | Generator | Fixed settings in addition to n, k, seed |
|---|---|---|
| 0 | `block_correlated_qubo` | `n_blocks=2`, `block_strength=.72`, `cross_strength=.16`, `linear_scale=.55` |
| 1 | `weighted_densest_k_subgraph_qubo` | `density=.55`, `weight_range=(.5,1.5)`, `weight_distribution="uniform"`, `planted_community_strength=1.8` |
| 2 | `weighted_max_k_vertex_cover_qubo` | `density=.55`, `weight_range=(.5,1.5)`, `weight_distribution="uniform"`, `community_strength=1.5`, `hub_strength=1.5` |
| 3 | `weak_structure_null_qubo` | `density=.55`, `coefficient_scale=1` |

| f | Essential development ID / seed | Reserved development ID / seed | Locked validation ID / seed |
|---|---|---|---|
| 0 | 0 / 420000000 | 1 / 420000001 | 8 / 420100000 |
| 1 | 2 / 420001000 | 3 / 420001001 | 9 / 420101000 |
| 2 | 4 / 420002000 | 5 / 420002001 | 10 / 420102000 |
| 3 | 6 / 420003000 | 7 / 420003001 | 11 / 420103000 |

Essential development: four QUBOs at `(n,k)=(6,3)`, `d=20`.
Validation: four QUBOs at `(n,k)=(9,4)`, `d=126`.
Reserved development: four further `(6,3)` QUBOs, not part of the essential eight.
Expected IDs are the explicit sets above, not IDs inferred from observed output.

Before generation, establish identifier noncollision using permitted public
manifests/seed-use metadata. Do not inspect protected blind-test records. If a
collision cannot be checked without such access, stop for an external identifier
check. Any necessary seed-namespace revision requires recorded approval before
outcomes; never replace an individual unfavorable or constant instance.

Activate all four reserved development instances, before validation, if a grid-33
policy is proposed or the selected policy would change from default P17 to P9.
This includes adopting a modified eligible production grid policy; do not activate
only favorable families. All activated records become required development cases,
and their failures remain visible. Changing any other production search recipe or
bounds is outside eligibility and requires a separate calibration design, not
silent activation of this reserve. Validation outcomes never activate development
reserves or select a revised policy.

The validation panel is a prospective procedural holdout, not secure blind custody.
Changing both n and k is a stress test, not separate evidence of size and cardinality
transfer. Four validation instances cannot establish a population failure rate.

## 5. Mixer, initialization, and audit matrices

Use exactly `m=n` physical mixer edges for every condition, matching the actual ring
count for n=6 and n=9. Graph IDs are fixed in this order:

| Graph ID | Condition | Construction |
|---|---|---|
| 0 | Ring | `ring_mixer(n)` |
| 1 | Historical strong absolute Q | `structure_conditioned_mixer(Q,m)` |
| 2 | Weak absolute Q | `score_conditioned_mixer(abs(Q),m,prefer="low")` |
| 3 | Low conditional RMS | `score_conditioned_mixer(exchange_profile_scores(Q,c,k),m,prefer="low")` |
| 4 | High conditional RMS | Same RMS matrix, `prefer="high"` |
| 5 | Shuffled RMS, low | Shuffled RMS matrix, `prefer="low"` |
| 6 | Shuffled RMS, high | Same shuffled matrix, `prefer="high"` |
| 7 | Random connected | `random_connected_mixer(n,m,seed=520000000+instance_id)` |

Shuffle upper-triangle RMS entries in lexicographic (i,j) order with
`np.random.default_rng(620000000+instance_id).permutation`; mirror them and set
the diagonal to zero. Record the permutation. This tests score correspondence,
not an independent score distribution. One random graph is a calibration stress
case, not a random-comparator mean estimate. Never redraw a graph based on results.

Initialization IDs: 0=`uniform`, 1=`mixer_low`, 2=`mixer_high`. Never pool these
conditions or choose a best initialization per method. Spectral comparisons
across graphs change both topology and initial state; preparation cost is not
accounted for, and no pure causal topology or hardware claim follows.

Construct all graphs before any QAOA optimization. Preserve labeled duplicate
graphs, but reuse computations for identical problem/edge/initialization contracts.
Use the lowest graph ID as canonical for duplicates; record requested and actual
canonical search seeds. Labels are not independent graph or instance replicates.

The essential matrix is `8 x 8 x 3 = 192` labeled landscapes and `192 x 4 = 768`
primary landscape-domain combinations. Activating all reserves adds 96 landscapes
and 384 combinations: totals 288 and 1,152. No sample is redefined retrospectively.

Unconditional finer-reference audits select these graph IDs, with ALL three
initializations, on the essential development AND validation instance of each family:

| f | Audit graph | Audited instance IDs |
|---|---|---|
| 0 | Ring, ID 0 | 0, 8 |
| 1 | Random, ID 7 | 2, 9 |
| 2 | Low RMS, ID 3 | 4, 10 |
| 3 | High RMS, ID 4 | 6, 11 |

This gives exactly `4 x 2 x 3 = 24` labeled audit landscapes, chosen before
outcomes with both RMS directions represented. If reserves activate, apply the
same family-specific graph/all-initialization audit to IDs 1,3,5,7, adding 12
audits. Exact duplicate computations may be reused without removing audit labels.

## 6. Domains, periods, and production-policy eligibility

| Box ID | Name | u interval | Physical beta interval | Role |
|---|---|---|---|---|
| 0 | B00 | [0,2*pi] | [0,pi] | Only eligible production domain |
| 1 | B10 | [0,4*pi] | [0,pi] | Gamma-only diagnostic |
| 2 | B01 | [0,2*pi] | [0,2*pi] | Beta-only diagnostic |
| 3 | B11 | [0,4*pi] | [0,2*pi] | Joint diagnostic |
| 4 | B22 | [0,8*pi] | [0,4*pi] | Required further-domain challenges |
| 5 | Bsign | [0,4*pi] | [-2*pi,2*pi] | Required signed-sector challenges |
| 6 | Bv | [0,2*pi] | [0,pi*W_ring/W_G] | Optional common-spectral-time sensitivity |

A sufficient cost-unitary period T requires `T*(D_x-D_y) in 2*pi*Z` for every
feasible pair. Normalization to [0,1] does not imply T=2*pi. Ideal rational costs
(0,1,3) give a 6*pi normalized-phase period; a float64 1/3 represents that ideal
only approximately. Approximate rational fitting is not a proof of period.

A mixer period requires `T*(lambda_j-lambda_0) in 2*pi*Z` for all relevant
eigenvalues. For one-particle K3, eigenvalues (2,-1,-1) give period 2*pi/3, not pi.
Other graph spectra have incommensurate differences. Any configuration-specific
reduction needs proof; this protocol performs no automatic period reduction.

For states real up to global phase, `g(-u,-beta)=g(u,beta)`. This does not justify
independent sign reversal or both-positive restriction as unrestricted optimization.
Signed-box evidence concerns its tested finite sector, not all real angles.

Eligible policies PN call the unchanged `optimize_p1(...,grid_size=N,
gamma_scale="feasible_span")` on B00. Run P9 and P17 routinely. Default proposed
selection is P17 if it passes all required development E/K gates. P9 is separately
reported for alignment compatibility and is eligible only if it independently
passes; do not transfer P17 evidence to P9.

If P17 has a demonstrated numerical failure with resolved references, evaluate
P33 on every essential development landscape and activate all reserves. Assess
P9, P17, and P33 on all activated development cases. Prefer P17 if it passes;
otherwise prefer passing P33, then passing P9. Selecting P9 also requires all
reserves, even if P33 is not used. No policy is selected from unresolved or
guard-band decisions. Grid size is not assumed monotonic in achieved accuracy.

If no eligible policy passes, report the failure/inconclusiveness and stop before
validation. Choosing an eligible PN changes no code during this stage; later
adoption of settings is a separate action. Bounds, methods, tolerances, initialization
selection, and legacy behavior cannot be changed under these eligibility rules.

Lock the chosen N before validation. Evaluate that exact PN once per validation
landscape; P9/P17 may also be reported as predefined diagnostics, and P33 is run
there only if activated before validation. None may replace a failed locked policy
using validation performance. References on B10/B01/B11/B22/Bsign qualify no expanded
production policy. Their methods are implemented only in an isolated future driver.

## 7. Routine empirical reference and independent routes

For each canonical landscape, build one endpoint-inclusive B11 grid with 64 u
intervals and `N_beta=smallest_power_of_two_at_least(max(64,8*W_max))`, where W_max
is the maximum feasible-space spectral width over its instance's graph conditions.
The same spacing is used across graphs and initializations. Restrict grid points
by exact box membership for the four primary boxes; upper faces are included.
Larger boxes preserve spacing by increasing interval counts, not by spreading a
fixed number of nodes. Refined grids halve both spacings within the same box.
Grid, local, DE, polish, and boundary components of each search must all use
that box's identical declared bounds; reject out-of-box returned candidates.

Observable u frequencies have magnitude at most 1; beta frequencies are bounded
by `W_G=lambda_max-lambda_min`. Thus the mesh obeys
`h_beta <= pi/(4*W_max)` on B11. This motivates sampling; it does not certify
minimum accuracy. A conservative bound is `|delta g| <= |delta u|+W_G*|delta beta|`.
The chosen mesh does not establish a `1e-4` global enclosure. No global certificate
or additional branch-and-bound infrastructure is required or claimed.

For each box, keep two separately logged routes:

**Route A, grid/local/boundary:** use at most four starts. Reserve slots for the
best grid point and a distinct best eligible production candidate, then add
lowest-value grid points separated by at least two cells in the maximum
grid-index distance; break ties by value/u/beta. The production start is exempt
from separation. Refine with bounded analytic-gradient L-BFGS-B, using production
`maxiter=120`, `ftol=1e-13`, `gtol=1e-9`, and a 250-objective-evaluation attempt cap.
Fewer available separated starts are recorded, not replaced by outcome-chosen seeds.

**Route B, independent DE then polish:** do not seed it with Route A winners.
For primary box IDs 0–3 use
`seed=720000000+10000*instance_id+100*canonical_graph_id+
10*initialization_id+2*box_id+replicate`, replicate 0 routinely and 1 on escalation.
This preserves the Stage 4A primary seeds. For challenge/optional box IDs 4–6 use
`seed=730000000+10000*instance_id+100*canonical_graph_id+
10*initialization_id+2*(box_id-4)+replicate`. Extending the original formula directly
to box 5 would collide with another initialization's primary seed; the separate
namespace prevents that collision without changing instance or primary seeds.
Freeze NumPy/SciPy versions;
use `strategy="best1bin"`, `popsize=8`, `maxiter=63`, `tol=1e-8`, `atol=1e-10`,
`mutation=(.5,1)`, `recombination=.7`, `init="latinhypercube"`, `updating="immediate"`,
`workers=1`, `polish=False`. With two free coordinates this is at most 1,024 DE
objective calls. Separately polish its best finite feasible candidate with the
same analytic L-BFGS-B settings and 250-call cap. Compare POLISHED route results;
raw DE-to-polished-local differences are not a calibration-failure statistic.

Explicitly scan primary box faces and corners using the shared grid. There are
six full traces: u=0,2*pi,4*pi and beta=0,pi,2*pi. Split brackets at internal box
cuts, retain endpoint candidates, and refine sampled strict/nonflat minima with
bounded scalar minimization. Use `xatol=1e-10`, at most 60 calls per bracket and
150 newly computed objective values per full trace. If a nonflat required trace
cannot be completed, escalate or mark it unresolved; do not silently discard it.
Use four traces, with the same caps, for each additional challenge box.
Record distance to each lower/upper face divided by its box width, the existing
`1e-8` relative boundary flag, and the separate 1% near-boundary trigger. Report
flat-face flags separately from nonflat boundary evidence.

Beta=0 is analytically probability-flat for every initialization. At u=0 an exact
mixer eigenstate is also probability-flat in beta. Avoid redundant refinement of
these faces. Before treating an approximately spectral state as flat, check the
eigenstate residual and the bound `2*beta_max*residual <= 1e-10`, as well as sampled
variation <=1e-10. Otherwise perform the nonflat scan. A near-boundary ridge is
not automatically evidence that a wider beta domain is necessary.

Only after computing independent a and b, pool feasible freshly evaluated
candidates from both routes, faces, production, and contained boxes to obtain r.
Require both independent routes to support the selected value within the R limit;
a shared injected winner alone is not route agreement. Reuse candidates in wider
final pools, but do not pretend that reuse creates an independent route.

Solver success, interruption, and candidate usefulness remain separate. Preserve
the best feasible observed value when an attempt is interrupted. If the prescribed
reference agreement is nevertheless supported by complete routes, a solver's
success flag alone is not a veto; any unmet required check remains unresolved.

## 8. Refinement, triggers, and limits on escalation

Scheduled audits and triggers use the same rules on development and validation.
Their execution is allowed reference refinement, not production-policy tuning.
Do not choose extra effort because a particular RMS direction looks favorable.

Trigger one reference-escalation bundle for a landscape if any required box has:

- R>1e-5 or a required nonflat face is incomplete;
- a material/ambiguous E or K decision, including either member of an affected pair;
- a performance threshold within its guard band;
- a solution within 1% of a nonflat face, or material/ambiguous domain sensitivity;
- unresolved rank-distortion diagnostics; or membership in the scheduled audit subset.

Bundle: halve both mesh spacings for affected primary boxes (reuse one refined B11
grid when appropriate); increase separated Route A starts to at most eight; use
replicate-1 DE with `maxiter=126` (at most 2,032 calls), with its separate local
polish; apply Powell to at most the two best distinct candidate locations per
affected box (`maxiter=180`, `ftol=1e-12`, `xtol=1e-10`, 250-call cap per attempt).
Complete affected boundary traces with at most 150 additional computed values per
trace. Preserve earlier candidates and route identities.

For a scheduled audit with no numerical trigger, only the finer grid, fresh
four-start Route A refinements, and boundary rechecks are compulsory. The retained
independent polished DE route remains an independent comparator because its box
and objective have not changed. If agreement fails, apply the remaining bundle.

Define `r_coarse` before that bundle and `r_fine` after complete local/global
refinement, including retained candidates. Require `F=r_coarse-r_fine <=1e-5` and
resolved independent-route agreement. A lower RAW fine-grid minimum is not evidence
of production inaccuracy; F compares complete refined references. Compare the
production deficit against the stronger final reference separately.

Only one finer-grid level and one replicate-1 DE route are allowed per box in this
study. Scheduled audits share this allowance rather than generating infinite
escalation loops. Repeated guard-band ambiguity or unresolved references after the
available bundle yield INCONCLUSIVE. The policy may demonstrably fail if an already
resolved stronger reference exceeds its E/K limit; reference incompleteness alone
does not prove a policy failure. Increasing budgets, adding algorithms, or further
domain ladders needs a separately recorded proposal.

## 9. Required further-domain and signed-sector challenges

Mandatory sentinels are the original combinations, on essential development AND
validation instances:

| f | Graph ID | Initialization ID | Essential instances |
|---|---|---|---|
| 0 | 0, ring | 0, uniform | 0, 8 |
| 1 | 4, high RMS | 2, mixer_high | 2, 9 |
| 2 | 3, low RMS | 1, mixer_low | 4, 10 |
| 3 | 7, random | 0, uniform | 6, 11 |

Thus eight mandatory sentinel landscapes each receive B22 AND Bsign: 16 additional
landscape-domain searches. Activated reserved instances receive the corresponding
family sentinel, adding four landscapes and eight searches.

Also challenge both boxes for every landscape with a B11 optimum within 1% of an
upper face, a material/ambiguous B00-to-B11 gain, or a clearly separated ranking
reversal across primary domains. Challenge both members of each affected pair.
Ambiguous ordering magnitudes that could determine a reversal trigger the challenge
conservatively. Required challenge sets are recorded explicitly, not inferred from
which cases completed. Repeat the same rules on validation without changing policy.

Use the same polished reference routes, mesh-spacing rules, face checks, caps,
and one-bundle escalation for challenges. B22 contains B11; Bsign contains B11.
Retain contained candidates in final pools. Mandatory sentinel searches are not
evidence that every unchallenged landscape is stable outside B11.

For each required B22 result compute BOTH

`A_direct = r(B00)-r(B22)` and `A_step = r(B11)-r(B22)`.

Apply the `1e-3` gain threshold and `1e-5` guard to both, as well as B00-to-B10,
B00-to-B01, and B00-to-B11. This prevents individually small ladder gains hiding
a material cumulative improvement. A material A_direct fails the claimed B00
adequacy over its tested further envelope; a material A_step fails the B11 plateau
claim. A material step also defeats B00 adequacy by nesting. No expanded production
optimizer is qualified by these findings. Unresolved challenge references or gains
in the guard band mean unresolved adequacy, not a demonstrated failure or pass.

For signed challenges compute

`S_signed = r(B11)-r(Bsign)`.

The same u and positive-beta extent make this a diagnostic of adding the negative
beta sector, rather than confounding it with simultaneous positive-time expansion.
Retain the signed candidate and its actual beta. A material S_signed prevents a
claim that positive-time searches cover relevant signed-sector optima. A resolved
B00 numerical policy can still be accepted WITH EXPLICIT LIMITATIONS under the
predeclared bounded target. It cannot be labeled globally or unrestricted-angle
adequate. Report `r(B00)-r(Bsign)` additionally as total improvement, without
calling it a pure sign effect. Nonmaterial signed gains support only the tested
finite sector, never all real angles.

Report primary/further-envelope ordering changes by initialization, with contrast
orientation fixed as g_j-g_i. Near-ties or ambiguous magnitudes carry no reliable
ordering claim. Clearly separated domain-dependent ranking reversals
require explicit scientific limitations even when individual gains are small.

## 10. Spectral, initialization, and arithmetic controls

Mandatory spectral records use the ACTUAL feasible-space H at its n,k:
lambda_min/max; W; `rho=max(abs(lambda))`; low/high multiplicities under the native
tolerance; next-distinct extremal gaps and neighboring separations; physical and
feasible-graph degree min/max/mean/variance; low/high uniform fidelities; projection
versus fallback branch; reference-state residuals; beta and `v=beta*W`.

Physical adjacency spectra are not substitutes for feasible-space spectra except
in the single-particle/hole case. Low degeneracy, near-degeneracy, spectrum, degrees,
and initial overlap are diagnostics, not performance predictors. Record g(0,0) and
initial optimum probability; improvement relative to the initial distribution is
secondary and does not identify a pure topology effect.

For every essential landscape, check the coordinate-preserving identity on a 5x5
endpoint-inclusive B00 grid. With `c=(lambda_max+lambda_min)/2`,
`H_tilde=(H-cI)/W`, `v=beta*W`, compare probabilities and g under corresponding
coordinates. Map native `[0,B]` to `[0,B*W]`; retain the ORIGINAL initial state.
Do not reconstruct it under a rescaled matrix's absolute eigenspace tolerance.
Changing coordinates on mapped domains leaves the state set unchanged; separate
optimizer discrepancies there would be numerical search effects.

The optional common-spectral-time optimization uses Bv, equivalent to common
`v in [0,pi*W_ring]`. The ring reproduces B00, while other physical beta bounds
change. It is a separately labeled sensitivity convention, not automatically a
fairer resource match or a causal test of spectral-width mediation. Spectral shape,
eigenvectors, degrees, and initial states still differ. Its omission is permitted
only with an explicit native-H/common-physical-beta scope and exclusion of
common-spectral-time conclusions; omission does not show spectral scale irrelevant.

If optional budget remains after all mandatory work, evaluate Bv on the 12
essential-development audit landscapes (one fixed graph per family, all three
initializations), at most 250,000 additional actual evaluations from the reserve.
If adopted, use the same empirical-reference criteria; incomplete optional records
are reported as such and support no interpretation. No optional outcome changes
the production policy, required matrix, or thresholds. A full Bv matrix is not
automatically authorized by this proposal.

Independent pointwise oracles compare the native spectral evaluator with direct
matrix-exponential evolution at all selected primary reference/production candidates,
the nine unique primary-grid corners, and eight fixed interior points per essential
landscape. Interior seed: `820000000+10000*instance_id+100*canonical_graph_id+
10*initialization_id`, sampling within B11. Check analytic derivatives

`d_u psi = -i*U_beta*D*U_u*s`, `d_beta psi=-i*H*psi`,
`d g = 2*Re(<d psi|D psi>)`.

Before using analytic gradients, compare central finite differences at interior
points with steps `h_u=1e-5`, `h_beta=1e-5/max(1,W)`; use points at least these
distances inside their box. A discrepancy is diagnosed, not hidden by adjusting
tolerances. Pointwise probability/g/normalization tolerance is `1e-10`, gradient
absolute tolerance `1e-6`; equality passes. All selected-point standard reevaluations
also use `1e-10`. Invalid arithmetic or a demonstrated oracle mismatch fails the
driver qualification; unfinished comparisons are inconclusive. No global search
accuracy is inferred from these pointwise tolerances.

On all graph/initialization conditions of the four essential development QUBOs,
check power-of-two coefficient multipliers `2^-20` and `2^20` on a 5x5 B00 grid.
Freeze original graphs and initial states; compare normalized costs, probabilities,
and g, and check inverse physical-gamma scaling using declared fixed u values.
Do not recompute score-selected graphs or replay raw phases. A large additive
shift can destroy represented distinctions; an optional modest-shift diagnostic
must report coefficient-rounding error and is not an unconditional equality test.

On the same four QUBOs, cyclically relabel the QUBO AND an already constructed graph.
Compare explicitly transported states on the 5x5 grid; probabilities must be
compared in the mapped feasible-basis order. Transported-state equality is a
pointwise gate. Separately reconstruct spectral references and report low-fallback
sensitivity: an altered degenerate-low selection is permitted by the existing
contract, NOT a violation of the `1e-10` transported-state equality test. Changes
above `1e-3` in g are material reference-selection sensitivity and require a label
limitation; the guard band handles materiality ambiguity. Do not rebuild tied
score-selected graphs here, silently repair the initializer, or average away the
sensitivity. Uniform/unique-high comparisons retain their expected equivariance.

## 11. Operational acceptance table

Store separate statuses for arithmetic, reference completeness, each PN's B00
E/K reliability, primary/further positive-time adequacy, signed-sector adequacy,
initialization selection sensitivity, and optional spectral-time interpretation.
Select no RMS direction. A failed graph/initialization case remains in every count.

Classification is applied to the LOCKED production policy and declared target,
not to whichever diagnostic PN happens to perform best on validation:

| Classification | Exact entry conditions and consequence |
|---|---|
| FAILED | Demonstrated arithmetic/oracle/mandatory contract violation; resolved E or K above its failure band for the locked PN; validation defeats that PN; or a resolved domain gain/ranking counterexample defeats a broader adequacy target declared essential BEFORE generation. Report every defeated claim and case. |
| INCONCLUSIVE | No decisive failure above, but a required row is constant/unresolved, a reference/challenge/oracle is unfinished or unresolved, a mandatory acceptance statistic remains in its guard band, required validation was not completed, or a resource ceiling interrupted required work. No readiness claim. |
| ACCEPTED WITH EXPLICIT LIMITATIONS | All required rows, references, oracles, validation, and locked-PN B00 E/K checks pass; the predeclared target permits bounded acceptance; but a resolved primary/further/sign challenge or labeling sensitivity restricts broader interpretation, or common-spectral-time conclusions are excluded because optional diagnostics were omitted/incomplete. Explicitly show any broader adequacy claim as FAILED, not passed. No expanded-domain production qualification. |
| ACCEPTED | All required checks and locked-PN validation pass; no required primary/further/signed-domain or reliable-ordering counterexample is found in the sampled challenge envelope; no additional labeling restriction applies; and optional spectral-time diagnostics, if needed to remove the corresponding interpretive limitation, are complete. Qualification remains empirical, finite-panel, B00-only and native-model specific. |

A decisive failure takes precedence over unfinished work, but unfinished statuses
are still reported. In the bounded default, a resolved failure of broader adequacy
does not manufacture an E/K failure inside B00; it leads to restricted acceptance
only if every mandatory numerical and challenge decision is resolved. Unknown
broader adequacy is INCONCLUSIVE, not restricted acceptance by omission. Known
signed/further improvements cannot be erased through a favorable average or scope
switch after data. A near-tie with resolved K is reported without ordering and is
not automatically a numerical failure.
Apply the rows in the order FAILED, INCONCLUSIVE, LIMITED, then ACCEPTED; ACCEPTED
requires no remaining limitation predicate from the preceding row. Omission or
incompletion of the optional Bv panel is an explicit spectral-interpretation
limitation, not unresolved mandatory work. Completing its 12 development cases
can remove only that omission limitation for those cases; it does not establish
spectral fairness or extend the optional result to the entire matrix.

Example decision pattern: B00 optimization reliable + material B22 improvement +
no calibrated expanded production policy => B00 numerical status PASS, envelope
adequacy FAILED, expanded-production status NOT TESTED; overall ACCEPTED WITH
EXPLICIT LIMITATIONS for the predeclared bounded target, or FAILED if broader
adequacy was an essential approved target. Never label the wider production policy
accepted. A missing/unresolved B22 reference instead gives INCONCLUSIVE.

No classification establishes low/high RMS superiority, an unrestricted global
optimum, population failure rates, hardware fairness, spectral-state preparation
efficiency, trainability benefit, quantum advantage, or confirmatory readiness.

## 12. Budget definitions, independent envelope, and scheduling

All ceilings are subordinate to approval of this proposal:

| Account | Hard ceiling |
|---|---|
| Essential workload account | 6,000,000 actual objective/state evaluations |
| Shared escalation, reserved-instance, and optional account | 2,000,000 additional evaluations |
| Whole study | 8,000,000 evaluations |
| Optional diagnostics | 250,000, drawn from unspent shared reserve, not an extra allowance |
| Canonical landscape, all boxes/controls combined | 120,000 evaluations, subordinate to both global accounts |
| Cumulative CPU time | Eight CPU-hours, one declared numerical thread |

Charge initial required searches, scheduled 24 audits, mandatory eight-sentinel
challenge pairs, and required oracles to the essential account. Triggered additional
bundles/challenges, P33, and activated reserves charge the shared account. If
scheduled mandatory work exhausts the essential account, it may use unspent shared
reserve before optional work; record that transfer and stop at eight million.
Optional work starts only after all required decisions are complete and cannot
consume resources reserved for unfinished mandatory work. Any first exhausted
hard ceiling interrupts further computation; completion is not assumed.

Counters must distinguish:

- `objective_requests`: every solver/grid/oracle request, including cache hits;
- `unique_parameter_points`: distinct coefficient/model/initial-state hash and
  exact float64 u/beta tuples, not rounded parameter buckets;
- `actual_evaluations`: every freshly computed objective/state, including uncached
  repeats and independent oracle calculations; THESE charge the hard budget;
- cache hits, gradient-only operations reusing a state, matrix-exponential calls,
  eigendecomposition counts, and CPU/wall time, recorded separately.

Batching counts each computed parameter point. Cached reads cost requests but not
actual evaluations. Oracle or transformed-cost work is not free because its native
parameter tuple appeared earlier. Finite-difference objective probes charge actual
evaluations; analytic gradient work and dense matrix construction charge CPU time.
Final reporting/reevaluation calls also charge the budget, even though the public
optimizer's reported counter excludes its final reporting call.

Allocate charges to the canonical landscape and computing domain that first needed
the point. Retain membership in other boxes and duplicate labels. Per-domain request
counts may overlap; their sums are not sums of unique cached evaluations. H reuse
across different QUBOs saves eigendecompositions, not cost-dependent objectives.

Independent workload accounting, before outcome-dependent deduplication:

| Quantity | Essential | All reserves active |
|---|---|---|
| QUBOs | 8: four dimension 20, four dimension 126 | 12: eight dimension 20, four dimension 126 |
| Instance/graph labels | 64 | 96 |
| Distinct H upper bound, sharing one ring per size | 58: 29 small + 29 large | 86: 57 small + 29 large |
| Reusable H/initialization pairs, upper bound | 174 | 258 |
| Labeled cost-dependent landscapes | 192 | 288 |
| Primary reference boxes | 768 | 1,152 |
| Routine four-start local attempts | 3,072 | 4,608 |
| Routine DE runs / separate polishes | 768 / 768 | 1,152 / 1,152 |
| Routine P9/P17 production runs | 384 | 576 |
| Unconditional finer audits | 24 | 36 |
| Mandatory sentinel boxes beyond primary boxes | 16 | 24 |

Since exchange-graph row degree is at most m, `W<=2*m` gives conservative
`N_beta<=128` for n=6,m=6 and `N_beta<=256` for n=9,m=9. This is a bound,
not a computed spectrum. A coarse B11 grid uses `65*(N_beta+1)` points; a
half-spacing B11 grid uses `129*(2*N_beta+1)`. Reuse old nodes exactly.

| Essential nominal component | Evaluation allowance/envelope |
|---|---|
| Coarse shared grids | 811,200 to 2,408,640 |
| New grid points for scheduled 24 fine audits | 297,984 to 890,880 |
| Routine four-start analytic local attempts | At most 768,000 |
| Routine DE | At most 786,432 |
| Separately logged DE polishes | At most 192,000 |
| Scheduled fresh four-start fine-grid local attempts | At most 96,000 |
| P9/P17 production grid requests | 71,040; production local calls are additional |
| Routine primary face refinements, six traces/landscape | At most 172,800 |
| Scheduled finer face rechecks | At most 21,600 |
| New grid points for mandatory B22/Bsign pairs | 132,608 to 396,800, before overlap with audits |
| Mandatory challenge local/DE/polish/face allowances | At most 45,984 |

The sum of this nominal envelope is **3,395,648 to 5,850,176**, excluding
production local refinements, pointwise oracles, other escalations and controls.
It is not a promised evaluation demand: solver allowances are ceilings, cache
reuse can reduce it, and some scheduled numerical attempts may terminate earlier.
The six-million essential account is therefore NOT guaranteed to complete.

For all reserves, P9/P17 grid requests become 106,560. If P33 is activated on all
12 instances, all three production grids request 420,192 points: 349,152 more
than the essential P9/P17 grids. A universal half-spacing primary grid alone
could require 12,718,368 points on all 288 landscapes under the conservative
width bounds; such a full escalation does not fit and is not automatically run.
Budget exhaustion must give an honest unfinished/INCONCLUSIVE status.
The per-landscape ceilings alone permit 23,040,000 evaluations for 192 canonical
landscapes, or 34,560,000 for 288 before deduplication. They are not additive
allocations: the eight-million global ceiling always overrides those sums.

Even a scheduled audited validation sentinel can exceed 120,000 points when a
conservative N_beta=256 fine grid and both expanded challenges are combined.
Do not waive its cap, drop its signed challenge, or claim a pass; retain the
completed data and mark the required decision unresolved. Actual widths may be
smaller, but no unperformed measurement is assumed in this design.

Dense H diagonalization costs O(d^3); native spectral state evaluation costs O(d^2),
with two dense matrix-vector products plus phases and reductions. Dimension 126
has about 39.7 times dimension-20 matrix-vector arithmetic. Direct matrix exponentials
are O(d^3) and remain pointwise oracles, not the grid engine. Costs, basis, H spectra,
states, duplicate graphs, and exact shared points should be cached. Verify scalar
versus batched evaluations before batching. These are operation counts, not measured
runtime. The eight-CPU-hour ceiling includes oracle, gradient, construction and
linear-algebra work; record wall time too. No assertion of completion within it is made.

Scheduling priority: essential development coverage and initial references first;
scheduled audits and required challenges second; triggered reference resolution
and any activated reserve/policy checks third; lock the policy and decision table;
then perform the corresponding validation work. Use round-robin instance/graph/
initialization ordering within each wave. Do not begin optional diagnostics while
any required row is unfinished. Do not open validation merely because development
ran out of budget; record the unrun holdout and INCONCLUSIVE qualification.

## 13. Development and holdout protections

Development may diagnose the isolated driver and assess only predeclared PN
policies and reference refinements. Its threshold values, graph directions,
generator settings, and acceptance target are not selected by favorable rankings.
Any driver correction is documented, hashed, and revalidated before holdout access.

Before any validation QUBO is generated or its landscape inspected, lock and record:
selected N on B00; driver/version hashes; target and restricted-acceptance rule;
all required IDs/labels; whether reserves were activated; thresholds; challenge
triggers; audit subset; and remaining budget. This runtime lock occurs only after
approval and does not describe this proposal as already frozen.

Validation is executed once with its predetermined reference/escalation rules.
A second numerical route is a prescribed computational replicate, not a second
holdout for policy tuning. If validation demonstrates a locked-policy failure,
retain it and do not substitute P9/P17/P33 based on those outcomes. Tuning from
validation changes makes it development data and requires a newly approved,
fresh validation design. No additional holdout is silently generated here.

All calibration instances and results remain exploratory numerical diagnostics.
Later mechanism instances must be new; no calibration record is confirmatory RMS
benefit evidence. No benchmark blind-test records are accessed. Do not reactivate
sample-size, multiplicity, or confirmatory inference infrastructure for this small
numerical study. QUBO instances are experimental units; graph, initialization,
optimizer seed and search-start conditions are not independent QUBO replicates.

## 14. Outputs, provenance, and integrity

A future run uses a fresh explicitly selected output directory outside preserved
historical results. Preflight every deterministic destination, including the
manifest and decision memo; reject occupied paths and parent traversal. No overwrite
option or historical artifact regeneration is part of this proposal.

Required deterministic outputs and minimum contents:

| Artifact | Required fields/content |
|---|---|
| `calibration_manifest.json` | baseline and driver SHA, proposed-document hash, approval/lock records, target, policy N/B00, expected IDs, all seeds/settings, budgets, versions/BLAS/thread settings |
| `instances.jsonl` | family, split/reserve activation, n/k/d, seed, generator version/parameters, coefficient hashes, status/error, expected versus observed IDs |
| `graphs.jsonl` | graph IDs/labels, edge list/hash, m, construction/shuffle/random seeds, canonical alias, degree/spectrum diagnostics, initialization hashes/branches/overlaps/residuals |
| `candidates.jsonl` | landscape/domain, exact u/beta and physical gamma, normalized objective/raw expectation meanings, route/start/DE seed, tolerances, freshly checked values, solver status, interruptions, evaluation/counter/CPU provenance |
| `landscapes.jsonl` | all requested labels, production PN results, independent route values, complete coarse/fine references, R/F/E, challenge membership and gains, face distances, oracle/invariance checks, missing/error states |
| `paired_contrasts.csv` | explicit instance/initialization and graph pair, production/reference contrasts, K, ordering/near-tie/band classification, domain changes |
| `budget.json` | request/unique/actual/cache/oracle/gradient counters by landscape/domain/account, transfers, CPU/wall time, stop reason, incomplete required work |
| `decision.md` | locked-policy outcome, separate claim statuses, all failures/ambiguities and affected/expected counts, limitations, and exact supported scope |

Store undefined metrics as JSON null or CSV blank, not NaN or favorable imputation.
Use explicit expected IDs and the Cartesian labels/domain/challenge sets from the
manifest to verify completeness. Duplicated computations retain every label and
canonical provenance. Missing/nonconstant failures are never inferred away from
only observed rows. Preserve solver messages and selected-source information;
success does not certify a global minimum. Numerical traces and preselected/failure
landscape plots may be optional exports after required outputs are secured.

## 15. Ordered implementation handoff and scientific boundaries

After separate scientific approval and implementation authorization:

1. Verify the pinned merged baseline, approved proposed-document hash, clean
   intended scope, and permitted seed-use metadata. Do not inspect protected data.
2. Implement an isolated calibration driver and its numerical recording/budget
   logic. Preserve production optimize/simulator/mixer/generator semantics and
   historical scripts; new bounds belong to diagnostic references only.
3. Verify normalized objective use, recorded u, mapped spectral coordinates,
   independent oracle/gradient checks, exact-cache keys, counter accounting,
   nested boxes and completeness behavior before study execution. Tests of a
   future driver are separate from calibration performance evidence.
4. If execution is separately authorized, generate only essential development
   cases, construct all graphs first, and perform routine references and policies.
5. Apply scheduled audits and predefined triggers, retaining counterexamples.
   Activate all reserves only under their declared policy conditions. Select no
   graph direction from results; stop if no eligible policy qualifies.
6. Record the policy/target/threshold/driver/budget lock before accessing validation.
7. Execute validation once under that lock, preserving failures and unrun cases.
8. Finish mandatory outputs and decision statuses before optional diagnostics.
   Apply the operational table; propose follow-up designs rather than claiming
   expanded production qualification or extending budgets automatically.

No calibration implementation or execution is performed by the documentation
revision itself. No large confirmatory benchmark, PR, merge, or scientific
performance experiment is authorized by this document. Numerical qualification
supports only subsequent separately approved, scoped mechanism work using new
instances and the validated policy; it establishes no RMS benefit or global/hardware
advantage. Common physical beta matches allowed native evolution time, not equal
spectral motion, gate count, depth or hardware cost. Initialization sensitivity and
score-tie label dependence remain explicit scientific interpretation limits.

## 16. Change log: Stage 4B corrections incorporated

- Blocker A: production qualification now explicitly B00-only. Expanded references
  diagnose broader targets, and a wider production recipe remains NOT TESTED.
- Blocker B: primary/cumulative/further and signed gains have oriented formulas,
  guarded decisions and explicit failed-versus-unresolved claim statuses. K is
  defined independently of the ordering threshold. F compares complete refined
  references, not raw grids.
- Acceptance: the four classifications and bounded-target fallback are operational;
  known broader failures remain visible, and unresolved mandatory work cannot pass.
- Guard-band reconciliation: the `1e-5` performance ambiguity band is retained;
  applying it to the nonnegative `1e-5` reference stopping limits would eliminate
  their pass region, so R/F use explicit direct stopping rules instead.
- Size: four development plus four validation instances are essential; four original
  second-development seeds are reserved for declared policy changes. All graph and
  initialization labels remain; audit IDs are fully specified and balanced across
  both RMS directions.
- Reference: routine four-start analytic local search plus one independent DE and
  its separately recorded polish replace repeated dual-local/two-DE ensembles.
  Additional methods and finer checks have outcome-neutral numerical triggers.
- Resources: six-million essential/two-million shared/eight-million total accounts,
  120,000 per landscape and eight CPU-hours are reconciled with a conservative
  envelope and explicit INCONCLUSIVE stopping, not assumed completion. Actual
  uncached computations, including oracles, are budgeted separately from cache hits.
- Independence: original seeds/settings, validation lock, retained failures, and
  separation from confirmatory RMS outcomes are preserved. Optional spectral-time
  analysis is labeled and does not silently change the scientific target.
- Seed extension: primary search and generator seeds are unchanged; additional-box
  DE seeds use a separate deterministic namespace to prevent initialization/box
  collisions in a direct extension of the original four-box formula.

The original Stage 4A temporary proposal is preserved unchanged. This repository
document is self-contained and is APPROVED FOR CALIBRATION-DRIVER IMPLEMENTATION — CALIBRATION NOT EXECUTED.
