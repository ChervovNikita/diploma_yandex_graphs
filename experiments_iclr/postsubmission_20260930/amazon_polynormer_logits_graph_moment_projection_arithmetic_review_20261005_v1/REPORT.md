# Bounded arithmetic review: probability-space posterior projection repair

5 October 2026. **FORMULATION_SUPPORTED_PENDING_WITNESS_SOURCE_AND_QUALIFICATION.** This is an independent mathematical assessment of the author-proposed repair, coordinated with root and the author. No V3 source is approved by this note. No failure witness, scientific payload or numerical function was accessed or executed.

## Observed boundary and scope

The inspected stderr confirms V2 failed inside the singular operator9 `simplex_qp` with `QP active-face solution not found`, called from the first group's posterior projection. Root reports split0/shared/fold0, 6.11 seconds, approximately500.9MB and no first-head fit or completed fold score. Those costs and absence claims come from root; this review did not inspect output payloads or inventory to reproduce them. The log alone does not identify the numerical cause. The original V2 failure and source are preserved.

The proposed alternative keeps the dense floor0.0125, sum-one constraint, squared distance to the diffused anchor-only posterior, all15 free faces, ordering and objective/norm ties. It changes the semidefinite projection arithmetic; ridge-moment arithmetic is intended to remain unchanged. V2 protocol's explicit KKT/pseudoinverse recipe changes under its numerical-repair allowance (`PROTOCOL.json` arithmetic lines390–400), so the eventual release must record the exact numerical deviation and predecessor. This is a custody requirement within the existing workflow, not a new study or gate.

## Face formulation and equivalence

Let P contain the four member probability rows, t the five-class diffused posterior, l=0.0125, and F a nonempty free set of size k. Inactive weights I are l; free weights sum to s=1−|I|l. Set the free-face center c=s/k and choose an orthonormal Helmert matrix U with k rows and k−1 columns satisfying U^T U=I and 1^T U=0. Every equality-feasible free vector is

`w_F=c·1+U z`, with `w_I=l`.

Let w0 be the corresponding center with inactive floor weights. The original face objective is exactly

`min_z ||B z−d||²`, where `B=P_F^T U` and `d=t−P^T w0`.

The author's reference-centered construction `B=(P_F−P_ref)^T U` is identical in exact arithmetic, because the common reference term is multiplied by `1^T U=0`. In FP64 it suppresses leakage from a nearly zero Helmert column sum and gives exactly zero contrast rows for identical represented member probabilities. Its tiny roundoff discrepancy from direct uncentered multiplication is covered by unchanged sum and original-gradient checks; it must not replace those checks. k=1 has no z and is evaluated as the fixed face vertex.

Direct SVD yields the Moore–Penrose minimum-norm least-squares z. Because the center is orthogonal to U,

`||w||²=|I|l²+s²/k+||z||²`.

Thus exact-rank SVD gives minimum full weight norm among equality-feasible minimizers of that face. The face candidate still needs the lower-bound and KKT checks. A singular face's unrestricted minimum-norm point can violate a floor even when another point in its minimizer set is feasible; all15 faces must remain enumerated. This still captures the globally constrained minimum-norm optimizer: on that optimizer's actual free face, it is interior to all remaining floors, so its norm is stationary along every primary-objective null direction and equals the unrestricted minimum-norm face solution. If a floor limits that norm minimization, the actual free face is smaller and also enumerated.

## Objective, gradients and tie semantics

Select candidates using `||P^T w−t||² / scale`, with the existing per-node `scale=max(maxabs(P P^T),maxabs(P t),1e−12)`. The stable residual objective differs from `w^T A w−2b^T w` only by `||t||²`, a face-independent constant, so division by the same scale preserves exact differences and the declared1e−14 numerical objective tie rule. Compare squared weight norm with the same1e−14 tolerance; retain first face for remaining ties. No ridge is introduced.

Compute original gradient `g=P(P^T w−t)` directly. In exact arithmetic this equals `A w−b`; factor2 of the full objective derivative cancels from KKT. Free g values must share their equality level; inactive g must be no smaller than that level. Scaled g/scale and raw g retain distinct stationarity/dual diagnostics and their existing1e−10 tolerances. Existing floor/sum checks and post-cleanup KKT rechecks remain necessary. Truncated-SVD residuals and face feasibility cannot substitute for those original checks.

## Realistic conditioning risk

Nearly identical member probability vectors produce small contrast singular values. Concentrated probabilities can also make several class contrasts tiny or represented identically. Both are plausible for shared banks; their prevalence and their relevance to the actual failed row remain unknown before root's witness. A Gram construction squares the singular-value conditioning, and an augmented KKT relative pseudoinverse cutoff can discard contrast directions against a much larger common/equality scale. A direct5-by-at-most3 contrast SVD avoids this squaring and removes the equality variable from the numerical rank decision. It does not make arbitrarily small contrast directions exactly resolvable.

## Proposed absolute machine-rank cutoff

The author's proposal is to retain singular values strictly greater than `tau=64·eps64·max(1,sigma_max)`, in probability units. This is an explicit effective-rank approximation. It is not a test that a mathematical singular value is exactly zero and not a ridge. The eventual source must record the rule, retained-rank counts, and direct scaled/raw residuals so the arithmetic is auditable. The cutoff should be fixed for all operator9 groups, not adjusted to the witness or a scored outcome.

For normalized probability rows, `sigma_max(B)≤||P_F||_F≤sqrt(k)≤2`. For any floor-feasible face vector, `||z||<1`. Therefore omitting its component in a sub-tau singular direction changes its reconstructed probability vector by at most tau, with tau at most approximately2.84e−14. This is a local truncation bound for a feasible vector, not a guarantee on the final optimizer or its weight norm. Two feasible probabilities and t have residual norm at most sqrt(2), so the corresponding conservative raw objective perturbation is at most `2sqrt(2)tau+tau²`, approximately8.1e−14. A diagonal of A is at least1/5, hence the existing scale is at least0.2 and the conservative scaled bound is approximately4.1e−13.

Consequently a blanket statement that this cutoff always lies below the1e−14 objective tie tolerance would be false. Direct objective evaluation across every feasible face mitigates selection error, but rank truncation still changes the finite-precision minimum-norm/tie interpretation near the threshold. Near an exact degenerate problem, the weights can move substantially while probabilities remain extremely close. That distinction should be reported without a claim of exact-rank recovery or scientific effect.

For the same local perturbation, each raw gradient component moves by at most tau; free-level subtraction moves its residual by at most2tau. Dividing by scale gives a conservative scaled residual perturbation of about2.9e−13. This is far below1e−10 KKT tolerance, which explains why those checks can accept a machine-rank approximation while not proving the exact weight tie. Existing objective/tie qualification remains relevant.

## Bounded handoff

Root will collect the exact failure witness without scores, and qualify the actual new projection function on witness and known/reference cases. The affected-only qualification can cover identical/duplicated members, constrained minimum-norm ties, near-identical members, target outside the dense hull, k=1 faces, and singular values around the fixed cutoff. These are the numerical cases directly implicated by the repair; this review neither executes them nor adds unrelated checks. Prior head, sparse and ridge-moment qualification can be reused through exact unchanged-source proof. No repeated150-update engineering fit, literature expansion, original acquisition/replay or scientific-arm change is called for.

After the witness and sealed V3 are available, the remaining source audit is the bounded solver/projection delta: exact fixed cutoff, Helmert basis and face construction, stable objective, scaled/raw KKT, cleanup/minimum-norm/face ties, universal operator9 call sites, witness/extractor scope, and unchanged nonprojection source. This note supports the formulation and documents its numerical limits; it does not authorize execution.
