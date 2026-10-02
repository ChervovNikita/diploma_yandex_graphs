# Post-closure diagnostic from preserved artifacts

This is a fixed descriptive analysis plan written from source before inspecting running-study outcomes. Execute only after the **whole registered 72-phase/30-fit cohort closes** and root admits analysis. It creates no arm, continuation run, new JVP/forward, checkpoint replay, score selection or final-label read. No diagnostic in this file has been computed. Use registered source TRAIN labels only for the signed mechanism calculation; final-pool labels and final report remain outside this plan.

## A. What is and is not preserved

The initializer computes four class-centered TRAIN JVP tensors `F_b = Gamma J t_b`, then discards them. `INITIALIZATION.json.method` preserves six `source_tangent_pair_rms`, accepted alpha/branch, `common_gradient_squared_norm`, `common_scalar`, algebra errors and every line-search trial's six finite pair RMS and training member/pool CE. The driver saves full warm logits, initialized member logits and installed factor slices. Continuation saves TRAIN/validation scalar traces per update and full member logits only at the declared native midpoint (or an absence reason) and the selected endpoint. There is no dense per-update prediction trajectory and no saved graph-cotangent/JVP vector. Sources: S6–S8 in the derivation.

Do not infer complete JVP orientation from pair distances. Do not run the model to fill missing timepoints. Retain common/unchanged fallbacks, failures and missing midpoint reasons rather than filtering them from a utility denominator.

## B. Recover the JVP route Gram from six scalar distances

Let `m = n*C`, `H4 = I4 - 11^T/4`. The stored order is `(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)`. Form the symmetric squared-distance matrix `D_J^2` from squared RMS values and zero diagonal. Then

`G_J = -0.5 H4 D_J^2 H4`

is the route-centered Gram `H4 F F^T H4 / m`. In ideal route centering it equals `F F^T/m`. Squared Euclidean distances determine this Gram even though they do not determine orientation relative to training errors, classes or examples. Preserve full precision and report small numerical negative eigenvalues rather than silently replacing the matrix with a fitted positive matrix.

From the accepted line-search pair RMS values similarly obtain `G_init`. Compare `G_init` with `alpha^2 G_J`, and separately report all six

`finite_pair_RMS / (alpha * JVP_pair_RMS)`

when the denominator is nonzero. For every preserved finite graph trial, report the same quantities at its own alpha. Common-only/unchanged branches have no installed graph contrast and are marked inapplicable; their attempted JVP geometry may still be shown as attempted, not installed.

Under the smooth bound in (4), ideal pair errors satisfy

`|finite_pair_RMS/alpha - JVP_pair_RMS| <= B_z alpha (||d_i||^2+||d_j||^2)/(2 sqrt(m))`.

If `W=H4 Gamma Z_init` and `E_W=W+alpha H4 F`, then

`||E_W||_F <= (B_z alpha^2/2) (sum_b ||d_b||^4)^(1/2)`,

`||G_init-alpha^2 G_J||_F <= [2 alpha ||H4 F||_F ||E_W||_F + ||E_W||_F^2]/m`.

These are conditional bounds, not calibrated acceptance thresholds: no Hessian bound was measured. Large mismatch limits the finite first-order account. Gram agreement alone cannot establish the desired sign or graph orientation. It can even hold after a sign reversal.

## C. Test the signed graph-error response without new JVPs

After the closure/admission above, use the already preserved warm logits and exact frozen source TRAIN rows/labels to reconstruct `r=(softmax(z_warm)-Y)/n`. Use the registered existing topology to apply the same fixed cubic filters and TRAIN mask, forming `q_b=(E^T H_b E-I/4)r`. This is arithmetic on preserved source artifacts; it needs no model or checkpoint. It is not performed during this review.

For every accepted graph candidate, compute

`Psi_init = sum_b <q_b, Gamma(z_init,b-z_warm)>`

`S_init = -Psi_init/alpha`.

Because `sum_b q_b=0`, the same value is obtained by subtracting the initialized route mean instead of the warm logits. Under the exact, smooth, nondegenerate interpretation,

`S_init = rho A_* + O(alpha)`

with absolute error at most `(B_z alpha/2) sum_b ||q_b|| ||d_b||^2`, and the ideal value is available from recorded source scalars:

`rho A_* = rho^2/lambda = 0.25*common_gradient_squared_norm/common_scalar`.

The last identity assumes the normalization floor is inactive, the branch is the actual graph candidate, and recorded finite algebra errors are negligible relative to the effect. Report the algebra errors and do not turn this formula into an exact machine certificate. A nonnegative `Psi_init` means the **finite** installed predictions do not reduce the intended aggregate detached graph contrast, even when the ideal derivative is negative. This would refute a claim of finite realization at that step; it does not refute the conditional linear characterization.

For the topology-permuted arm, evaluate both its own saved-permutation cotangents and the original-topology cotangents. For random, common-only and warm-copy arms, evaluate the same original-topology functional as a descriptive comparison. Use each actual alpha and branch; never impute a common alpha or silently compare attempted graph tangents with installed common factors. Near-zero alpha, zero `common_scalar`, unresolved raw tangent norm and fallbacks receive explicit inapplicability reasons for derivative-ratio calculations. The raw signed `Psi` can still be reported when defined.

The ideal graph direction optimizes this very functional by construction, so seeing a favorable local graph score is not independent evidence of task utility. The useful check is whether that orientation is actually realized at the accepted finite point rather than being swamped by nonlinear/precision effects.

## D. Measure retention, not a persistent training guarantee

At initialized, frozen native midpoint and selected deterministic predictions, set `W_t,b = Gamma(z_t,b-mean_m z_t,m)` and report

`orientation_t = -sum_b <q_b,W_t,b> / (||Q||_F ||W_t||_F)`,

where `Q` stacks the four fixed warm `q_b`. Report undefined values for zero denominator. Also report `||W_t||_F` and signed numerator separately. Use the same fixed route/band correspondence; do not relabel members to maximize alignment. Repeat original-topology scores across all arms and distinguish the permuted arm's own cotangents.

Loss of orientation at the frozen midpoint would undermine a claim that graph-error-directed route specialization persists under the actual continuation. It would not rule out every transient effect of initialization. A selected timepoint uses the existing validation selector and is not an independent confirmatory observation; a missing midpoint stays missing. Mean member CE and independent dropout do not enforce retention of the initial projected constraints.

## E. What these artifacts cannot isolate

- The stored pair summaries omit JVP orientation, and neither `Jg` nor common-step logits at the graph alpha are generally saved. Thus they cannot independently verify the same-alpha pooled `O(alpha^2)` comparison when the actual arms choose different alpha values.
- Current random and graph arms match parameter tangent norm only. Differences in source Gram, maximum direction norm, accepted alpha, branch and nonlinear spread remain part of the complete operation. These diagnostics do not replace the previously proposed prospective same-alpha Gram-matched control.
- Initialization graph alignment and later retention do not imply helpful decisions or calibrated uncertainty. Utility must be assessed through the already registered full-cohort task contrasts and charged costs, under root's separate comparison/report authority. No final labels are needed for the mechanism checks above.
- No result from this previously exposed, outcome-aware small cohort would establish industrial-scale, universal or confirmatory superiority.

## F. Refutation table

| Claim being assessed | Evidence against it | Boundary of conclusion |
|---|---|---|
| Code realizes the ideal centered, gradient-orthogonal construction | Recorded algebra/partition violations beyond admitted tolerances, wrong installed-slice correspondence or unresolved numerical degeneracy | Runtime/source correspondence; do not call a tolerance-bounded check exact |
| First-order prediction geometry is a useful approximation at the installed step | Finite Gram/pair behavior substantially departs from the saved JVP geometry | Limits finite linear interpretation; no measured Hessian constant supplies a theorem threshold |
| Installed graph perturbation reduces its intended detached graph-error contrast | `Psi_init >= 0`, or a signed realization ratio inconsistent with the ideal coefficient at the installed step | Refutes finite realization at that step, not the conditional derivative formula |
| Graph-oriented specialization persists through continuation | Initial orientation vanishes/reverses at the frozen midpoint while spread remains, or is indistinguishable from the descriptive controls | Weakens persistence account; cannot rule out every transient path effect |
| The complete initializer is practically useful beyond warming and generic spread | The already registered graph/common/random/permuted/warm contrasts show no meaningful beneficial task effect, or fallback/cost burden defeats it | Exploratory utility assessment of the complete operation; current controls do not isolate orientation alone |

No thresholds, selections, arm modifications or scoring rules are inferred from unseen results. Report these as descriptive mechanism diagnostics with their explicit limitations.
