# The sealed label-only route as a class-compatibility operator

**For root, 8 October 2026.** Analytic interpretation of the sealed core only. No model execution, outcomes, source change, new retrieval, initializer selection or modification of the frozen screen.

## What each route computes

Each route forms a feature-conditioned weighted histogram of the visible neighbors' TRAIN labels, then applies a learned class-to-class logit correction. It does not send raw feature values or label-dependent scores through this hop.

Use column vectors and write `D(a)` for a diagonal matrix. The source `FactorLinear` computes the column map `D(s_m) W D(r_m) x`. Let:

- `E[C,L]` be the shared label embedding table, with `L=64`; `Eᵀ e_c` embeds anchor class c.
- `W_V[L,L]` and `W_O[C,L]` be the shared bias-free value and output weights.
- `rV_m,sV_m,rO_m,sO_m` be their private input/output factor rows.

The exact effective **class-compatibility matrix** is

`T_m = D(sO_m) W_O D(rO_m ⊙ sV_m) W_V D(rV_m) Eᵀ`,

with shape `C×C`: output classes index rows, anchor-label classes index columns. The merged middle diagonal follows because the value-output factors commute. Source row-vector arithmetic is the transpose of this expression.

For a target i, define

`g_m(i,c) = b × sum_{j→i, nonself} alpha_m(i,j) × 1[j is a visible TRAIN anchor] × 1[y_j=c]`.

Here `b=579/290` during the frozen common-Q training operation and `b=1` during legal inference. Attention `alpha_m` uses detached native H through factorized feature-only Q/K maps, and softmax normalization includes **all incoming nonself edge records**, not just labeled neighbors. Duplicate records retain their multiplicity.

Then the entire correction and served member are

`delta_m(i) = T_m g_m(i)`,

`p_m(i) = softmax(z0(i) + T_m g_m(i))`.

Thus T_m is global within a route at a fixed learned state; its feature dependence enters through the weighted histogram g_m. It is an additive **logit** compatibility matrix, not a row-stochastic transition, conditional probability or Bayesian posterior. Its entries can be negative. The histogram need not sum to one: at inference its mass is the attention assigned to allowed labeled neighbors, at most one. Training inverse scaling can raise that mass above one.

If there are no visible neighboring label values, `g_m=0` and the correction is exactly zero. Attention changes within one anchor class can cancel completely in this histogram. More generally, two label neighborhoods with identical g_m are indistinguishable to this value path, even if their individual source-node weights differ. Nonlinear dependence on graph features remains in native H and the attention softmax; label-value aggregation/output is linear.

## Latent label separation is not identifiable from the prediction

Let `A=D(a)` and `B=D(b)` be any common invertible diagonal matrices in the 64-dimensional latent spaces. Change the shared parameters by

`E' = E A`,

`W_V' = B W_V A⁻¹`,

`W_O' = W_O B⁻¹`,

and leave every private factor and Q/K parameter unchanged. Substitution gives

`T'_m = D(sO_m) W_O B⁻¹ D(rO_m⊙sV_m) B W_V A⁻¹ D(rV_m) A Eᵀ = T_m`.

The equality uses commutation of diagonal matrices. It holds simultaneously for every route and permitted label field in real arithmetic. Common rescaling can change label-embedding norms, pairwise distances and, with anisotropic scaling, angles; B can likewise change route-message geometry. Predictions remain unchanged. Linear class centering does not remove this coordinate freedom. Arbitrary dense basis changes are **not** claimed: they need not commute with the private diagonals.

This is a parameterization ambiguity, not a new theorem or a claim that Adam/regularizers/training trajectories are invariant under the transformation. No transformed model or numerical comparison was constructed. At the sealed zero-output initialization, every T_m is zero regardless of the nonzero embedding table and differing Q/K starts; latent or attention differences then cannot alter the base prediction.

## Decision-visible quantities for later error diagnosis

Raw T_m differences are insufficient too: a class-common logit shift does not change softmax. Use output row-centering `J=I−11ᵀ/C`, or directly compare class-margin rows `T_m[y,:]−T_m[c,:]`. The observable correction is `J T_m g_m(i)`.

For two routes m and l, the exact descriptive decomposition

`J(delta_m−delta_l) = J(T_m−T_l)g_l + J T_m(g_m−g_l)`

separates a compatibility-map difference from a weighted-label-context difference using l as a fixed algebraic reference. It is not a causal attribution: all matrices and attention paths learn together.

After the **whole frozen family** closes, a compact diagnosis can retain:

1. Each route's row-centered T_m and actual labeled attention mass/histogram.
2. Route histogram differences, including cases where different edge attention gives identical class histograms.
3. Actual correction margins against each target's rival, base margins, repaired predictions and new errors.
4. Cases where histogram differences lie in the decision-null directions of T_m, or T_m differences act only as common logit shifts.

No latent-distance score can replace pooled/member accuracy, NLL or full-population repairs/harms. Structural or error cohorts remain descriptive and cannot rescue the frozen gate.

The earlier Wiki15 report measured nearly unanimous **decisions** and little pooled rescue. It did not measure latent collapse or establish this label-space ambiguity as the cause; those earlier routes were a different architecture. The current corrector's trained decisions/outcomes are not opened by this note.

## Scope of any later class-basis idea

Only after complete results and the above diagnosis could a separately attributed class-space parameterization or regularization be worth considering—for example a fixed one-hot label basis when `C<=L`, or a constraint expressed on decision-visible T_m rather than arbitrary latent distances. Such a change affects parameterization, private-factor interpretation and optimization; whole-family capacity equivalence would need checking. It is not cleared novelty, a selected initializer, a current rescue or an admitted experiment. No coefficient/grid or source modification is proposed.

UniMP, C&S and GMNN remain the known label-input/correction ancestry. Root's saved UniMP author-code scope confirms its original label-conditioned scores/values, unscaled masking and different operator; this sealed feature-only-score route is not a source-faithful UniMP reproduction. No new primary reading credit, absence claim or quality guarantee follows from this interpretation.
