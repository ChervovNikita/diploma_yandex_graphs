# Counterfactual marginal value of private learning: design synthesis

**No distinct, useful successor is established by the assessed definition.** Its natural differentiable forms either recover served-loss meta-learning, combine ordinary adaptation benefit with loss-matched ambiguity, or isolate a label-independent interaction that can improve while served predictions remain wrong and unchanged. This is a mathematical rejection of the proposed *additional credit mechanism*, not a claim that every nonlinear counterfactual objective is impossible or that attributed objective variants cannot help empirically.

This synthesis reuses the quality triage, method decision, saved correction-credit assessment, GNCL/loss-matched theory, DICE/FoRDE specification, BMAML/ANIL ancestry and the scoped MCL/COMA/DRPG team-credit assessment. **New primary papers: 0. Primary rereads: 0. Full-paper reads: 0.** Source identities, previously inspected scopes and hashes are bound in `SOURCE_BINDINGS.json`. No model, outcome payload, server, new bibliography, source edit or frozen-pilot change is involved.

## 1. Define the counterfactual precisely

Hold the episode queries, support graph, outer evaluation map, private starting parameters/moments and inner randomness fixed. Let

\[
z_m^0=f(\theta,\phi_m),\quad
z_m^1=f(\theta,U_m(\theta;\phi_m,s_m,I_m)),\quad
d_m=z_m^1-z_m^0,
\]

\[
z^0=M^{-1}\sum_mz_m^0,\qquad
z^1=M^{-1}\sum_mz_m^1,\qquad
\bar d=z^1-z^0.
\]

Every counterfactual retains all M predictor slots. Freezing only update m gives `z^{-m}=z^1-d_m/M`. It does **not** remove member m, renormalize by M−1, refit other routes, change the support or commit a counterfactual optimizer state. Its terminal marginal credit is

\[
c_m=\ell_y(z^{-m})-\ell_y(z^1),\qquad C=M^{-1}\sum_mc_m.
\]

All following identities are pointwise and extend to the same fixed nonnegative query average. They therefore do not require independent graph queries. They do require the stated additive raw-logit pool and route-local private updates. Endpoint conditioning cannot change the identities at fixed episode inputs.

The intended failure mechanism is credible as a hypothesis: an update may improve a route's own inner loss while contributing little or causing harm to the ensemble's remaining outer residual. But the standard derivative of **served outer loss through that update already measures its local contribution**. The question is whether the counterfactual adds something beyond that derivative.

## 2. Routing marginal update credit recovers the existing meta derivative

For the derivative with respect to the adapted private block `phi'_m`, the frozen-update counterfactual does not contain that block. With peers and starting state held fixed,

\[
\nabla_{\phi'_m}c_m=-\nabla_{\phi'_m}L_{\rm serve}^1.
\]

Consequently, maximizing each route's credit only through its own update sensitivity gives

\[
-\sum_m(D_\theta\phi'_m)^T\nabla_{\phi'_m}c_m
=\sum_m(D_\theta\phi'_m)^T\nabla_{\phi'_m}L_{\rm serve}^1.
\]

This is exactly the private-learning credit term of ordinary served-loss bilevel meta-learning, including the live optimizer transformation. Adding its direct shared derivative recovers that complete meta-gradient. A mean rather than a sum introduces a scalar rescaling; it is not an identical native Adam transition without matching normalization and state conventions.

The frozen rule already includes this served-loss pathway with half weight. Its other half superposes member-competence gradients. Removing that anchor or increasing served-loss weight changes a known loss tradeoff; it does not introduce marginal credit that was previously absent. Fully differentiating all counterfactual shared and peer paths instead gives the different scalar objective below, so that implementation must not be conflated with routed credit.

## 3. Full differentiation gives adaptation benefit plus ambiguity

For CE/BCE on raw logits, write `ell_y(z)=A(z)-y^T z`, with `A=logsumexp` for class logits and `A=softplus` for binary logits. The counterfactual centroid is

\[
z^c=M^{-1}\sum_mz^{-m}=z^1-\bar d/M.
\]

Therefore, exactly,

\[
C=\underbrace{\ell_y(z^c)-\ell_y(z^1)}_{\text{partial pre/post adaptation benefit}}
+\underbrace{M^{-1}\sum_mA(z^{-m})-A(z^c)}_{D_{\rm cf}\ge0}.
\]

Unlike normalized **member removal** in the saved correction scout, update-freezing credit need not lose all label dependence: the two centroids differ. This distinction matters. However, the surviving label term is ordinary improvement along the aggregate adaptation displacement; the extra symmetric component is a Jensen ambiguity of counterfactual outputs. GNCL and loss-matched ensemble theory already establish that objective family. This is not literally GNCL Eq.5 on the original member outputs for arbitrary M and nonlinear update maps.

For small update-induced displacements,

\[
C=-M^{-1}\nabla\ell_y(z^1)^T\bar d
+(2M^2)^{-1}M^{-1}\sum_md_m^T\nabla^2A(z^1)d_m
+O(\max_m\|d_m\|^3).
\]

The first-order credit is simply scaled aggregate adaptation improvement. The remaining curvature term rewards update dispersion, not necessarily useful served corrections. A candidate objective `J=L_serve^1-lambda*C` is thus an attributed adaptation/ambiguity tradeoff, with a known risk of improving the reward by worsening counterfactuals. Adding a competence anchor does not change this identification.

## 4. Isolating the ensemble interaction restores label cancellation

Subtract update m's benefit when no other update has occurred:

\[
b_m=\ell_y(z^0)-\ell_y(z^0+d_m/M),\qquad i_m=c_m-b_m.
\]

Then

\[
i_m=A(z^1-d_m/M)-A(z^1)-A(z^0)+A(z^0+d_m/M).
\]

Its explicit outer-label term cancels exactly. Inner labels still affect the learned displacements; the statement is label independence **at fixed outputs/displacements**, not label-free training. For squared loss `ell(z)=0.5*||z-y||^2`, this reduces exactly to

\[
i_m=-M^{-2}d_m^T\sum_{j\ne m}d_j.
\]

For CE/BCE the small-displacement leading term is the same negative cross-correlation in the loss-curvature metric. The exact finite-displacement form is a convex interaction objective, not a claim of literal equality to every original-route NCL implementation.

A symmetric coalition alternative also has no new aggregate credit. Define `z^S=z^0+M^{-1}sum_(m in S)d_m` and `v(S)=ell_y(z^0)-ell_y(z^S)`. Along any ordering of updates, incremental credits telescope to `v(all)`. Averaging those orderings gives the same sum for Shapley-style credits: `sum_m credit_m=L_serve^0-L_serve^1`. Its symmetric optimization is pre/post adaptation meta-learning; stopping the baseline leaves ordinary post-adaptation loss up to scale. This is a derivation, not a newly inspected Shapley paper. Individual coalition credit assigned to different parameter blocks may define a different field; it requires its own complete routing and utility argument and is not established by the symmetric aggregate.

## 5. Two concrete exclusions

**A credit reward can rise with no served gain.** Take binary y=1, M=2, equal initial logits b<0, and private update displacements `d_1=a`, `d_2=-a`, a≠0. The adapted ensemble still serves b and remains on the wrong side of zero. Yet

\[
C=\tfrac12[A(b-a/2)+A(b+a/2)]-A(b)>0,
\qquad M^{-1}\sum_mi_m=2C>0.
\]

Both terminal credit and pure interaction reward increase from zero while the served logit remains exactly fixed. This is not merely a hypothetical output permutation: a scalar private head `b+phi*x`, initialized at phi=0, admits opposite one-step BCE/SGD updates from positive inner examples with features x and −x; an outer feature 1 produces the stated opposite displacements. It is an analytic learning-rule witness, not a native NCN/Adam qualification or measured graph result.

For single-query binary BCE, `sign(c_m)=sign((2y-1)*d_m)` because BCE is monotone in the true-label logit. Thus a per-query clipped harm gate can be a different scalar objective, but its sign test only asks whether that route's update moves its logit toward the label; peers affect magnitude, not sign. A query-aggregated gate need not share this simplification. Neither variant supplies a supported new ensemble-specific operator merely by adding clipping.

**A capable single can implement the whole construction.** One predictor with shared theta, a private block vector `(phi_1,...,phi_M)`, output `M^-1 sum_m f(theta,phi_m)`, and the same block optimizer/streams exactly reproduces all adapted and counterfactual outputs and gradients. This is an exact functional/update equivalence to a branched single with the same capacity and paid paths, not a claim that a smaller ordinary NCN single always matches it. At M=1, terminal credit is ordinary pre/post adaptation benefit and the interaction term is identically zero. Hence none of these objectives establishes a representational advantage over a capable single or isolates damage caused by parameter sharing.

## Prior boundary and design decision

| Retained source family | Conclusion for this proposed successor |
|---|---|
| ANIL/BMAML, graph-meta scopes and the current rule | Differentiating shared features through private learning and served ensemble loss is prior. Routed marginal update credit reproduces that chain-rule term. |
| GNCL and Wood et al. loss-matched diversity | Full symmetric terminal credit contains ordinary adaptation benefit plus counterfactual ambiguity. Pure interaction reduces locally to negative correlation and exactly to it for squared loss. |
| DICE and FoRDE | No literal equality: this objective neither estimates conditional feature redundancy nor uses an input-gradient particle kernel. Their non-equality does not clear the surviving meta-learning/ambiguity operator as novel. |
| Saved member-removal correction scout | Its complete label cancellation cannot be copied to update freezing. The new centroid calculation above supplies the correct boundary. |
| Scoped MCL/TreeNets and COMA/DRPG | Specialization and counterfactual/default-action team credit are established. A deterministic differentiable learner update is not a stochastic actor action, so no REINFORCE unbiased-baseline theorem is imported. |
| Learner collusion | Increasing a committee-oriented credit while weakening competence is a concrete risk. The analytic fixed-served-logit witness already shows why the credit itself cannot certify prediction quality. |

The smallest decisive screen for the proposed *new mechanism* is the algebra and two-member witness above; it needs no model fitting. A distinct useful successor does not survive this assessment, so no new representative experiment, coefficient, gate or implementation is promoted. Attributed objective utility remains an empirical possibility, but it is insufficient to promise superiority over capable singles and true independent ensembles or methodological novelty. The existing frozen study remains unchanged.

**Empirical facts used here:** none beyond the existence and definitions of retained methods and the frozen design. Earlier predictive successes/failures mentioned in input reports were not reopened or used to choose an objective. Every equality, limiting expansion and counterexample above is mathematical reasoning under its declared assumptions. This is not a global literature absence proof or a manuscript verdict.
