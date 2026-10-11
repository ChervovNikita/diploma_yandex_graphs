# Shared fast-factor draws induce conditional map dependence

**Defensible statement:** independent coordinates in a fast-factor prior do not imply independent incidence maps. Within one fixed component and layer, reusing one sampled factor vector in E's endpoint builder can correlate maps on distinct, even disjoint, edges after conditioning on their input contexts and fitted shared weights. This differs from BSNNv2's inspected product variational family. It is a known random-function consequence of the Rank-1 parameterization, not a new graph uncertainty principle or prediction guarantee.

This statement concerns the proposed stochastic port. Original point-factor E has no within-component factor posterior draws. No frozen proposal, acquisition source, data or paper was changed for this note.

## Exact law at the builder site

Fix a route/component, a layer, all incoming hidden states and one source or target builder. Let c_a be the ordered endpoint context at incidence/edge record a. The byte-pinned `FactorLinear.forward` puts its bias outside output scaling:

\[
g_a=D_s W D_r c_a+b.
\]

Here one r and one s are drawn once and reused for every record in that complete route. Assume independent r and s, independent coordinates within each, finite second moments, means mu_r/mu_s and diagonal covariances Sigma_r/Sigma_s. Define

\[
v_a=W D_{\mu_r}c_a,\qquad
K_{ab}=W D_{c_a}\Sigma_rD_{c_b}W^\top.
\]

Then the exact raw-builder cross covariance is

\[
\operatorname{Cov}(g_a,g_b)
=(\mu_s\mu_s^\top+\Sigma_s)\odot K_{ab}
+\Sigma_s\odot(v_av_b^\top),
\]

where the plus signs are ordinary addition and odot is entrywise multiplication. Bias is deterministic. This formula follows by expanding E[s_i s_j] E[(WD_r c_a)_i (WD_r c_b)_j] and subtracting the product of means. It needs no dependence between factor coordinates. A shared random coefficient can contribute to many incidence evaluations.

The law is indexed by endpoint features and fitted W. It is not a prescribed kernel of graph distance, adjacency, cycle consistency or node-frame transport. Zero/canceling contexts, zero weights or zero effective variation can make a particular cross covariance vanish. Compact latent dimension alone also does not certify low covariance rank after bilinear products and nonlinearities.

## Counterexample that survives E's tanh and paired transport

Consider two distinct nonself ordered edges with four distinct endpoints and identical fixed ordered hidden-state contexts. In E's d=4 source builder, select one context coordinate equal to one and one shared weight equal to one, with all other source weights/biases zero. Set the target builder weights/biases to zero. Let its active source input/output factors R and S be independent nondegenerate Gaussians, for example with means one and any positive scale. These are admissible parameter values; this is not a claim about actual WikiCS inputs or fitted states.

Both raw source maps have the same random (1,1) entry RS. Under the actual helper,

\[
F^{src}_a=F^{src}_b
=I+\tfrac12\tanh(RS)E_{11},\quad
F^{tgt}_a=F^{tgt}_b=I.
\]

Consequently both effective transports P=F_tgt^T F_src are identical and nonconstant, with

\[
\operatorname{Cov}((P_a)_{11},(P_b)_{11})
=\tfrac14\operatorname{Var}(\tanh(RS))>0.
\]

Identical nonconstant random variables cannot be independent. An incidence-product sampler has zero cross covariance between these two independently drawn map coordinates. Even hypothetically matching their one-map marginals would not match this joint law; BSNN's native Gaussian/Cayley marginal choices need not match this toy marginal.

The example is deliberately at the geometry/transport object **before message aggregation**. It does not infer map dependence from final-logit covariance: BSNN's independent geometry draws can already yield correlated node predictions through overlapping graph computations. Other sampled native factors and earlier layers can additionally randomize E's contexts, so the fixed-context formula must not be promoted to an unconditional whole-network covariance formula.

At E's actual zero-builder initialization, both W and b are zero and all maps equal I regardless of nonzero factor scales. The proposed scale values alone therefore do not even guarantee initial geometry variance. Self records bypass the VALUE transport; the counterexample uses nonself records.

## What the closest inspected sources already say

| Primary scope | Explicit fact and boundary |
| --- | --- |
| [Rank-1 BNN, arXiv2005.07186](https://arxiv.org/abs/2005.07186), saved §§3.1–3.2, Eq1–3; blocks28–33/39–46 | Places distributions on rank-one factors around deterministic W and describes factors shared across rows/columns. The displayed edge-map covariance above is not printed in this inspected scope; it is the elementary push-forward when that random map is evaluated repeatedly. Block33 also permits per-example factor sampling: that can estimate marginal training expectations, but independent draws per edge would define a different joint geometry sampler. The saved carrier's precise ar5iv revision is unqualified. |
| [MPNP, arXiv2009.13895](https://arxiv.org/abs/2009.13895), saved Aggregation/Decoder/Generation, blocks36/38/40–44, Eq4 | Explicitly samples one global z and concatenates the same sample with each target; Eq4 is conditionally factorized given z. Marginalizing z already supplies dependent graph predictions from independent latent coordinates. It does not state this diagonal-builder geometry law or use E's factor placement. |
| [BSNNv2, arXiv2410.09590v2](https://arxiv.org/html/2410.09590v2#S3.SS2), §3.2 S3.SS2.p1–p2/Eq21; §3.4 p2–p6 | Explicitly factorizes q over incidences and predicts their parameters with a shared endpoint MLP; samples iid sheaves one per layer and ensembles predictive forwards. Fixed shared learner parameters are not sampled global factor coordinates. The stated product q cannot realize the counterexample's nondegenerate identical maps. This is a variational-family comparison, not a claim about its exact posterior or independent predictions. |
| [BGCN, arXiv1811.11103v1](https://arxiv.org/html/1811.11103v1), saved §4/Eq5–6/Algorithm1 conclusions only | Already integrates sampled graphs and graph-conditioned weight uncertainty, with MAP/community and MC-dropout approximations. Broad graph/model marginalization is occupied. The saved scope does not establish this particular builder law; no new primary reread or exact-law attribution to BGCN is made. |

Thus the useful distinction is **where the product assumption lives**: fast coordinates of one reused random function versus separate incidence maps. Calling it graph-level correlation describes the indexed push-forward, but its cause remains ordinary random-function covariance. Placement inside geometry gives a precise comparator interpretation; it supplies no additional acquisition rule, calibrated posterior claim, member competence result or new contribution beyond the attributed Rank-1 port. No further source search was needed and no broad novelty/absence clearance is issued.

## Learned sigmas and the shared-weight gauge

For arbitrary positive diagonal scalings A and B, apply the same scalings to every route sharing this builder and transform

\[
r'=Ar,\quad s'=Bs,\quad W'=B^{-1}WA^{-1},\quad b'=b.
\]

This preserves every raw builder evaluation pathwise. Scaling factor means and standard deviations by the same A/B therefore changes raw sigma values arbitrarily while preserving the entire conditional map/transport law, including tanh. It is an internal parameter gauge, with no change of graph coordinates required. Even without a full gauge transformation, an output scale's effect depends on W and its current input; a large sigma beside zero W has no geometry effect.

If q **and** its prior p are pushed forward consistently, KL is invariant under this invertible change of variables. The frozen proposal instead fixes prior centres/scales in the original native coordinates; that KL is generally changed by the rescaling. Its fixed prior breaks the exact full-objective symmetry and imposes a coordinate convention. It does not make raw fitted sigmas a parameterization-independent measure of epistemic or geometry uncertainty, and no equivalent AdamW training trajectory follows.

Interpretable quantities require the actual push-forward to raw maps, effective paired transports or predictions under a stated fitted parameterization/context. This note recommends no extra diagnostic or experiment. It only limits future sigma/geometry wording while preserving the frozen comparison.

## Scope

Reused saved primary custody and revisited only the decisive Rank-1, MPNP and BSNN passages above; BGCN is saved conclusion reuse. Read the established FactorLinear and E helper/initialization as text. No new primary identity, public retrieval, full paper, proof/code audit, numerical computation, model/data access, server, reviewer or source/proposal/paper edit occurred. The equations and counterexample are analytic arguments, not a scientific result from current runs.
