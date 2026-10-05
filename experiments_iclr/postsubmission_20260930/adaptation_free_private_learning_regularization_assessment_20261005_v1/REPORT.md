# Private-learning lookahead for adaptation-free ensemble inference

5 October 2026. Source and saved-literature assessment. No model, sampler, numerical diagnostic, checkpoint, prediction or outcome access; no server contact. The fixed 30+9 study, selectors and quality/attribution gates are unchanged. The adopted screening correction is followed: finite-sample or optimization benefit can be a legitimate contribution without a predictor that a branched single cannot represent or a new primitive.

**The coherent explanation is training-time lookahead for a private update that actually commits.** Recomputing that update after the shared step gives an exact same-episode committed-state objective identity. It removes a logical mismatch between virtual learning and adaptation-free serving. It establishes neither Adam descent nor generalization, ranking improvement, useful specialization or novelty.

## 1. Exact state and source binding

For episode t, freeze the old shared state theta, old private rows Phi, old private Adam moments/step counts s, old shared Adam state sigma, graph/features, masked support S, all inner/outer query identities, and each inner dropout stream's starting state. The private losses use their separately normalized positive and negative means; outer evaluation is dropout-off. Shared theta includes the encoder and dense head bases/biases; F4 private rows contain r/s, LayerNorm affine coordinates and beta. Neither old private parameters nor old moments are differentiated through earlier episodes.

| Bound source | Inspected lines | Semantics used |
| --- | --- | --- |
| `shared_backbone_private_transfer_training_source_20261005_v2/private_adam.py` | 18–63, 72–94 | Native functional Adam response, old moments detached, exact zero-history branch and fixed-history Jacobian |
| Same folder, `transfer_step.py` | 6–35, 44–71, 76–121, 124–171 | Fixed loss normalization, mask replay, virtual private response, live/detached shared derivative, native shared commit, recomputed private commit from old parameters/moments |
| Same folder, `models.py` | 129–157; retained partition audit | Exhaustive roles, constant native buffers and richer capable-single role difference |
| Same folder, `run.py` | 111–152 | Native shared Adam constants, NCN/JK architecture, committed-state mean-logit serving and full-TRAIN-support VALID inference |

This mathematical identity concerns `commit="recomputed"`. The optional stale-virtual source branch is not the active frozen cohort. Exact here means equality of the declared maps with the same inputs and random realizations; no new floating-point parity measurement is claimed.

## 2. Exact committed-state lookahead identity

Write each actual private response as

\[
g_m(\vartheta)=\nabla_{\phi_m}L_{I,m,S}(\vartheta,\phi_m),\qquad
\delta_m(\vartheta)=A_{s_m}(g_m(\vartheta)).
\]

Define, with everything except the candidate shared parameters vartheta fixed,

\[
J_t(\vartheta)=F_{O,S}\bigl(\vartheta,\Phi+\delta(\vartheta)\bigr).
\]

The source evaluates J_t(theta), commits one native shared Adam displacement u, then recomputes the private gradients at theta+u using the old private parameters/moments and the same masks. Consequently,

\[
\theta^+=\theta+u,\quad
\Phi^+=\Phi+\delta(\theta^+),\quad
\boxed{F_{O,S}(\theta^+,\Phi^+)=J_t(\theta^+).}
\]

The committed new private moments also come from this recomputation; they become next episode's frozen history. Virtual moments are discarded. The identity holds for live and detached rules at a common starting state, although their shared displacements generally differ. The saved scalar `virtual_outer_objective` is J_t(theta), not J_t(theta+u); no post-commit outer evaluation is retained by that scalar.

Thus the differentiated response anticipates the training-time state that will be committed. The deployed predictor uses already committed private weights and needs no test-time update. This is a short lookahead with persistent state, not differentiation through the full optimizer history. Outer labels influence the shared block; they do not directly commit a private gradient.

The equality concerns this episode's TRAIN outer queries on S. VALID uses full TRAIN support, different queries and fixed ranking negatives. It does not identify J_t with the final full-support served risk.

## 3. Exact credit and actual Adam movement

At the virtual state, let d be the partial shared derivative with adapted private rows held constant. With a_m the outer private derivative there, the live derivative is exactly

\[
\nabla J_t=d+c,\qquad
c=\sum_m(D_\theta g_m)^T(D_gA_{s_m})^Ta_m.
\]

Detached uses d at the same virtual parameter values. Let U_sigma be the actual native shared Adam displacement map at the same old shared state. Then the same-state interventions are

\[
u_l=U_\sigma(d+c),\qquad u_d=U_\sigma(d).
\]

Neither u_l=-alpha(d+c) nor u_l-u_d=-alpha c is assumed. Momentum, second moments, bias correction and epsilon matter in both maps. In the first zero-history private step,

\[
A_0(g)=-\eta g/(|g|+\epsilon),\qquad
D_gA_0=-\eta\epsilon/(|g|+\epsilon)^2.
\]

The derivative at zero is finite; no everywhere second-differentiability claim follows. Away from zero, sign-like normalization may make sensitivity small, but no magnitude of c is inferred. In general separate responses, histories and nonlinear predictions do not collapse to Adam on the mean gradient. That distinction supplies no favorable sign or need for four members.

## 4. Restrained explanatory expansions

### Response alignment: exact identity, then approximation

For fixed theta and delta=A_s(g), if the outer loss is absolutely continuous along the private segment,

\[
J_t(\theta)-F_{O,S}(\theta,\Phi)
=\int_0^1\nabla_\Phi F_{O,S}(\theta,\Phi+\tau\delta)^T\delta\,d\tau.
\]

If the private gradient is L_phi-Lipschitz on that segment,

\[
J_t(\theta)=F_{O,S}(\theta,\Phi)
+\nabla_\Phi F_{O,S}(\theta,\Phi)^T\delta+r,
\quad |r|\le\tfrac12 L_\phi\|\delta\|^2.
\]

With the additional local smoothness needed for second-order Taylor expansion, the next term is one half delta^T H_PhiPhi F delta. This curvature term need not be nonnegative in parameter space. ReLU/other activation boundaries preclude a global smoothness assertion; the line-integral statement only requires the indicated path regularity, while Taylor claims require their extra local conditions.

The alignment is with the actual Adam response, not a raw inner gradient. The classical small-SGD dot-product expansion is already MLDG ancestry; it cannot be substituted for this source. The response-dependent addition can have either sign and is not a fixed nonnegative norm penalty. “Regularization” here describes an induced training bias toward features and bases whose private learning transfers to other queries.

### Shared movement: conditional local progress only

If J_t has an L_theta-Lipschitz gradient on the relevant shared segments, then

\[
J_t(\theta+u)=J_t(\theta)+(d+c)^Tu+r_u,
\quad |r_u|\le\tfrac12L_\theta\|u\|^2.
\]

An explicit sufficient condition for live to improve the same-state committed surrogate over detached is

\[
(d+c)^T(u_l-u_d)
+\tfrac12L_\theta(\|u_l\|^2+\|u_d\|^2)<0.
\]

This is conditional standard Taylor reasoning, not a certified condition of the current network, a new theorem or an added gate. Native Adam can move uphill because of its history. A nonzero c, a negative raw gradient dot, or the direct-component projection d^T(u_l-u_d) alone establishes no decrease of J_t. The plan's exact discarded-copy loss comparisons can assess local full-rule effects if separately authorized; the source-only assessment computes none.

## 5. The finite-sample/optimization hypothesis still needed

A precise useful hypothesis is: **under the fixed TRAIN episode distribution and native Adam histories, credit through private learning makes the returned committed predictor generalize better because it favors private corrections carrying transferable signal over corrections fitting endpoint-local noise, and sharing reduces estimation/optimization error enough to offset its bias.** A coherent account needs the following empirical premises:

1. **Transferable signal:** inner queries contain label-relevant corrections that help different outer queries after the declared masked support and dropout construction. Endpoint separation must retain enough useful information; it does not itself make the queries independent. Old private histories may already contain labels involving the current outer endpoints.
2. **Influential, useful movement:** the mixed credit changes the actual native shared displacement in a useful direction and the recomputed private step preserves that effect. Neither a correct derivative nor a large norm supplies this premise. The exact identity in Section 2 aligns the current commitment; longer-term trajectory benefit remains empirical.
3. **Estimation benefit from the chosen coupling:** finite graph supervision makes unrestricted private evidence noisy enough that shared bases/features plus limited private rows improve the bias/variance or optimization tradeoff. The retained shrinkage example illustrates this possibility under explicit linear/second-moment assumptions. It is not an Adam, GNN or MRR result; a capable regularized single can share that bias.
4. **Serving and metric transfer:** useful behavior survives full-support inference, the absence of current query adaptation, later persistent updates and checkpoint selection, and improves the frozen ranking metric. BCE progress on masked TRAIN outer queries does not guarantee MRR/Hits improvement.

Training removes the union of positive target edges for both paired geometries but leaves common graph context and nodes. VALID restores full TRAIN support. Mask-induced neighborhood/common-neighbor changes therefore condition the hypothesis and can make its usefulness dataset dependent. Root's recent Pubmed TRAIN census is design evidence, not accuracy evidence; it adds no inference-risk estimate or gate here.

No unbiased cross-fitting claim, independent-task interpretation, effective sample size, variance reduction magnitude or generalization bound follows from endpoint eligibility. No premise demands that the function be unrepresentable by a branched single. Actual incremental quality over competent distinct training procedures remains required.

## 6. Fixed outer competence pressure

With the source's separate positive/negative normalization, let F_pool be BCE of mean raw logits, F_member the mean member BCE, and G=F_member-F_pool. Convexity gives the exact identity

\[
F_O=\tfrac12F_{pool}+\tfrac12F_{member}
=F_{pool}+\tfrac12G,\qquad G\ge0.
\]

The objective presses members to be competent and penalizes the Jensen gap. It supplies no explicit reward for disagreement or complementary errors. Complementarity can emerge through the private states, but parameter/logit dispersion and negative gradient dots are insufficient evidence. The retained squared-error pooling identity depends on error cross-moments and supplies no ranking guarantee.

## 7. What the unchanged controls can resolve

| Fixed comparison | Claim it can inform | Remaining limitation |
| --- | --- | --- |
| F4 live versus detached; endpoint versus matched-random interaction | Whether response credit and eligibility affect the trained served predictor | Trained states diverge; not the same-state identity; similar point estimates do not prove equivalence or a generic cause |
| Capable single live/detached/joint | Competent quality alternative and generic private-learning explanation | Dense head roles, private capacity, label access and histories differ from F4 |
| Row0 F1 endpoint live/detached/joint | Credit/member-count behavior with matched shared/private roles and four streams | Row count, initial pool, separate histories and ensemble loss change; no F1 random cells, hence no count × geometry × credit interaction |
| Initially matched untied four live; independent native four and retained true independent anchor | Whether the shared recipe provides quality beyond distinct competent ensembles | No untied factorial; jointly trained native four is not independently trained ensembling; original anchor budgets differ |
| Paid F4 joint and other ordinary controls | Value of the complete learning rule versus paid ordinary training | Three joint Adam commits differ from one shared plus one recomputed private commit |

The fixed diagnostics already separate selected-state competence/complementarity, exposure and later optional TRAIN response probes. Saved scalar histories cannot reconstruct historical mixed credit or post-commit outer losses. No logging, fit, probe, threshold or comparator is added. Original gates remain the promotion policy. Three seed/host-confounded blocks on one graph give exploratory evidence; fresh tasks with a locked recipe are required for confirmation. Failed interaction support narrows an interaction claim; it does not logically refute all finite-sample benefit.

## 8. Scoped prior and reading accounting

The v64 index was consulted first. Its MLDG/MetaReg records already resolve adaptation-free training-only transfer; ANIL/BMAML resolve shared-feature/private-learning ancestry. Later saved scoped graph/DG packets were reused for commitment and correlation boundaries. No genuinely missing adjacent scope is needed to justify the algebra or its limits; therefore this assessment performs **zero public searches, new retrievals, raw-primary rereads, new bounded primary reads or full-paper reads**. Saved conclusions/scope metadata were reused; raw papers and selected primary-excerpt files were not opened. An initial overbroad index filter incidentally displayed embedded saved GraphLoRA paragraph/equation containers; they supply no conclusion or reading credit here. The later HOPPER/NBA scout was consulted only through its report and adopted screening correction, not for a new propagation claim.

| Precise primary identity and retained scope | Conclusion reused here |
| --- | --- |
| Li et al., [MLDG, arXiv:1710.03463v1](https://arxiv.org/abs/1710.03463v1), PDF pp.2–4, supervised methodology through Final-Test, Algorithm 1, analysis through Eq.7 | Training-only transfer objective and gradient-agreement interpretation, with adaptation-free final deployment, are prior |
| Balaji et al., [MetaReg, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html), Figure 1 caption, §§3.1–3.4, Eqs.1–3, Algorithm 1 | Shared features, persistent source heads and differentiated virtual private updates are prior; features are frozen for regularizer learning and a fresh single is finally trained |
| Raghu et al., [ANIL, arXiv:1909.09157v2](https://arxiv.org/html/1909.09157v2), §4 and Appendix C.1, saved blocks46–57/111–126 | Head-only adaptation and shared-feature meta-updates are prior, without importing results |
| Yoon et al., [BMAML, arXiv:1806.03836v4](https://arxiv.org/html/1806.03836v4), §§3.1–3.2/Eqs.2–7, §5 saved paragraphs43/47 | Ensemble meta-learning and shared extractor/private classifiers are prior; no native-Adam or ranking guarantee |
| Hwang et al., [SELAR, arXiv:2103.00771v1](https://arxiv.org/html/2103.00771v1), §III/III-A/III-B, Eqs.1–9, Algorithm 1 especially lines6–12 | Virtual update, slow meta-update and recomputed persistent commitment are graph prior; the slow variable is a loss-weight function, not this shared block |
| Li et al., [Episodic DG, arXiv:1902.00113v1](https://arxiv.org/html/1902.00113v1), introductory S1.p1.1, §§3.1–3.5/Eqs.1–6/Algorithm 1 | Persistent private training partners and fixed deployment are prior; private partners are constant in cross-domain feature loss, not this differentiated response |
| Foret et al., [SAM, arXiv:2010.01412v1](https://arxiv.org/html/2010.01412v1), saved method blocks14–28/89–91, method math0–6 | Training perturbations can shape learning; current descent-response lookahead is not SAM's local maximization and inherits no flatness or i.i.d. bound |

The exact endpoint-conditioned graph composition is not established as an exact match by these retained scopes. That limited observation is not global novelty clearance. Any substantive contribution must identify and attribute the coupled eligibility/roles/state/objective/commitment choices and demonstrate the required quality; this identity alone provides an explanation, not a novelty verdict.

`INPUT_BINDINGS.json` binds source/protocol documents and only the safe ledger adoption selector. `READ_SCOPE_REUSE.json` records precise retained scope custody and zero new reading credit. `MANIFEST.json` and `SEAL.json` cover this new analysis packet only. Frozen artifacts and the literature index were not changed.
