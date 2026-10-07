# Centered additive low-rank member corrections: bounded independent assessment

2026-10-07. Unadopted direction assessment. CURRENT_SUPPLEMENT, its prior chain, and base index72 were consulted before local saved literature/source passages. The parent then closed source expansion and pilot design after the exact equivalence was established. No pilot or code modification is authored here. No training, model/numerical imports, fixtures, raw prediction/checkpoint reads, remote access, original-score changes, or prior-file changes occurred.

## Decision

**Do not promote this as a new architecture or an established remedy for the observed member-competence failures.** With an unrestricted trainable shared W and the same rank-r raw adapters, the centered and unrestricted jointly trained adapter families represent exactly the same member functions. Centering changes optimization coordinates and regularization unless their geometry is transported. Nonlinear private hidden states allow a nonzero pooled correction, but they do so in the unrestricted family too.

The saved phase already contains exact zero-sum low-rank corrections using a common frozen output basis and Helmert member contrasts. Joint shared-weight/private-LoRA ensemble learning is also explicit in the retained TabLoRA primary method. Those facts close claims that mean/deviation organization, zero-sum adapters, or a covariance identity alone supplies a new learning mechanism. A bounded metadata search adds no novelty clearance.

The closed WikiCS evidence motivates a competence problem. It does not diagnose adapter common-mode contamination, inadequate rank, or an optimizer failure that centering repairs. Following the parent's latest instruction, stop this direction at the bounded no-adoption assessment; no conditional pilot design is proposed.

## 1. Exact family equivalence

Let raw adapters be

\[
\Delta_m=B_mA_m,\qquad \operatorname{rank}(\Delta_m)\le r,
\quad\bar\Delta=M^{-1}\sum_m\Delta_m.
\]

The candidate uses

\[
E_m=W_c+C_m,\qquad C_m=\Delta_m-\bar\Delta,
\qquad\sum_m C_m=0.
\]

The unrestricted adapter family uses \(E_m=W_u+\Delta_m\). For exactly the same A_m,B_m,

\[
W_c=W_u+\bar\Delta
\quad\Longleftrightarrow\quad
W_u=W_c-\bar\Delta.
\]

Consequently

\[
W_c+\Delta_m-\bar\Delta=W_u+\Delta_m
\]

for every member. The map \((W_u,A,B)\mapsto(W_u+\bar\Delta(A,B),A,B)\) is invertible: it retains all factor coordinates and shifts the unrestricted full shared matrix. No rank increase of the raw adapters is needed for this equivalence. It holds at every independently eligible affine site and therefore through all nonlinear member trajectories, logits and pooling when every effective affine map and bias is matched.

This requires a freely trainable shared matrix with the same allowed domain. Frozen W, a restricted shared subspace, incompatible ties to another site/branch, or parameter penalties not transported by this map can invalidate the same-protocol equivalence. Those alternatives are separate hypotheses; none is added to rescue this direction. A difference from BE multiplicative factors is an additive-adapter choice already present in the prior literature, not a consequence of centering.

## 2. The rank caveat is necessary

Raw rank-r adapters do **not** imply rank-r centered deviations. Since C_m is a linear combination of all M raw matrices,

\[
\operatorname{rank}(C_m)\le\min(d_o,d_i,Mr),
\qquad
C_m-C_n=\Delta_m-\Delta_n,
\quad \operatorname{rank}(C_m-C_n)\le2r.
\]

For M=2,r=1, take \(\Delta_1=\mathrm{diag}(1,0)\) and \(\Delta_2=\mathrm{diag}(0,1)\). Then \(C_1=\mathrm{diag}(1/2,-1/2)\) has rank2. This is exact symbolic algebra, not a numerical fixture.

If the intended restriction is instead \(\operatorname{rank}(C_m)\le r\) **and** \(\sum_m C_m=0\), subtracting the mean of independent LoRA matrices does not implement it. Centering B and A separately also does not generally center BA. A rank projection after matrix centering need not preserve zero sum.

A common output basis U and member-contrast coefficients can impose both restrictions:

\[
C_m=U\sum_{q=1}^{M-1}H_{mq}A_q,
\qquad 1^\top H=0,
\quad \operatorname{rank}(U)\le r.
\]

Then every C_m has rank at most r and their mean is zero. This has a common-subspace restriction and differs from general independent raw adapters. The inspected saved `FrozenOutputNeighbor` source implements exactly this form with frozen orthonormal U, a Helmert H, and zero-initialized A_q. It is an existing local construction, not a new candidate introduced here. Making U trainable changes its learning rule but does not establish a new quality mechanism.

## 3. The pooled covariance identity and its limits

At one affine layer let private inputs be h_m and outputs

\[
u_m=(W_c+C_m)h_m+b_m,\qquad
\bar h=M^{-1}\sum_m h_m.
\]

Then

\[
\bar u=W_c\bar h+\bar b
+M^{-1}\sum_m C_m(h_m-\bar h).
\]

The last term may be nonzero when adapters and states vary together. For two members, C_1=D,C_2=−D and h_1=h+a,h_2=h−a give the correction Da. Its sign in a target-versus-competitor margin is arbitrary. Its norm satisfies the generic bound

\[
\left\|M^{-1}\sum_m C_m(h_m-\bar h)\right\|
\le \sqrt{M^{-1}\sum_m\|C_m\|_F^2}
\sqrt{M^{-1}\sum_m\|h_m-\bar h\|^2}.
\]

Neither expression connects the correction to truth or competence. A larger covariance term can strengthen a common wrong class.

If h_m are identical, the centered **preactivation** contribution vanishes. Member nonlinearities can still make \(M^{-1}\sum_m\sigma(u_m)\) differ from \(\sigma(\bar u)\); zero-mean weights are not zero-mean predictions. At later layers, independently evolving nonlinear states can create the displayed term. The identity is layerwise and does not equal the final probability-pool correction through arbitrary nonlinear maps.

For unrestricted adapters the same expression is

\[
\bar u=(W_u+\bar\Delta)\bar h+\bar b
+M^{-1}\sum_m(\Delta_m-\bar\Delta)(h_m-\bar h).
\]

Using W_c=W_u+barDelta gives the same output. The covariance explanation therefore does not distinguish the two families and supplies no additional ensemble expressivity.

## 4. What actually changes under ordinary optimization

For any differentiable joint loss L, let \(G_m=\partial L/\partial E_m\), including any cross-member dependence of a pooled loss. In centered coordinates,

\[
\partial_{W_c}L=\sum_mG_m,\qquad
\partial_{\Delta_k}L=G_k-\bar G,
\qquad \bar G=M^{-1}\sum_mG_m.
\]

The factor gradients become

\[
\partial_{B_k}L=(G_k-\bar G)A_k^\top,
\qquad
\partial_{A_k}L=B_k^\top(G_k-\bar G).
\]

In unrestricted coordinates the corresponding raw adapter gradient is G_k. Centering removes common matrix-gradient credit from the private route and gives it to the shared route. It does not create supervision or demonstrate that this reallocation is helpful.

Even direct full-matrix SGD differs. With shared learning rate eta_W and adapter rate eta_D, centered effective updates are

\[
\delta E_m=-\eta_W\sum_kG_k-\eta_D(G_m-\bar G),
\]

whereas unrestricted updates are

\[
\delta E_m=-\eta_W\sum_kG_k-\eta_DG_m.
\]

The difference is a common eta_D barG displacement. Bilinear factorization and Adam produce more complicated differences. Equal numerical learning rates and ordinary diagonal Adam states are not coordinate-equivalent optimization. The nonlinear shared-matrix shift depends on all factors, so exact metric/state transport generally mixes blocks and correlations. Parameter decay also changes: \(\|W_u\|^2\) and \(\|W_c\|^2\) are different penalties at matched functions. Factor scaling gauges remain within each BA pair.

There is also a symmetry limitation. If all effective members, inputs, stochastic paths and losses are identical, then G_m are identical and the centered raw-adapter gradients vanish. At B_m=0, private random A_m alone does not overcome this cancellation. A specified asymmetry or independently varying stochastic path can break this equality, but zero-sum organization does not itself create competent specialization. No new initializer or training rule is proposed here.

## 5. Closest retained prior and source evidence

| Source and scope | Consequence and limit |
|---|---|
| [TabLoRA,2607.10077v1](https://arxiv.org/html/2607.10077v1), saved `METHOD_EXCERPTS.json`, §§3.3–3.5 Eqs.4–10 | Explicit jointly trainable shared W and private BA corrections from scratch, nonlinear private MLP states, mean own loss and raw-logit averaging. Closest consulted whole-family precedent for the unrestricted model. No result or publication-validity claim is transferred. |
| [LoRA-Ensemble,2405.14438v5](https://arxiv.org/pdf/2405.14438v5), saved scoped §2 and initialization/gradient appendices | Frozen pretrained base, independent low-rank attention adapters/private heads, nonlinear trajectories and standard ensemble aggregation. Its frozen-base policy differs; its gradient and initialization discussion does not certify this centered rule. Saved result interpretations are not adopted. |
| [DSA,2408.04150v1](https://arxiv.org/html/2408.04150v1), saved conclusions/scopes | Shared/private residual adapter ensemble structure and own supervision already target competent structural diversity. Structural difference does not establish corrective information or independence. Conclusions reused, no new primary reread. |
| Saved `spectral_privacy` adapter snapshot, `FrozenOutputNeighbor` and `helmert`, lines1–18 and66–118 | Exact U times centered member-contrast coefficients, original shared map/bias retained, zero private initialization. Direct local zero-sum low-rank precedent; frozen basis and selected source SAGE map are narrower than independent trainable BA factors. Static reading only. |
| [Regularized multi-task learning, Evgeniou/Pontil,2004](https://doi.org/10.1145/1014052.1014067) | Classic shared-plus-task-deviation bibliographic lead. OpenAlex identity/DOI verified; primary method body not recovered/read in this task. It is not counted as a verified exact zero-sum or rank-constrained predecessor. Source expansion stopped before a primary request. |
| [Weight normalization](https://arxiv.org/abs/1602.07868v3), [LoRA-Pro](https://arxiv.org/html/2407.18242v1), [LoRA-RITE](https://arxiv.org/html/2410.20625v1), prior gauge assessment | Previously saved scoped methods establish the distinction between function-equivalent factorization and optimizer geometry/state transport. Conclusions reused here without new primary reads or theorem transfer. |

The multitask identity alone is not used to infer its exact regularization or arithmetic-mean convention. The direct TabLoRA and local centered-source evidence already suffice to reject a novelty claim based on these primitives. Three bounded metadata requests are retained; search ranking and failures do not establish global absence.

## 6. What the actual closed development failures support

The parent explicitly directed this assessment to three saved, fully closed development reports. Their numerical summaries were read; raw prediction/checkpoint payloads were not. They concern one WikiCS graph/split and a merged validation+stopping population repeatedly used for checkpoint selection, not held-out confirmation.

The24-cell interpretation reports unit factors plus contrast improving served accuracy by0.5056 percentage points and member mean by about0.480 points, with only about0.025 points extra pool benefit. That arm still trails ordinary independent4 by0.4108 points. Random first-factor initialization increases coverage but costs1.199 points of mean member accuracy relative to unit factors and loses0.5878 points of served accuracy. The primary initialized-plus-contrast gain is small and mixed across seeds, with worst-member damage and no reliable competence-preserving improvement.

The current common-competitor obstruction is precise: if every unchanged member assigns a common wrong class greater probability than truth, any convex probability pool retains that wrong ordering. The retained report observes almost all all-member-wrong instances with such a competitor. Changing weights before prediction could repair rankings, but a nonzero covariance term or increased latent diversity does not show that it will do so.

The separately closed21-endpoint private-correction recipe loses to the competent continued single in every paired seed; its mean development accuracy is80.2427% versus81.6205%. It also trails the independent reference. Selection-rule and runtime differences prevent attributing every gap to sharing alone. The result is nevertheless direct negative evidence against assuming that extra private corrections yield stronger predictors.

These failures support requiring competent, task-relevant member changes. They establish neither rank deficiency nor common adapter-scale contamination. Centered coordinates can remove common private credit, but that credit may be useful; restricting it does not ensure positive truth margins. The candidate's entire function class already exists in unrestricted jointly trained adapters, so the observed failures cannot support an architectural rescue claim for centering. The remaining optimizer hypothesis has no current diagnosis or predictive evidence and is not advanced to a pilot.

## Read accounting and closure

Zero new primary method identities; two retained primary method excerpt revisits, one saved adapter-source scope, and saved scoped conclusions reused. No full-paper or proof audit, author reproduction, numerical experiment, or novelty certificate. Metadata discovery is distinct from primary reading.

`READ_SCOPES.json` records exact retained scopes and their limits. `INPUT_BINDINGS.json` binds the pointer/base/prior chain, read reports and saved sources. `CONCLUSIONS.json` records the equivalence/rank caveat and no-adoption decision. `MANIFEST.json` covers only this new folder. The source expansion and conditional pilot requested initially were stopped by the parent's later instruction; they are not outstanding work.
