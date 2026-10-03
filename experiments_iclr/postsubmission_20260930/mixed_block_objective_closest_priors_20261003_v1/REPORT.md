# Bounded closest-prior pass for the frozen mixed objective

3 October 2026. Literature/source interpretation only. Read `literature_memory/index_v28` and the saved mixed-objective scout first. Jeffares et al. and Song/Chai were reused from saved conclusions, without reopening their primary texts. Exactly two genuinely new relevant primary methods were read: ONE `1806.04606v2` and PCL `2006.04147v2`. No subagents, graph payloads, labels, current outcomes, training, canonical-record changes, new implementation, hyperparameter-grid changes, or extra experimental gates.

## Finding

**The two additional primary scopes establish close partial overlap, not exact equality to the frozen policy.** Both methods train shared trunks and private branches with member and ensemble supervision. PCL supplies a particularly direct block-routing precedent: its ensemble CE bypasses individual peer classifier heads, while training shared/private feature blocks and a separate fusion classifier. It does **not** remove member CE from shared features, and its private feature blocks also receive ensemble CE. ONE's ensemble CE reaches its private branches directly. Both include distillation losses and serve different default predictors.

The frozen learner remains a block-dependent use of established own/pool objectives within these inspected scopes. No published complete equivalent was established in this bounded pass; this is not a novelty certificate or an exhaustive absence claim. The prior scout's locally saved exact 32-step mixed-policy comparator remains acknowledged. Full-training use tests duration and utility rather than discovery of block-specific training.

No experiment outcome or literature score changes this conclusion. The existing four-policy comparison remains the utility test. This packet adds attribution and scope accounting, not another method proposal or gate.

## 1. Exact object being compared

Let `M=4`, `z_i=z_i(W,phi_i)` with route-local private factors, `z_bar=mean_i z_i`, `p_i=softmax(z_i)`, and `p_star=softmax(z_bar)`. `W` includes every shared trainable parameter, including shared classifier/norm/bias parameters; `phi_i` denotes the private fast factors. Each expression below includes the same TRAIN example mean already frozen by root; there is no additional class rescaling.

    L_pool = CE(z_bar,y)
    L_own  = (1/M) sum_i CE(z_i,y)
    g_W    = (1/M) sum_i J_W,i^T (p_star-y)
    g_phi,i = (1/M) J_phi,i^T (p_i-y)

All gradients are collected from the same pre-update forward state and committed in one simultaneous optimizer transition. Deployment is `softmax(mean raw logits)`. The original `1/M` in each private gradient is retained. A prior's unscaled per-branch loss, changed gate, EMA model, distillation term, or alternating update cannot be silently substituted.

The saved scout already derives `L_own=L_pool+D`, with `D=mean_i KL(p_star||p_i)`. The target removes only the shared `grad D` relative to own/own and keeps the private `grad D` relative to pool/pool. Its generic nonconservative field and lack of a simultaneous descent guarantee are inherited conclusions, not new derivations here. Nonconservativity by itself is not an exclusion proof for detached online KD fields; compare the actual cotangents and update paths.

## 2. ONE: ensemble CE reaches the private branches

Lan, Zhu and Gong, **Knowledge Distillation by On-the-Fly Native Ensemble**, NIPS 2018. Inspected `1806.04606v2`, revised 8 September 2018. The complete method is Section 3, PDF pages 3-5, Eqs. 1-7 and Algorithm 1. Deployment and the no-gating variant were resolved with page 7 Section 4.4. Figures/formulas on pages 4-5 were visually checked.

The model shares low layers, adds private final blocks/classifiers and a gate `g(W,x)` with positive softmax weights, and forms

    z_e = sum_i g_i z_i,   sum_i g_i=1,
    L_ONE = sum_i CE(z_i,y) + CE(z_e,y)
          + T^2 sum_i KL(q_e^T || p_i^T),  T=3.

Here `M=m+1` in the paper; its default is three branches. The branch hard losses are **summed**, not averaged, in Eq. 7. The method/algorithm specifies one simultaneous SGD update of the joint model. No additional shared/private gradient filter or shared-only ensemble backward path is stated in the complete method scope.

### Exact hard-loss paths, independent of KD target detachment

The gate consumes shared low features, not a private high-layer output. For private branch parameters `phi_i`, the hard-loss gradient is

    grad_phi,i L_hard = J_phi,i^T [(p_i-y) + g_i (p_e-y)],
    p_e=softmax(z_e).

Thus the private branch receives **both** own CE and ensemble CE. Shared parameters receive the sum of own-CE branch gradients, the weighted ensemble cotangent, and the gate-gradient contribution:

    grad_W L_hard = sum_i J_W,i^T [(p_i-y)+g_i(p_e-y)]
                 + sum_i grad_W g_i * [z_i^T(p_e-y)].

These equations derive the hard part of the printed forward/loss; they are not a code qualification. Setting `g_i=1/M` removes the last gate term but leaves `J_phi,i^T[(p_i-y)+(p_e-y)/M]`, and shared own CE remains. A common scalar division of the complete loss cannot generally remove these unwanted terms or reproduce the target's different blocks.

The paper's KL equations do not explicitly settle target-detachment semantics. With a stopped teacher target, the direct student logit gradient of `T^2 KL(q||softmax(z_i/T))` is exactly `T*(p_i^T-q)`. If the online teacher target is differentiable, additional teacher-path derivatives appear. Either interpretation leaves the hard-loss paths above and adds supervision beyond the frozen own-CE private cotangent. Do not turn the paper's approximate `T^-2` scale explanation into an exact finite-temperature identity.

### Served predictor and limits

Default ONE discards auxiliaries and serves one ordinary live branch, `theta_0`. Optional ONE-E retains all branches; Eq. 3 and Section 4.4 identify its learned gating combination, with uniform averaging explicitly treated as a separate no-gating ablation. It is not the frozen uniform mean-logit M4 predictor. Averaging reported **branch error rates** in the evaluation setup is also not a deployed prediction pool.

The inspected setup uses image classification, SGD/Nesterov and shared low blocks/private last blocks, not rank-one fast factors throughout a graph model. No author code was read; batch-loss reduction, any unreported implementation backward rescaling and actual teacher detachment remain unqualified. The primary text supplies strong shared/member/ensemble-loss ancestry, without exact mixed-policy equality or a transferable graph-quality guarantee.

## 3. PCL: natural block routing through separate classifier heads

Wu and Gong, **Peer Collaborative Learning for Online Knowledge Distillation**, AAAI 2021, DOI `10.1609/aaai.v35i12.17234`. Inspected `2006.04147v2`, revised 3 March 2021. The complete method is PDF pages 3-5, Eqs. 1-9 and Algorithm 1; page 7 specifies ensemble deployment. Equations and Algorithm 1 on pages 4-5 were visually checked.

PCL shares low-layer parameter values and separates high layers. Each peer receives its own random augmentation. Let

    h_i = f_i(W,psi_i,x_i),
    z_i = B_i h_i,
    z_t = A concat(h_1,...,h_M) = sum_i A_i h_i,
    L_PCL = sum_i CE(z_i,y) + CE(z_t,y) + L_pe + L_pm.

`B_i` is an individual peer classifier and `A` is a separate feature-fusion classifier. The teacher is **not** the arithmetic mean of peer logits. The hard peer CE is summed without `1/M` in Eq. 2. The printed KD coefficients are

    L_pe = omega(epoch) T^2 sum_i KL(q_t^T || p_i^T),
    L_pm = omega(epoch) T^2/(M-1)
           * sum_i sum_(j!=i) KL(q_EMA,j^T || p_i^T).

Thus `1/(M-1)` averages peer **teacher sources for each student**, not the hard member losses across students. The epoch ramp-up, EMA smoothing and `T=3` are material method parts. They are not parameters of the frozen policy.

### Exact hard-loss routing

For the printed architecture, with `s_i=p_i-y` and `s_t=softmax(z_t)-y`,

    grad_B,i L_hard = s_i h_i^T,
    grad_A,i L_hard = s_t h_i^T,
    grad_W L_hard = sum_i (partial_W h_i)^T [B_i^T s_i + A_i^T s_t],
    grad_psi,i L_hard = (partial_psi,i h_i)^T [B_i^T s_i + A_i^T s_t].

Biases have the corresponding residual gradients. These paths give **real differential loss exposure across parameter blocks**: each `B_i` receives no direct teacher hard CE; `A` receives no direct member hard CE. Yet shared `W` receives member and ensemble CE, and private feature `psi_i` also receives both. Labeling all private blocks as `B_i` would omit the separately trainable high layers and would change the compared parameterization.

KD further changes each peer classifier's cotangent. If the current ensemble target and EMA targets are fixed during a backward pass, its direct logit contribution is

    omega T * [(p_i^T-q_t^T)
              + (p_i^T-mean_(j!=i) q_EMA,j^T)].

If the online feature-fusion target is differentiable, it additionally affects feature/fusion paths. The primary method does not explicitly resolve that target-detachment choice. The hard-path distinction above is valid either way and already prevents generic equality to the frozen policy. The EMA targets have their own state update in Algorithm 1; no source-level autodiff/state qualification is claimed.

### Update and served predictor

Algorithm 1 first updates the peer/student model with Eq. 1, then updates the mean teachers through the EMA equations. Default PCL serves the first peer's **EMA model**. PCL-E deploys the EMA peer features with the separately learned fusion classifier (the classifier is also EMA-aggregated). Neither is the unchanged uniform mean raw-logit pool. Live peer parameters are jointly updated, but the additional temporal teacher state is a separate transition absent from the frozen learner.

The scope supports attribution for natural architectural loss routing and combined peer/ensemble supervision. It does not specify shared pooled CE only and private uniform mean-member CE only. The printed hard-loss scaling differs, and actual code reductions/stop-gradients were not inspected. Image-task improvement, branch-variance curves and ensemble accuracy do not establish the frozen HGT policy's efficacy or exact optimizer equivalence.

## 4. Comparison summary

| Method | Shared trainable feature/core blocks | Private predictive blocks | Loss scaling and served predictor |
|---|---|---|---|
| Frozen policy | Only gradient of raw-logit pooled CE | Only their slice of uniform mean-member hard CE | Original `1/M`; same-state one optimizer transition; uniform mean raw logits |
| Saved Jeffares et al. | One scalar own/pool interpolation in inspected method | Same scalar loss's appropriate block derivatives | Scalar coefficient applies across blocks; score/probability pool distinctions retained from scout |
| Saved Song/Chai | Averaged/rescaled backward from hard+peer-soft head losses | Hard+peer-soft branch loss at branch scale | Shared backward `1/H`; one retained head; not shared raw-logit pooled hard CE |
| ONE v2 | Sum member CE + gated ensemble CE + KD, including gate paths | Own CE + gated ensemble CE + KD | Printed member sum; default one live branch; optional learned-gate ensemble |
| PCL v2 | Sum peer CE + feature-fusion CE + KD | Private features receive both hard losses; individual peer heads bypass teacher hard CE but receive own CE and KD | Printed member sum; `1/(M-1)` only for teacher-source KD; default one EMA branch; optional EMA feature-fusion model |

Constant scaling can sometimes approximately cancel in idealized Adam settings, but it does not establish equality of the raw gradient fields, epsilon/decay effects, moments, group settings or complete native optimizer transitions. The target keeps its frozen `1/M` and no new scaling comparison is introduced.

The positive rationale from the scout remains conditional: shared capacity fits the served committee; private factors receive their individual true-label residuals even when the committee is already confident. The inspected distillation papers establish adjacent learning goals but do not prove this assignment helps. Removing shared own-loss regularization can hurt; private competence can improve while the served pool worsens. Existing own/own, pool/pool and reverse-policy comparisons address that claim. No new result-based gate, selector, coefficient, optimizer rule, or inference change is suggested.

## 5. Scope, access and accounting

- New relevant scoped primary methods: **2**. New whole-paper/proof certifications: **0**.
- Previously read primary texts reopened: **0**. Jeffares, Song/Chai, GNCL, TabM/BE, TreeNets and the saved 32-step comparator were reused from the v28/scout record.
- New author-source reads, implementations, scientific runs and current-outcome inspections: **0**.
- The metadata-only search covered shared/private losses, gradients, shared trunks, detached/decoupled ensembles and exact online-KD titles. Broad OpenAlex/Bing results had poor relevance. No survey was method-read. Exact-title resolution selected ONE and PCL; other candidate titles were not primary-read. Search quality/absence is not novelty evidence.
- The two downloaded latest arXiv URLs were bound to exact v2 versions by both landing-page history and PDF header; future citations use versioned URLs. No duplicate existing source tree or scientific fixture was copied.

`PAPER_CONCLUSIONS.json` is suitable for root's later memory-index adoption; `READ_SCOPES.json` separates complete method scope, limited deployment context and incidental table exposure. `REUSED_REFERENCES.json` binds the v28 and scout bytes. Root owns canonical records and the frozen experiment. This packet changes neither.
