# Shared pooled loss, private member loss: conditional quality experiment

3 October 2026. **Retain one attributed utility question:** does full-training shared-pool/private-own CE improve the served classifier while retaining competent private routes, compared with own/own, pool/pool and the reverse own/pool policy? The split gradient is not a newly discovered principle: it already appears as a 32-step comparator in the saved shared-gradient proposal. Its full-policy predictive utility is unmeasured. Two new scoped primary methods do not specify the exact full mixed learner in their inspected methods; this is a bounded distinction, not exhaustive published-method absence or a novelty claim.

No implementation, training, numerical gate, protocol/index/status edit, dataset/label/fitted-state/current-run-output access or subagent. An initial memory lookup incidentally displayed historical scalar summaries in objective_alignment_v1/OBJECTIVE_SUMMARY_v1.md; none is used to choose a model, outcome threshold, mechanism or conclusion here, and its underlying result packs remain unopened. No current HGT outcome is claimed.

## 1. Exact policy and gradient difference

For M=4 independent member routes, z_m=z_m(theta,phi_m), theta contains **all shared trainable parameters** and phi_m contains every route-indexed fast factor. Private parameters affect only their own route; common-message/peer-feedback architectures are outside this formula. Use the same mean TRAIN normalization and weights for every loss, mean raw logits at inference, and

```
zbar = mean_m z_m;  pstar=softmax(zbar);  p_m=softmax(z_m),
L_pool = CE(zbar,y),
L_own  = mean_m CE(z_m,y) = L_pool + D,
D = mean_m logsumexp(z_m)-logsumexp(zbar)
  = mean_m KL(pstar || p_m) >= 0.
```

All equations extend by the common weighted TRAIN average. D is label-independent at fixed logits. Write sstar=pstar-y, s_m=p_m-y, J_theta,m=partial_theta z_m and J_phi,m=partial_phi_m z_m. Then

```
a = grad_theta L_pool = (1/M) sum_m J_theta,m^T sstar,
a+c = grad_theta L_own = (1/M) sum_m J_theta,m^T s_m,
c = grad_theta D = (1/M) sum_m J_theta,m^T (p_m-pstar),

b_m = grad_phi_m L_pool = (1/M) J_phi,m^T sstar,
b_m+d_m = grad_phi_m L_own = (1/M) J_phi,m^T s_m,
d_m = grad_phi_m D = (1/M) J_phi,m^T (p_m-pstar).
```

Let b,d stack the private blocks. The four policies supply the optimizer with:

| Shared/private loss | Shared gradient | Private gradient |
|---|---|---|
| own/own, native reference | a+c | b+d |
| pool/pool | a | b |
| **pool/own, candidate** | a | b+d |
| own/pool, reverse control | a+c | b |

The candidate removes **only grad_theta D** from native training. Unlike the earlier peer-detached pooled-loss construction, it retains a different private cotangent s_m rather than sstar, so it does not reduce to ordinary pooled training in general. No inference change, routing, repulsion or tuning coefficient is added.

Normalization is essential: “private own CE” means the private slice of the uniform mean-member loss, including 1/M. Replacing it by unscaled CE_m multiplies the private gradient by M and introduces a learning-rate/optimizer intervention. Do not make that change in this four-policy comparison. Identical parameter groups, native optimizer settings and weight decay remain explicit.

## 2. Why this could improve quality, and what it does not show

The shared core is trained against the prediction actually served. It no longer pays the *additional* member-versus-pool ambiguity gradient. Private factors still receive each member's true-label residual; a weak member can therefore receive a substantial correction even when the pool's residual is already small. This assigns committee fitting to the shared representation and individual fitting to private capacity. It is a concrete training-role hypothesis rather than an instruction to maximize spread.

The conditional opportunity is greatest when shared capacity is useful for the served prediction while the private factors can repair individual route errors. The reverse control asks whether that assignment of roles matters, rather than any block receiving a different loss. This is an empirical optimization/regularization question applicable beyond graphs, not a new graph operator or a guarantee of complementary explanations.

“Relieves consensus pressure” needs a precise qualification. Removing grad_theta D does not necessarily preserve member logit contrasts. If all shared Jacobians are identical, a shared displacement leaves centered raw-logit contrasts unchanged to first order; nevertheless D can change through the common confidence and the softmax curvature. In the binary common-shift example z_1=theta+t, z_2=theta-t,

```
grad_theta D = [sigmoid(theta+t)+sigmoid(theta-t)]/2 - sigmoid(theta).
```

This need not vanish, although theta changes neither raw-logit contrast. It can have either sign and does not inherently hurt prediction. Private own loss still contains D and can reduce useful contrasts. Thus lower D, less hidden separation, or larger disagreement is not the desired endpoint.

The split can fail when private capacity cannot keep members competent, when native own gradients supply useful shared regularization, or when the pooled shared update exploits compensating private errors. Correcting private factors can itself worsen the pool. Fully identical routes give s_m=sstar and all four policies agree at that state; ordinary member initialization/dropout must provide any subsequent difference. There is no forced diversification.

## 3. Nonconservative field and no global descent guarantee

The candidate vector field is V=(partial_theta L_pool, partial_phi L_own). In a smooth region its cross-partials differ by the mixed Hessian of D:

```
partial_phi V_theta - (partial_theta V_phi)^T = -H_theta,phi D.
```

Unless that block vanishes, V is not the Euclidean gradient of one scalar potential. If D is locally separable in theta and phi, a local potential can exist; generic nonconservativity does not mean every special case fails, nor rule out some Lyapunov function.

A simple legal witness uses two independent private intercepts z_1=theta+phi_1 and z_2=theta+phi_2. The cross derivatives are

```
partial_phi_1 V_theta = (1/2) sigmoid'(theta+(phi_1+phi_2)/2),
partial_theta V_phi_1 = (1/2) sigmoid'(theta+phi_1),
```

which are unequal at generic divergent states. No numerical fixture is required.

For simultaneous raw SGD with positive block step sizes eta_theta, eta_phi, the first-order loss changes are

```
Delta L_pool = -eta_theta ||a||^2 - eta_phi b^T(b+d),
Delta L_own  = -eta_theta (a+c)^T a - eta_phi ||b+d||^2.
```

Either expression can be positive. Sufficient alignment of the displayed cross terms supports local decrease; it is not guaranteed by the policy. The shared pooled block alone is a local pooled-descent direction under sufficiently small SGD, with private state fixed; the private own block alone decreases own loss under analogous conditions. Simultaneous composition need not decrease either. Alternating updates are a different learner and do not supply a common objective guarantee either.

These are raw-gradient statements, not claims about native AdamW displacement. Moments, epsilon, preconditioning, decay and finite curvature matter. At a common state, removing c from the supplied gradient does not generally change the AdamW step by exactly eta_theta*c. The moment trajectories subsequently diverge between policies. No finite acceptance/backtracking guard is added to make the hypothesis look guaranteed.

A fixed scalar GNCL interpolation L_lambda=lambda L_pool+(1-lambda)L_own puts the same ambiguity coefficient 1-lambda on both blocks. Generically it cannot set that coefficient to zero on theta and one on phi simultaneously. The candidate is therefore a **block-dependent use of established objectives**, not GNCL's single scalar loss with a renamed lambda, and not a new gradient/projection principle.

## 4. Closest priors and exact-overlap limits

| Source consulted | Complete operation relevant to this question |
|---|---|
| Saved [GNCL, 2011.02952v2, Eq.5](https://arxiv.org/html/2011.02952v2#S4.E5) | Scalar own/pool interpolation for the learner. It supplies the objective ancestry; the generic mixed-block field is not its one scalar Euclidean gradient. |
| Saved [TabM, 2410.24210v3](https://arxiv.org/abs/2410.24210v3), and [BatchEnsemble, 2002.06715v2](https://arxiv.org/abs/2002.06715v2) | Shared/private factor architecture and own-member supervision are prior. TabM native probability pooling must not be silently assigned this mean-logit formula. No exact mixed policy is specified in the retained scopes. |
| Saved [TreeNets, 1511.06314v1, §5](https://arxiv.org/abs/1511.06314v1) | Shared branches and ensemble-aware losses are established. Full pooled training is not the split private-own policy. |
| Saved shared-gradient/GEM packets | The saved proposal explicitly includes **shared pooled/private own for 32 steps**, then native continuation. This is exact policy overlap during that window; full-training use is a duration/utility extension, not discovery of the split. GEM protects response/loss inequalities through projection, a different update. No saved empirical mixed-policy result is established here. |
| **New scoped** Jeffares, Liu, Crabbe and van der Schaar, [Joint Training of Deep Ensembles Fails Due to Learner Collusion, 2301.11323v1](https://arxiv.org/html/2301.11323v1), 2023-01-26 | §5 Eq.3 uses (1-beta) mean-member loss + beta ensemble loss. Appendix E derives distinct score/probability-pool CE gradients. Its failure mechanism discusses compensating outputs and supplies a reason to retain competence checks; it does not establish failure of every shared-HGT pooled policy. No shared-pool/private-own partition is specified in these inspected methods. The common score cotangent is not proof that parameter gradients are identical when route Jacobians differ. |
| **New scoped** Song and Chai, [Collaborative Learning for Deep Neural Networks, 1805.11761v1](https://arxiv.org/pdf/1805.11761v1), 2018-05-30 | §§3.1–3.3 define shared intermediate layers, each head's hard CE plus peer-consensus soft CE, simultaneous updates, and backward rescaling by 1/H into shared layers while private branches retain their scale. This is direct precedent for distinguishing shared/private backward treatment. At beta=1 its shared gradient is averaged own-hard CE, not pooled CE; with soft loss active the private objective is not own-hard CE. It serves one retained head. Peer-target detachment/source semantics and proofs are unqualified; no author code or numerical benefit is transferred. |

No exact **published full complete learner** equal to the proposed policy was established in these bounded scopes. The two new sources narrow overlap; they do not certify global absence. The locally saved 32-step policy is acknowledged. A positive complete comparison could support a useful training-policy extension of GNNM with these ingredients attributed; novelty cannot be the gradient split, the Jensen identity, or block optimization alone.

## 5. One representative four-policy experiment

Parent's preferred representative task is the **complete released HGB DBLP and ACM graphs**, the already frozen native-compatible HGT **global_BE** architecture, and all **five existing paired seed/split blocks**. There is no CP residual: all member-indexed fast factors are private, while core matrices, typed/relation parameters, norms, biases and the classifier are shared. This resolves shared q/u ambiguity before testing the loss policy. Bound the exhaustive shared/private parameter partition from module semantics before any new fit; no learned parameter can receive both gradients or be omitted.

Compare exactly the four table policies for the **entire native training budget**, with the existing ordinary global-BE initialization, four member RNG streams, graph/feature choices, optimizer/group/decay settings, full native update count, validation cadence, pooled selector, checkpoint eligibility and mean-logit serving. No warm residual initializer, intervention-time search, step multiplier, loss coefficient, new capacity, router or diversity penalty. All block gradients at an update must come from the same pre-update forward state. Commit one optimizer/state transition; do not update theta before evaluating phi or update forward buffers twice.

This gives **40 graph/block/policy terminals**. Prefer reusing the **10 frozen own/own terminals only after all HGT35 fits close and are audited**, if source version, architecture/grouping, initialization tensors, feature/graph/split identities, native recipe, streams, normalization and selector are exactly compatible. Reuse every eligible baseline block, never only favorable states. Do not change or retrain its selected checkpoint. This leaves **30 new ensemble fits** if all ten baselines match. If compatibility fails, the affected baseline needs a newly paired fit or the reuse claim must be dropped; the failure is not evidence against the policy. Root owns that source/init correspondence and use-history decision. No baseline outcome or initial tensor is opened in this scout.

Primary comparison is served pooled VALIDATION NLL against own/own **and** pool/pool, with reverse own/pool testing the proposed assignment of roles. Retain raw Micro/Macro-F1, every member's validation NLL/F1, mean own NLL, D under the exact pool, fixed final endpoint, complete seed/block differences and costs at the same joint selected state. Selection uses the same existing rule; no favorable per-member checkpoint selection or outcome-based variant change. These are exposed development tasks and overlapping/fixed graph splits, not independent graph-population replications or unused confirmation.

**Decisive falsifier:** if pool/own does not improve served quality beyond both native own/own and pool/pool across the complete paired comparison, the proposed predictive extension is unsupported. Better TRAIN fit, member competence alone, more disagreement or removal of shared grad D cannot rescue that claim. If reverse own/pool matches the gain, the claimed advantage of assigning pooled loss specifically to shared capacity is unestablished. If pool/pool matches served NLL but has weaker members, the mixed policy may preserve member competence, but a new served-prediction gain is still unestablished. If gains require rescaled private gradients, different schedules, selectors or extra tuning, this fixed policy was not the tested explanation. No new numerical acceptance threshold/gate is supplied.

A favorable result still needs the already planned competent native and independent-ensemble comparisons for broad utility claims. No confirmation task or heldout opening is added here; original scores remain fixed. Ordinary validation differences and three quantities own/pool/D are descriptive evidence, not causal proof that shared consensus pressure was the cause.

The architecture and serving work are identical for all four policies. Mixed policies may need two reverse sweeps from a retained four-member forward graph; qualification must determine whether retained-graph memory is feasible or paired recomputation is required. Charge all complete member work, recomputation, peak memory, optimizer state, qualification/failures and training time. Equal update counts do not imply equal paid compute. No measured timing, memory claim, GPU reservation or compute-based rejection is made.

## Source-only next step and reading scope

Root can first bind the exhaustive module-based partition, loss reductions, same-state gradient collection, native state/RNG transition and baseline compatibility. That is the remaining source preparation; this scout writes no implementation. The representative comparison can then be frozen without a parameter grid. Saved canonical ledger/status and every live study remain unchanged.

New primary methods: **2 scoped, 0 full-paper reads**. Collusion scope: title/abstract/intro snippets; §4.2 definitions, non-proof §5 objective passages/Eq.3, Appendix E classification objective/gradient passages and Table2/math nodes. Collaborative scope: pages1–4 plus page5 method through §3.4; method equations1–6 and figures1–2 were inspected, with page5's experiment-setup opening incidentally visible in its scoped render. No results/proof reproduction, author code or later-version audit. READ_SCOPES.json preserves exact boundaries and versioned sources. Saved GNCL/TabM/TreeNets/GEM summaries were reused; none of their primary texts was reread in this task.
