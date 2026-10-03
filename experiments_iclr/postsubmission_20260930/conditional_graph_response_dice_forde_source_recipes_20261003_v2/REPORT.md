# Corrected DICE/FoRDE graph comparator source specification

Prepared 2026-10-03. **Scientific choices pending root adoption; source only, no numerical admission or execution.** The predecessor manifest `6b04e6cd5d44dece9e3b2f88bb2fa68c7e918ac6a1bd674701525dc158fc9047` and independent review seal `369c5ec281d2ad66d19c732599f896c8a8872000803dd12f84f85feb2c339fd2` remain unchanged.

This successor fixes the author-code FoRDE member-axis bandwidth and supplies prepared, unexecuted force/Gram source. It separates a simpler proposed source-style comparison from the unadopted matched guarded continuation. It uses the already pinned paper excerpts and author files; no broad literature revisit occurred.

## Proposed main comparison

Use native TRAIN-node CE with native polynomial tokens. Compare CE-only, DICE Appendix F.2 unit-noise CR, and author-code FoRDE Identity, starting from the same source-qualified warm bank. Freeze common maps and shared statistics; permit **all existing predictor private R/S factors**, including stem, readout and classifier boundary factors, to train. This is a frozen-common graph adaptation, not a reproduction of independently trainable published networks. The broader partition is a pending scientific choice; v4's intermediate-only `begin_private_continuation` is not this partition.

Use ordinary SGD with Nesterov momentum 0.9 and coupled parameter decay, with graph-qualified learning rate/schedule and training duration required before execution. Source decay reference is 5e-4. Use sums of mean-per-member CE, retaining the original absolute force scale. Do not add graph-removal CE, candidate response-energy constraints, backtracking or a fixed 200-attempt ceiling to this main protocol. Every normally completed optimizer cycle advances its optimizer/auxiliary state. No Bayesian or full-method superiority claim follows from fixed common maps or these selected published ablations.

Keep the established mean-raw-logit serving endpoint primary for comparison. Also report the saved FoRDE author's mean-probability endpoint from the same logits and checkpoint, separately labeled. Neither endpoint requires refitting.

`SOURCE_STYLE_PROTOCOL.json` contains the proposed execution contract. Root has not adopted it. `MATCHED_GUARDED_PROTOCOL_NOT_ADOPTED.json` retains the narrower frozen-intermediate, three-view-CE, candidate-guarded alternatives as a separate scientific question, with corrected FoRDE bandwidth and explicit joint state policy. It does not activate the prior candidate study.

## Exact FoRDE force and efficient graph implementation

For `[live i, reference j, target b]` distances,

\[
D_{ijb}=\|s_{ib}-\operatorname{sg}(s_{jb})\|^2,
\quad h_{jb}=\operatorname{sg}(\operatorname{median}_iD_{ijb})/\log M+10^{-12},
\]

\[
K_{ij}=B^{-1}\sum_b e^{-D_{ijb}/h_{jb}},
\quad R_{\rm sum}=B^{-1}\sum_i\log\sum_jK_{ij}.
\]

The median includes each column's self distance and averages the central two values for even M. References and h are detached. Self kernels stay in the denominator. The batch mean stays inside log. Minimize `sum_m mean_b CE_m + R_sum`; the positive log KDE supplies repulsion. The score is the author's true-class **raw logit**. Norm denominator is `sqrt(||g_X||²+1e-24)`. A full-matrix median or log-probability score is a separately named variant.

The actual saved PolyFormer block uses only per-row LayerNorm, linear/elementwise operations and attention across polynomial orders (`native_polyformer.py:39–59,89–109`). With dropout inactive, target token-row slicing preserves the full native output for those targets. Hence the summed true-class score VJP with respect to **selected token rows** gives separate target coefficient gradients `q[m,b,k,f]`; it does not sum overlapping whole-graph input gradients.

Let `a[b,k,:]=row_target(b)(P^k)` and `G[b,k,l]=<a[b,k,:],a[b,l,:]>`. For fixed immutable P,

\[
g_{X,m,b}=\sum_k a_{b,k}\otimes q_{m,b,k},\qquad
\langle g_{X,i,b},g_{X,j,b}\rangle
=\sum_{k,l}G_{b,k,l}\langle q_{i,b,k},q_{j,b,l}\rangle.
\]

Compute normalization and pair distances by this identity while keeping derivatives through q live. This is an exact mathematical evaluation of the same full-feature kernel; it is not PCA, a changed metric, an approximation, or a new method. The prepared `core/forde_identity.py` forms G using selected basis rows propagated through the declared powers of the native uncoalesced float32 COO operator, and computes the source force without allocating `[M,B,N,F]` derivatives. It does not coalesce, normalize, project, truncate support, change dtype or differentiate P.

Float32 arithmetic orders differ between Gram contraction and direct pullback. Runtime equivalence and roundoff must be measured against the direct oracle before deployment; mathematical equality does not imply bitwise equality. Negative quadratic values or nonfinite results fail explicitly rather than silently clamping a different force. A qualified direct-pullback implementation remains the authority if Gram arithmetic cannot meet the adopted tolerance.

## DICE and state choices fixed here

The DICE loss is `sum_m mean CE_m + delta/(M-1)*sum_(i<j) S_ij`, where `S_ij=mean_(b,s) 10*tanh(a_w/10)`. Deterministic h receives independent unit Gaussian noise; there is no VCEB or sampled classifier loss. Delta 0.1, ratio two negatives per positive, four replicas, four RMSProp updates and RMSProp LR 0.003 are source-setting references pending graph competence/adoption, not established graph optima.

One positive noise bank is shared across all pairs. The first discriminator step reuses the predictor objective's joint samples. Later discriminator steps redraw the joint noise. Partners are drawn per `(auxiliary step, pair, anchor, negative index)` from the complete same-class TRAIN population excluding the anchor; each partner ID is reused across four independent noisy replicas. Negative feature noises are fresh and independent. The prepared standard-library partner planner makes this choice explicit without reading labels or executing sampling in this packet.

All detached base partner means are computed **before** the predictor step. Predictor backward uses the base discriminator frozen with live input gradients; its SGD step precedes four RMSProp updates on detached base means. On normal cycle completion, both optimizers have zero gradients (`set_to_none=True`), predictor private flags are restored, common flags are false, and discriminator flags are true. The source-style protocol stops on an exception and resumes only from a completed-cycle checkpoint; it requires no per-step dual-optimizer copy for candidate backtracking. The separate guarded protocol retains full outer joint rollback and commits unscaled auxiliary updates only when a nonzero predictor displacement is accepted. `DICE_NOISE_AND_STATE.md` specifies both.

## Immediate implementation gates

1. Adopt an actual all-layer BE composition/warm receipt and the main protocol's exact named all-private partition; qualify selected-row native forwards before using local VJPs.
2. Run the clustered M4, asymmetric median, direct feature-pullback/Gram and frozen-reference mixed-derivative oracles in `ORACLES.md` only after separate numerical authorization. The code here was AST-parsed, not imported or executed.
3. Bind a normal runtime, graph-qualified optimizer schedule/duration and checkpoint selector. Qualify source-style CE competence and DICE estimator learning before any comparison. Fixed image LR or 200 attempts do not certify competence.
4. Profile immutable native token storage and selected-row Gram preparation under `RESOURCE_PLAN.md`. No measured memory/time/hardware claim is made.

No data, models or outcome records were loaded. No sealed predecessor/review files were modified. No installation, remote access or Git operation occurred. This packet has no fit or launch authority.

