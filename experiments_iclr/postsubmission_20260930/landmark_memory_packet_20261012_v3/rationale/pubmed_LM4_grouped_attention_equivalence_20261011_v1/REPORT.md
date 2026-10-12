# PubMed LM4: exact attention equivalence and remaining constraint

2026-10-11. **Scoped algebraic conclusion; no execution or nomination change.** Authority is the sealed PubMed PolyFormer LM4 v1 construction. Saved primary equations suffice for the exact question; no new primary source or manuscript reviewer was needed. Experimental implementation remains separate and allocation-only.

## Conclusion

**LM4 is exactly expressible using ordinary inducing cross-attention followed by four grouped decoder-attention heads, tied projections, a shared native classifier, four own losses and averaged predictions.** This is an exact refactoring of its real-arithmetic computational graph, not a new memory operator. It is a constrained grouped attention model, rather than a general unconstrained Transformer or a reproduction of full Set Transformer/Graph Perceiver IO. Keeping branch-specific logits and softmax is essential to the equivalence.

The remaining meaningful package is: common projected content/read maps across four fixed landmark groups; private query-offset rows; separate per-group read normalization and per-branch supervision; source-induced parameterization/optimizer interaction; a shared source-native class head; and fixed graph-distance biases. These are explicit constraints and graph inputs built from known operations. Their usefulness needs evidence; renaming groups “members” does not establish novelty.

## 1. Constructive exact refactor

Let fixed `U` be the normalized source cache, `P=U[landmarks]` the64×256 anchor matrix, `E=concat(E0,…,E3)` the64×32 free offsets and `D` the N×64 capped-distance matrix. In row-vector convention:

`Q=P Aq+E`

`Z=softmax_rows(Q (U Ak)^T/sqrt32-D^T/4) (U Av)`.

Stacking the four original16-query writes gives this **identical64-query write**. Its softmax normalizes over input nodes, separately for each slot; slots never compete during this first exchange. Split Z and D into the original16-slot groups. Then:

`alpha_m=softmax_rows((U Bq)(Z_m Bk)^T/sqrt32-D_m/4)`

`R_m=alpha_m (Z_m Bv)`.

This is a four-head decoder cross-attention operation: give each head the same projected node query, tie its key/value/query projections, and use a head-specific mask allowing only its own16 of the64 latent slots. Inside the group retain the original finite distance bias. “Mask” here describes the fixed group restriction; it does not replace LM4's finite graph bias with an unreachable-vertex mask.

Retain all four head outputs rather than mix them into one latent. Apply shared Wo, replicated native residual H, branch-specific training dropout and the shared factorized lin3, followed by each branch softmax and their arithmetic probability mean. A block-diagonal output projection can retain four outputs in a concatenated-head implementation. The resulting computation and mean-own-CE gradients are exactly LM4 when the same physical parameter objects, optimizer groups and dropout draws remain tied. Merely initializing four independent copies equally is not equivalent training. Different vectorization may change floating-point summation or RNG consumption; the mathematical identity is not a bitwise qualification receipt.

This construction uses standard attention operators with specific tying/masking/output constraints. It is not a bijection from LM4 parameters to the entire space of unconstrained multi-head attention: that space also permits different head projections and head mixing. Full MAB/ISAB includes LayerNorm/feedforward operations absent here, and full Perceiver IO can include latent self-attention absent here.

## 2. The landmark-query parameterization is redundant in function class

P is frozen and E is unrestricted. Therefore every64×32 learned query table `Q*` can be realized, for any Aq, by `E=Q*-P Aq`. Conversely only `Q=P Aq+E` enters the first attention exchange. A free learned inducing-query table realizes the same query-function class. Projected landmark features provide initialization and parameterization; they do not restrict learned queries to graph-feature anchors.

**This does not license removing Aq from the sealed implementation.** Write `G=∂loss/∂Q`. The exact gradients are `∂loss/∂Aq=P^T G` and `∂loss/∂E=G`. Even equal-rate plain gradient descent with zero decay would induce

`ΔQ=-eta (P P^T+I) G`,

rather than direct-query descent `ΔQ=-eta G`. The shared Aq path couples slot updates through a source-feature Gram matrix, including slots in different groups. Actual coordinate-wise Adam and the declared decay make that interaction parameter-dependent; folding to Q generally changes optimizer states and training trajectories. This is a concrete reparameterization/optimization interaction, not a newly acquired graph-information source or novelty claim. Slot-to-landmark association still remains explicit in D even when E moves queries away from their initial anchors.

At eval, the common factorized native head is affine: `C=diag(r) Wlin3^T diag(s)`, with its unchanged bias b. Thus branch logits satisfy

`ell_m=H C+b+.25 alpha_m Z_m Bv Wo C`.

The linear products can be folded for serving functions, but not generally for source-bound training/decay. No additional private native predictor or recurrent state is hidden in this expression.

## 3. What cannot be collapsed without changing the model

**Grouped read is different from joint64 read.** At a corresponding parameter state define read scores g_m and `t_m(v)=sum_j exp(g_m(v,j))`. The S64 joint read is exactly

`R_joint(v)=sum_m [t_m(v)/sum_k t_k(v)] R_m(v)`.

The joint decoder therefore assigns input-dependent group mass through the four partition functions. LM4 normalizes each group independently and preserves four residual/logit paths. This explains the existing S64 contrast without asserting a new attention principle: ordinary grouped/masked attention expresses the same difference.

**Averaged probabilities are not averaged logits or residuals.** In general `mean_m softmax(ell_m) != softmax(mean_m ell_m)`. For three-class branch logits[0,0,0] and[log3,0,0], repeated twice each, the first-class probability mean is7/15; its mean-logit probability is `sqrt3/(sqrt3+2)`, a different value. A single output can reproduce the mixture using nonlinear log-sum-exp of normalized branch logits, but an ordinary linear mean of residuals followed by one classifier is not that refactor. This is the established ensemble mixture operation, not an originality argument.

**Mean own CE is ordinary branch supervision.** It differs from pooled-probability CE: Jensen gives `mean[-log p_m(y)] >= -log mean[p_m(y)]`. Own CE gives every branch class-logit gradient `(p_m-onehot(y))/4`; pooled CE gives weight `p_m(y)/sum_j p_j(y)` instead of1/4. Thus the frozen objective pressures every route toward competence, while pooled CE can emphasize the route already supporting truth. Shared maps/head receive the sum of branch gradients. This is the same interaction in a tied multi-output attention network; no independent mechanism arises from calling it ensemble training.

**Graph evidence remains, through a known pathway.** Node→landmark write followed by landmark→node read is a global bipartite attention path, with capped shortest-path biases in both directions and graph-aware source H. Equal local source representations can read different summaries when their distance rows differ. All graph pairs retain finite bias, including disconnected pairs, so this is not edge-local message passing. Ordinary graph-biased inducing attention reproduces the complete pathway. The model has no raw feature information beyond the frozen source and supplied distances, and no guarantee that its remote summaries add useful class evidence.

## 4. Saved closest primary collisions

Set Transformer1810.00825v3,§2.2 eqs3–5, defines projected attention and concatenated heads;§3.1 eqs9–10 gives input→inducing→input exchanges. The saved full MAB equations6–7 also document the normalization/feedforward distinction. Graph Perceiver IO2209.06418v2,§§3.1–3.3/4.1–4.3, explicitly supplies latent write, node-query readback, graph smoothing and pretrained output queries. MemGNN2002.09518v1,§3.1 eqs1–3, already uses shared graph queries against head-specific memory keys, though its Student-t assignments/coarsening/head aggregation are not LM4's equations. Graphormer2106.05234v5,§3.1.2 eq6, supplies shortest-path attention bias. The exact normal form follows by algebra from these saved operator definitions and the frozen LM4 specification. A broader firstness/absence claim remains unresolved and is not needed to answer this question.

## 5. One decisive comparison after utility

If LM4 passes its current frozen utility/member screen, a scientifically useful **single sharing comparison** would retain its entire grouped attention graph, source H/U,64 landmarks/four16-slot groups,E,Aq parameterization,common Wo/native head,own-CE objective,probability mean,dropout draws,initialization and selector/horizon law, while releasing only **Ak,Av,Bq,Bk,Bv** into four independently optimized head-specific copies. Start copies from the same paired values; do not change source ancestors or give a head all64 slots. Aq/E optimizer geometry stays matched, so its redundant parameterization does not become another changed ingredient.

That comparison tests whether a common content/read basis across private queries is useful relative to ordinary head-specific projections on the **same evidence and objective**. The relaxed model has more parameters and optimizer state; retain their measured cost and do not call this a parameter-count-matched or pure optimization comparison. A shared-model advantage with competent members, served repairs and protected losses would support this specified tying/regularization allocation. A relaxed-model match or win would weaken a sharing-specific explanation while preserving any original augmentation utility. Neither outcome makes inducing attention novel.

This is an evidence requirement, not an added nominee/arm, fit request, numeric gate or execution authorization. Root would bind any criteria prospectively before separate work. Current S64 changes normalization and prediction/objective structure; T16 changes offset sharing/capacity; I4 changes source multiplicity and full slot access. None isolates this particular content-map sharing constraint.

## Confidence-population clarification

Root separately reported that Physics GRU's literal frozen “CAL NLL” guard will remain a whole-CAL confidence guard, with separately labelled added calibrated DEVELOPMENT protection. Raw/member/continuation criteria remain unchanged. The PubMed LM4 nomination already specifies DEVELOPMENT NLL after fittingT on CAL. This algebra report does not substitute one evaluation population for another or amend either nomination.

## Custody

Only the frozen nomination, exact factor source text and saved Set Transformer/Graph Perceiver/MemGNN/Graphormer primary passages were inspected. No model/scientific library import, data/checkpoint/prediction array, numerical experiment, GPU/SSH, new literature retrieval, reviewer or implementation edit occurred. The existing nomination/literature/interpretation seals remain unchanged. Source equivalence is mathematical, not predictive or novelty evidence.
