# Relation factors, useful ensemble diversity, and a same-initial-function ablation

2026-10-03. Literature/theory recommendation only. **No change to the adopted 35-fit HGT family or its gate; no execution or new loss adoption.**

## Conclusion

Relation-specific fast factors can change which evidence a member uses and which competing classes it confuses. Useful label-relevant complementarity is possible, but coefficient rank, hidden-state separation, factor norm, disagreement, or a contrastive loss does not establish it. The served mean-logit predictor must improve while member competence remains acceptable. A benefit could instead be a better shared relation residual or better individual members.

The missing causal comparison is now small and concrete: compare **full CP, common-gradient-only c, and frozen c**, from the exact same unfitted CP function and paired random streams. This tests the utility of adapting member coefficients under the declared optimizer, without introducing a new architecture, contrastive objective, router, or grid. It is a separate prospective mechanism study; it supplies no novelty claim.

## Reused conclusions first

Index v26 SHA `0b07746440f0376f235cec368a4a581c2c11835a1fefe0153b3bc3b946aea8ad` was consulted before retrieval. Fifteen selected conclusion records were reused: TabM; StarSSE; FoRDE; graph-expert diversity; graph ensemble collapse; R-GCN; SEA; continuing shared-normalization diversity; REEF/DRSA; GNN-FiLM; conditional CP sharing; ADaMoRE; LHGEL; recent tensor adapters. HGEN implementation/challenger work was excluded.

Task-error diversity, true-label-gradient repulsion, continuous shared-backbone diversification, graph expert diversity, and conditional diagonal/tensor adapters are established. The retained graph-collapse conclusion concerns selected GCN/GATv2 settings and excludes graph transformers; it does not prove HGT route collapse. The existing decision-diversity and shared-gradient records already state the mean-logit Jensen identity and own/pool trade-off. Those are reused mathematical conclusions, not new results here.

## One genuinely new primary method scope; one named retained revisit

**Wood et al., A Unified Theory of Diversity in Ensemble Learning**, JMLR 24(359):1–49, published December 2023. New scoped read of pages 9–14: centroid-combiner ambiguity, bias/variance/diversity trade-off, general-loss effects, and cross-entropy Corollary 9. Pages 9, 10 and 14 were rendered and inspected to verify equations. No full paper/proof, author-code, or empirical reproduction. The normalized geometric mean is exactly softmax of mean logits; diversity subtracts from ensemble loss only alongside the changing member-quality terms. The paper explicitly does not establish that maximizing diversity improves risk. [Primary](https://jmlr.org/papers/v24/23-0041.html).

**Pang et al., Improving Adversarial Robustness via Promoting Ensemble Diversity**, arXiv:1901.08846v3, 2019-05-29. ADP is absent from v26's paper records, but the historical decision-diversity report explicitly lists it as previously inspected. Today's §§3.1–3.3 read and Eq.3–6 check therefore count as **one retained revisit for a named unresolved dimensional question**, not a new paper. ADP excludes the true-class probability, normalizes the remaining probability vectors and maximizes log determinant of their Gram matrix, together with ensemble entropy. It uses probability pooling, separate full networks, an adversarial objective, and an unlimited-output-capacity analysis. These are not guarantees for shared HGT/mean logits. [Exact primary](https://arxiv.org/html/1901.08846v3).

There is a decisive actual-shape obstruction: DBLP has C=4 classes and M=4 members. The non-true-class vectors lie in R^(C−1)=R^3. Their M×M Gram has rank at most 3, so its determinant is identically zero. Bare ADP logdet is undefined. The same problem holds whenever M>C−1. Adding epsilon or using a pseudodeterminant changes the objective and does not inherit ADP's analyzed method. This rules out a direct port, rather than ruling out all useful diversity regularization.

## Rigorous falsifiers of separation as an explanation

Let z_m be member logits, zbar their mean, p_m=softmax(z_m), and pstar=softmax(zbar). The established exact identity is

```
mean_m CE(z_m,y) = CE(zbar,y) + A,
A = mean_m logsumexp(z_m) − logsumexp(zbar)
  = mean_m KL(pstar || p_m) >= 0.
```

A is label-independent at fixed logits. In a CP-minus-control comparison,

```
Delta pooled NLL = Delta mean-member NLL − Delta A.
```

An increased A can be exactly offset by worse members. Ordinary mean-member CE penalizes A at fixed pooled logits; it does not directly preserve a useful specialization. Wood supplies the matching loss/pooling interpretation; the exact Jensen identity was already recorded in this project. Do not substitute probability-pool ambiguity, error correlation, or an embedding loss for this predictor.

Three precise cautions follow:

- `z'_m = zbar + alpha*(z_m−zbar)` leaves the served predictor exactly unchanged for every alpha, even though member separation changes. At alpha=0 all output routes collapse. Thus a served predictor's improvement cannot be attributed merely to its current centered output spread.
- A perturbation of final hidden features in the classifier's nullspace changes embedding distance without changing logits. This is a counterexample to a generic embedding-separation guarantee, not a claim that every such perturbation is reachable by CP factors.
- In the identical-route affine limit, an antithetic rank-one perturbation `c_m*v` cancels from the mean when mean(c)=0. Actual HGT has different base factors and nonlinear private states, so this is a limiting falsifier, not a prediction of actual cancellation. Nonzero Jacobians do not guarantee a beneficial contraction with the label loss.

Label-relevant error dependence and ambiguity analysis is owned by the root; no duplicate analyzer is implemented here. These identities are bookkeeping/falsification, not causal or generalization certificates.

## Minimal separate coordinate-update study

The three arms below are necessary to distinguish a changing shared residual from changing member contrasts. Use all five already fixed DBLP seed/split blocks, for **15 declared terminals**. Start each pair/triple from the same newly constructed, unfitted CP state, with identical q/u/a/b/core, four member streams, recipe, loss, mean-logit pooling, native selector, and budgets. No fitted-state inheritance, validation-selected intervention time, resampling, new seeds, or broad tuning. Exact source/replay qualification and release remain root-owned.

| Arm | c operation after ordinary backward, before AdamW | What remains trainable |
| --- | --- | --- |
| Full CP | Native c.grad unchanged | c, q/u, base fast factors, shared core and all native parameters |
| Common-gradient-only c | Each layer: c.grad becomes its member mean broadcast over the existing four coordinates | All original parameters and original AdamW state remain |
| Frozen c | Retain c at its copied initial value with no gradient/update/decay | q/u, base fast factors, shared core and all native parameters |

Attribution: fixed c0 acts as a fixed member descriptor in the established conditional CP/gating parameterization (1611.09345v1); freezing one factor mode is an ordinary ablation. The fixed orthogonal gradient projector is standard constrained-optimization geometry. Retained PCGrad/GEM provide related gradient-manipulation precedent, but neither is exactly this unconditional projector. No complete published identical three-arm HGT study was established or required to make the control scientifically useful.

Common-gradient-only retains the actual four c coordinates; do not replace them with a new scalar, average c before the forward, or average parameter values after the step. Project each layer independently with `G=11^T/4`. Record native and projected c gradients. Keep the original AdamW moments/step convention, zero initial moments, epsilon, learning-rate schedule and decoupled decay; do not inherit a fitted full-CP optimizer. Frozen c should remain serialized under its original name, but never be updated or decayed. Its optimizer-state allocation differs and must be disclosed.

Let P=I−11^T/4. With equal initial c moments, the common-gradient arm has equal moments in every c coordinate and a common adaptive displacement k_t:

```
c_(t+1) = (1−lr_t*wd)*c_t − k_t*1,
P c_(t+1) = (1−lr_t*wd)*P c_t.
```

The c mean can learn; centered contrast direction is fixed, and its magnitude changes through the known native decay. Three steps of fabricated scalar AdamW arithmetic verified this recurrence to 8.7e−19, with equal moments. This is not a source/runtime fixture. Finite arithmetic defects must still be qualified in the actual source.

Full vs common-gradient tests the incremental utility of coordinate-specific c gradient components under this optimizer. Full vs frozen tests the complete permission for c coordinates to change, including common shifts and optimizer/decay effects. Common-gradient vs frozen includes shared-mean adaptation **and** known contrast decay; it is not a pure shared-component contrast. q/u continue to learn in every arm, so all three retain learned member × relation interaction. Different gradient fields also change later q/u/core trajectories; no comparison identifies one isolated causal scalar effect.

## One representative diagnostic and interpretation rule

CP has a scale/sign gauge: scaling c can be undone by reciprocal scaling of q or u. Raw c norms are therefore inadequate. Per layer, decompose c=mean(c)*1+Pc and q=mean(q)*1+Qq, with Q the relation-centering projector. The double-centered interaction tensor is `(Pc) ⊗ (Qq) ⊗ u`; its norm is `||Pc||*||Qq||*||u||`.

The representative diagnostic is the sign-invariant departure of the centered c profile from its initial direction:

```
D = 1 − ((Pc)^T(Pc0))^2 / (||Pc||^2*||Pc0||^2).
```

Common-gradient and frozen controls should have D≈0, within predeclared finite tolerances. Record the double-centered tensor norm as the validity companion: if it or Pc is effectively zero, mark the direction diagnostic undefined/inactive. A positive D in full CP proves a changed relative member profile, not useful diversity. Full c can also develop a nonzero mean and hence a common relation residual; common-gradient deliberately permits this alternative explanation.

For an incremental-utility screen, prospectively reuse the policy margin without altering the existing gate: all 15 terminals valid; full CP mean selected raw NLL at least 0.005 nats better than **each** control; at least 4/5 paired wins against each; mean macro-F1 nondecline. Paired intervals stay descriptive. If full matches common-gradient, adaptive private c contrasts are not shown necessary. If full matches frozen, allowing c to adapt is not shown necessary. Neither result rules out useful q/u adaptation or fixed-profile relation conditioning. A full-arm win supports this coordinate-update policy on the declared graph, not a new operator, a global diversity theorem, or an isolated whole-interaction mechanism. The previously adopted native-challenger and ACM/heldout requirements remain intact.

## Scope and discovery limits

Writes are confined to this new directory. No original dataset, label payload, checkpoint, score file, GPU, scientific remote execution, or source/protocol edit was performed. A historical fitted-state numerical summary appeared incidentally in a bounded keyword-search output; it was ignored and supplies no conclusion or measurement. Primary-paper example figures/qualitative results in the selected scopes were incidental and are not transferred to HGT.

The first guessed arXiv ID for Wood was wrong (spin-qubit metadata); it received no method-read credit and is retained as a rejected discovery. The correct JMLR primary supplies the new read. Crossref title discovery was irrelevant and provides no negative literature evidence. No exhaustive novelty search, full-paper count, cumulative read total, author-code reproduction, source compatibility, utility, or cost qualification is asserted.
