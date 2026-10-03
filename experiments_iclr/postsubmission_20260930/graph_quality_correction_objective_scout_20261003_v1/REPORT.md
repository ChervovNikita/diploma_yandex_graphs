# Shared-ensemble correction credit: negative scout

3 October 2026. **No distinct trainable quality modification is recommended from this scout.** One hypothesis was examined: train private factors using each member's leave-one-out contribution to correcting the served classifier, while keeping the shared core and uniform inference pool. Two apparent realizations reduce to established pooled-loss/ambiguity machinery. This rejects the claimed label-specific correction mechanism, not every possible empirical benefit of those objectives or every GNNM extension.

This packet reuses the saved literature conclusions first. It opens no datasets, result files, fitted state or active model source; it performs no server execution, training, numerical gate or grid. Parent reports the completed Squirrel support pilot as a practical tie and the live HGT utility as unproven; neither was independently reopened. Resource availability is not a scientific rejection reason.

## Assumptions and proposed hypothesis

There are M≥2 differentiable member routes z_m(v;theta,phi_m) in R^C. theta is the entire shared core, and phi_m is private to route m. Routes may have different nonlinear graph states and different shared Jacobians. The served classifier is softmax(zbar), where zbar=M^-1 sum_m z_m. All losses use the same TRAIN nodes, labels, nonnegative node weights and scalar normalization. No inference router, probability averaging, teacher labels or hidden representation penalty is introduced.

Hypothesis: “A member's decrease of CE relative to the committee without that member gives an additional, label-specific signal that teaches private factors complementary correct-label corrections.” The apparent attraction is to reward corrective contributions rather than arbitrary separation. The following algebra defeats that description for the symmetric realization.

## 1. Averaged removal credit loses its label term

Define A(z)=logsumexp(z), ell(z,y)=A(z)-z_y, and

```
z_-m = (sum_(j != m) z_j)/(M-1),
B = mean_m [ell(z_-m,y)-ell(zbar,y)].
```

Both full and removed committees are normalized mean-logit predictors, using their respective member counts. Because mean_m z_-m=zbar, exactly

```
B = mean_m A(z_-m)-A(zbar)
  = mean_m KL(softmax(zbar) || softmax(z_-m)) >= 0.
```

The expression is **label-independent at fixed logits**. The result holds pointwise, then under the common weighted TRAIN average; it does not require identical Jacobians, an affine predictor, independent parameters or small perturbations. An individual removal credit can depend on y. Its uniform sum/mean cancels the linear target-logit term.

Writing c_m=z_m-zbar gives z_-m=zbar-c_m/(M-1). Thus B is the ordinary loss-matched ambiguity of a committee whose contrasts have been reversed and contracted. Maximizing it can change disagreement without changing the served prediction. For example, binary true-positive margins with fixed mean b<0 and M=4 contrasts (+a,-a,+a,-a) give

```
B = [softplus(b+a/3)+softplus(b-a/3)]/2-softplus(b) > 0  (a != 0),
```

although the served classifier remains wrong with probability sigmoid(b)<1/2. This is an algebraic counterexample, not a measured graph outcome.

A supervised base CE plus a reward for B still contains label supervision through that base CE. The conclusion is that **B adds an ambiguity tradeoff, not a new label-specific correction principle**. For M=2 the removed routes are the original peers and B equals the ordinary own-versus-pool Jensen gap: mean-own CE minus lambda B is exactly GNCL's own/pool interpolation for 0≤lambda≤1. For M=4, B is ambiguity of the contracted leave-out routes, so it must not be called literally the same original-route GNCL formula. No useful reason for this particular contraction was established here. A positive empirical effect remains possible and would need attribution as an objective adaptation.

Positive-part clipping, winner weights, label-dependent masks or an asymmetric credit sum can retain label dependence. Each changes this assessed operation. MCL/TreeNets, SEA and error-reweighted graph boosting are already close specialization/error-credit priors. None of those alternatives is promoted here without a separately specified complete operation and discriminating rationale.

## 2. Detaching peers does not create a new correction update

A second implementation might form the pooled loss separately for each route while detaching its peers:

```
q_m = [z_m + sum_(j != m) stop_gradient(z_j)]/M,
L_local = sum_m ell(q_m,y).
```

Every forward q_m equals zbar. Let s=softmax(zbar)-one_hot(y), J_theta,m=partial z_m/partial theta and J_phi,m=partial z_m/partial phi_m. Automatic differentiation gives

```
grad_theta L_local = (1/M) sum_m J_theta,m^T s,
grad_phi,m L_local = (1/M) J_phi,m^T s.
```

These are **exactly the ordinary pooled-CE gradients**, including the shared-core gradient. The fact that the numerical scalar L_local equals M times pooled CE does not make its stopped-gradient derivative M times that gradient. Replacing the sum over local losses by a mean scales every gradient by 1/M; multiplying that mean by M restores the exact gradient identity. Arbitrary scaling must not be described as an identical AdamW transition because epsilon, decay and optimizer state matter.

Exact update equivalence uses the same forward graph, node weights, deterministic or matched stochastic realizations, buffers and optimizer inputs/state. It also assumes route m's private parameters do not affect other routes through a peer-message feedback path. Cross-route parameter dependence invalidates the private-gradient equation; that would be a different architecture/operation. Shared theta alone does **not** invalidate the identity. Separate peer recomputation with independent dropout, or a different optimizer transition, is likewise a changed operation rather than complementary credit.

No graph-specific correction is introduced by putting the same node weighting on every local loss. Member-dependent topology-derived masks change the aggregate gradient and require their own attribution; the existing graph-error refresh/shared-gradient proposals already investigate specific such responses.

## Closest prior and overlap

| Saved verified source | Relevant established operation and boundary |
|---|---|
| Buschjäger, Pfahler and Morik, *Generalized Negative Correlation Learning for Deep Ensembling*, [arXiv:2011.02952v2, §4.1 Eq.5](https://arxiv.org/html/2011.02952v2#S4.E5) | Member/pool loss interpolation and loss-sensitive output diversity. The M=2 reduction above is exact; the M=4 contracted leave-out ambiguity is attributed to this family without claiming literal equality to Eq.5 on the original members. |
| Wood et al., *A Unified Theory of Diversity in Ensemble Learning*, [JMLR 24(359), 2023](https://jmlr.org/papers/v24/23-0041.html), saved pp.9–14 | Loss-matched centroid/ambiguity; mean logits correspond to a normalized geometric probability pool. This supplies the combiner interpretation. Spread alone does not imply better risk. |
| Zou, *Self-Error Adjustment: Theory and Practice of Balancing Individual Performance and Diversity in Ensemble Learning*, [arXiv:2508.04948v1, §III-A](https://arxiv.org/html/2508.04948v1) | Task-error/complementary-target member training. Inspected classification uses one-hot regression, and independent-coordinate constant/detachment arguments cannot silently transfer to a shared core. No SEA implementation equivalence is claimed. |
| Lee et al., *Why M Heads are Better than One*, [arXiv:1511.06314v1, §§5/6.2](https://arxiv.org/abs/1511.06314); *Stochastic Multiple Choice Learning*, [arXiv:1606.07839v3, §§2–3](https://arxiv.org/abs/1606.07839v3) | Shared branches, oracle/winner assignment and ensemble specialization. They are direct attribution for replacing symmetric credit with lowest-loss member assignments, rather than a new correction principle. |
| *AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models*, [arXiv:1908.05081](https://arxiv.org/abs/1908.05081), saved method conclusion | TRAIN-error reweighting, staged graph classifier specialization and centered-log-probability combination. Graph weighting of corrective credit is not a new task-level principle by itself. |

The main arguments reuse saved source summaries. A final GNCL title-verification search also exposed retained source keyword hits, accounted below. These sources establish the stated ingredients; they do not certify exhaustive nearest-prior closure or an exact duplicate of every possible shared-HGT composition.

## Decision and experiment boundary

**Do not launch this as a new quality mechanism.** The decisive falsifier is algebraic: symmetric removal credit is label-independent ambiguity, and correctly normalized peer-detached pooled backward reproduces ordinary pooled-CE gradients under the stated architecture. A graph/GPU experiment is unnecessary to settle those claims.

This is not a claim that a contracted-ambiguity regularizer cannot help. It is a finding that this scout supplied no credible distinct gap or predictive argument for that exact variant beyond known ensemble-objective tradeoffs. The already saved shared-pooled/native objective comparator is the appropriate existing control if parent pursues objective utility. No duplicate comparator, numerical gate, tuning grid or new driver is proposed. The parent's live HGT study and retained graph-response hypotheses are unchanged.

## Read accounting

Consulted index v27 and saved distinct-idea reports before public discovery. New primary method scopes: 0. One narrow retained GNCL keyword revisit occurred during title/identifier verification: the saved metadata/abstract plus source heading and experimental-setting hits (gncl_blocks.json lines 725, 779 and 803 and an adjacent conclusion excerpt) appeared in search output. No new method or numerical result premise was taken from those incidental hits; no full-paper read is claimed. Four broad, four focused and four title-level OpenAlex metadata queries are saved under discovery/. Their mostly survey/stacking/causal-cross-fitting results supplied no method evidence and no absence claim. No metadata-only ensemble-meta-learning title is used to assert method equivalence.
