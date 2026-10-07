# What the own/pool mixture changes

7 October 2026. Exact loss algebra for the current risks, with saved GNCL/ambiguity ancestry. No scientific run, model/data/current outcomes, server contact, new variant or coefficient choice. Root supplied the existing fixed λ=.5; this note explains it rather than selecting a strength.

Let `D=L_own−L_pool`. Under the declared convex averaging contracts,

**`J=(1−λ)L_own+λL_pool = L_own−λD = L_pool+(1−λ)D`, with `D≥0`.**

This is the established GNCL member/pool interpolation. The two tasks' gaps have different meanings.

## 1. WikiCS: probability-pool CE

For one labelled node, let `p_m=softmax(z_m)`, `q_m=p_m(y)>0`, `Q=sum_m q_m`, `r_m=q_m/Q`, and let U be uniform over the **M members**. Then

```
L_own = −(1/M) sum_m log q_m
L_pool = −log(Q/M)
D = log(Q/M) − (1/M)sum_m log q_m
  = (1/M)sum_m log[(1/M)/r_m]
  = KL(U || r_y).
```

The orientation is U-to-responsibilities, not the reverse. This gap depends on the true class through q_m; it is not a class-distribution KL or hidden-feature diversity statistic. It is zero whenever the members assign the same correct-class probability, even if their other class probabilities differ.

The logit derivatives are

```
∂_{z_m} D = (1/M−r_m)(p_m−onehot(y))
∂_{z_m} J = [(1−λ)/M+λr_m](p_m−onehot(y)).
```

Thus pool supervision puts more credit on a member with a larger correct-class probability. For λ<1 the multiplier has floor `(1−λ)/M`; this is not a lower bound on a parameter-gradient norm or a competence guarantee.

## 2. Binary tasks: mean-logit BCE

Let `A(z)=softplus(z)=log(1+exp(z))`, `z̄=mean_m z_m`, and `ℓ(z,y)=A(z)−yz`. The label term cancels exactly:

```
D = mean_m ℓ(z_m,y) − ℓ(z̄,y)
  = mean_m A(z_m) − A(z̄).
```

This is the nonnegative Jensen gap of A, independent of y **at a fixed example and logits**. It is also the mean Bregman divergence

```
D = mean_m B_A(z_m,z̄),
B_A(u,v)=A(u)−A(v)−A'(v)(u−v).
```

With `p*=sigmoid(z̄)` and `p_m=sigmoid(z_m)`, this equals `mean_m KL(Bern(p*) || Bern(p_m))`. Here p* is sigmoid of the mean logit; it is not the mean of member probabilities. The mean first-order Bregman term vanishes because `mean_m(z_m−z̄)=0`.

```
∂_{z_m}D = [sigmoid(z_m)−sigmoid(z̄)]/M
∂_{z_m}J = [(1−λ)(sigmoid(z_m)−y)
             +λ(sigmoid(z̄)−y)]/M.
```

Every member receives the same pool residual, pulled through its own parameter Jacobian. Wiki's correct-class responsibilities do not apply here. For finite logits, strict convexity makes D=0 iff all member logits agree.

## 3. Preserve the exact reductions and views

All identities above are per supervised object/output. Apply the **same linear reduction** to own loss, pool loss and gap.

For one Collab view, with prescribed positive set P and negative set N,

`L_own=mean_{i∈P} L_own,i + mean_{i∈N} L_own,i`,

and identically for L_pool, J and D. In particular,

`D_Collab=mean_{i∈P}D_i + mean_{i∈N}D_i`.

There is no extra `.5` between the groups and no concatenated mean over unequal positive/negative counts. Pointwise y cancels from D_i; group membership and its denominators still come from the declared supervision contract. For MolHIV, use the declared mean over finite TRAIN target/output entries in the batch, with identical weights and masks for own and pool risks.

For the two stochastic views a,b,

`J_two=.5[J_a+J_b]`, `D_two=.5[D_a+D_b]`.

Compute each view's member pool first. Averaging predictions across views before taking the loss gives another risk. The `.5` averages the **two views**, not Collab's two label groups. These equalities hold at each realized old-state forward; stochastic training views are not dropout-off serving predictions.

## 4. Internal-only I versus global G

For the supervised part, partition parameters into shared core θ, private prediction boundaries ψ and internal factors φ. Write Pφ for the coordinate projection onto φ. Then

```
G: V_G = ∇L_own − λ∇D = ∇J
I: V_I = ∇L_own − λPφ∇D
I−G = λ(I−Pφ)∇D.
```

Equivalently, θ and ψ retain `∂L_pool+∂D`, while φ receives `∂L_pool+(1−λ)∂D`. At the current fixed λ=.5, the internal block receives half the gap derivative; shared/boundary blocks retain it fully. Any matched internal-only alignment auxiliary is added with the same permissions and cancels from this comparison.

The change is **where an established risk derivative enters the existing model**. It is not a separate graph-diversity force. Different internal Jacobians can turn the task residual into different changes in messages/attention, but their graph-evidence use must be demonstrated. Own-risk supervision does not freeze shared features, boundaries, predictions or competence: all depend on the changing internal factors. Include every real dependency when taking the private pullback; detaching an upstream representation can remove the desired derivative.

The equations describe supplied raw gradients at one common state. Adam moments/preconditioning and future states need not match. I is generally not the gradient of the global scalar J, so these identities provide no simultaneous descent or step guarantee. Neither exact Jensen gaps nor lower CE/BCE establishes superior accuracy, MolHIV ROC AUC or Collab Hits@50. Fixed-pool changes can alter member confidence/disagreement without adding useful evidence; J still penalizes a larger gap by `(1−λ)D` when λ<1.

## 5. Symmetry and coefficient limits

At exactly equal member predictions, ∇D=0. The gap term alone cannot break an exact deterministic collapse. Existing distinct initialization/stochastic views can supply asymmetry; the identity does not guarantee they will produce useful specialization. Wiki additionally has zero gap gradient whenever all q_m coincide.

At fixed predictions, `∂J/∂λ=−D≤0`. Optimizing λ **within GNCL's [0,1] range** favors endpoint1 whenever D>0; at D=0 it is indifferent. A truly unbounded λ can drive J below all bounds at a fixed positive gap. This differs from minimizing a positive auxiliary coefficient, which can favor weight0. It supports keeping the existing fixed strength rather than inventing an adaptive mixture rule; it does not prove any joint optimizer must converge to1.

## Attribution and scope

[GNCL `2011.02952v2`, §4.1 Eq.5](https://arxiv.org/html/2011.02952v2#S4.E5) explicitly provides the member/pool interpolation and λ∈[0,1]. Its checked §2 passage cites generalized ambiguity for exponential families (Hansen/Heskes) and earlier ambiguity work. Its own §3 decomposition uses a second-order Taylor remainder; the NLL/CE passages explicitly discuss nonzero remainders. **Do not identify the exact D above with GNCL Eq.4's Hessian quadratic after dropping that remainder.** The two explicit formulas were not stated in the checked GNCL passages; this is a scoped statement, not a full-paper absence claim.

The score-pool Jensen/KL identity is also already in the saved3October mixed-objective scout; saved Jeffares AppendixE scopes distinguish probability versus score pooling. The probability-responsibility KL identity and the binary Bregman expression here are elementary log-sum/convex-duality applications to those established objectives, not new theorems or learning principles. No additional citation search was necessary. Exact saved-primary passages and reuse bindings accompany this note.

New primary identities/retrievals/full-paper/proof credits: **0**. Targeted saved-GNCL passage checks are disclosed as an additional local scope within an already read paper, not another paper credit. No current family or canonical ledger was edited.
