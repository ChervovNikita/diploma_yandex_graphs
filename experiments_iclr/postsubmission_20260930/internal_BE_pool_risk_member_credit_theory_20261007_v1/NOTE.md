# Ensemble risk gradients at individual member outputs

Probability-pooled multiclass log loss and hard-label binary mean-logit BCE
rescale each member's own per-object gradient by a positive scalar. Multiclass
mean-logit CE can change its class-vector direction; mean-output MSE can reverse
its sign. These are derivatives of established ensemble risks and their GNCL
interpolation. A graph policy using them still needs a specific useful graph
mechanism and a correctly attributed block allocation.

## Convention and scope

Fix one labelled object and one realized stochastic view, M members, finite
real logits and a constant lambda in [0,1]. Let F=(1/M)sum_m ell_m be mean own
risk, L ensemble risk, and J_lambda=(1-lambda)F+lambda L. A private block phi_m
affects only member m's output unless explicitly stated otherwise. Output
Jacobians are unrestricted. Common nonnegative object/view reduction weights
multiply all equations; Collab's positive/negative group reductions fit this
convention. Apply the identities separately to the two realized views before
averaging their losses. Pooling the views first defines a different objective.

The statements concern raw supervised gradients at a fixed parameter state.
Representation auxiliaries, optimizer history and deliberate block permissions
are separate contributions. No coefficient, task, computation or experiment is
adopted by these equations.

## Multiclass log loss on mean probabilities

For p_m=softmax(z_m), P=(1/M)sum_m p_m, hard class y and unit vector e_y,
ell_m=-log p_m(y) and L=-log P(y). Direct differentiation gives

```text
rho_m = p_m(y) / sum_k p_k(y)
dL/dz_m = rho_m * (p_m-e_y)
dJ_lambda/dz_m = [(1-lambda)/M + lambda*rho_m] * (p_m-e_y).
```

The responsibility rho_m is positive and sums to one across members. Thus each
private per-object gradient equals its own CE gradient times a detached weight
w_m=(1-lambda)/M+lambda*rho_m. Its logit direction stays unchanged. A member with
larger correct-class probability receives a larger multiplier; this does not
order gradient norms or guarantee that weak members improve. Lambda<1 gives a
multiplier floor (1-lambda)/M, not a competence guarantee.

This is ordinary fixed-gate mixture-likelihood responsibility. For fixed prior
weights pi_m, rho_m=pi_m*p_m(y)/sum_k pi_k*p_k(y). Learned gate logits add their
own gradient pi_m-rho_m, so treating gate parameters as another weighted member
block would omit a term. Graph MoEs with hidden/message mixtures need their own
computation graphs; they are not automatically this output-mixture likelihood.

Two useful exact identities clarify what is being weighted:

```text
L = -log[(1/M)sum_m exp(-ell_m)]
L = sum_m rho_m*ell_m + KL(rho || uniform_M)
F-L = log AM_m[p_m(y)] - log GM_m[p_m(y)] >= 0.
```

Differentiating sum_m stopgrad(w_m)*ell_m gives the same current first-order
supervised gradient as J_lambda. Their scalar loss values are different.
Differentiating non-detached w_m*ell_m would add unwanted weight derivatives.
An autodifferentiated Hessian through stopped weights omits responsibility
derivatives; first-order equivalence does not supply higher-order equivalence.

## Binary BCE on the mean raw logit

Let y be exactly 0 or 1, s_m=sigmoid(z_m), bar_z=mean_m z_m and s=sigmoid(bar_z).
For ell_m=softplus(z_m)-y*z_m and L=softplus(bar_z)-y*bar_z,

```text
dL/dz_m = (s-y)/M
dJ_lambda/dz_m = [(1-lambda)*(s_m-y) + lambda*(s-y)]/M
              = w_m*(s_m-y)
w_m = (1-lambda)/M + lambda*a_m
a_m = (s-y)/[M*(s_m-y)]
    = s/(M*s_m)                         if y=0
    = (1-s)/[M*(1-s_m)]                 if y=1.
```

Finite real logits make a_m positive. Hence private per-object BCE credit is
again detached positive own-loss weighting. The pool residual is the same for
every member; the weights express it relative to each member's own residual.
They generally do not sum to one and can be large when an own residual is small.
The ratio is a mathematical identity, not a prescription to form unstable
floating-point divisions. With soft binary targets, the ratio can be negative
or undefined at an own zero residual; the hard-label positivity claim then fails.

Positive per-object weights can change a batch or two-view aggregate parameter
direction because its constituent gradients differ. They do not establish a new
per-object binary credit direction.

## Multiclass CE on the mean raw logits

Let bar_z=mean_m z_m and q=softmax(bar_z). Then

```text
dL/dz_m = (q-e_y)/M
dJ_lambda/dz_m = [(1-lambda)*(p_m-e_y) + lambda*(q-e_y)]/M.
```

The pool derivative is scalar-collinear with p_m-e_y exactly when the
conditional nontruth-class distributions agree. The mixed derivative has the
same condition for lambda>0; lambda=0 trivially gives own risk:

```text
q_c/(1-q_y) = p_m(c)/(1-p_m(y)) for every c != y.
```

When they agree, the pool multiplier is
(1-q_y)/[M*(1-p_m(y))]. With two classes the condition is automatic, agreeing
with the binary case; with three or more classes it generally fails. The pool
gradient then changes relative wrong-class credit while retaining the CE sign
pattern: negative truth coordinate, positive nontruth coordinates. A parameter
Jacobian can erase this output-direction distinction; it is not a guaranteed
parameter-update or accuracy difference.

Mean-logit softmax is the normalized geometric probability pool,
q_c proportional to product_m p_m(c)^(1/M). It differs from the arithmetic
probability pool used by WikiCS. Replacing one with the other changes the stated
risk/combiner contract and needs explicit attribution.

For both multiclass mean-logit CE and binary mean-logit BCE, the exact ambiguity
gap is label-independent at fixed outputs:

```text
F-L = mean_m KL(q || p_m) >= 0                  multiclass
F-L = mean_m KL(Bern(s) || Bern(s_m)) >= 0       binary
J_lambda = F-lambda*(F-L).
```

The KL direction is pool-to-member. These are Jensen/Bregman identities for
logsumexp/softplus, not new diversity or credit primitives.

## Scalar MSE on the mean output

Use ell_m=(z_m-y)^2 without a factor 1/2. Put e_m=z_m-y and
bar_e=mean_m e_m. The requested mixture, including lambda=1/2, gives

```text
dJ_lambda/dz_m = (2/M)*[(1-lambda)*e_m + lambda*bar_e]
dJ_half/dz_m   = (e_m+bar_e)/M
F-L = mean_m (z_m-bar_z)^2 >= 0.
```

Relative to own mean-risk gradient2*e_m/M, the sign reverses exactly when
bar_e and e_m have opposite signs and lambda*abs(bar_e) exceeds
(1-lambda)*abs(e_m). At lambda=1/2 this requires abs(bar_e)>abs(e_m).
At e_m=0 but bar_e!=0 and lambda>0, pool credit is nonzero while own credit is
zero. Consequently a positive detached own-loss weight cannot represent this
rule everywhere. A signed ratio can represent nonzero own residuals, but it
loses the earlier positivity and zero-residual interpretation.

This is standard regression compensation: a member can move farther from its
own target to offset other members' errors. It can lower pool error while
weakening a member. Keeping heads or shared weights on own risk controls their
assigned gradient; simultaneous internal-factor changes can still worsen member
predictions. Protected heads do not guarantee competence or generalization.

There is also a useful limit on the apparent new signal. In unconstrained output
coordinates with the same gradient-descent step size for all members,

```text
mean_m dJ_lambda/dz_m = (2/M)*bar_e.
```

The instantaneous mean-output step is independent of lambda; changing lambda
changes how disagreement contracts and permits individual compensation. Native
parameter Jacobians, sharing and Adam break that simple output-coordinate update
description. None of these equations guarantees improved graph regression.

Regression can expose signed residual disagreements that probability-pooled CE
does not create at the individual output. Any future native CoGNN/ZINC question
must first bind its actual training loss and evaluation metric. If training is
MAE/L1, its derivatives use residual signs: at lambda=1/2 opposite nonzero signs
cancel rather than reverse. The displayed MSE sign-flip claim does not transfer
unchanged. No CoGNN/ZINC task, loss change or compute is adopted here.

## Shared blocks and auxiliary qualifications

The probability/BCE detached-weighted sum over all members reproduces the
supervised first-order gradient for shared parameters too, by the chain rule.
If phi_m also affects other member outputs through peer messages, its gradient
is the sum of their weighted own-loss pullbacks; only the single-own-loss
private simplification fails. Fixed shared parameters alone do not invalidate
the identities.

Giving shared/boundary blocks unweighted F while only internal factors receive
J_lambda is a different block allocation. A single globally weighted own loss
would expose every block to those weights. The stated mixed-permission field
can be nonpotential, as the saved independent block-game note already explains.
Identical raw gradients yield identical first-order optimizer inputs at the same
state/history, but positive per-object weights alone do not constrain actual
Adam displacements or guarantee descent of every risk.

Alignment/residual representation losses add their own representation-Jacobian
pullbacks, usually outside any scalar own-CE weighting. Loss-weight equivalence
for the supervised term does not remove these contributions. The current
Wiki12 A/R attribution study measures those auxiliaries; these pool-risk
derivatives do not identify its mechanism or reinterpret its outcomes.

## Prior comparison and implication

| Saved prior scope | Consequence for the interpretation |
|---|---|
| [GNCL v2 §4.1 Eq.5](https://arxiv.org/html/2011.02952v2#S4.E5), plus the saved pinned author-code audit | Mean-own/pool interpolation is established. The inspected implementation backpropagates one scalar mixture through all reachable parameters; a restricted internal-block permission is an attributed allocation change. |
| [Unified Theory of Diversity in Ensemble Learning](https://jmlr.org/papers/v24/23-0041.html), saved centroid/ambiguity scope; classic NCL/GNCL regression ancestry | Loss-matched ambiguity and the geometric probability combiner are established. MSE's variance gap and raw-logit CE/BCE Jensen gaps should be attributed accordingly. |
| Saved [GMoE](https://arxiv.org/abs/2304.02806v2), [Node-MoE](https://arxiv.org/abs/2406.03464v1) and GraphMETRO scopes | Graph-aware expert weighting, specialization, balancing and selective gradient exposure already exist. The likelihood responsibility derived above is ordinary mixture calculus; those hidden/message architectures are not claimed to implement this exact formula. |
| Saved [AdaGCN](https://arxiv.org/abs/1908.05081) method scope and boosting ancestry | Error-weighted examples and ensemble-residual correction have prior ancestry. Boosting's staged new-learner/frozen-bank updates differ from simultaneous updates of existing shared/private members. The probability responsibility favors current correct-class mass, which is distinct from boosting's usual hard-example emphasis. |

The appropriate interpretation is an attributed ensemble-risk/block-allocation
control. Cases 1/2 alone cannot support a new per-object credit-direction claim;
cases 3/4 offer different directions already supplied by their standard pooling
losses. Graph placement, protected competence and useful serving behavior remain
the questions. No novelty, confirmation, optimizer remedy, expensive graph-policy
experiment or regression task follows from calling this a credit primitive.
