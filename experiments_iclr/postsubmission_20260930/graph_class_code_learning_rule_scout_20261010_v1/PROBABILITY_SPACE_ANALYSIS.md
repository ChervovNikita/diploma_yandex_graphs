# What the class-partition risk changes

These are exact elementary identities for the proposed loss with a fixed question
pack. They are not numerical experiments, novel theorem claims or guarantees
about trained graph models.

## 1. Proper risk and a competence anchor

Let A_j be the2×C membership matrix for one binary class partition. For a class
probability vector p, A_j p contains its two group probabilities. Given true
class y, the group loss is `−log (A_j p)_(group_j(y))`.

For a fixed conditional label law q and fixed masks,

```
E_q[L(p,y)] − E_q[L(q,y)]
 = KL(q||p) + λ/J * sum_j KL(A_j q || A_j p).
```

With λ=.5 and J14 this is the report's loss. The first term is strictly proper;
the additional terms are proper coarse-label scores. The well-specified target
distribution stays q, rather than being pushed toward member disagreement.
Masks constructed from a finite TRAIN sample do not make development scores
unbiased or grant independent-task evidence. The identity conditions on a
fixed pack; it does not analyze code selection or generalization.

Pointwise, the group containing y has mass at least p_y. Hence each group loss
is between0 and full CE, giving `CE <= L <= (1+λ)CE`. No arbitrary reward can
drive the objective to−infinity. A finite training minimum, native convergence,
validation accuracy and mean/worst member quality are separate empirical matters.

## 2. Every class-probability direction is visible to the pack

Let B be the C×(C−1) signed question matrix. The stipulated rank of
`[B_a−B_0]_(a=1..C−1)` is C−1, equivalently the C×C matrix[1,B] is nonsingular.
Let S=(B+1)/2 be positive-group membership. If p,q are probability vectors with
`S^T p = S^T q`, their difference d satisfies

```
1^T d = 0,
B^T d = 2*S^T d − 1*1^T d = 0.
```

Nonsingularity gives d=0. Thus all question probabilities together determine p.
For this auxiliary, extra hidden coordinates cannot satisfy a separate decoder
while leaving every class probability unchanged: it is computed from those
class probabilities themselves. This does not say its scalar loss identifies
p on one observed label, nor that every representation gradient is nonzero.
Ordinary fine-class CE is retained and no conditional-independence assumption
among questions is made.

## 3. It changes a structured risk geometry

At q with positive class and group masses, the population excess score's
Hessian with respect to p is

```
H(q) = diag(1/q) + λ/J * sum_j A_j^T diag(1/(A_j q)) A_j,
```

restricted to probability-simplex tangent directions. The additional terms
are positive semidefinite. Different packs can emphasize different class-mass
directions while keeping the same optimum. This is a loss-space interpretation,
not coordinate-invariant model optimization or a new natural-gradient algorithm.
The actual model update also depends on its prediction Jacobian, optimizer,
dropout and current state. Graph W merely influences which A_j are selected;
it does not certify a task-aligned curvature or predictive gain.

For one label y and group S containing y, the derivative with respect to native
class logits is `p_c − 1[c in S]*p_c/sum_(a in S)p_a`. This redistributes
pressure according to class groups. It can increase a wrong class inside S;
other questions and own CE must resolve the fine-class decision. No per-node
margin or accuracy safety theorem is claimed.

## 4. Exact prior identity and symmetry limit

For any one partition,

```
−log p_y − λ log p_(S_y)
 = −log[p_y/p_(S_y)] − (1+λ)log p_(S_y).
```

This is depth-two HXE with weights1 and1+λ. The proposed objective is an average
of established proper hierarchical scores with a full-class anchor. It is not
a new elementary loss. ECOC supplies multiple class partitions, spectral ECOC
supplies class-graph conditioning, and shared ECNN supplies parameter-sharing
ancestry. Only the complete member-owned graph-cut construction/coupling is a
conditional proposed increment; exact novelty remains unestablished.

If all member predictions and shared prediction Jacobians are identical, the
shared-gradient average equals the all-pack single's corresponding gradient.
Different private paths can receive different pack gradients, but equality or
weak differences are possible. At a well-specified optimum every member may
predict q. No mandatory symmetry breaking, posterior independence or accuracy
gain follows. Complete single/common-pack/graph-free/untied controls, full repairs
versus harms and independent confirmation decide whether this finite-training
hypothesis is useful.
