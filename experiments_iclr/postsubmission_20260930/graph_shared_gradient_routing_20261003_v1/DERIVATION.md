# Shared motion, conditional margins and projection

Source-only local mathematics. No trained model, dataset, checkpoint or logits were accessed or executed.

## Erasure is a response property

Let shared parameters be theta, private parameters phi_m, member logits z_m, pooled logits z_bar=mean_m z_m and shared Jacobians J_m=partial_theta z_m. For frozen TRAIN weights and correct-margin cotangents t_y,

    S_m = sum_v w_m(v) t_y(v)^T (z_m(v)-z_bar(v)),
    h_m = sum_v w_m(v) (J_m(v)-J_bar(v))^T t_y(v),
    Delta_shared S_m = h_m^T d_theta + O(||d_theta||^2).

Thus negative h_m^T d_theta means local erosion of this declared useful margin functional. It is not a theorem about all member errors or generalization. Stop gradients through current support and graph weights: differentiating their selection would create a different operation.

If J_m=J_bar on the protected roots, h_m=0 and common shared motion cannot change these centered logit contrasts to first order. In a fixed affine model with exactly equal member shared Jacobians, the preservation is exact. Private updates, different route Jacobians, nonlinear shared motion and differing state can break this null. Sharing can also improve every member together; contraction of a probability difference due solely to higher common confidence is not logit-contrast erasure.

The exact CE/BCE decomposition remains L_own=L_pool+D, with D a nonnegative, label-independent-at-fixed-logits Jensen/KL gap. It does not say D is useful. Raw SGD driven by gradient D can contract contrasts; a shared block changes them only through differing member Jacobians. Actual AdamW displacements depend on moments, scaling and decay. We use the actual proposed displacement in the constraint rather than equating it to a raw averaged gradient.

## Aligned loss gradients can erase a correct contrast

A symbolic two-member binary affine boundary witness has target y=1, common probability p=3/4, shared logit Jacobians J1=(-4,0), J2=(-4,-4). At equal margins its CE gradients are

    g1=(1,0), g2=(1,1), g1^T g2=1>0,
    a=g_pool=(1,1/2), d0=−(g1+g2)/2=(−1,−1/2).

The member-1 binary correct-margin contrast gradient is h=J1−J_bar=(0,2). Then h^T d0=−1: the shared own-CE step erodes that contrast despite positively aligned member gradients. PCGrad's negative-pair trigger does not act.

At the equal-margin boundary the protected advantage itself is zero, so our support is absent. Give member 1 a private affine intercept +epsilon and member 2 −epsilon, keeping their pooled logit fixed and the common margin positive. For sufficiently small epsilon>0 both members remain correct, member 1 has larger target probability and positive relative margin, its h remains (0,2), and the strictly positive gradient alignment and negative erosion persist by continuity. This supplies a valid active-margin neighborhood; it is not a graph or source-qualified construction. The scalar graph is trivial and supplies no evidence for topology's value.

For the boundary vectors the minimum pooled-neutral correcting displacement is

    c=(−1/4,1/2), d_star=(−5/4,0),
    a^T c=0, h^T d_star=0,
    a^T d_star=a^T d0=−5/4, ||c||/||d0||=1/2.

Both member loss derivatives still indicate descent. In this affine binary limit the pooled logits after the compared steps are identical exactly, while the protected relative logit margin is retained. Consequently immediate pooled improvement is not promised. A later own-CE update can differ through softmax curvature, as in the already sealed CE note. Retaining disagreement can increase own CE relative to the native proposal at the same pool; a finite own guard against the private-only reference allows ordinary progress without requiring this particular Jensen contraction.

All displayed vector arithmetic is symbolic. No optimization trajectory or numeric model forward was run. Step vectors scale by eta; the finite affine/continuity statements do not transfer as native AdamW guarantees.

## Minimum correction and feasibility

At reference B after the native private proposal, let a=gradient_theta L_pool(B), H=[h_1,...,h_r], b=−H^T d0. For a nonzero a define P_a=I−aa^T/(a^T a), B_H=P_a H. Corrections preserving the native pooled directional effect lie in the range of P_a. The problem is

    min_c 0.5||c||^2 subject to a^T c=0, H^T c>=b.

Its dual is

    max_(lambda>=0) lambda^T b−0.5 lambda^T G lambda,
    G=H^T P_a H=B_H^T B_H,
    c_star=B_H lambda_star.

With at most four specialist columns, deterministic enumeration of active sets can qualify the KKT solution; singular/rank-deficient sets need a pseudoinverse and consistency checks. The primal/dual formulas are established constrained optimization, not a new theorem or driver.

If h is parallel to a and h^T d0<0, no pooled-neutral correction can repair it. More generally Farkas' obstruction is a nonnegative lambda with B_H lambda=0 but lambda^T b>0. This can arise even in a large parameter space: parameter count does not guarantee a feasible useful correction. A correction-norm cap can also exclude every feasible solution. Preserve native fallback and the reason.

At a feasible accepted local solution,

    a^T(d0+c_star)=a^T d0,
    H^T(d0+c_star)>=0.

These are first-order statements about the reference dropout-off map with private parameters fixed at phi_plus. Actual finite scores/pooled CE need independent finite guards. Full shared/private trajectory changes, noisy training and carried native moments can undo the benefit later.

## Graph conditioning is the choice of response constraints

The nonnegative two-hop P^2 transport is applied to positive, currently correct TRAIN margin advantages. Requiring current member target probability to exceed the pool prevents a merely larger class-centered contrast from qualifying on its own. Requiring a_m>0 ensures the protected margin is positive. Fixed hard ties and zero-support handling are part of the operation.

Topology can change the weighting of these specialist supports and therefore H, the feasible cone and c. This is more specific than calling each member a task, but it is still a graph-conditioned application of response preservation. P=I and node-permuted P are exact same-operator controls. At identical routes the construction is null; it cannot manufacture useful complementarity from nothing. Preservation of current TRAIN specialists can entrench overfitting or impede necessary corrections, so heldout predictive tests are indispensable.

The protected S_m is an aggregate. Preserving it does not preserve every node margin, all outputs, member accuracy or error independence. Different positive supports can induce dependent or incompatible constraints. The graph prior has no favorable-sign or label-independence theorem.
