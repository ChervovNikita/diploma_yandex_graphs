# Pooled CE curvature and centered private contrasts

Source-only mathematical analysis, 3 October 2026. No model, dataset, logits, checkpoint, remote host or GPU was accessed or executed. These identities are explanatory mathematics, not novelty or generalization results.

## 1. Exact CE/BCE identity and pool convention

At one observed target let z_m be member logits, z_bar=mean_m z_m, p_m=softmax(z_m), p_g=softmax(z_bar), and A(z)=logsumexp(z). CE(z,y)=A(z)-z_y gives

    mean_m CE(z_m,y) = CE(z_bar,y) + D,
    D = mean_m A(z_m)-A(z_bar)
      = mean_m KL(p_g || p_m) >= 0.

Indeed log p_m=z_m-A(z_m)1. Averaging the KL linear terms cancels mean_m(z_m-z_bar), leaving exactly the displayed Jensen gap. The KL orientation is **geometric pool to each member**. For fixed logits D is independent of y. It is unchanged by a separate scalar class-logit shift for each member/node. Equal supervised row weights and the same observed-target mask across members are required when aggregating this identity. Missing BCE targets are simply excluded under the same mask/weights; a different mask or normalization for each member changes the decomposition.

For scalar BCE use A(z)=softplus(z), p_m=Bernoulli(sigmoid(z_m)) and p_g=Bernoulli(sigmoid(z_bar)); the same identity and KL orientation hold. Geometric averaging operates on binary odds. None of these identities concerns the arithmetic probability mixture p_a=mean_m p_m. Its NLL gap log p_a,y-mean_m log p_m,y is a different, label-dependent quantity.

This verifies the root POOLED_OBJECTIVE_NOTE.md as stated. The identity, own-versus-pooled objective distinction and its GNCL interpolation were already retained in the bottleneck/decision-diversity reports. No novelty follows from writing the gap as KL.

## 2. Instantaneous output splitting and next-update CE dynamics differ

Let z(phi)=J phi+b be a common fixed affine **logit** model, with TRAIN block J_T. Members start at phi_m=c+delta_m, mean delta_m=0. Their initial mean logits are exactly z(c), including off-TRAIN rows. With separately applied own-CE full-batch SGD steps of effective step eta,

    c_next = c - eta J_T^T(mean_m p_m-Y),
    c_baseline_next = c - eta J_T^T(p_g-Y).

Thus the exact mean-update difference is -eta J_T^T(mean_m p_m-p_g). A mean-member loss implementation may multiply each private gradient by 1/M; eta must include that source scaling. Shared-parameter updates and AdamW moments are not represented by this simple independent affine formula.

For squared loss the gradient is affine in phi, so the centered mean trajectory remains identical under common full-batch affine updates. CE is different because softmax is nonlinear. The CE difference is already possible with an affine predictor, without any logit Hessian.

For a general nonlinear logit map, instantaneous balanced splitting changes mean logits by approximately 0.5*mean_m Hessian(z)[delta_m,delta_m]. Its pooled-loss term is the residual-weighted **logit curvature**, as in the retained Splitting Steepest Descent matrix. In an affine head that Hessian is zero; initial pooled CE cannot improve through this instantaneous mechanism. Own-member CE nevertheless increases by the nonnegative Jensen penalty, and its next update can differ as described below. These are different objects, not competing names for one Hessian.

The current initializer's bound slice includes **stem.S and head.R**. Stem perturbations generally need not be affine in the outputs, so the affine-head mean-logit guarantee cannot be silently assigned to that complete slice. A head-only counterfactual must qualify exact affine dependence on the chosen private input factor and identical initial hidden states. The sealed support-amendment and current source remain unchanged.

## 3. Exact small-contrast coefficient

At a TRAIN row define a_m=J_v delta_m, C_v=mean_m a_m a_m^T, and p=softmax(z(c)_v). The softmax second directional derivative is

    D2 softmax(z)[a,a]_i
      = p_i * ((a_i-p^T a)^2 - (p^T(a elementwise-squared)-(p^T a)^2)).

Consequently mean_m p_m-p equals, to second order,

    b_v = 0.5 * p elementwise-multiplied by
      [diag(C_v)-2 C_v p+(2 p^T C_v p-p^T diag(C_v))*1].

The class sum of b_v is zero. For a generic centered bank the remainder is cubic. For paired antithetic contrasts (+a,-a,+b,-b), all odd contrast moments cancel and the remainder is fourth order under the smooth affine softmax model. This is a **third derivative of CE**, since its gradient is softmax-Y; it is not the residual-weighted logit Hessian of instantaneous splitting.

Let c_B be the baseline after one SGD step and g_B=gradient L_pool(c_B). The leading next-step pooled-loss difference is

    L_pool(c_next)-L_pool(c_B)
      = -eta g_B^T J_T^T b + O(contrast^4)

for antithetic affine contrasts at fixed eta. Replacing g_B by the pre-step gradient adds an eta-squared times contrast-squared approximation term. The expression has no fixed sign. Parameter covariance enters C_v=J_v C_phi J_v^T, so this leading coefficient is linear in C_phi. Optimizing a PSD covariance under a norm/Fisher budget therefore gives an ordinary restricted eigenvalue/generalized-eigenvalue problem. Its eigenvector machinery is not new; the changed criterion would be post-update pooled CE, rather than immediate output splitting or maximal spread.

## 4. A binary sign witness

For a scalar binary margin z, centered small contrast variance v, and p=sigmoid(z),

    mean sigmoid(z+contrast)-p
      = 0.5*p*(1-p)*(1-2p)*v + O(contrast^4)

for paired antithetic contrasts. At p=3/4 the coefficient is -3v/64. Under independent logit SGD the mean margin therefore moves an extra +3*eta*v/64. Let p_B=sigmoid(z-eta*(3/4-y)) be the baseline probability after its step. At fixed eta the leading post-step pooled-loss difference is (3*eta*v/64)*(p_B-y), with a fourth-order contrast remainder. It is negative for y=1 and positive for y=0. Expanding additionally for a small step gives -3*eta*v/256 for y=1 and +9*eta*v/256 for y=0, with an additional O(eta^2*v) term. The corresponding p=1/4 cases reverse class labels.

Thus centered contrast can accelerate already-correct confident margins while slowing correction of confidently wrong margins in this simple setting. Graph/task parameter coupling can change these signs; it supplies no guarantee that graph-error directions choose useful covariance. This scalar witness does not realize the proposed common-gradient projection or a graph candidate. It is symbolic coefficient arithmetic, not a trained model, dataset result or optimization experiment.

## 5. What survives in a real shared-backbone ensemble

The exact CE/BCE loss decomposition always holds for the declared mean-logit pool and weights. Its parameter derivative is g_own=g_pool+gradient D. Whether gradient D helps the actual pooled loss depends on its alignment with the actual update metric, moments, decay and finite displacement. Gradient D is label independent for fixed logits, but its selected initialization can depend on TRAIN labels.

With shared trainable weights, private multiplicative factors and nonlinear bodies, the averaged shared gradient also contains changes in member Jacobians. Expansions include mixed parameter derivatives of the logits and loss, not only the softmax coefficient above. Finite initial pooled logits can change at second order under stem/private-body perturbations. AdamW's nonlinear moment transformations and route-specific dropout invalidate a simple SGD mean formula; zero second moments can also make a formal Hessian of the optimizer map unsuitable. The same copied optimizer state and an explicitly defined finite trial are essential.

A deterministic dropout-off one-step trial can be used as a **surrogate selector**. Its favorable result does not certify the next native noisy update, convergence or generalization. Labels reused for inner training and outer selection increase overfitting risk; ordinary VALIDATION/heldout comparisons remain necessary. A positive construction coefficient or larger unlabeled-root response is insufficient.

In a head-only private architecture with shared hidden states and direct mean-logit pooled training, centered linear head factors can instead remain invisible to the served predictor under appropriately identical affine private updates/shared state. That is a useful null limit. It is not generally the current all-layer BE continuation: trainable private body factors, different moments or member state can break it. No broad pooled-training null theorem is asserted for the actual source.
