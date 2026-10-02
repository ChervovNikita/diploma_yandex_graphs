# Support algebra and construction limits

Let the common deterministic output closure be `z(theta)` with full-node Jacobian `J`. The compact TRAIN residual `r=(softmax(z_T)-onehot(y_T))/T` is injected into full output rows as `R=E r`. Only TRAIN labels enter. The degree-3 Bernstein filters `H_m` use the pinned symmetric normalized adjacency, satisfy `sum_m H_m=I`, and act identically on class coordinates.

The initializer uses raw band gradients:

```text
g = J^T R
h_full,m = J^T H_m R
h_remask,m = J^T P_T H_m R
sum_m h_support,m = g
```

For comparison with centered-cotangent notes, define `q_full,m=(H_m-I/4)R` and `q_remask,m=P_T q_full,m`. Subtracting `g/4` from each raw h disappears under the pinned projection orthogonal to g. Therefore those centered-q notes and the raw-band implementation yield the same projected tangents at the common boundary. The support delta is `J_U^T (H_m R)_U`; this adds unlabeled **output-root** cotangents. TRAIN-root backpropagation already differentiates through unlabeled input neighbors wherever they affect TRAIN predictions.

The candidate preserves double-precision projection/centering, the single common Frobenius cap, cast checks and `d_m=-g-tangent_m`. Thus the mean first-order full output movement remains `-Jg`, and every route has TRAIN derivative `g^T d_m=-||g||^2`, up to the existing numerical certificate. These are construction identities, not predictive guarantees. Full and remasked tangents have no general norm or rank ordering.

## Signed finite check

The initializer's shared diagnostic is

```text
C_full(alpha) = -sum_m <q_full,m,
                  center_classes(z(theta0+alpha*d_m)
                               - z(theta0-alpha*g))>
slope_full = sum_m <q_full,m, center_classes(J*tangent_m)>
prediction = alpha*slope_full
```

The residual already contains `1/T`. The diagnostic is a sum over members, nodes and classes; it has no additional division by four, N or T. Its separate Gram divides by `N*C`. The finite changes are class-centered to remove scalar logit gauge. The q is detached at the common warm point.

For graph full support, this is support-matched and equals `sum_m h_full,m^T tangent_m` at first order. With a cap scalar lambda and uncapped projected vectors u, use the actual capped expression `lambda*sum_m ||u_m||^2` in exact arithmetic. It is not generally `sum_m ||tangent_m||^2`. For remasked support, the stored slope is the cross-support quantity `sum_m <q_full,m,J*tangent_remask,m>` and has no norm-squared or positivity guarantee. A support-matched remask functional would instead use `q_remask`; this packet does not compute it. Random tangents also lack a positive-alignment guarantee.

One extra common forward is taken at the **exact candidate alpha**, locally cached. Neither this signed functional nor the full JVP Gram controls branch acceptance or alpha choice. Comparing independently chosen support-arm alphas is not a paired same-alpha study.

## Known limits and attribution

Detached warm probabilities `p0` and complementary targets `p0-eta*q` produce the exact warm soft-CE logit gradient `eta*q`. Proper probability targets require a common `eta <= min_(q>0) p0/q`; this can be tiny or numerically zero. Clipping/renormalization changes the direction. The source uses signed cotangents directly and does not require target construction. The pullback primitive is ordinary one-step distillation from graph-error-corrected complementary targets, not a new primitive.

In a frozen common linear predictor with squared loss and identical full-batch SGD updates, or one shared fixed linear preconditioner, initial zero-mean route offsets remain zero-mean. The pooled output trajectory equals common descent at every later step. The stdlib fixture proves this for 15 exact rational updates. Evolving nonlinear bodies, classification CE, member-specific sampling and nonlinear optimizer/moment transformations can break those assumptions; their actual later utility requires measurement.

The two-node exact cubic-band witness proves that full support can add a projected private direction while remasking removes it. Its middle bands are zero and its tangents duplicate. The unchanged TRAIN separation guard would reject it. A separate three-node rational witness proves that the shared-full-q slope for remasked tangents can differ from their support-matched norm expression. Neither witness is a native four-distinct-member acceptance or quality result.

Correct & Smooth supplies error diffusion/output correction; BernNet supplies the attributed Bernstein bank; GNTK and NTK ensemble work establish graph-dependent tangent/kernel machinery. Saved complementary-target/distillation conclusions cover the pullback equivalence. The bounded remaining question is whether wider output cotangent support improves later paid continuation at the same admitted private slice. This is an amendment of the existing initializer and supplies no novelty, accuracy, calibration, generalization or convergence claim.

Saved mathematical inputs and their hashes are in `SOURCE_BINDINGS.json`. No primary paper was newly read for this source packet. Resource limits or unqualified native AD keep execution deferred; they do not make the scientific hypothesis false. Broader quality/scale proposals in the saved input note are not adopted as execution instructions by this packet.
