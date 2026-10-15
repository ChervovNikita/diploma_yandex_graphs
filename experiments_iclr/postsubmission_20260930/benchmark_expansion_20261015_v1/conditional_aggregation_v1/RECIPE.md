# Conditional raw-logit aggregation v1

Status: source preparation and local synthetic tensor checks only. Native task
qualification, selection reconstruction, and any scientific fit remain pending.
This is a small mixture/stacking-inspired candidate, with no novelty or quality
claim. The parent approved this fixed recipe before implementation/freezing.

## Fixed matched controls

Four routes share the unchanged native backbone and retain the existing boundary
factors. Use three readouts trained under the same global objective:

| ID | Scores and weights | Added parameters |
| --- | --- | ---: |
| `aggregation_mean_globalJ_v1` | fixed weights 1/4 | 0 |
| `aggregation_global_globalJ_v1` | four route intercepts b; .05+.8 softmax(b) | 4 |
| `aggregation_conditional_globalJ_v1` | b plus a shared tiny route descriptor MLP; .05+.8 softmax(s) | 20 |

The softmax axis is the four routes for each query. Every route weight is at
least .05 and all weights sum to one up to floating point precision. No
parameter matching or equal-capacity fits are included. Native M1, independent
I4, original boundary mean-own F, and the separately retained source baselines
remain campaign comparisons. The mean-globalJ cell is the primary training
control for this candidate; own-only F is a different objective.

## Prediction-only conditional features

For multiclass native logits z[m,q,k], let p[m,q]=softmax(z[m,q]) over the native
class axis. For one binary logit, lift sigmoid(z) to [1-p,p]. Binary link values
are source-label confidence surrogates under the source sampling law; they are
not assumed to be calibrated probabilities of natural edges.

For each route and query the two inputs are:

1. c=1-H(p)/log(K), normalized entropy confidence.
2. d=JS(p,p_bar)/log(2), with p_bar the four-route mean probability and JS the
   symmetric equal-mass Jensen-Shannon divergence.

Logs use clamp_min(1e-12); the probabilities themselves are not floored or
renormalized. Zero-probability products contribute zero. Roundoff excursions in
c,d are clamped to [0,1]. Native FP32 and FP64 are supported; lower-precision
native losses/AMP are unqualified and rejected by the core. Features are fully
detached, have no labels/preheads/IDs/teacher inputs, and are invariant to an
arbitrary class-common shift within each multiclass route/query.

s[m,q]=b[m]+v^T GELU(W[c,d]+a), hidden width4. There is no output bias.
W has eight entries; a and v four each; b four. W is iid Gaussian with standard
deviation 1/sqrt(2), drawn in CPU FP64 with an isolated generator seeded by
job seed+1301009, then cast to the native dtype/device. a=v=b=0. Global b=0.
Construction consumes no global RNG. All three readouts initialize at exactly
the native uniform raw-logit mean. At initialization W/a gradients are zero
because v=0; v/b can learn immediately. This is expected.

Seed provenance: an earlier discussion proposed offset1901003. Before any
numerical execution, this package fixed offset1301009; the parent explicitly
approved this convention before freezing/qualification. The change is an
implementation seed convention, not an experimentally selected setting.

## Serving and objective

z_served[q]=sum_m alpha[m,q] z[m,q]. Implementation uses the algebraically
equivalent centered form mean(z)+sum_m (alpha-1/4)z to preserve exact mean
arithmetic at initialization without removing the live gate derivative.

Let ell_m be each caller's live, unchanged native per-route task scalar and R
the source regularizer. F=mean_m(ell_m)+R. Let the unchanged source callback
evaluate the ACTUAL served logits: L=ell_native(z_served)+R. Every trainable
shared/native, private-boundary, and aggregation parameter receives the same
computation-graph objective J=.5F+.5L, with one backward and one original
optimizer transition. R therefore occurs exactly once in J. No private-only
credit split, added objective, extra view, or copied backbone is used.

Native CE, molecular observed-label BCE, NCN/NCNC separate positive and negative
means, and BUDDY joint mixed-batch mean remain distinct source reductions. The
generic core accepts live scalar losses and an actual-served-loss callback; it
does not infer labels or rebuild reductions. Complete native reduction groups
must reach the caller unchanged; microbatch-normalized surrogates are invalid.
Only the molecular TRAIN/VALID transform is prepared in this package. Node and
link native ports are pending, despite generic tensor support for their loss
domains.

Detaching descriptors removes their derivative to native logits. The direct
pooled prediction gradient is still weighted by learned alpha. Shared/private
gradients and training trajectories are not claimed to remain unchanged.

## Prospective interpretation

Read a selected gated bank under its trained gate and a fixed uniform mean to
estimate the readout effect on that SAME state. Separately trained mean-J and
gated-J banks compare the whole joint pipeline. The global-J learned weights
control tests a fixed route preference. No VALID gate fitting, checkpoint gate
refitting, feature selection, feature grid, or hyperparameter search is allowed.
Future selection must use the declared actual served output with the same
native controller and criterion. This package does not extend final TEST
serving: reconstruction of these wrapped selected states remains unsupported.

The history's mean-cancelled member alternatives motivate this question;
they do not establish benefit. Prediction repair, ranking repair, native loss,
member loss, weight dispersion and resource cost should be inspected under a
frozen later analysis plan. Retaining F is no member-competence guarantee and
convex scalar pooling is no bound on ranking quality. No every-member>M1
criterion is imposed. C1/C4/C8 combinations are not prepared here.
