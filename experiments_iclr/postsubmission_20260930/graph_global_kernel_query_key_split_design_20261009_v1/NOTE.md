# A private query/key split of the native positive global kernel

9 October 2026. Inactive source-only hypothesis and bounded protocol recommendation.

The native tied query/key core averages values through a reversible transition
matrix at each fixed feature state. Small private scales applied separately
before sigmoid can add directions that violate that core's reversibility while
retaining one shared large projection. This is a concrete possible operator
change. It supplies no evidence that the native whole GNN lacks directed effects,
that the added direction is useful, or that the construction is novel.

Only authorized local text source was read. No dataset, numerical array, weight,
current outcome, remote machine, scientific execution, launch or live-source
change is involved. The arithmetic below is analytic; no numerical fixture was
executed. Prior overlap is a separate assessment and is not cleared here.

## The exact fixed-core restriction

The pinned `GlobalAttn.forward` sets
`q=k=sigmoid(k_lin(x))`. For each head it computes `K^T V`, multiplies by Q,
and divides by `Q sum_j K_j`, with no global support mask. Consequently, for
fixed features and parameters, its value aggregation has core

\[
C_{ij}=u_i^T u_j>0,\quad d_i=\sum_j C_{ij},\quad
P_{ij}=C_{ij}/d_i,\quad
\pi_i=d_i/\sum_k d_k,\qquad
\pi_iP_{ij}=\pi_jP_{ji}.
\]

Positivity here is in exact real arithmetic with finite logits and at least one
channel. Thus P is row stochastic, irreducible and reversible, with the stated
unique stationary mass. It need not be symmetric: unequal row sums give unequal
`P_ij` and `P_ji`. This distinction matters when interpreting directed attention.
The same argument holds separately for each member and head even when private
factors produce different feature maps. It does not assert floating-point
denominator stability. [S1–S3]

This is one conditional aggregation core. Native LayerNorm, multiplication by
`h+beta`, output projection, ReLU and dropout follow it. Local GAT attention,
local residual/gating paths, evolving feature states, products of layer cores
and combinations of different member functions can already produce directed
effects. Products of reversible matrices need not be reversible, and mixtures
with different stationary distributions need not share detailed balance. The
served probability ensemble is not itself a transition-matrix average. No
whole-GNN expressivity failure follows from the displayed restriction.

## A source-compatible pre-sigmoid split

The existing `FactorLinear` applies private input factor r, a shared weight W,
private output factor s, **then adds the shared bias b**. An exact extension is

\[
a_{mi}=W(x_{mi}\odot r_m),\qquad
q_{mi}=\sigma\bigl(a_{mi}\odot(s_m+\delta_m)+b\bigr),\qquad
k_{mi}=\sigma\bigl(a_{mi}\odot(s_m-\delta_m)+b\bigr).
\]

Only delta is added. Each member retains one projection result a; query and key
use different output scales before sigmoid. At `delta=0` this preserves the
existing factorized function for every r/s/W/b state, including its native
unit-factor start. At that state, the old parameters' mathematical gradients
also equal the native gradients for the same loss; the new delta gradient can
be nonzero. No additional random draw is needed to set delta to zero.

Writing `sigmoid(t*(s±delta))` without defining t and the bias convention would
silently change the present affine map if t included its bias. Bias should
remain outside the scale as above. This is a specification correction, not a
new initialization principle. The pair `s+delta,s-delta` is simply an invertible
reparameterization of two independent output-scale vectors. Sigmoid keeps the
resulting affinities positive even if an output scale becomes negative.

Queries and keys still use the same projected coordinates and input factor r;
the scales are shared across nodes within a member. They do not supply arbitrary
independent projection directions, a node router or a new edge-support mask.
The unnormalized and row-normalized cores retain rank at most D. A small added
parameter count also imposes no bound on the learned split's magnitude.

By contrast, node-independent diagonal factors applied **after** the same
activation give `Q=U D_q`, `K=U D_k` and
`QK^T=U D_q D_k U^T`, which is symmetric because the diagonal factors commute.
When that kernel is positive, its row normalization is still reversible.
For the particular post-sigmoid pair `s±delta`, the product is
`diag(s^2-delta^2)`: its delta derivative is zero at delta=0. A zero-start
post-sigmoid split therefore remains dormant under ordinary exact first-order
training. It is not a competent active control for the proposed new tangent.

## What the added tangent actually does

Hold features and all old parameters fixed. At delta=0 let
`u_i=sigmoid(a_i*s+b)` and, for a small perturbation epsilon*h, let
`e_i=a_i*sigmoid'(a_i*s+b)*h`. Then

\[
q_i=u_i+\epsilon e_i+O(\epsilon^2),\quad
k_i=u_i-\epsilon e_i+O(\epsilon^2),\quad
C_{ij}(\epsilon)=u_i^Tu_j+\epsilon A_{ij}+O(\epsilon^2),\quad
A_{ij}=e_i^Tu_j-u_i^Te_j=-A_{ji}.
\]

The s tangent instead changes both activations in the same direction and gives
a symmetric kernel tangent. The delta tangent is antisymmetric, with zero
diagonal to first order. Row normalization retains a valid zero-row-sum
transition tangent:
`dP_ij/d epsilon = (A_ij-P_ij sum_k A_ik)/d_i`.

For any differentiable downstream loss H, the delta gradient at the tied state
is the node sum of `a_i*sigmoid'(a_i*s+b)*(partial_q_i H-partial_k_i H)`.
Query and key have different roles in the normalized aggregation, so this
difference need not vanish at the symmetric start. The gradient can nevertheless
vanish or be small when a channel's a is zero, sigmoid saturates, downstream
sensitivities cancel, or the realized tangent supplies no circulation. Identical
deterministic members still receive identical gradients; private delta banks do
not themselves create member diversity or a competent specialization.

Kernel asymmetry alone is insufficient to prove nonreversibility. A positive
matrix can be nonsymmetric yet have a reversible row normalization after a
different diagonal symmetrization. With one feature channel,
`C_ij=q_i k_j` always produces identical normalized rows and is reversible.
More generally, a first-order ratio `A_ij/C_ij=c_i-c_j` has zero circulation
around every cycle. Nonzero delta therefore does not guarantee a useful or
even nonreversible core.

## An exact cycle witness, including the native-start tangent

For a positive P, reversibility requires equal forward and backward products
on every cycle. Row denominators cancel from the ratio on a three-node cycle,
so it suffices to compare the corresponding unnormalized C products.

Take two coordinates, zero bias, first projection values
`(-log 3,0,log 3)` and second projection value zero. First query scale 1 and
key scale -1, with both second scales 1, give
`q_i=(u_i,1/2)`, `k_i=(1-u_i,1/2)` and
`u=(1/4,1/2,3/4)`. In the s/delta parameterization this finite witness has
first `s=0,delta=1`, rather than keeping s at its native value.

| Edge | C value | Reverse edge | C value |
| --- | --- | --- | --- |
| 1→2 | 3/8 | 2→1 | 5/8 |
| 2→3 | 3/8 | 3→2 | 5/8 |
| 3→1 | 13/16 | 1→3 | 5/16 |

The two products are exactly `117/1024` and `125/1024`, so the normalized core
is nonreversible. This verifies the proposed witness analytically.

There is also a nonzero tangent at the native tied state `s=1,delta=0` for
those same projection values. Perturb first delta only. Write `l=log 3`.
The activation derivatives are `(-3l/16,0,3l/16)`, giving
`A_12=A_23=-3l/32` and `A_31=3l/16`. Native tied edge values are
`C_12=3/8`, `C_23=5/8`, `C_31=7/16`. Therefore

\[
\left.\frac{d}{d\epsilon}\log
\frac{C_{12}C_{23}C_{31}}{C_{21}C_{32}C_{13}}
\right|_{\epsilon=0}
=2\left(-\frac l4-\frac{3l}{20}+\frac{3l}{7}\right)
=\frac{2\log3}{35}\ne0.
\]

Thus arbitrarily small splits can leave reversibility; the finite sign-flip
witness is not needed to establish that local possibility. The construction
extends exactly to any width D≥2: let the remaining D−1 coordinates be the
same constant `1/(2 sqrt(D−1))` in both q and k. Their squared contributions
sum to 1/4. Finite shared biases and zero projection rows can supply those
constants, and the first-coordinate tangent is unchanged. Hence the fixed
512-channel recipe does not invalidate the algebraic witness. It remains an
existence argument for the operator form, not a statement about the actual
graph's learned features or trajectory.

## Parameter and compute consequences

The exact recipe has four members, two global projection maps, one head and
512 output channels. One added delta bank per map adds **4,096 trainable
scalars in two tensors**. Existing r/s factors and shared W/b are retained;
no additional large projection matrix or member forward is required.

Each active layer/member still computes the shared projection once, but creates
two scaled preactivations and two sigmoid activations instead of one activation
aliased as q and k. Extra forward and reverse work and stored activations scale
as O(ND); the existing linear-attention contractions remain O(ND²), with no
required dense N×N kernel storage. This is a structural count, not a measured
memory, speed or efficiency result. Delta is inactive during the original local
stage and should retain None-gradient treatment there.

A full untied-Q/K alternative adds one 512×512 shared query matrix per layer,
plus bias and its private factors: 524,288 additional matrix scalars across
the two layers, 1,024 bias scalars and 8,192 r/s scalars under this same BE
construction. It also adds the corresponding projection work. Native source
contains that untied branch, but the current Polynormer constructor does not
expose it, and the current recipient/release source rejects untied Q/K or new
private tensors. None of these alternatives is already admitted by that source.

## One inactive, competent operator comparison

Keep this hypothesis separate from the completed relation-credit interpretation.
The smallest useful prospective panel uses **one fixed ordinary mean-member
F objective** and the original two CE views for all four operator conditions:

| Condition | Purpose |
| --- | --- |
| Native tied kernel | Exact architecture/reference with the existing private local scorers and factors fixed as common design choices |
| Active reversible extra scale | Add equally many private c scalars, `q=k=u*exp(c/2)`, c=0; preserves the initial function and permits a nonzero symmetric kernel tangent |
| Pre-sigmoid query/key split | Add delta as specified above, delta=0; tests the restricted directional operator package |
| Full independent Q/K maps | Competent larger operator reference using the native untied branch, with copy-matched initial maps for the attribution comparison |

The reversible c condition controls an active added private scale pathway,
trainable count and initial function. It cannot perfectly match the split's
tangent geometry, activation placement or update conditioning. A split advantage
over it would support this operator package; it would not by itself prove that
breaking reversibility caused the advantage. Nonzero cycle circulation in the
learned fixed core would verify that the available direction was actually used,
without establishing mediation or quality. Any diagnostic probes must be fixed
before scores; absence on a finite probe set cannot certify reversibility.

All cells need fresh separately owned source and native qualification, complete
graphs and label populations, full width/depth/horizon, paired initial functions
and stream opportunity, one accepted Adam transition per update, original stage
restore and selection, the same served probability mean, and inclusive work
accounting. Copy matching is a causal-control choice, not a new initialization
claim or proof that the full Q/K reference is optimal. Keep every cell, failure,
seed and cost; freeze genuinely unused role/seed choices and unchanged utility
and competence criteria before outcomes. This note asserts no unused IDs and
authorizes no fit.

The primary split-versus-tied contrast asks whether the package helps on that
fixed setup. Split versus the active reversible scale asks whether an active
extra private scale explains it. Split versus full Q/K asks how the restricted
operator compares with a competent more flexible alternative at its actual
cost. Member mean/worst quality, full-population repairs and introduced errors,
coverage and pooled quality remain necessary alongside served accuracy.
None of these cells alone establishes an ensemble or sharing advantage; those
broader claims still require competent native single/independent ensemble and
objective-matched untied-feature references. New-role/graph confirmation remains
a separate claim requirement.

The current relationJ/allJ/phiJ family changes supervision recipients while
retaining q=k. Its outcomes cannot establish this new operator's utility or
an interaction with collective credit. No credit-policy factorial, tuning grid,
new training primitive, novelty, quality result or generalization claim is
supplied here. The proposal is a bounded operator hypothesis for a later genuine
full comparison.

## Exact source scope

- S1: `wikics_staged_private_graph_residual_method_preparation_20261007_v4/vendor/native_polynormer.py`,
  complete `GlobalAttn` and native downstream/local structure.
- S2: `portable_internal_be_public_interface_20261007_v2/core/factors.py`,
  `FactorLinear` affine/bias order and private bank ownership.
- S3: `portable_internal_be_public_interface_20261007_v2/recipes/wikics.json`
  and `graph_relation_private_credit_source_20261008_v2/permissions.py`,
  pinned shape, head/map/member counts and current strict tied-QK policy.

All paths are under the single authorized research root. Source bytes are
bound in `SOURCE_BINDINGS.json`. No primary literature was retrieved or read
for this note; existing source and analytic reasoning supply its bounded claims.
