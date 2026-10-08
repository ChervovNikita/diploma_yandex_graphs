# Query-conditioned label-value capacity: one inactive extension

8 October 2026. This sealed source-only note assesses one conditional extension.
It does not implement or adopt it, choose a verdict, propose coefficient grids,
or authorize a run. Current staged P0 source and frozen B remain untouched.

## Limited algebra: what changes and what does not

For clarity, write the class-mass vector as z_m(i), the learned bias-free value
map as V_m ∈ R^(64×C), the label message as g_m(i)=V_m z_m(i), and the bias-free
class readout as R_m ∈ R^(C×64), using R_m for the supplied readout T_m
when g_m is already 64D. The composite class-to-logit map is R_m V_m.
When the permitted neighborhood contains just class c, z_m(i)=a_mi e_c, with
positive a_mi, so g_m(i)=a_mi v_mc and ell_m(i)=a_mi R_m v_mc. Here v_mc=V_m e_c
is the learned 64D class value direction. Attention changes a_mi, while each
route's preferred class remains argmax_k (R_m v_mc)_k. This statement concerns a
single route before probability pooling/native mixing. Changing a_mi can still
change its confidence and the later mixture's decision.

Assess one extension: a **shared receiver-feature multiplicative gate**, applied
to the aggregated 64D label message before each route's class readout:

    gamma_i = 2 sigmoid(A stopgrad(H_i) + b) ∈ (0,2)^64
    gated_g_m(i) = gamma_i ⊙ g_m(i)
    gated_ell_m(i) = R_m gated_g_m(i)

The same trainable A,b is used by all four routes. There is no additive FiLM shift
and no readout bias. At A=b=0 the gate is the identity. This is a single capacity
change; masks, feature-only attention scores, all-nonself-neighbor denominator,
source scaling, route supervision, selection, serving weights and support-based
native fallback retain the supplied comparison specification. No values are
chosen by a sweep here.

For one visible class, the class-k versus class-l margin becomes

    a_mi Σ_d (R_m[k,d]−R_m[l,d]) gamma_i[d] v_mc[d].

Independent coordinate modulation can change the sign of that margin and can
change the route's winning class. A concrete existence example uses two active
coordinates inside the 64D message: v_mc=(1,1), R_m=[[1,0],[0,3/4]]. The ungated
winner is class 1. Gates (3/2,1/2) and (1/2,3/2) yield logits proportional to
(3/2,3/8) and (1/2,9/8), respectively, with winners 1 and 2. Both gates are
attainable with a scalar frozen H=±log(3), gate rows A=(1,−1), and b=0. This proves
possible direction-dependent decisions, not useful learning, expressivity of the
whole model, or a change for every trained value/readout.

**Placement matters.** If e_c is literally a one-hot coordinate and a positive
diagonal gate acts on it before the dense value map, it gives only
V_m diag(gamma) e_c=gamma[c] V_m e_c: the original fixed class direction persists.
A common scalar gate across all 64 coordinates also cannot change a route's
preferred class. Even after the dense map, some learned vectors/readouts/gate
ranges remain in one decision region. The extension must modulate the dense
message's active coordinates, and its attainable gates must cross a relevant
class decision boundary.

For finite H, empty permitted label context gives z_m=0, g_m=0, and exactly
zero gated logits. Gate-generator bias b does not break this property because it
only enters the multiplier. A conventional standalone additive shift beta(H)
would break it. Zero logits become uniform probabilities under softmax, so the
explicit no-anchor native fallback remains necessary. Frozen B and detached H
prevent route losses from updating B; they do not prevent the new gate from
learning a classifier on H. Every masked TRAIN query label must still be removed
from every route context before propagation, including any derived label field.

## Attribution and shortcut risk

The saved GNN-FiLM conclusion for **arXiv:1906.12192v5** describes a
receiver-hidden-state generator that performs featurewise affine transformations
of incoming messages. The present gate is a restricted multiplicative-only
instance of that established conditioning primitive. Applied after a linear
aggregation, a receiver-specific diagonal gate also distributes over its incoming
label messages. No equality with the complete GNN-FiLM architecture is claimed;
its nonlinearity placements and full operators are not audited here. FiLM and BE
components carry no novelty claim.

Zero-without-context is a support property, not evidence that label identity is
used. If class value vectors share a useful common component, or converge toward
one vector v_0, then a message can become approximately a_i v_0 and the output
approximately a_i R_m diag(v_0) gamma(H_i). This is an H-based nonlinear classifier
activated by retained anchor mass. Attention/support may add another signal even
when anchor class identities contribute nothing. A learned query gate therefore
changes the interpretation of a “label-only” route: its final direction can come
from frozen feature information once any label context is present.

The parent supplied counts, without any numeric artifact read: 2512 no-label and
1733 one-visible-class nodes among 5274 dev nodes; the true class occurs in only
49/276, 62/270 and 49/266 native errors in the one-class cohort. These measurements
are not independently verified here. The extension is exactly inactive on the
2512 no-context nodes under the preserved fallback. Its new directional capacity
is relevant to the one-class cohort, but the low truth-presence counts do not
establish useful label evidence. It could repair a native error by interpreting
label identity together with H, by exploiting negative evidence, or by acting as
an extra feature classifier. These mechanisms need different interpretations.

## Incisive matched comparisons, if this hypothesis is later authorized

The capable nonlinear joint four-head SINGLE must receive the same detached H,
masked label messages, gate family, attention/support information, supervision
weights, selector and serving combination. It must be able to make the same
query-conditioned class-direction changes. Giving only BE this gate would test
extra conditional capacity alongside parameter sharing. An untied four-route
control should use the identical shared query gate and protocol, with BE sharing
removed from the value/readout maps. Gate capacity must be included in every
comparison; no new loss, selector or mixture advantage follows from this note.

A particularly relevant **label-identity-erased control** replaces each retained
TRAIN anchor one-hot class vector with the same unit-mass vector u=1_C/C before
the existing bias-free value map. It preserves anchor positions, common query
exclusion, feature-only scores, denominator, source scaling and support-based
fallback. The supervision targets remain the original TRAIN targets. Its message
is exactly a_mi V_m u, so the query gate can use H, anchor presence/positions and
mass, but cannot receive retained anchor class identity through this route.
Meaningful assessment would train and select this control through the same
protocol; a post-training swap alone is a distribution change. A fixed global
class permutation is an invertible relabeling and does not erase identity after
retraining.

The one conditional utility question is: **at fixed frozen B and the same
information, masking, selection and serving rule, does target-conditioned label
value modulation improve decisions through useful label-identity interaction,
beyond an anchor-activated feature classifier and the matched joint/untied
capacity?** Gains that persist when identity is erased would not establish that
label identities drive the benefit. Capacity algebra alone supplies no utility
answer.

No primary-paper reread, code audit, scientific server, dataset, model/checkpoint,
scientific outcome/numeric file, PDF compilation or GENLINK was accessed. No fit,
launch or scientific-source change occurred. New files are confined to this fresh
folder under P.
