# Distinct member-learning lead: private-factor confident choice credit

This is a literature and method proposal only. It changes which supervised
gradient each member receives; it does not condition projector inputs, change
the graph, add labels, acquire teachers or extend the active fit grid. No model,
tensor, dataset, author implementation or current experiment outcome was executed
or read for this scout.

## Primary mechanism and exact attribution

Lee, Hwang, Park and Shin, **Confident Multiple Choice Learning**, arXiv
1706.03475v2, accepted at ICML 2017, is the new primary source. Its method explains
a failure of ordinary Multiple Choice Learning: a member can be accurate on its
specialty but confidently wrong elsewhere, so the committee can lose an available
correct answer. CMCL changes both training credit and non-specialist confidence.

For each TRAIN observation, it assigns one member the supervised prediction
loss. Other members minimize `KL(U || P)`, pulling their predictions toward a
uniform distribution. Algorithm1 chooses the member minimizing the **complete
assignment cost**, including the penalties of other members. The equivalent
member-dependent cost is supervised loss minus that member's uniform penalty.
Choosing the minimum supervised loss alone would implement a different algorithm.

The paper's prose describes KL in the opposite direction from its displayed
Eq3/Eq4 and Algorithm1. The displayed formula and gradient derivation explicitly
use **uniform-to-predictive** KL. Reverse KL is not an interchangeable port.
Its stochastic-label variant estimates that gradient with uniformly drawn labels.
The first proposed graph test should use exact uniform cross entropy; no extra
label-sampling mechanism is needed.

CMCL also already proposes stochastic cross-member lower-layer feature sharing.
That feature-sharing ingredient is prior and cannot be presented as a new graph
ensemble principle. Its image results were not adopted as evidence for GNNM.
The full paper, proofs, figures and author implementation were not audited here.

## Bounded graph hypothesis

Use the ordinary factual native37 SeHGNN factor bank, retaining the complete graph,
all native positive-label channels, native widths and all six private factor sites.
This is separate from typed-label-context V3: no additional masses, generator,
context halves, input masks or reconstruction objective are added.

Keep complete own supervision for every member. Add CMCL specialization credit
**only to member-owned factors**, while the shared core continues to receive the
ordinary mean own loss. The hypothesis is that private routes can learn different
competence regions without the uniform penalty flattening the shared graph
representation needed by every route. This is a two-block learning rule, not the
gradient of one scalar loss over all parameters.

For native IMDB's five simultaneous Bernoulli outputs, use a node-level assignment
of the entire five-bit target vector. Define each member's node loss as the mean
of five binary losses. Its uniform penalty is the mean of five Bernoulli
uniform-to-predictive KL terms. This is not a five-way softmax classification task.
For a logit z, the stable uniform penalty is
`softplus(z) - z/2 - log(2)`, whose derivative is `sigmoid(z) - 1/2`.

Let E be mean own loss over all TRAIN nodes and members, and C be CMCL's assigned
supervised plus non-assigned uniform losses, with the same node/member averaging.
Freeze assignment q while taking gradients. The proposed update is:

- Shared core: gradient of E only.
- Each member's private factors: gradient of E plus lambda times its part of C.
- One deduplicated native Adam update; shared moments and factor ownership remain
  explicit. This is not a second optimizer step or a change to the serving rule.

Lambda and the uniform-penalty coefficient must be fixed in a later source-reviewed
protocol before any fit. This packet chooses no favorable coefficient or seed.
The full own-loss floor preserves each member's access to every TRAIN target;
vanilla CMCL does not preserve that floor.

Assignment must be computed from complete TRAIN logits and TRAIN truth only, with
no VALID loss in responsibility calculation. Ties require a fixed, independently
seeded balanced rule; identical initial factor functions do not themselves create
useful specialties. A future implementation must resolve extra forward/backward
cost, scratch native buffers, AMP scaling and owned RNG before numerical authority.
Parameter-only VJPs can prevent auxiliary shared gradients; merely detaching logits
would also remove the desired private credit. No executable prototype is included.

Native nonaffine LayerNorm across five logits couples output components. Therefore
per-bit winner assignments should not be silently substituted: insisting that four
unassigned logits be zero while a single assigned logit is nonzero conflicts with
the layer's zero-mean output constraint. Whole-node uniformity is representable,
but its effect on five-bit competence still needs testing.

## Decisive controls for a later study

| Fresh control | What it separates |
| --- | --- |
| Ordinary full own-loss factor bank | Whether new credit improves competence at all |
| Vanilla CMCL adapted exactly to Bernoulli node vectors | Whether the published mechanism already explains the benefit |
| Own-loss floor plus CMCL credit on every parameter | Whether protecting the shared core matters |
| Own-loss floor plus CMCL credit on private factors only | Proposed two-block rule |
| Own-loss floor plus uniform regularization on all members, without assignments | Specialization versus a general confidence penalty |
| Independently initialized, individually selected four-body native ensemble | Whether sharing remains competitive with a strong ordinary ensemble |

Every control needs the same information, architecture, roles, initialization
policy, complete horizon, selector and fixed pool; native body storage differences
must be charged. A primary-consistent mean-probability pool is a reasonable default
for this separate study, but it must be frozen before fitting. Existing completed
scores are not a matched reference and are not recalculated here. Full validation,
selected state, replay and actual serving cost must be included.

## Priority and evidence needed

The root-authorized factual error readout shows that the shared bank usually lacks
a correct member alternative. Suppressing off-specialty confidence alone mainly
addresses the smaller lost-alternative category. This lead is worthwhile only if
the **specialization credit also creates competent alternatives**, rather than
just making wrong predictions less confident. It cannot fix information absent
from the predictor or prove that private factor capacity is sufficient.

A representative later test must improve available-correct coverage and the fixed
committee prediction across complete roles, while retaining acceptable own-member
risk. Better entropy, disagreement or oracle loss alone would not justify a study.
If the private-only rule is no better than ordinary CMCL or a uniform penalty,
attribute the published component and stop treating parameter-block credit as the
explanation. Stronger evidence on fresh graphs or held-out roles is required before
any broad methodological claim.

This is a different learning mechanism from input conditioning and the earlier
source-supply correction, but it is an extension of established CMCL. Novelty,
superiority, calibration, causal benefit and acceptance are not established.
