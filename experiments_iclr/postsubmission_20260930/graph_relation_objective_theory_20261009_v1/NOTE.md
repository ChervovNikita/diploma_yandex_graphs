# What the fixed graph-relation credit comparison can establish

9 October 2026. Pre-result theory and contribution assessment.

The rule gives a member's private graph-relation parameters more corrective
credit on a TRAIN node when that member supplies a larger share of the pool's
correct-class probability. Its remaining parameters receive equal member
supervision. This can change which labelled examples shape each member's graph
relations. The scientific question is whether that particular allocation helps
the fixed served pool while retaining useful individual predictors. The loss
mixture, probability-pool derivatives, parameter sharing and selective gradients
have established ancestry; the present comparison tests their exact allocation.

This note uses saved literature scopes, the two requested design notes and
pinned text source. It opens no current outcomes, arrays, weights or dataset,
contacts no server, changes no study source, and proposes no fit or new gate.
Historical closed-result prose incidentally visible in reused notes supplies
no evidence or design choice here. Source byte bindings are saved separately.

## The actual two-view objective

Let `v=1,2`, `i=1,...,N` and `m=1,...,M`, with `N=580` complete TRAIN nodes and
`M=4` members. For a fixed realized stochastic view, write
`p_vim=softmax(z_vim)`, `q_vim=p_vim(y_i)` and `e_i=onehot(y_i)`. Define

\[
f_{vi}=-\frac1M\sum_m\log q_{vim},\qquad
\ell_{vi}=-\log\!\left(\frac1M\sum_m q_{vim}\right),\qquad
(F,L,J)=\frac1{2N}\sum_{v,i}
\bigl(f_{vi},\ell_{vi},(1-\beta)f_{vi}+\beta\ell_{vi}\bigr),
\quad\beta=\tfrac12.
\]

The pinned `own_supervision` computes mean-node CE separately per member;
`output_credit` then averages members and views. The pool helper uses
log-softmax and logsumexp to compute the displayed probability-average CE
without a clamp. Each view has its own four-member pool. Averaging all eight
view/member probabilities before CE would couple credit across views and
produce another objective; CE of mean logits would also differ. Representations
have zero output cotangent because no auxiliary term is present. [S1–S4]

Set the per-node, per-view responsibilities to
`rho_vim=q_vim/sum_k q_vik`. Since
`grad_z_m q_m=q_m(e_i-p_m)`, the pool derivative is
`rho_m(p_m-e_i)`. Hence the exact full-objective logit partial is

\[
\nabla_{z_{vim}}J=\frac1{2N}
\underbrace{\left(\frac{1-\beta}{M}+\beta\rho_{vim}\right)}_{w_{vim}}
(p_{vim}-e_i),\qquad
\nabla_{z_{vim}}F=\frac1{2NM}(p_{vim}-e_i).
\]

Thus J changes the magnitude of a member's ordinary CE cotangent for that
node/view, rather than introducing a new residual direction. The weights sum
to one over members. In exact arithmetic with finite logits, `1/8<w<5/8`; compared with F's
`1/4`, the multiplier lies strictly between `1/2` and `5/2`, with those values
as limiting bounds. These bounds concern each logit cotangent, not a summed
parameter-gradient norm or useful optimizer movement. More credit means greater
*relative correct-class probability*; the recipient can still be wrong. The
code differentiates the true J and then detaches its cotangent for replay.
Differentiating a weighted CE through these state-dependent weights would add
terms and implement a different derivative. [S1–S4]

## The individual/pool Jensen gap

For uniform member weights `U_m=1/M`, elementary algebra gives

\[
f_{vi}-\ell_{vi}
=\frac1M\sum_m\log\frac{1/M}{\rho_{vim}}
=\mathrm{KL}(U\Vert\rho_{vi})\ge0,
\qquad
D=\frac1{2N}\sum_{v,i}\mathrm{KL}(U\Vert\rho_{vi}),
\qquad F=L+D,\quad J=F-\beta D=L+(1-\beta)D.
\]

This is dispersion of true-class responsibilities across members. It is not
KL between complete predictive distributions, hidden-state separation, or a
measure of distinct useful neighborhoods. At fixed F, increasing the gap lowers
J; at fixed L, increasing it raises J when `beta<1`. Neither conditional
statement proves that optimizing J creates useful diversity. The source never
optimizes D alone.

Equal correct-class probabilities make responsibilities uniform, give `D=0`
for that node/view, and make F/J logit partials identical even if wrong-class
probabilities differ. Identical deterministic members therefore receive no
explicit symmetry-breaking signal. Different stochastic paths can create
unequal credit without guaranteeing beneficial specialization. If one rival
outranks the truth in every member, it also outranks the truth in their fixed
probability mean. There is no term specifically detecting that common rival.
The mixture and its diversity-gap interpretation are established prior; the
KL expression is an elementary exact specialization for this pool. [L1–L2]

## Allocation changes the gradient field

Let `P_X` project onto a fixed parameter recipient set. The accumulated
gradient supplied to native Adam is

\[
V_X=\nabla F+\beta P_X(\nabla L-\nabla F)
=\nabla F-\beta P_X\nabla D.
\]

| Policy | Coordinates receiving J | Coordinates receiving F |
| --- | --- | --- |
| alphaF | None | All |
| allJ | All | None |
| phiJ | 56 internal dense-factor tensors | Complement, including local scorers |
| relationJ | 14 private local scorer banks and four tied-QK factor banks | Complement, including shared weights, normalizations, boundary factors and other internal factors |

The alphaF and allJ fields are gradients of F and J respectively. For a mixed
policy, F/J partials are assembled into one vector; no single global scalar
objective follows. In a smooth region, a scalar potential would require the
mixed recipient/complement derivatives of `J-F` to satisfy the corresponding
integrability condition. The source establishes no such condition. A reported
F or J value is consequently a diagnostic for relationJ/phiJ, not a certificate
that their complete update descends that scalar. [S1–S3]

All member/view VJPs use the same old parameters and realized views, accumulate
before one Adam transition, and preserve inactive paths' None gradients. Adam's
moments and coordinate normalization act on `V_X`; they need not preserve the
direction of a plain gradient step. Same-state differences in gradient inputs
do not identify differences between eventual training trajectories. [S1, S5]

The graph interpretation is concrete but bounded. Private local scorers can
alter neighbor weights at fixed incoming states and values. The tied factors
alter the native `q=k=sigmoid(k_lin(x))` kernel maps, rather than independent Q
and K maps. Across layers, relation changes also alter later features and
values. This allocation adds neither graph evidence nor an attention operator,
and local GATv1's static-ranking restriction remains. [S3, S6, L3]

## Why competence might improve or deteriorate

Restricting pool credit leaves broad feature and boundary supervision on F at
the same state. The own-CE component also keeps positive correction on every
recipient for every label. These are plausible reasons to test the allocation
as a way to retain competent members while learning useful relation choices.
They are not preservation constraints: allJ also contains the same positive
own-CE component, and F-trained shared coordinates can help one member while
hurting another.

For member m's private relation block `a_m`, let `g_vim` be its unreduced
node CE gradient. Because each bank row affects only its own member path,

\[
\nabla_{a_m}J=\frac1{2N}\sum_{v,i}w_{vim}g_{vim},\qquad
\nabla_{a_m}F=\frac1{2NM}\sum_{v,i}g_{vim}.
\]

Changing positive example weights changes cancellation among these vectors.
It can emphasize a useful relation gradient, remove an adverse interaction,
or amplify one. A labelled graph node's weight multiplies its entire upstream
pullback, not an independently selected edge. Reused parameters, overlapping
receptive fields and global attention prevent a per-node sign argument from
becoming a member-risk argument. No bound on total gradient norm follows from
the bounds on w.

A minimal analytic counterexample shows why the positive floor is insufficient.
Consider one private scalar parameter with two node CE derivatives `a` and
`-b`, where `b>0`, and positive weights `w_1<w_2`. Choose
`b<a<(w_2/w_1)b`. Its own mean CE derivative is positive, whereas the J-weighted
derivative `w_1 a-w_2 b` is negative. A sufficiently small plain J-gradient step
therefore raises its own mean CE to first order. Such local CE derivatives can
be realized by binary logits with chosen nonzero parameter sensitivities;
duplicating the view preserves the argument. This refutes a general inference
from positive node weights to member descent. It is not a claim that this
configuration occurs in the native graph model. The actual full shared Adam
update supplies still less basis for a descent guarantee.

The saved competence-projection note is an inactive proposed safeguard. Its
constraints are absent from the current source. The current fixed competence
screen tolerates small specified degradation; passing it would be empirical
retention within that screen, not mathematical preservation of every member,
node or training step. [S7]

## Decisive interpretation of the existing controls

| Fixed contrast | What a supported complete comparison would identify | Remaining limit |
| --- | --- | --- |
| relationJ minus alphaF | Effect of supplying J to the chosen relation block, under the same copied-scorer architecture and recipe | Local scorer and tied-QK changes occur together |
| relationJ versus allJ | Whether restriction of the established mixture helps relative to scalar J on the same architecture | Does not identify a new loss or a new optimizer |
| relationJ versus phiJ | Whether the exact relation recipient package helps relative to broader internal-factor recipients | Both include the four QK banks; the contrast adds 14 scorer banks and removes 52 other factor tensors jointly |

Matching or better allJ/phiJ leaves a special advantage of relation placement
unsupported by this family. RelationJ exceeding both, with the unchanged
primary utility and competence screen met, would support the empirical value
of this exact allocation under the fixed setup. No sign is assumed here.

The frozen full-population readouts already distinguish several interpretations.
Pooled NLL can improve without changed predicted labels. Accuracy can improve
through changed probability balance even when any-correct coverage does not
increase. Increased coverage supports newly correct member predictions but
does not by itself identify the responsible parameter block. Paired repairs
and introduced errors quantify net prediction changes; strict common-rival
support addresses unanimous pairwise failure; member mean/worst accuracy and
NLL assess competence. Lower TRAIN J, a larger Jensen gap or greater attention
variation cannot substitute for these readouts. These are interpretations of
existing controls, not added post-result admission criteria. [S8]

This family can assess allocation on the one fixed graph, split roles, backbone,
selector and three optimizer seeds. Claims about sharing itself require an
objective-matched untied comparison; claims of superiority to ordinary single
models or independently trained ensembles require competent corresponding
references. Neither those claims nor confirmation on new graphs or roles
follows from the present controls. No follow-up fit is requested here.

## What is already established in the saved closest literature

| Saved prior | Established fact relevant to this rule | Bounded difference |
| --- | --- | --- |
| GNCL [L1] | Eq5 interpolates individual and pool risk; the inspected author upper-mode fit sends one scalar mixture through its optimizer | Here the exact probability pool and mixed parameter recipients are specified; inspected code is not an exact reproduction of this WikiCS pool |
| Jeffares et al. [L2] | Individual-versus-ensemble loss gaps, own/pool interpolation, member-compensation risks, and Appendix E's distinct probability/score-average CE gradients | The displayed responsibilities specialize its established probability derivative; the paper does not prove the present recipient policy succeeds or fails |
| Collaborative Learning, ONE and PCL [L4] | Live shared/private branches, member and collective supervision, differential loss exposure and backward treatment | Their peer/KD losses, gates or fused-feature teachers and deployment differ; selective gradients themselves are prior |
| GAT/GATv2, Polynormer and BatchEnsemble [L3] | Learned graph relations, native global kernel attention, shared matrices and private factors | The candidate allocates supervision to existing sites; it invents neither attention nor sharing |
| Saved CoGNN communication-credit proposal [L5] | Project ancestry already proposes own-risk shared/features plus half-mixture private communication feedback | Native scorer/QK recipients replace the proposed sampled communication machinery |
| Fed-GAME and GNNMoE retained method scopes [L5] | Graph-attention aggregation of federated client updates and gated intermediate graph experts, respectively | Those scopes specify neither the present four-classifier fixed pool nor this exact F/J recipient rule |

The candidate's plausible contribution is therefore an empirically useful,
fully specified allocation of an established collective-risk mixture to private
graph decisions. Its responsibility algebra, Jensen identity, positive floor
and mixed-field classification do not establish novelty, useful gradient size,
convergence, generalization or competence. No complete published collision was
established in the saved bounded scopes; publication priority also remains
unresolved. The inaccessible *Graph ensemble neural network* body,
DOI `10.1016/j.inffus.2024.102461`, continues to prevent clearance of that closest
locator.

## Source grounding and reading limits

All local paths below are relative to the single authorized research root
`/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930`.
`SOURCE_BINDINGS.json` records the exact text bytes read. Existing scoped
conclusions are reused, without new paper retrieval, whole-paper/proof credit,
runtime certification or outcome evidence.

- S1: `graph_relation_private_credit_source_20261008_v2/replay_adapter.py`,
  `output_credit` and `collect`.
- S2: `portable_internal_be_public_interface_20261007_v2/core/objectives.py`,
  `own_supervision` only for the objective claim.
- S3: `graph_relation_private_credit_source_20261008_v2/permissions.py`,
  exact partition and selector assignments.
- S4: `public_internal_be_private_steering_adapter_20261007_v1/adapter.py`,
  `served_pool_supervision`, and the v2 source's `train.py` binding to it.
- S5: `portable_context_steering_public_interface_20261008_v1/context_recompute.py`,
  retained old-state/two-view replay and single transition.
- S6: `wikics_staged_private_graph_residual_method_preparation_20261007_v4/vendor/native_polynormer.py`,
  native local layer and tied-QK kernel source; private-row ownership is also
  grounded in `core/factors.py` and `private_local_attention.py`.
- S7: `graph_relation_member_competence_projection_design_20261008_v1/NOTE.md`.
- S8: `graph_relation_private_credit_source_20261008_v2/PROTOCOL.json` and
  `SOURCE_REVIEW.md`; protocol metadata only, never a result body.
- L1: Buschjager, Pfahler and Morik, *Generalized Negative Correlation Learning
  for Deep Ensembling*, [2011.02952v2](https://arxiv.org/abs/2011.02952v2),
  saved Eq5 and pinned author-code excerpts in
  `live_shared_own_internal_pool_operational_prior_20261007_v1`.
- L2: Jeffares et al., *Joint Training of Deep Ensembles Fails Due to Learner
  Collusion*, [2301.11323v1](https://arxiv.org/abs/2301.11323v1), saved
  selected passages/equation and Appendix E scopes in
  `graph_mixed_block_objectives_scout_20261003_v1/primary`.
- L3: Reused native/BE and GAT/GATv2 scopes in
  `graph_relation_private_credit_benefit_boundary_20261008_v1/NOTE.md`.
- L4: Song and Chai [1805.11761v1](https://arxiv.org/abs/1805.11761v1),
  ONE [1806.04606v2](https://arxiv.org/abs/1806.04606v2), and PCL
  [2006.04147v2](https://arxiv.org/abs/2006.04147v2), retained method conclusions
  in the same benefit-boundary note and the saved prior recommendation.
  No new author-code or target-detachment certification is supplied.
- L5: `attention_only_member_pool_credit_prior_design_20261008_v1/RECOMMENDATION.md`
  and `LITERATURE_MEMORY_INVENTORY.md`: CoGNN proposal ancestry and bounded
  Fed-GAME [2603.01363v1](https://arxiv.org/abs/2603.01363v1) Sections2.1–2.4,
  GNNMoE [2412.08193v2](https://arxiv.org/abs/2412.08193v2) Section3.
  Published-body equivalence and broader absence are unestablished.
