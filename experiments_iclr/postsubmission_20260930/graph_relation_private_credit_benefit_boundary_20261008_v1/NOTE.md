# What private graph-attention ensemble feedback can establish

8 October 2026. Scientific interpretation and one conditional experiment design.
The running graph-relation twelve-fit family is not opened or changed here.
Saved method scopes, supplement v22, earlier proposals and the pinned native
attention source are reused. There are no new primary-paper reads or results.

## The useful hypothesis

Four predictors share feature-processing weights but retain private attention
parameters. The candidate gives those private parameters feedback from both
their own predictions and the served probability average. Shared weights and
the remaining private factors continue learning from the individual losses.
The hypothesis is that this arrangement learns useful differences in graph
evidence without weakening the individual predictors.

That is a plausible training hypothesis, not an established novelty or benefit.
It neither explicitly separates embeddings nor selects the best member at
inference. All four predictors remain active and their probabilities are
averaged. No reinforcement-learning estimator, serving router, contrastive
auxiliary or learned mixture coefficient is present.

## The exact learning signal

For one TRAIN node in one realized stochastic view, let p_m be member m's
class-probability vector, y the true class, and M=4. Define

    F = -(1/M) sum_m log p_m(y)
    Lpool = -log[(1/M) sum_m p_m(y)]
    J = (1-beta) F + beta Lpool,  beta=.5.

Each view has its own four-member pool. The two view losses are averaged
afterward. Pooling all eight view/member predictions first would change J.

Let rho_m = p_m(y)/sum_k p_k(y). The exact logit partial is

    partial_z_m J = w_m [p_m-onehot(y)]
    w_m = (1-beta)/M + beta rho_m.

Thus private relation parameters receive a **reweighted ordinary true-label CE
pullback**, not a new residual direction. Here w_m is in [.125,.625]; compared
with F's weight .25, the multiplier is in [.5,2.5]. These are limiting bounds
(finite softmax probabilities lie strictly inside them). The floor is on a
per-node logit cotangent multiplier, not a parameter-gradient norm, member
accuracy or optimizer-descent guarantee. The responsibility must enter through
the actual J derivative: differentiating weighted CE through w_m would add
incorrect terms.

The saved exact identity is

    F-Lpool = KL(U || rho_y),
    J = F-beta KL(U || rho_y) = Lpool+(1-beta) KL(U || rho_y),

where U is uniform over members. This is true-class responsibility dispersion,
not hidden-state contrast or KL between complete class distributions. At fixed
F, a greater pooling gap lowers J; at fixed Lpool, beta<1 still penalizes that
gap. Neither statement means arbitrary disagreement is useful. This algebra
is an elementary specialization of the established own/pool-risk mixture [1],
already saved in the project; it is not a new theorem.

The running relationJ field is, for the exact relation-coordinate projection
P_R,

    V_R = grad F - beta P_R grad(F-Lpool).

It is generally a nonpotential block field. The common-state derivative
identity does not imply a scalar-loss descent theorem for its simultaneous
native Adam transition. The positive own-risk component is not a preservation
constraint on shared features or predictions.

## Why common wrong rivals can persist

If every member gives a rival class c strictly more probability than y, then
their unchanged probability mean also ranks c above y. A new convex aggregator
cannot remove that unanimous pairwise ranking. Members must change their
predictions, or a predictor outside that fixed pool must supply new evidence.
This elementary averaging fact does not diagnose why those members were wrong.

If all p_m(y) coincide, rho_m=1/M and the J/F logit partials are **exactly
equal**, even when the members disagree on other classes. If correct-class
probabilities are close relative to their mean, responsibilities are close to
uniform and the extra relation-credit signal is small. Absolute closeness of
very small probabilities alone is insufficient. There is no special term
detecting or repelling a shared wrong rival.
Identical deterministic paths therefore receive no deliberate symmetry-breaking
signal. Different dropout paths can create unequal responsibilities, but their
useful specialization is not guaranteed.

Even when all members are wrong, one can have greater p_m(y) and receive more
relation credit. That may strengthen its partial correct evidence. It may also
prefer an already confident but still incorrect member, give weaker members
less corrective training, or concentrate on easy examples. A nonzero floor
does not eliminate these possibilities. Every member still receives ordinary
CE supervision through the other F-trained coordinates.

With independently adjustable logits for one isolated labelled example, a
plain gradient step with this positive multiplier increases the true-versus-rival
logit margin to first order. F already has that property. Actual attention
parameters are reused across nodes and layers, so this argument is not a
graph-model or Adam guarantee.

For a private relation block a_m, write g_im as the gradient of node i's
unreduced own CE, in one fixed realized view with N TRAIN nodes. Then

    g_J,a_m-g_F,a_m = (beta/N) sum_i (rho_im-1/M) g_im.

For any evaluated node t and rival c,
let d_tm=z_tm(y_t)-z_tm(c). Holding every other coordinate fixed, a small plain
gradient relation step gives

    change d_tm = -(eta/N) sum_i w_im <grad_a_m d_tm, g_im> + O(eta^2).

This exposes the actual uncertainty: the inner products can have either sign.
A graph-labelled training node can affect another node through its receptive
field or global attention. Reweighting can favor useful evidence directions,
reduce an adverse interaction, or amplify it. The running full update also
moves F-trained coordinates and uses Adam/history, so the displayed isolated
first-order expression predicts neither its final margin nor its trajectory.
It explains why lower TRAIN J or more attention disagreement is insufficient.

## What is specifically graph-related

Private local scorer rows can alter relative source weights at a fixed incoming
hidden-state/value tensor. The global tied-QK factors can alter the native
kernel affinities without being private value-projection factors. This is a
parameter intervention on graph relations. Across layers, the changed
aggregates also change later states and values; attention and feature effects
are not separable along the entire learned trajectory.

The native local scorer is GATv1. Its additive score followed by a monotone
LeakyReLU cannot rank neighbors differently as a function of the receiver
within a head [5]. Private copies do not remove that static-ranking restriction.
When every incoming preactivation lies in the same LeakyReLU linear region,
the receiver-side additive term cancels from the neighborhood softmax, so that
particular direct scorer path has zero derivative. This conditional local
calculation does not assert zero total derivative through earlier node states.

The native global operation uses q=k=sigmoid(k_lin(x)). Its per-head normalized
kernel weights are proportional to k_i^T k_j. Changing tied factors changes
both query and key maps together; it does not add independent query/key control
or invent query-dependent global attention. Native value maps, normalization,
multiplicative h/beta branch and later transformations remain part of the model.

The current recipient sets matter: relationJ changes 14 private local-scorer
banks and four tied-QK factor banks together. phiJ already includes those same
four QK banks, plus 52 other dense-factor tensors. Consequently a relationJ
advantage cannot separately identify local versus global attention, and the
relationJ/phiJ contrast combines inclusion of local scorers with exclusion of
other dense factors. No universal claim for attention-free backbones, graph
classification, heterogeneous graphs or link prediction follows.

## Closest methods and the remaining claim boundary

| Prior or saved proposal | Established contribution and actual difference |
| --- | --- |
| GNCL [1] and learner-collusion analysis [2] | Individual/pool-risk interpolation and its failure risks are established. The inspected GNCL author path applies a scalar mixture to all parameters. The candidate changes recipients; it does not introduce a new ensemble loss. Probability averaging must not be silently replaced by mean-logit CE. |
| Collaborative Learning [3], ONE and PCL [4] | Live shared/private branches, member plus collective supervision, differential loss exposure and backward treatment have direct prior. Their peer/KD losses, gates or feature-fusion classifiers and default deployment differ from the present fixed pool and exact scorer/QK recipients. Selective gradients are not the novelty. |
| GAT/GATv2 [5], native Polynormer and BE | Learned graph relations, shared dense weights and private factors are established. The candidate supplies a particular supervision allocation to known parameter sites, not a new attention operator or sharing principle. |
| CoGNN [6] and the saved communication-credit proposal | Learned graph communication is prior. The project proposal already states own-risk shared/features plus half-mixture private communication feedback. Using prepared native scorers/QK factors replaces the proposed sampled sender/receiver machinery; it is not a newly discovered general communication-credit idea. |
| Fed-GAME and GNNMoE [7] | The v22 inspected scopes describe attention over federated client updates and gated intermediate graph experts, respectively. They do not specify this four-classifier scorer/QK F/J rule within those scopes. This bounded distinction is not exhaustive novelty clearance. |

The known-versus-new distinction is therefore narrow. A useful result could
show that **this exact relation-recipient allocation** consistently improves
shared-backbone graph predictors beyond equal-architecture own risk, scalar J
and broader factor J, while maintaining competence. A sharing-specific claim
additionally needs an objective-matched untied reference. Superiority to
ordinary ensembles needs competent independently initialized/trained models
with their own selectors. Those are different claims.

No exact complete published collision has been established in the saved
scopes. No publication priority is established either. The inaccessible
Information Fusion paper *Graph ensemble neural network*,
DOI10.1016/j.inffus.2024.102461, remains unresolved; its inaccessible body cannot
be treated as evidence of absence. Generic combinations of known components,
small parameter counts, a nonpotential field or successful engineering cannot
by themselves establish methodological novelty.

## Falsifiers for the current mechanism

- Failing the frozen primary utility/competence screen leaves this specified
  rule unsupported; no beta/initializer sweep follows automatically.
- Matching or better allJ/phiJ removes a demonstrated special advantage of
  relation placement. A generic known-risk gain can still be recorded honestly.
- Increased attention variation or lower J without net full-population repairs
  is not useful diversity. Report introduced errors as well as repairs.
- Unchanged common-rival support and any-correct coverage alongside only better
  NLL supports confidence adjustment, not demonstrated new corrective evidence.
  Better accuracy need not require greater any-correct coverage, so these are
  interpretation limits rather than extra post hoc admission thresholds.
- An apparent advantage that disappears under a competent single/eight-view or
  ordinary ensemble reference leaves the corresponding ensemble claim unsupported.
- Effects restricted to reused split roles or optimizer seeds do not establish
  confirmation across new data roles or graphs.

These statements interpret the existing frozen screen. They do not add an
outcome-dependent gate or change the running family.

## One conditional next experiment: tying by relation supervision

**Inactive design only.** If the whole twelve-fit family passes its unchanged
screen and supports relation placement over allJ/phiJ, run one fixed factorial
study on three prospectively chosen genuinely unused official WikiCS split-role
blocks: shared versus untied feature maps, crossed with F versus relationJ.
One paired seed is fixed per block before scoring; all four conditions run the
complete graph, native width/depth and full 1100-epoch recipe. This is twelve
new complete fits, not a coefficient or initializer grid. Root first audits
role/test exposure and freezes exact split/seed IDs, recipe, source and costs.
TRAIN and prospectively declared development roles supply this comparison;
official TEST stays closed until a separately authorized final evaluation.
Overlapping roles on the same graph are disclosed; this is split robustness,
not three independent graphs or general cross-task confirmation.

The untied condition begins with four physical copies of the shared condition's
fresh native matrices. It preserves the same initial per-member functions,
private scorer/factor banks, two views, streams, probability mean and mean-loss
scaling; subsequently those copied maps can evolve separately. The true-label
own/pool mixture is sent only to the corresponding private local scorer/QK
parameters. All remaining coordinates receive F. Same-state accumulation,
one native optimizer transition and the **same joint selector/stage restore**
are required in all cells. Source/native resource qualification is a separate
prerequisite; identical initialization and normalization are design choices,
not a claim of optimal ordinary-ensemble training.

Estimate both within-architecture relationJ-minus-F contrasts and their
difference. A positive interaction supports that tying changes the utility of
this recipient policy under this exact recipe. It does not prove sharing
improves accuracy absolutely: shared relationJ must also be compared directly
with both untied cells. An equally large untied gain supports a general
attention-supervision adaptation rather than a sharing-specific benefit.

Use the complete prespecified populations and both benefit/harm readouts;
retain all four conditions and all costs. Numerical opening follows whole
family closure. No stochastic-view or optimizer-normalization adjustments are
tuned after looking at results. Copied starts and joint selectors intentionally
isolate tying/credit under matched opportunity; **untied F is not automatically
the strongest ordinary independent ensemble**, which requires independently
initialized models and their original own selectors. Those competent
benchmarks remain necessary before any superiority claim. No second new
mechanism experiment is proposed if the present family fails.

## Citations and reading limits

1. Buschjager, Pfahler and Morik, *Generalized Negative Correlation Learning for
   Deep Ensembling*, [2011.02952v2](https://arxiv.org/html/2011.02952v2), saved
   Section4.1/Eq5 and pinned author-code scope. The exact responsibility algebra
   is saved elementary application, not GNCL's Taylor quadratic without remainder.
2. Jeffares et al., *Joint Training of Deep Ensembles Fails Due to Learner
   Collusion*, [2301.11323v1](https://arxiv.org/html/2301.11323v1), saved method
   and probability/score-pooling appendix scopes; no theorem transferred.
3. Song and Chai, *Collaborative Learning for Deep Neural Networks*,
   [1805.11761v1](https://arxiv.org/abs/1805.11761v1), saved method conclusions.
4. Lan, Zhu and Gong, *Knowledge Distillation by On-the-Fly Native Ensemble*,
   NIPS2018, [1806.04606v2](https://arxiv.org/abs/1806.04606v2); Wu and Gong,
   *Peer Collaborative Learning for Online Knowledge Distillation*, AAAI2021,
   [2006.04147v2](https://arxiv.org/abs/2006.04147v2). Saved complete method and
   deployment scopes; author implementation and target-detachment not qualified.
5. Brody, Alon and Yahav, *How Attentive Are Graph Attention Networks?*,
   ICLR2022, [2105.14491v3](https://arxiv.org/abs/2105.14491v3), saved method
   pp2-5/Eqs2-7. Appendix theorem proof not certified here.
6. Finkelshtein et al., *Cooperative Graph Neural Networks*,
   [2310.01267v2](https://arxiv.org/html/2310.01267v2), ICML2024 metadata and
   saved method/source scopes; proceedings/source equivalence not certified.
7. Fed-GAME [2603.01363v1](https://arxiv.org/html/2603.01363v1), saved
   Sections2.1-2.4/Eqs1-3; GNNMoE
   [2412.08193v2](https://arxiv.org/html/2412.08193v2), saved Section3/Eqs2-7.
   These are bounded preprint method scopes; no new published-body equivalence.

This note reuses scoped conclusions without inflating paper counts. The native
Polynormer text at commit fc8c276c9c5dfbd616d83f65338a3392188a5e08 was reread
only to check the q=k kernel operation. No paper-body retrieval, new primary
scope, proof or implementation certification occurred. No current prediction,
checkpoint, dataset, label, pending fit score or numerical array was opened.
Historical closed-result summaries inside reused notes were incidentally
exposed; they supply no new numerical evidence for this study. No remote
contact, scientific execution, source edit to a frozen study, canonical memory
mutation or launch authorization is supplied. Input byte bindings accompany
this note. Root owns any admission, publication and future scientific claim.
