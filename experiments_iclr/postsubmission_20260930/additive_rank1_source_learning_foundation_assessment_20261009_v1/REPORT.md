# Additive rank-one adapters and direct joint source learning

9 October 2026. **A worthwhile known-method foundation to compare, with no established graph-specific advantage.** Additive rank-one adapters offer different affine directions from multiplicative BE; direct source-gradient integration changes the learning policy. They must be separated. TabLoRA already jointly learns shared W/private low-rank adapters from scratch; LoRA already supplies random incoming/zero outgoing initialization. This is an attributed variant, not a new adapter or ensemble principle, and no run or implementation is admitted.

## Closed context and what it diagnoses

Root supplied a fully closed pilot summary: assigned-source loses to both genuine references by approximately1.15/1.16 micro-F1 percentage points in every pair and exactly ties shared-own-only F1. Selected states retain1/3/2 accepted corrections and private L2 doses0.00861/0.04958/0.03382. U/D source use is nonzero but near-identical across members/arms. No outcome artifact was reopened or recomputed here.

Thus the frozen candidate failed. It was not a zero-intervention arm. Exact F1 ties do not imply equal logits/BCE, and nonzero parameter motion does not imply useful function motion. The summary supports **no demonstrated useful source specialization under that recipe**. It does not diagnose a binding BE matrix constraint, prove all source gradients are ineffective, or justify a larger dose/horizon. Selected-state accepted counts are not substituted for the full training history. The closed source, results, quality freeze and failure remain intact; this note supplies no lambda/dose/epoch rescue.

## Existing ancestry, consulted first

| Saved source/scope | Direct overlap and limit |
|---|---|
| [TabLoRA,2607.10077v1](https://arxiv.org/html/2607.10077v1), saved §§3.3–3.5/Eqs4–10 | Shared W and private BA trained jointly from scratch, nonlinear member trajectories and mean-own loss. Its group-private heads and mean-logit/softmax serving differ from the present shared native heads and Bernoulli probability mean. Direct whole-family ancestry; no graph quality transfer. |
| [LoRA,2106.09685v1](https://arxiv.org/abs/2106.09685v1), saved pp3–4; [LoRA-Ensemble,2405.14438v5](https://arxiv.org/abs/2405.14438v5), saved method/initialization scopes | Additive low-rank maps, random incoming/zero outgoing factors, independent adapters and ensembles are prior. Their frozen pretrained bases differ from jointly learned fresh W; teacher acquisition is unnecessary for the proposed from-scratch variant. |
| Saved outgoing-zero/AdaptFormer/EVA assessment | Different initial dictionaries/zero ends change accessible tangents while preserving the initial base function. Sign-related dictionaries can have identical functional tangents; zero outgoing does not guarantee competent members or incoming-factor learning. |
| Saved GraphLoRA/HG-Adapter/GaRA/graph-conditioned adapter scout | Low-rank graph adaptation, typed structural conditioning and graph-generated factors already exist. A static member/group adapter is not an edge-conditioned graph transport or a new semantic expert mechanism. |
| Saved source-credit prior review | The source gradient is signed responsibility-weighted factual BCE minus source-ablated BCE. HGEN/semantic ensembles, GNCL and source views establish related ingredients. The present finite guards and actual probability serving are material parts of its declared policy. |

The saved centered-additive assessment also establishes that subtracting the adapter mean with a freely learned shared W is a change of coordinates, not a new member-function family. No centering repair is added. Existing scopes already answer the method question: **zero new primary scopes/retrievals, zero full-paper credit**.

## Exact affine-family differences

Use column-vector notation with shared `W∈R^(d_out×d_in)` and shared outside bias b. Compare

`BE: W_m=D(s_m) W D(r_m)`,

`additive: W_m=W+u_m v_m^T`.

Each has d_out+d_in private scalars per member **per actual group/site**, without a private bias or learned extra scale. The proposed grouped replacement must preserve the original group boundaries and six native sites; a layer-wide adapter spanning groups would change capacity. Equal scalar counts do not match effective directions, optimizer geometry, regularization, activation work or usable first-step dimensions.

For two rows i,k and columns j,l, with nonzero denominator and scales, BE preserves the cross ratio

`(W_m,ij W_m,kl)/(W_m,il W_m,kj)=(W_ij W_kl)/(W_il W_kj)`.

It also preserves W's zero entries. All members share these ratios at a fixed layer even while their common W learns. Additive rank-one changes can fill zeros and change ratios: `W=[[1,1],[1,2]]` has ratio2; adding `e1 e1^T` gives `[[2,1],[1,2]]` with ratio4. Those two member matrices cannot belong to one BE diagonal orbit.

**There is no general capacity containment.** Additive rank-one members obey `rank(W_m−W_n)≤2`. BE can have full-rank pair differences: in dimension3, members I and2I are BE-realizable but their difference has rank3, excluding a common-base rank-one additive realization. The affine families are generally incomparable. Multiplicative scale changes can make high-rank corrections; additive rank one changes a chosen coupled input/output direction.

These are layer-conditional constraints. Nonlinear activations, biases, normalization, evolving shared W, multiple sites and member-specific semantic attention can represent the same final predictor through other settings. A failed fixed-W cross-ratio representation is **not** a whole-network impossibility or an accuracy theorem. Raw matrix distance is not source use or correct decision diversity.

## Exact initial Jacobians and the initializer confound

At `r=s=1`, the BE tangent is

`delta W=diag(delta s)W+W diag(delta r)`.

For `Y=W X+u(v^T X)+b`, with `u=0` and a finite random v,

`delta W=delta u v^T`, `grad_u=G v`, `grad_v=G^T u=0`,

where G is the effective-weight cotangent. All members initially have the same native function and old-parameter derivatives; their different v values provide different candidate tangents, not different initial predictions. Only d_out outgoing coordinates are initially live; BE usually exposes more independent first-order scale directions. Once u moves, v can become live, but zero cotangents, limited input span, normalization or downstream nullspaces can prevent it.

For a plain first SGD step, `delta W_m=−eta G v_m v_m^T`. Distinct useful dictionaries can yield different changes; v and−v have the same projector and first function step. Under Adam, coordinate moments/epsilon and factor scale change this equation; gradient magnitude or vector norm is not a benefit certificate. Both factorizations have scaling gauges and different coupled decay semantics. The outgoing-zero start is a singular bilinear point, not a generic rank-one-manifold tangent. These statements are manual algebra, with no numerical fixture or model run.

The exact SeHGNN semantics remain decisive. First grouped projection inputs removed with their raw family can be zero, giving zero probe-adapter derivatives and a positive weighted factual-BCE remainder. Later LayerNorm/native biases mix retained channels, so probe dependence can remain. Native semantic gamma starts at0: Q/K/V predictor derivatives are zero then for **either** parameterization, until ordinary learning activates that path. An additive adapter cannot bypass this gate. Native residual/head/shared bias remain capable explanations. Factors in semantic processing of precomputed graph/TRAIN-label channels do not create learned per-edge geometry.

## Direct joint learning is a separate change, with a risk tradeoff

A proposed integrated rule would give shared/native weights ordinary own-task F, and eligible private factors F plus a fixed source-J gradient in the same old-state Adam update, from the beginning. Peer values remain detached under the exact declared source semantics. It must specify reductions, recipients, peer mode/state, source exposures, optimizer states and source-work cost before outcomes. Restricting J to private parameters is not unrestricted minimization of F+lambda J over every parameter.

The old method instead follows own Adam with scheduled projected/finite source corrections, preserves its native schedule, and admits a step only under full/probe/absent-risk anchors and finite guards. Integrated unguarded Adam is neither the same step nor a dose match: J now contributes to adaptive moments, interacts with own gradients and supplies many earlier opportunities. Comparing additive-integrated to old BE-guarded conflates parameterization, initialization, objective exposure, optimizer history and safety constraints.

The guard tradeoff is concrete. For an observed Bernoulli event, a recipient's source contrast is supplied-pool NLL minus absent-pool NLL. With frozen positive peer mass c, its derivative with respect to its absent-event probability q is `+1/(q+c)` up to the declared averaging factor. Minimizing J can therefore lower q and **worsen absent-view evidence**. The old absent-risk guard prevents that route at its declared step. Own full-input task loss alone does not protect source-ablated competence. Removing guards can permit useful larger joint moves, but also source gaming, wrong-member damage and noisy early specialization. No performance/competence guarantee survives their removal. Retaining guards around a combined Adam step would be another explicitly specified policy, not ordinary unguarded Adam; no new transaction framework is designed here.

## A bounded hypothesis and the crossed controls it requires

The remaining falsifiable question is narrow:

> At the same verified source sites/information and complete native training/selection/serving opportunity, does additive rank-one private adaptation improve competent task learning over multiplicative BE, and does persistent source-assigned private credit add useful source-specific correct decisions within either parameterization?

The current result does not establish that rank restrictions caused failure. This is an **existing-method foundation and a prospective interaction question**, not a defensible new graph-specific mechanism. The first foundation comparison can use ordinary F alone; source credit earns a separate result. No compound win is attributed.

| Fresh prospective shared condition | Contrast |
|---|---|
| BE + ordinary F | Foundation reference. Preserve the closed BE recipe as historical evidence; do not relabel it. |
| Additive rank1 + ordinary F | Parameterization/zero-output dictionary package under task learning. A pure parameterization claim still needs its initializer control. |
| BE + directly integrated source credit | Effect of the new learning policy within BE. |
| Additive rank1 + directly integrated source credit | Same policy within the alternative coordinates; assess the crossed interaction. |

For a fixed task endpoint Q, report the policy effect `Q_(Add,J)−Q_(Add,F)` and `Q_(BE,J)−Q_(BE,F)`, plus their difference. A win only in the compound cell does not identify either main effect. Match source views/credit/exposures and paid forwards within the two J cells; raw factor-L2 doses are not comparable between parameterizations. Use actual prediction changes, own risks and objective/component gradients to disclose different physical update strength. No coefficient/horizon/scale grid or retrospective rescue is proposed.

Initializer attribution needs an additive copied/common incoming dictionary versus independent random dictionaries, both with outgoing zero, matched native starts and declared scaling. Sign-flip copies are a known functional-null control and do not deserve a redundant fitted arm. If randomness explains the gain, call it initializer/optimization utility; do not infer that the affine family restriction was binding. A swapped-zero initializer changes the initial tangent too and is not silently an identical control.

A **true independent4** must own four fresh full bodies/optimizers and the same adapter/source operation/recipient restrictions, actual own checkpoints and probability serving. Detached committee losses couple learning but do not share storage. Include its ordinary-F counterpart to bound sharing claims; never use these models as teacher donors. Existing genuine native/factor references remain competence anchors, with current consumed validation/seed exposure acknowledged.

Capacity must also be capable: a single rank4 adapter has the same four rank-one private-vector budget at a site and can test extra low-rank capacity. It does not reproduce separately evolving nonlinear trajectories. An ensemble-necessity claim additionally needs a same-information joint model with the four live paths and a capable learned decision rule, charging every path/readout. Match source access and labels; a one-path scalar head on a compressed embedding is insufficient. No weak single or identity-initialized but inactive reference establishes a sharing advantage.

Success requires useful full-input pool and member task quality against competent references, retained mean/worst member competence, all-label BCE/F1 and whole-population repairs versus harms, plus complete current U/D and assigned-versus-off-diagonal source responses. Different u/v, attention or source gradients alone fail. If additive F explains the result, the finding is ordinary adapter capacity/optimization. If COMMON/same-source exposure or source-view supervision explains J, persistent assignment is unestablished. A source-specific claim therefore retains those known attribution controls rather than promoting the four-cell contrast into proof of semantic specialization.

## Disposition and scope

Keep additive joint-from-scratch adaptation as a **standard foundation worth considering for competence**, with precise layer/tangent differences and strong direct prior. Keep direct source credit as a separate, currently unadmitted policy question because it changes the risk protections. No new graph-specific claim is established and no additional study is automatically warranted by the closed failure. Root decides whether a separately frozen comparison is useful after diagnosing the complete current pilot.

Only saved theory/prior excerpts, source specifications and root's supplied aggregate context were used. No current outcome file, checkpoint/logit/label/dataset, model import, fixture, code/framework/fit, research server or GPU access occurred. No new primary request was necessary. Sources, scores, gates, running state and the closed recipe remain untouched. Exact reused identities, scopes and supplied-context provenance are saved alongside this report.
