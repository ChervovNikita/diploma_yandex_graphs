# Outgoing-zero nonlinear message initialization

## Assessment

The suggested alternative changes the new branch's initial accessible first-order functions. With finite incoming V and outgoing B0, the first outgoing gradient sees `P tanh(HV)` immediately. With incoming V0 and finite B, the first incoming tangent is linear in `PH`. This is a real distinction at a fixed native interface, with a specific equal-aggregate example below. It supplies no whole-NCN impossibility, optimizer-success or novelty theorem.

The alternative is worth a fixed mechanism comparison within the existing growth family. It is not a new larger method family: LoRA already uses outgoing-zero initialization, EVA supplies activation-SVD incoming initialization, and AdaptFormer already uses a nonlinear bottleneck with a finite down-projection and zero up-projection. The possible graph-specific utility is where the nonlinear dictionary is placed before aggregation, how it is initialized and whether it improves a competent native model beyond the matched controls. Existing growth v1/v2 remain sealed and unchanged.

No numerical/model/data/checkpoint/score work, source changes, GPU launch or server access occurred. Four first scoped primary method reads were made; EVA upgrades the Oct4 metadata-only lead. GradMax and the existing growth/source conclusions were reused without new credit.

## Precise interface proposition

Fix current H, native support P and the interface cotangent G=dL/dY. Let

`Y=PH + P φ(HV) B`, with V shaped d×r and B shaped r×d'.

There are no extra biases or normalization in the new branch. Native tail/JK operations must remain in their original arithmetic. Assume finite differentiable operations, fixed dropout realization and the exact same support for comparison. φ is tanh for the proposed experiment.

### Incoming zero, outgoing finite

At V0 and B=B0, with φ(0)=0 and φ'(0)=1:

`Y0=PH`, `grad_B=0`, `grad_V=H^T P^T G B0^T`.

The initial new-branch tangent is

`T_in={PH ΔV B0 : ΔV arbitrary}`.

All pre-existing parameter derivatives match the baseline in real arithmetic. For ordinary SGD on this branch with H held fixed, the first-order correction is

`ΔY_in=−η (PH)(PH)^T G B0^T B0`.

This has broad node directions in the linear aggregate PH and a restricted output subspace determined by B0. At two nodes with equal PH rows, every such new-branch tangent has equal rows. This statement concerns the branch/interface with fixed H, not every learnable old parameter or the full decoder.

### Incoming finite, outgoing zero

At V=V0 finite and B0, define `Φ=P φ(HV0)`. Then

`Y0=PH`, `grad_V=0`, `grad_B=Φ^T G`.

The initial new-branch tangent is

`T_out={Φ ΔB : ΔB arbitrary}`.

Again, the baseline function and old-parameter derivatives match in real arithmetic. With H/V held fixed for an outgoing SGD step,

`ΔY_out=−η Φ Φ^T G`.

That expression is exact for the B update because the branch is linear in B. Once B becomes nonzero, incoming gradients can become live, but this requires nonzero relevant cotangents and nonzero activation derivatives; it is not guaranteed. Saturation, dead features, a zero outgoing gradient or later cancellations can prevent it.

These tangent spaces are generally **incomparable**. Outgoing-zero allows unrestricted output directions but only the r chosen nonlinear node-feature directions; incoming-zero offers the PH node span but only the r chosen output directions. There is no universal tangent containment or universal advantage. If φ is identity, both policies depend only on PH and neither repairs equal-PH aliasing.

The formulas describe SGD/local geometry. The actual source uses Adam, whose reset first step is approximately coordinate sign normalized. Larger raw gradient norm is not a benefit certificate, and swapped zero ends alter which parameter block receives its first moment/update. Shared-loss reduction, epsilon and parameter scale matter. Real function/gradient inclusion does not certify FP32 bitwise agreement: both policies need actual copied-native logits and old-gradient gates.

## Equal-aggregate illustration

Use the parent's proposed two disconnected two-leaf stars. The central target feature is0 in each component. The leaves are `(0,2)` in the first and `(1,1)` in the second. Centers have degree2 and leaves degree1. With native

`P=D_(1+deg)^(-1/2)(A+I)D_(1+deg)^(-1/2)`,

each center-to-leaf coefficient is1/sqrt6. Thus both target rows of PH equal2/sqrt6. For scalar incoming v1:

`Φ_first=(tanh0+tanh2)/sqrt6`, `Φ_second=2 tanh1/sqrt6`.

They differ because tanh is strictly concave on positive inputs. For an interface cotangent supported only at the two targets, with rows g and−g, the incoming-zero branch gradient cancels: `grad_V=(2/sqrt6)(g−g) B0^T=0`. The outgoing-zero gradient is `(Φ_first−Φ_second)g`, nonzero when g is nonzero. Its first outgoing SGD update gives target correction difference `−η(Φ_first−Φ_second)^2 g`.

This proves a layer-conditional representation/tangent distinction, including a possible supervision cancellation. It does not show that real TRAIN errors are caused by that alias, that other old parameters cannot separate the queries, or that the NCN common-neighbor decoder has identical information. Stochastic dropout, changed masks, updates to H and other native paths can also break equality. DeepSets' feature-before-sum construction and GIN's injective-multiset analysis are prior ancestry, not new principles established by this example.

A finite rank2 tanh dictionary is not injective over all neighborhoods. With no bias, tanh is odd: equal-degree symmetric `(-a,+a)` neighborhoods and `(0,0)` can still have the same nonlinear aggregate. Additional biases, even features or larger dictionaries would change this fixed architecture and require a new protocol. A capable nonlinear single can exploit the same example, so the illustration supplies no ensemble-only capacity result.

## Primary initialization evidence

| Source and scoped read | Supported operation | Limit for this proposal |
|---|---|---|
| Original LoRA, arXiv2106.09685v1, pp3–4 | `W0+BA`, with incoming A random Gaussian and outgoing B0; W0 frozen. The added linear update starts at0 and can be folded into W for serving. | Zero outgoing is direct prior. With identity activation its first features remain linear in the original aggregate. Our nonlinear message branch cannot generally be folded away, and our old body stays learnable. |
| PiSSA, arXiv2404.02948v1, pp2–3 | Both factors start from principal weight SVD; their product plus a frozen residual reconstructs W. | Baseline preservation relies on subtracting the initial *linear* product from the residual base. A generic finite `P tanh(HV0) B0` cannot be compensated by a linear PH map for all H. Factor letters differ across papers: determine incoming/outgoing by dimensions and dataflow. No nonlinear PiSSA port is certified. |
| EVA, arXiv2410.07170v1, pp3–5 | Incoming LoRA factor from right singular vectors of downstream activations, outgoing factor0, optional global rank redistribution by explained variance. | Data-driven finite incoming directions with outgoing-zero are already prior. This is the first method scope upgrading the Oct4 metadata-only lead, not a new metadata discovery. We do not adopt its rank redistribution or printed variance-ratio formula as a qualified graph recipe. |
| AdaptFormer, arXiv2205.13535v1, pp4–5, p6 initialization paragraph, p12 pseudocode | Down→ReLU→Up residual bottleneck, finite Kaiming down weights and zero up weights/biases; original body frozen. | Direct nonlinear outgoing-zero ancestry. Main prose uses Kaiming Normal and pseudocode Uniform, but both unambiguously retain finite incoming/zero outgoing. Our native site is preaggregation and all original shared/private weights learn; usefulness at that site is still untested. |

GradMax, already method-read in the preceding note, is the incoming-zero/nonzero-outgoing comparison with φ'(0)=1 and gradient-selected outgoing bases. Neither its raw-gradient objective nor EVA's gradient-norm plots imply Adam superiority. No author implementation, full-paper/proof/results or target reproduction is certified by this packet.

## Fixed representative initialization design

Use the exact existing HeaRT Citeseer native site: dropout-off H from warm xemb, one normalized PureConv, taildropout0 and scalar JK. Use the same root-bound terminal20 TRAIN-only common warm state and the same prospectively fixed first cycle0 endpoint/random union-masked calibration episode. Retain native backbone/head operations and all trainable shared/dense/BE leaves. M4/rank2 adds4096 parameters; a single with eight incoming units adds the same number. New biases are absent. Packed8-channel extra P computation remains applicable.

For the primary outgoing-zero graph initialization, reuse the **same** matrices as the existing graph-output initializer:

`K_q=H^T P^T F_q(P)G`, `F_q=binom(3,q)(I+P)^(3−q)(I−P)^q/8`, q0..3.

Instead of top2 *right* singular vectors initializing outgoing B, use top2 **left** singular vectors as incoming directions V_q. Normalize each finite incoming column by the full-public-node RMS of its preactivation H v_j, giving one fixed preactivation RMS1. Set every B_q exactly0. Zero/nonfinite RMS or unusable rank stops without redraw or scale search. Columns are computed from the fixed TRAIN G/support only; all public node features are permitted, no VALID/TEST labels or teacher targets.

This is an error/graph-aware finite feature initializer, not a guarantee of maximal actual outgoing gradient. The left-SVD optimum concerns the corresponding *linear* proxy; finite tanh and graph-filtered G remove that exact objective. Graph bands only initialize directions and do not impose persistent frequency specialization.

Mandatory comparison policies are fixed before outcomes:

1. Existing incoming-zero graph-right-SVD initialization, retained from sealed growth v2.
2. Outgoing-zero graph-left-SVD, as above.
3. Outgoing-zero unfiltered top8-left-SVD partition: four disjoint2-column blocks of the same unfiltered `K=H^T P^T G` top8 left basis. This controls subspace coverage/diversity, not just repetition of top2.
4. Outgoing-zero feature-SVD partition: top8 right singular vectors of raw H, split into four disjoint pairs, with the same RMS rule; EVA-inspired activation control.
5. Outgoing-zero fixed seeded Gaussian incoming directions, with the same shape/RMS rule; LoRA/AdaptFormer-inspired random nonlinear control.

All incoming columns remain learnable after initialization. There is no rank, site, basis-strength, activation-strength or optimizer sweep. The five policies are distinct scientific controls, not candidates selected by VALID. One prospective identity-activation control is required before attributing a positive result to nonlinear neighborhood moments; it need not replace the first fixed quality screen.

A capable complete native **single8** receives the exact concatenation of the graph-left incoming columns and outgoing B0. It has the same nonlinear dictionary and new parameter count; its entire native nonlinear predictor learns. Also keep the no-growth native/shared references and a separately optimized same-growth native4 control. That control may use common warm copies for the first bounded paired experiment, but must retain the existing common-warm/pooled-selector limitation explicitly. A positive ensemble claim subsequently needs an ordinary genuinely independent native bank and a strong GNN backbone with its own nonlinear message transformations. A weak, frozen or unqualified single cannot establish ensemble necessity.

The representative quality screen keeps the existing warm20/post60/eval5/three-seed0–2/three-ordinary-pass source recipe and raw-logit pooling, with all old sources/results untouched. It is a future separate protocol/source amendment; this note authorizes no fit or implementation mutation.

## Actual gates and attribution requirements

Before any fit, require actual copied-native logits and old-gradient agreement for both zero-end policies on the same stochastic realization. Incoming-zero must have grad_B0 and live incoming gradients as originally qualified. Outgoing-zero must have grad_V0 and live outgoing gradients. On a discarded qualification copy, after one fixed outgoing update check whether incoming gradients become live; no updated qualifier state becomes scientific initial state. Report the actual function step and norms under the admitted Adam, never substitute the SGD equations for a runtime gate.

For outgoing-zero, initial B projectors are zero and therefore cannot certify route diversity. Inspect actual **node-feature** subspaces of `Φ_m=P tanh(HV_m)`, their usable ranks and projector separation. Sign flips of incoming directions produce sign-flipped tanh features and the same node-feature projector; they do not create tangent diversity. Arbitrary rotations within an incoming subspace can change tanh features, so inspect the realized nonlinear dictionary, not merely V's columns.

A useful TRAIN mechanism diagnostic projects Φ onto the complement of the PH column space, using an orthonormal basis Q for PH:

`Φ_res=Φ−Q(Q^T Φ)`.

Record its rank/energy and `Φ_res^T G`. Nonzero residual features show interface functions outside the incoming-zero branch's linear node span; nonzero residual-gradient correlation shows first-order relevance to that calibration cotangent. Neither certifies new directions outside the **full old-model score Jacobian**, which can already be rich/full row rank on finite queries. It also does not certify useful held-out common-error correction.

Stop the structural mechanism claim when the realized dictionary has no relevant nonlinear residual, unusable ranks or collapsed projectors. Stop before fit if native copy/old-gradient gates fail. Saturation, dead features and failure of the incoming block to become live are recorded failures, never rescued by a new scale. No larger gradient norm or disagreement-only result is success.

For quality, outgoing graph initialization must improve pooled served quality over incoming-zero and unfiltered/EVA/random controls without lowering equal-seed mean member competence. Report every seed/member and full cost. If identity activation explains the positive result, close the nonlinear-moment attribution. If the capable single8 matches, close demonstrated ensemble necessity. If a strong nonlinear message backbone explains the gain, describe the result as a capacity/site repair for this native encoder, not a general GNN limitation. TEST remains closed until every decision is frozen.

## Recommendation

Keep this as a precisely motivated initialization variant of nonlinear preaggregation growth. It fixes a possible immediate tangent weakness of incoming-zero while retaining the baseline, and has clear prior ancestry and counterexamples. It is not proof that the current native model's common errors are aggregation aliases. The independent-teacher compression note is the larger separate architecture/initialization direction; neither note alters the sealed current pilot or certifies performance or novelty.
