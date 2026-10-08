# Decision supervision for private graph attention

## Recommendation and evidence boundary

Retain **one inactive, attributed hypothesis**: ordinary member supervision
learns shared features and dense BE factors, while an own/pool-risk mixture
learns only copied private local attention scorers. The useful contribution,
if supported, would be a particular effective graph-ensemble training rule.
Private attention, probability-pool credit and selective gradients are known
ingredients. No new optimization principle, mathematical guarantee, quality
result or publication-priority claim is established.

The parent reports that complete Context9 failed its original Stage1 gate:
route-minus-common mean accuracy was −0.0316016 percentage points, with the
three-seed df2 interval [−0.10355, 0.04035]; route-minus-permuted was −0.0063203,
with interval [−0.22891, 0.21627]. No twelve-reference extension was admitted.
Those are parent-supplied closed results, not recomputed here. They support
changing the mechanism rather than varying the same alignment coefficient.
They do not establish that every contrastive method or graph target fails.

This note reads saved method conclusions and two new primary method scopes.
It opens no project array, checkpoint, prediction or current fit score, and
prepares no scientific code or run.

## The complete proposed learning rule

Use the existing four unit-factor Polynormer routes and the prepared private
local attention bank. Each native local GAT `att_src` and `att_dst` vector
becomes four copied rows, installed after native initialization and before a
fresh Adam. Retain the original dense BE factors, graph support, local/global
stages, values, normalization, task head, member RNG streams and probability
mean at inference. All four routes remain active. There is no serving router,
teacher bank, embedding repulsion, extra head or separately pretrained ensemble.

For each original stochastic CE view v, define Fv as mean-member own CE and Lv
as CE of the mean of the four class-probability vectors. Use
Jv = .5 Fv + .5 Lv. Average the two view objectives **after** constructing their
separate pools; pooling all eight member/view probabilities first is another
learner. Collect all partial derivatives from the same old model/buffer state
and realized views before one original Adam transition:

| Prediction parameter | Supervision |
|---|---|
| Shared feature maps, attention-value projections, normalizations and head | F = mean of the two Fv |
| Every existing dense BE factor, including input/output factors | F |
| Only the copied private local attention scorer rows alpha | J = mean of the two Jv |

The bank is the already prepared native `att_src`/`att_dst` bank, not a new
edge network or global Q/K/V adapter. No common scorer remains trainable behind
it. The frozen beta=.5 is a declared balanced attribution choice, not an
estimated optimum. No beta, initialization, coefficient or selector grid is
recommended.

For one example/view, the scorer derivative for member m is its ordinary own-CE
derivative multiplied by

`w_m = (1−beta)/4 + beta p_m(y)/sum_k p_k(y)`.

At beta=.5, w_m lies between .125 and .625 and sums to one over members. This
follows directly from the established probability-pool mixture. It is not a
new responsibility theorem. Use the true J partial or an exactly equivalent
first-order cotangent; differentiating a hand-built weighted-CE scalar through
the weight itself adds terms and does not implement this rule.

The own term keeps positive supervision on each scorer for every TRAIN label.
It is **not** a competence guarantee. At equal member probabilities, the
weights are exactly uniform and alphaJ matches alphaF for that example. There
is no deliberate symmetry-breaking signal at identical deterministic paths.
Native different stochastic paths and subsequent private factors can create
different credits; that does not guarantee useful specialization.

## Mechanism and meaningful failure cases

The narrow hypothesis is that output-level supervision can assign examples to
graph-neighbour scorers without requiring a chosen graph-signature similarity
to represent useful contexts. Members with greater correct-class probability
receive more ensemble credit in their scorer coordinates. Shared features and
dense factors continue receiving the complete own-label objective.

Private scorer rows can change attention while keeping the projected value
tensors fixed. Existing dense BE output factors generally couple changes in
the projected values and their attention scores. This is a direct parameter
ownership distinction, not a strict function-class separation theorem: dense
factors, later layers and normalization can compensate, and the additional
scorers can be redundant.

The native local layer is GATv1. A scorer bank does not remove its static
neighbour-ranking limitation or create a query-dependent GATv2 scorer.
Per-example **training credit** is not a per-example inference router. Each
scorer row is still a learned global parameter vector applied through the
native node representations.

Important falsifiers remain:

- Similar correct-class probabilities on common-error examples make credit
  approximately uniform, giving little change from alphaF.
- Shared features may lack the information needed by any scorer. An attention
  change cannot reconstruct information already lost in feature processing.
- Higher-confidence members may monopolize easier examples while difficult
  examples stay jointly wrong. The nonzero CE floor does not rule this out.
- Attention can differ through harmless temperature/confidence changes or
  irrelevant neighbours, with no net prediction repairs.
- The policy can weaken members, improve only pooled NLL, or reproduce an
  ordinary joint-risk advantage unrelated to graph-scorer placement.

The field is generally nonpotential because F and J have different recipient
blocks. Standard selective-gradient/game ancestry applies. Lower TRAIN J,
attention disagreement, a positive floor, or that mathematical classification
does not establish stable native Adam dynamics or heldout accuracy.

## Saved closest priors and internal overlap

| Prior or saved proposal | Exact relationship |
|---|---|
| GNCL, saved v2 Eq5 and author-code scope | Own/pool-risk interpolation is established. The inspected author fit uses one global scalar mixture backward. Restricting that mixture to scorer rows is a recipient change; it is not a new ensemble loss. |
| Joint Training of Deep Ensembles Fails Due to Learner Collusion, saved method scope | Own/ensemble risk mixtures and compensating-member failures require attribution and competence checks. Its analysis is not a proof that this scorer restriction succeeds or fails. |
| Song/Chai Collaborative Learning | Shared low features, private supervision, peer losses and shared-backward scaling are direct shared/private learning ancestry. It does not specify this GAT-scorer-only probability-risk field in the retained scope. |
| ONE | Private branches receive hard member and gated ensemble CE plus distillation; shared layers also receive those paths. Learned gates, summed losses, KD and deployment differ. |
| PCL | Shared/private peer features and selective loss exposure precede the proposal. A separate fused-feature teacher CE, peer/EMA KD and default single-peer serving differ. |
| TreeNets/CAMERO | Shared trunk, private branches and aggregated child supervision are established architecture ancestry. A small private attention bank is not a new sharing principle. |
| MolI / existing internal-risk interfaces | Shared/boundary blocks receive own risk; dense internal phi receives the GNCL mixture plus original phi-only alignment. Here every dense factor receives F, only local scorer alpha receives J, and no alignment is present. MolHIV mean-logit BCE credit also differs from WikiCS probability-pool responsibilities. |
| Prepared private attention COMMON/ROUTE design | The copied bank and constructor/source qualification already exist. That design targets contrastive context assignment. This supervised recipient rule is not an executed arm of it. |
| Saved CoGNN communication-credit proposal | Already proposes own-risk shared/features plus half-mixture private communication policies and a feature-credit control. The present proposal replaces an unprepared sampled action mechanism with the prepared native scorer bank. It is not a new general idea of rewarding useful graph communication. |
| GAT/GATv2, attention-disagreement regularization, SuGAr/DIVE | Learned graph attention and graph-evidence diversity are known. Diverse attention alone cannot be the contribution. |
| Saved graph-MoE scopes: NodeMoE, GraphMoRE, MORGAN, GC-MoE | Learned intermediate filters, geometry/expert choices and graph routing are prior. GC-MoE's inspected frozen independently pretrained experts differ from a freshly learned live common backbone. No absence of an exact operation is inferred from their titles. |

The saved `Graph ensemble neural network` DOI10.1016/j.inffus.2024.102461
remains method-inaccessible. Its title/metadata cannot clear this rule or
establish a collision. Unchanged failed access routes were not repeated.

## Two new bounded primary method checks

**Fed-GAME, arxiv2603.01363v1, 2026-03-02:** Section2.1–2.4, Equations1–3 and
the complete printed two-level training description. Each federated client
trains a persistent private forecasting model with local task loss plus a
proximal term. The server encodes final-layer parameter differences, uses
shared scoring experts and personalized noisy top-k gates to produce
client-update attention, and trains its aggregation module with L2/cosine
alignment to client updates. This is close graph-attention/private/shared
aggregation ancestry, but its graph is a graph of clients' parameter updates.
It does not train four native node classifiers under the proposed F/J partials
or serve their fixed mean probabilities. No exact complete-rule collision is
specified in the inspected method.

**GNNMoE, arxiv2412.08193v2, updated2025-02-12:** complete Section3, Equations2–7.
Soft node gates combine four propagation/transformation-order experts inside
PT blocks; hard Gumbel gates select FFN activation experts; learned residuals
and one final classifier produce node predictions. Equation7 gives the final
supervised softmax log-loss. Its method prose calls this binary CE, so author
code would be needed before claiming an exact native loss reproduction. The
method has one final prediction, rather than four separately supervised served
routes, and no printed scorer-only F/J derivative allocation. Metadata links
DOI10.1145/3701716.3715462; equality to the published body was not audited.

Neither scope establishes an exact complete collision. **Novelty remains
uncleared**, because the ingredients and broader communication-credit
composition are established and bounded searches cannot certify absence. If an
exact prior is later verified, abandon a novelty claim and classify the work as
an attributed reproduction/adaptation; keep all results regardless of sign.

Four arXiv discovery queries were saved before these reads. Other returned
titles/abstracts, including TESTAM and the various newer forecasting/graph MoE
methods, are locators only. Fed-GAME's cost/experiment sections and author code
were not read. GNNMoE's floated Table1 appeared inside Section3 during automatic
extraction, so its published numbers were incidentally exposed; they were not
used for model selection, quality claims or numerical comparisons. Neither
paper is credited as a full-paper, proof, code or reproduced-benchmark read.

## Representative prospective comparison

After **whole original Wiki12** terminal/source custody closes, use only its
three original plain anchors at seeds6101/6203/6307 and their original physical
GPU per seed. A later collector must bind the separate old-ten-plus-new-two
union closure; the failed old-owner closure remains false. Eligibility is
source/setup-based, never score-based.

Prepare and prospectively freeze two new conditions at all three paired seeds:

1. **alphaF:** copied private scorer bank; every parameter learns F.
2. **alphaJ:** identical bank/model and initialization; only alpha learns J.

This is six new complete fits and three existing eligible anchor records,
not nine new fits. Retain all original11701 nodes, 442907 ordered edges,
580 TRAIN labels, complete5274 development population, 512 hidden width,
seven local/two global layers, two CE views, full1100 epochs, strict-first
joint selector and local/global optimizer-state transition. No auxiliary panel
or hidden contrast is used. Install/reconstruct private banks before Adam and
selected-state loading; wrap all training shadow/replay/evaluation routes with
the same native scorer-row scope.

alphaF−plain measures the private scorer ownership/capacity intervention.
alphaJ−alphaF isolates output-risk credit into those scorers. alphaJ−plain is
a practical quality contrast; it alone identifies neither credit placement
nor graph-specific benefit.

Freeze the original plain error cohorts before candidate predictions open.
Primary outcomes use the full population and all seeds: served accuracy/NLL,
member mean/worst accuracy and NLL, any-correct coverage, pooled-only rescues,
paired repairs **and** introduced errors, and strict common wrong-rival support.
Preserve initial/final coefficient identity, checkpoint epochs and paid
forward/reverse/optimizer/serving costs. Attention diagnostics are secondary;
they cannot select favourable nodes or replace prediction quality. Three
optimizer seeds on one graph are exploratory paired evidence, not three
independent graphs.

The root should freeze a no-tuning screen before release. A defensible initial
screen requires consistent alphaJ−alphaF served-accuracy improvement across the
three seeds, a mean gain of at least .2pp, no worse mean pooled NLL, and the
existing pilot member-competence tolerances (.1pp mean/.2pp worst degradation).
Those are utility-screen choices, not statistical guarantees. Do not soften
them after opening. A negative alphaJ result remains a completed test of this
specified rule, not permission for a beta sweep.

Only a supported pilot should trigger unchanged full-task follow-up controls:
the same scorer-bank architecture with scalar GNCL on all parameters; with J
on dense internal factors instead of scorers; and capable single and ordinary
independently trained ensembles. An objective-matched untied counterpart is
needed before attributing benefit to sharing itself. Own-versus-joint selector
differences must remain explicit. Fresh unused WikiCS splits and a separately
chosen complete graph/backbone are required for confirmation. Attention-free
SAGE/GIN do not acquire this method without an architecture change; no universal
backbone or link/graph-level claim follows from this pilot.

The bank has CPU/constructor qualification, not this training-field/GPU/source
qualification. Any preparation must audit all gradient recipients, exact
two-view pooling/reductions, native local/global restore, copied scorer keys,
actual gradients and replay/serving scope on the competent full recipe. Extra
reverse collections and memory must be measured and charged. No speed, memory
saving, success, run admission or additional GPU request follows from this note.
