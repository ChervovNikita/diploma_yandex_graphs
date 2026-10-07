# A controlled next hypothesis: different label-compatible graph targets

**Recommendation:** after whole Wiki12 attribution is read, consider the one
source-prepared hypothesis below. It directly targets useful member learning
without the damaging Rademacher start. The ingredients are established; the
shared-route utility question is open. This packet is not admission to train.

## What changes

All four routes start with unit factors and receive ordinary own CE on every
TRAIN label in both original stochastic views. The native shared backbone,
internal factor sites, normalizations, classifier, optimizer, trajectory and
mean-probability serving are unchanged. There is no teacher, new branch model,
auxiliary projection head, graph modification or serving router.

Prepare four fixed context signatures from public features and graph edges:
root features X, the residual X-PX, neighborhood average PX, and P^2X. P is the
row-normalized directed destination aggregation A+I with exactly one self-loop;
it is a signature operator, not a change to the predictor. These standard
root/high-pass/low-pass/multi-hop ingredients require attribution. They do not
claim four new evidence sources or extra predictor capacity.

For each TRAIN node/context, select the 16 most similar OTHER same-class TRAIN
nodes by signature cosine, or all available if fewer. Include the node itself
as a cross-view positive. Ties use fixed TRAIN row order. The mask is fixed
before training, with labels from TRAIN only. Validation labels, predictions,
selected error cohorts and TEST do not enter target preparation. The original
at-most-512 deterministic linspace auxiliary panel restricts these masks each update; normalize after
restriction. Every row retains its own-object positive.

Let Q_m be that context's row-normalized positive mask. Route m receives
ordinary symmetric cross-view contrastive CE with target Q_m and temperature
0.2. The auxiliary coefficient is 0.05, fixed rather than learned. All candidate
parameters receive the scalar objective mean-own-CE plus this term. In
particular, this prototype does not introduce blockwise gradient projection,
nonpotential optimizer permissions or an undocumented competence guarantee.

Unselected same-class nodes remain denominator distractors. This is deliberate
within-class discrimination, not the original all-same-class positive relation.
It can be harmful and is a reason to require competence and full-population
quality checks. The signatures need not correspond to real semantic subclasses.
The method is not claimed to eliminate class collapse.

## The decisive simpler control

Use Q_bar=mean_m Q_m for EVERY route. Its positive mass, context information,
sample identities, contrastive denominator, views and auxiliary compute match
the candidate. It removes persistent route-to-context assignment. No temporal
cycling, extra labels or learned target producer is added.

At exactly equal member score matrices, route targets and common targets have
the same average scalar objective and the same average score cotangent. Their
private route cotangents can differ. This supplies purposeful symmetry breaking
at a unit start, rather than damaging input signs or maximizing arbitrary hidden
distance. In a stochastic native model, independently realized dropout means
members need not have exactly equal scores; no equal-gradient trajectory claim
follows. The complete private Jacobians may also erase the target difference.

An explicit same-context cycling source option is also prepared: all routes use
one Q_(update modulo4) each update. Over four updates on the unchanged original
auxiliary panel it averages to the common target. Evolving parameters remove an
exact cross-update gradient identity. It is therefore a separate
schedule diagnostic, not a cleaner primary comparison than Q_bar. It remains
an inert source option and is excluded from the proposed staged9-then12 study.
Do not add another trained arm or infer a cross-update gradient identity.

The permutation control applies one fixed, independently sampled permutation
to each route's entire mask. It stays within class and original deterministic
auxiliary-panel membership. Inside the panel, each route permutation also stays
within that route's restricted positive-degree bucket. It therefore preserves
EVERY scored anchor's exact positive count after restriction, positive labels
and self positives. Column-degree multiset is preserved; column-degree
assignment to individual nodes can change. This breaks factual context
relationships without changing scored positive-count opportunity. The seed and
relation hashes are frozen without utility selection. Before training require
a nontrivial changed-target total variation; do not retry a seed if the control
is effectively identical because buckets are too small.

## How this could acquire correct rankings

Different positive relations pull each route toward different examples of its
correct class. Gradients reach private factors inside nonlinear attention and
message stages, where they can alter which features/neighbors influence final
class margins. Shared weights continue to learn the average label objective.
This is not merely an output temperature or convex reweighting of unchanged
member predictions, so it can in principle change a common wrong competitor's
ranking. It can also put differences in a classifier-nullspace, learn nuisance
contexts, lose competence, or reproduce identical useful decisions. Prediction
audits must distinguish these cases; the mechanism is an empirical hypothesis.

## Closest priors and limits

- SupCon and BotSCL already provide supervised same-class cross-view alignment.
  BotSCL also provides shared query/key maps, channel scaling and signed
  neighborhood attention. Those features are not our contribution.
- Chen et al., arXiv2204.07596v1, separates useful subclass clustering from
  arbitrary within-class spread. Xue et al.,2305.16536v1, analyzes feature
  suppression/class collapse under specific distributions and encoders. Their
  transfer results do not prove that preserving context improves our original
  coarse-label accuracy or that Wiki24 actually suffered collapse.
- HLCL2303.06344v1 contrasts high/low-pass graph encoders with shared weights;
  graph-filter contexts and shared contrastive views are therefore established.
- PMGCL, Zheng and Cheng, DOI10.1007/s13042-025-02924-2 (published16Feb2026),
  states in its publisher abstract that it estimates true-positive probabilities
  with a Beta Mixture Model and uses multiple positives per anchor. Only the
  subscription preview abstract was accessible. This is a positive-mining
  prior, not a verified complete method-body comparison or results adoption.
- Retained CDLG, DICE, HGEN, DIVE, SuGAr, DIBS, FoRDE, BatchEnsemble and TabM
  remain direct pairing/diversity/sharing/initialization priors. We cannot claim
  novelty from giving each member a contrastive term or a different graph view.
- Exact complete-rule published duplication is unverified. The defensible next
  question is whether persistent label-compatible context targets induce useful
  complementary rankings through restricted private factors, beyond their
  shared-target and randomized-target controls.

Known ingredients are allowed in a useful extension. A positive controlled
result could establish a specific effective method, but cannot retroactively
make its primitives new or settle generality.

## Conditional complete pilot recommendation

Use complete WikiCS on the competent pinned Polynormer recipe, original TRAIN
580/full graph and complete selected-development5274; no TEST. Keep1100epochs,
two stochastic own-loss views, native stage transition and strict-first
checkpoint selection. Fresh paired seeds8101/8203/8307 are a proposed roster,
not yet frozen or launched. Prior development has already informed this idea;
these are exploratory optimizer replicates, not confirmation.

The smallest defensible recommendation is a prospectively specified staged
screen. Freeze both stage recipes and no-tuning decisions before any opening.

Stage1: three conditions x three seeds =9 complete fits:

1. Unit BE plus common Q_bar context alignment.
2. Unit BE plus route-specific Q_m context alignment (the only candidate).
3. Unit BE plus class-preserving-permuted route targets.

The primary mechanism comparison is2-1 and graph-context comparison2-3. Exact
Q_bar target matching addresses ordinary reduction in positive count/context
information without needing a fresh class-full arm to isolate assignment.
Existing class-full Wiki12 is a diagnostic reference only. Stage1 establishes
no superiority or ensemble-specific benefit.

Only if the fixed Stage1 quality/competence screen passes, run Stage2: four
source-frozen references x the same three seeds =12 complete fits:

4. Capable ordinary single, two-view own CE.
5. Ordinary independent4, four independently own-selected native predictors.
6. Single plus Q_bar context alignment at the same0.05 coefficient.
7. Ordinary untied4 plus corresponding member-specific Q_m context alignment.

Single6 receives the same per-object CE and average context target. Untied7 has
four separable objectives own_m+.05 alignment_m; all parameters of each separate
model receive its own loss, and no gradient flows through another model. Use
independent own checkpoint/stage-transition selection, rather than silently
changing it to joint selection. The shared bank retains the source mean-member
scaling and ordinary joint selection, with every shared/private factor/head
trainable. Matched objectives are essential: if the same regularizer explains
the quality gain in one or four untied models, do not attribute it specifically
to shared factors.

Total21 fits only when a worthwhile Stage1 signal survives, with no coefficient,
k, signature, initializer or selector grid. The two stages' controlled code,
data roles, seeds and contingent gate must be root-frozen prospectively; the
current packet is a recommendation, not that freeze. No superiority statement
is made until all12 Stage2 references close and are scored competently.

Before release: root must review the actual native dispatch, dataset/role
fingerprints, relation hashes, objective scaling and exact serving/reload;
qualify real fullgraph preprocessing and Torch/CUDA backward/replay; charge
preparation and all training/evaluation calls; and fit existing host limits
without disrupting other jobs. A disabled explicit full driver/selector/replay successor is now prepared in
`integration_successor_v1`. It has only source/stdlib fixture qualification;
actual native/fullgraph/CUDA preparation and execution remain pending. Existing v2 sequential-VJP execution
can be adapted without changing stochastic views; its extra forward costs must
remain visible. The small CPU functional fixture has passed. None of those native/CUDA
runtime checks has run in this packet.

After the whole9 Stage1 close, use fixed selected checkpoints and report every arm/seed,
accuracy, member mean/worst accuracy, pooled/member NLL and Brier, original
checkpoint epochs, costs, coverage and common wrong-competitor support. Freeze
plain comparison cohorts before candidate prediction opening and count paired
repairs AND introduced errors across the full population. Context-specific
cohorts are diagnostics only, never the primary endpoint. Three seeds on one
graph cannot supply independent-graph uncertainty; use descriptive paired
intervals without node pseudoreplication.

## Falsifiers and promotion threshold

- If route/common masks are nearly identical on the full TRAIN panel, do not
  call the resulting study differentiated steering. Inspect mask overlap and
  zero signature counts before training without reading fitted outcomes.
- If2 does not improve served accuracy over1 in all three seeds with a mean
  difference at least0.2points, persistent assignment is unsupported.
- If3 matches or exceeds2, graph-context semantics is unsupported. A gain
  attributable to arbitrary target partitions should be reported accordingly.
- If2 loses mean member accuracy by over0.1points or worst-member accuracy by
  over0.2points against1, or its mean pooled NLL is worse, do not promote it as
  competence-preserving. These are pilot criteria, not statistical guarantees.
- If hidden loss improves but common wrong-competitor support/served net
  repairs do not improve, the intended error mechanism remains unsupported.
- A useful quality candidate must also have positive paired accuracy differences
  against both ordinary and context-matched single/independent4 and no worse
  mean pooled NLL in Stage2. Otherwise retain
  only the bounded mechanism finding, not superiority.

A passing screen admits one unchanged method with fresh heldout confirmation
on unused official WikiCS splits and a separately chosen complete graph task,
with capable single/independent-ensemble controls and a source-qualified
efficient-ensemble/diversity comparison. Pick those identities and resource
plans prospectively before any heldout result. The current packet does not
reserve tasks or assert acceptance.

If Wiki12 establishes that alignment itself is harmful/unreproduced or the
backbone deficit has another stronger explanation, revise the hypothesis in a
new owned successor and preserve this candidate unchanged. Do not append this
family to an existing frozen study or alter any original-paper scores.
