# Contrastive objectives did not acquire useful prediction diversity

8 October2026. Scientific analysis of the complete Wiki12 plus canonical SupCon
family. This is an author-side empirical analysis, **not** the separate fresh
independent aggregate assessment. The running new private-attention twelve-fit
family was not opened, changed or used to choose a favorable seed.

## Complete result and population

All15 selected states cover the same5274 WikiCS split0 development nodes and
three fixed optimizer seeds. P/A/R/C denote plain, alignment-only,
class-centered cross-member residual contrast, and their combined package.
S is the canonical SupCon control. Every state completed1100 epochs and retains
its original accuracy-selected checkpoint and local/global mode. This same
development population selected the states; it is not independent TEST evidence.

The fresh primary C-minus-P changes are **−0.5119/−0.5499/+0.8532pp**, mean
**−0.0695pp**. Its mean NLL change is−0.04228nats, but member accuracy changes
largely track pooled accuracy. Canonical S-minus-P is−0.4551/−0.4361/+0.3034pp,
mean−0.1959pp; mean NLL worsens0.48559nats. Neither establishes useful improvement.
The earlier positive Wiki24 unit-plus-package result must not replace this
fresh primary contrast.

## Members make almost the same decisions

The following table retains every cell. Disagreement counts nodes where at
least two members' top1 predictions differ. Coverage counts nodes where any
member predicts truth. A pool harm has a correct member but an incorrect mean
probability prediction. D is mean member NLL minus served NLL.

| Seed | Arm | Accuracy % | NLL | Disagreement nodes | Coverage | Pool harms | D, nats |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
|6101|P|81.6458|1.17471|34|4315|9|.000766|
|6101|A|81.2856|1.12451|35|4297|10|.000499|
|6101|R|81.5700|1.12073|29|4311|9|.000494|
|6101|C|81.1339|1.05219|30|4290|11|.000255|
|6101|S|81.1907|1.05773|26|4298|16|.000225|
|6203|P|81.5889|1.09423|27|4308|5|.000735|
|6203|A|80.9443|1.08086|28|4280|11|.000307|
|6203|R|81.2287|1.35152|38|4298|14|.001206|
|6203|C|81.0391|1.12137|26|4280|6|.000684|
|6203|S|81.1528|2.79592|99|4308|28|.052957|
|6307|P|80.8115|1.13915|28|4271|9|.000343|
|6307|A|81.0770|.99012|22|4284|8|.000191|
|6307|R|81.1907|1.19873|34|4290|8|.000583|
|6307|C|81.6648|1.10768|37|4325|18|.000552|
|6307|S|81.1149|1.01119|26|4286|8|.000358|

In14 of15 states only22–38 nodes disagree:0.417–0.721% of the population.
The exception disagrees on99 nodes,1.877%. Thus all15 predict the same class
across members on at least98.12% of nodes. Correctness disagreement is only
15–30 nodes in the original12 and24/76/19 for S. This is **practical decision
redundancy**, not exact equality of functions or parameters.

All15 have **zero pooled-only rescues**. Every all-member-wrong node has a
strict common wrong rival, apart from one node in S/6203. That rival strictly
outranks truth in every member; any unchanged convex probability mixture also
ranks it above truth. An argmax router cannot produce a correct alternative
there. A new learned predictor or changed member must supply one.

The maximum possible fixed convex-mixture accuracy is bounded by coverage plus
the count of all-wrong nodes without a common rival. Across these15 states that
upper bound never exceeds4325 correct nodes. The earlier ordinary independent4
pool served4327 in each seed. This is descriptive historical context with
different training/selection identities, **not** a new paired superiority test.
It explains why changing only a nonnegative final averaging rule is insufficient.

## Confidence diversity is usually small too

For probability-mean serving and true class y, let
rho_m=p_m(y)/sum_k p_k(y), and U be uniform over four members. The existing
exact own/pool identity gives

    D = mean_i KL(U || rho_i).

The14 small gaps lie between.000191 and.001206nats. Pinsker plus averaging gives
mean_i||rho_i-U||_1 <= sqrt(2D), between.01955 and.04911 for those states.
This bounds average **correct-class responsibility imbalance** on the evaluated
population. It does not measure full-distribution equality, hidden geometry,
edge evidence, parameter correlation or the two dropout-active TRAIN views.
No raw logits or embeddings were loaded to make a stronger claim.

S/6203 is the exception: D=.05296nats and top1 disagreement increases72 nodes
versus P. Yet coverage remains4308, the mean member loses.5499pp, and pool harms
increase from5 to28. The pool loses23 correct nodes and NLL rises1.70170nats.
Greater variation in this cell therefore does not provide a quality gain.
NLL alone does not identify overconfidence, underconfidence or calibration error.
Its original selected epoch1073/global mode is retained; no alternate checkpoint
or temperature is selected to repair the result.

## Repairs and new errors largely cancel

The complete21 per-seed comparison rows are retained in
PAIRED_FLOW_DIAGNOSTICS.csv. Important full-population flows are:

| Contrast | Seed | Repairs | Harms | Net | Coverage gained/lost | Common rivals cleared/appeared |
| --- | --- | ---: | ---: | ---: | --- | --- |
|C−P|6101|161|188|−27|156/181|156/181|
|C−P|6203|103|132|−29|104/132|104/132|
|C−P|6307|128|83|+45|133/79|133/79|
|S−P|6101|137|161|−24|137/154|137/154|
|S−P|6203|189|212|−23|196/196|197/196|
|S−P|6307|97|81|+16|96/81|96/81|

C repairs392 and harms403 repeated seed-node instances. S repairs423 and
harms454. Within P's common-rival cohort C serves156/100/124 new correct nodes,
but new errors elsewhere defeat the overall gain. Most mistakes persist:
803/862/870 P common-rival nodes remain common-rival nodes under C. These
overlapping cohorts are not additive, and repeated nodes/seeds are not iid
experimental replicates.

A-minus-P nets−19/−34/+14 nodes; R-minus-P nets−4/−19/+20. R's mean NLL worsens
.08763nats. C-minus-A nets−8/+5/+31 and C-minus-R−23/−10/+25. The positive mean
C−A−R+P interaction does not rescue a failed C−P primary or prove synergy.
The prospectively matched S−A comparison nets−5/+11/+2 nodes, mean+.05056pp,
but mean NLL worsens.55645nats. No component or selected seed supplies a robust
replacement headline.

## What embedding repulsion did and did not establish

The source distinguishes two objectives. A aligns each member across two
stochastic views using same-class positives. R contrasts the same member/object
across views against other members after TRAIN class centering. C includes both.
Canonical SupCon is a same-label contrastive reference, **not** an inter-member
repulsion objective. These distinctions must remain explicit.

R/C were trained with an embedding-diversity objective, but their selected
outputs remain nearly redundant. The compact evidence contains no residual
angle/norm trajectory or final hidden-distance matrix. It therefore does not
show that embedding repulsion succeeded, nor that different embeddings collapse
into identical logits. It shows that these objectives did not deliver useful
decision diversity under this recipe.

The captured native pre-head vector has512 coordinates and a linear ten-class
readout. At a frozen head, many hidden directions can change without changing
logits; class centering does not ensure prediction relevance. Private heads and
upstream nonlinear computation complicate exact cross-member equivalence.
This is a possible explanation, already covered by saved prediction-tangent
and readout reasoning, not a measured nullspace diagnosis. The completed direct12
head diagnostic did not support its tested remedy; do not repeat that refit.

## Private attention can escape copies, but has no symmetry-breaking guarantee

The running rule privately owns local attention scorers and tied-QK factors.
Their J partial is ordinary own CE multiplied per TRAIN node/member by
(.5/4)+.5rho_m; all complement coordinates receive F. At equal correct-class
probabilities this equals the F partial, even with different wrong-class scores.
At exact copied/unit parameters and identical deterministic paths, row symmetry
is preserved. A positive CE floor, ownership change or pooled feedback does
not force different graph evidence.

Native different dropout paths produce different states and can break realized
gradient symmetry. Separate private scorers can then alter source weights
without first altering projected values at that interface; useful probability
advantages can receive larger relation credit. That is a plausible escape
mechanism. Own-risk alphaF already receives stochastic asymmetry too, so
relationJ must beat that control. Shared features, GATv1 static ranking,
tied global QK, confidence-only changes and graph-wide gradient interactions
can still produce redundant or weak routes. The tiny **serving** D values above
must not be substituted for unobserved TRAIN-view responsibilities in new12.

## One conditional intervention, explicitly established practice

No new graph-band, Fisher, reciprocal-factor or degree-semantic initializer is
justified by these results. Those ideas already have saved protocols/prior and
failure limits; the closed graph profile does not localize the deficit to one
degree class. Wiki24 already showed that first-adapter Rademacher randomization
increased coverage while weakening members and serving. A generic factor-noise
sweep would repeat that tradeoff rather than isolate its cause.

Retain only a **standard independently drawn native local-scorer control**, and
only if whole new12 closes without an already supported rule awaiting its
required references/confirmation and still shows insufficient useful alternatives.
The inspected Wiki24 and copied-attention protocols establish no executed arm
with independently drawn local scorer rows; this is a bounded history statement.
Independent scorers are ordinary multihead GAT practice, not methodological novelty.

Keep all shared maps and dense/QK factors at their existing native/unit starts.
For each of the14 local scorer banks, initialize four rows with independent
draws from the pinned native GAT reset law at the original per-row shape. Use
an isolated, prospectively fixed initializer RNG so shared initialization and
member dropout streams are not shifted. Do not tune variance, redraw on bad
initial scores, warm a teacher, or randomize another parameter site.

Cross copied versus independent scorer starts with alphaF versus relationJ.
The copied six full endpoints may be reused **only after** whole new12 closure
and source/recipe/selector/seed custody matching; then two new conditions × three
fixed seeds require six complete1100-epoch fits. No old score is recalculated.
The changed initial functions are a declared intervention, not parity failure.

Independent-F minus copied-F measures ordinary scorer-start utility. The
difference between J−F effects under the two starts measures whether relation
credit uses that asymmetry. A gain from independent-F alone supports known
attention initialization; it does not establish the new feedback contribution.
An initial/final spread increase with poorer members, NLL or serving rejects
usefulness. Source/runtime qualification and a complete-population utility screen
must be frozen separately before admission. Strong singles/ordinary ensembles,
objective-matched tying controls and unused confirmation remain necessary.
No source, grid, gate change or fit is admitted by this note.

## Attribution and scope

Saved BE/TabM, GAT/GATv2, Kim's GAT graph-BE formulation, GNCL, CDLG-inspired
residual contrast and canonical SupCon are ancestry. The pinned PyG GAT source
uses Glorot reset for att_src/att_dst; applying it independently at each bank
row is an adaptation of established initialization. No new primary paper is read.

Only the admitted closed compact15 and saved complete Wiki24/direct12 reports,
existing objective/native sources and initialization conclusions were read.
DERIVED_FACTS.json and the two CSV files perform arithmetic on those supplied
counts/metrics, not fresh model scoring. Original scores remain unchanged.
No dataset, label array, checkpoint, raw prediction, pending new12 outcome,
remote job, scientific fit, manuscript or canonical-memory pointer was accessed
or modified. Input bindings and immutable-source provenance accompany this note.
