# Two new primary reads: competence and purposeful specialization

15 October 2026. **Two new scoped primary reads relative to the checked retained
ledger and notes; no added recipe or experiment.** The practical lesson is to
train identifiable skills and test their value in the actual pool. Disagreement,
positive own-loss weight, and a symmetry argument do not establish competence.

## Sources and read accounting

| Source | Primary scope actually read | Status |
| --- | --- | --- |
| Wang et al., [RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts](https://arxiv.org/html/2010.01809v4), arXiv:2010.01809v4 | §3, Eqs7–17; §4 prose/captions including component-ablation definition; A.1 implementation paragraphs | New scoped primary method read. No author-code audit or numerical reproduction. |
| Zhang et al., [Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition](https://arxiv.org/html/2107.09249v4), arXiv:2107.09249v4 | §§3–4, Eqs1–4 and theorem statement; §5.1 and §§5.4–5.5 prose; B including Algorithms1–2; C.3; E; G–H | New scoped primary method read. Proof in A and numerical table bodies were not audited. |

Exact identities have no matches in v72's 260 paper records or 655 packet
records. A targeted retained-text check found only bibliography mentions in
Balanced Contrastive Learning and GTrans source text. These are prior exposure,
not retained method reads. The scan scope and exceptions are in `DEDUP.json`;
this is not a global novelty or exhaustive absence certificate.

The KDEX lead, DOI10.1145/3705323, remains metadata only: ACM returned403,
Crossref supplied no alternative full text, and search endpoints returned
challenges. It is **not** a third read. Version4 HTML is the evidence used below;
version3 retrievals are retained but not counted as another source or read.

## RIDE: own risk plus class-sensitive repulsion

RIDE jointly trains a shared early CNN with independent, channel-reduced late
branches. Its complete-bank prediction averages logits. It explicitly compares
classification loss on the pooled logits (Eq8) with a sum of each expert's own
classification loss (Eq9). The component ablation distinguishes own loss,
diversity, sequential routing and optional distillation. Default classification
uses LDAM; neither the architecture nor the complete published gain is evidence
for native GNNM.

Its additional training mechanism is negative ordered-pair KL (Eqs10–15):

```text
p_m = softmax(z_m / T),
L = sum_m own_loss(z_m,y)
    - lambda/(M-1) * sum_m sum_(n != m) KL(p_m || p_n).
```

T is a class-coordinate temperature derived from TRAIN frequencies; rarer
classes receive lower temperature, introduced after initial training. This is
frequency-sensitive output repulsion, not label-conditioned mutual information
estimation. The paper's statement that symmetric losses make the experts
“equally good and distinctive” is not a general guarantee: a symmetric objective
can have unequal realized members or collapsed solutions.

**Transferable principle:** give every expert direct supervised credit before
asking the bank to be complementary, and separate that ingredient from the
repulsion and serving rule. Current GNNM's F already does this. The J recipes
retain half F; RIDE does not establish which F/J balance works for our models.

The class-temperature KL term is absent from the checked25-recipe catalogue,
but it supplies no member-quality constraint. A faithful native-body adaptation
would retain the four existing routes and add only the stated output auxiliary,
with own-only and common-temperature controls. This would still be a new
adaptation. In binary BCE there is an extra definition problem: native scalar
z can be represented as two logits (0,z) or (-z/2,z/2); unequal class
temperatures make these normally equivalent representations different.
Moreover, these sources share only early layers, whereas our native GNN
operators remain shared throughout. No coefficient, temperature schedule or
native integration is selected here, and no repulsion experiment is warranted
before the current combination study is analyzed.

## SADE: diversity from distinct risk targets

SADE gives three experts different supervised tasks on the same training data.
With class-frequency vector pi and its reversed-order vector bar_pi, its losses
are CE(z,y), CE(z+log(pi),y), and
CE(z+log(pi)-lambda*log(bar_pi),y). Algorithm1 sums these losses through a shared
early backbone and private late branches. AppendixE adds a fourth expert by a
further inverse-softmax coefficient. This is purposeful specialization for class
priors, rather than arbitrary prediction separation.

For ordinary long-tailed evaluation it averages raw expert logits. Its separate
unknown-prior procedure freezes the bank, then fits global weights using two
augmented views of **unlabeled TEST**. Algorithm2 stops if any weight reaches
.05; this is a stopping rule, not our C6's differentiable weight floor. The prose
calls the stability score cosine similarity, while Eq4 prints a dot product of
softmax probability vectors. We do not resolve this through an unperformed code
audit or transfer its theorem to graphs. A confidently constant prediction can
also have high stability; stability is not itself supervised correctness.

**Transferable principle:** specify what different routes should know, compare
against all routes learning the same target, then evaluate competence in the
intended conditions and the pooled repairs/harms. That directly supports the
existing `context_affinity_common` versus `context_affinity_split` ingredient
control. Structural responsibility supplies another declared source of differing
training emphasis. Those are established learning ideas; the unresolved claim
is that our fixed graph targets, private input capacity and shared native body
produce useful alternatives under the actual mean or TRAIN-fitted gate.

Route-specific prior correction is absent from the25 recipes, but a binary
sanity check prevents overinterpreting it. Let r=log(pi_1/pi_0). A route trained
with BCE(y,z+a) has unrestricted population optimum

```text
z*(x) = logit P_TRAIN(y=1|x) - a.
```

For forward, uniform, inverse-lambda1 and inverse-lambda2 experts the offsets
are a={0,r,2r,3r}. Their ideal logits therefore differ only by constants. Any
fixed weighted logit pool has the same ranking as the original posterior; AUC
is unchanged. Threshold accuracy or calibration may change. Finite capacity,
regularization and shared gradients can change rankings in practice, so this
is a limitation of the evidence, not an impossibility theorem for learned
models. In multiclass tasks class offsets can change decisions; their usefulness
still requires a declared target-prior question.

A cautious mapping would apply these offsets inside own losses of the four
existing native routes, without changing operators, inputs or readout, using
TRAIN-only frequencies. The necessary comparator would include all routes
learning a common offset and a fixed prior-shift readout, so mere intercept
changes are not credited as new graph skills. This is an interpretation map,
not a frozen recipe. SADE's TEST adaptation, image augmentations, private CNN
stages, cosine classifier and Places pretraining are not part of that map.

## Decision for the current campaign

Keep the25-recipe roster intact. Complete the already authorized normalization
and combination analysis before proposing another scientific fit. The new
sources strengthen two existing questions: does private capacity support
competent alternatives, and does a meaningful split target outperform its
common-target control? They do not establish graph gains or promise that a gate
can recover information missing from its members.

Use the current study's complete predictions to interpret member quality,
conditional skill and actual pool benefit together. A useful route need not
beat M1 everywhere, but its disagreement needs an identifiable benefit to the
served predictor. No partial running-study scores were read for this note.

This packet retrieved public literature and wrote local research artifacts
only. It changes no source recipe, roster, controller or prior packet; launches
no model/pilot; reads no checkpoint/tensor/TEST data; uses no scientific remote
host or Git. No independent teacher or distillation proposal is adopted.
