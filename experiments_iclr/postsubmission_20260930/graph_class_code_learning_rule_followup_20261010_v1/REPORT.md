# Class-partition supervision: closest priors and split correction

## Decision

**Low-priority attributed baseline; no fits or engineering admitted.** The rule is
a composition of shared ECOC-style supervision and hierarchical cross-entropy.
The current inspection establishes no new learning principle, predictive gain,
or acceptance claim. Preserve the first packet and its disabled proposal; this
follow-up corrects its split description and strengthens prior attribution.

The conditional mechanistic idea remains intelligible: keep a full classifier in
every member, and add different fixed losses on sums of its class probabilities.
For the same correct-class probability, these losses treat different wrong-class
destinations differently. Shared parameters receive their average update, while
private paths receive different group-risk gradients. The rule does not require
four pretrained teachers. These facts do not ensure different competent decisions:
all members have the same well-specified optimum, a group loss can increase a
wrong class in the correct group, and identical predictions/Jacobians reduce the
shared update to the all-pack single's update. Properness, rank and positive
probability-space curvature are standard algebra, not novelty or anti-collapse
results. Those qualifications remain in the prior packet.

## Strongest collisions

| Prior | Verified bounded operation | Relation and limit |
|---|---|---|
| Deep N-ary ECOC, arXiv2009.10465 | Original TeX §§3–3.2 and coding paragraph: balanced class-group tasks; no, partial and full shared feature encoders; each private classifier follows its meta-class objective | Direct shared-backbone/class-code ensemble ancestry. The proposed rule retains full-class CE and derives group probabilities from each existing full-class posterior instead of serving a meta-class code decoder. That is an attributed distinction, not established novelty. |
| ECNN, arXiv1912.00181 / AAAI2021 | Prior packet's main method and sharing paragraph | Joint coded learners with shared components, code design and an additional entropy term already combine coding and parameter sharing. No new read credited here. |
| HXE, arXiv1912.09393 / CVPR2020 | Prior packet's §3.1 and AppendixA | For every partition, `-log p_y - λ log P(group_y)` equals depth-two HXE with fine conditional weight1 and coarse weight1+λ. The proposed multi-question risk averages these established losses. No new read credited here. |
| B-CNN, arXiv1709.09890 | §3.1–3.4 | Shared backbone with coarse/fine branches and a weighted CE sum; coarse-to-fine loss scheduling. It uses separate branch heads, not the proposed summed fine probabilities. Multiple levels alone are not a new mechanism. |
| Learn Class Hierarchy using CNNs, arXiv2005.08622 | §2 | Full classifier extended by a linear head per hierarchy level, CE sum and center loss. It does not establish the proposed different-pack ensemble, but adds explicit hierarchical supervision ancestry. |
| Anytime Inference with Distilled Hierarchical Neural Ensembles, arXiv2003.01474 / AAAI2021 | §§3–3.3 | A tree of shared computation with individual CE, sub-ensemble losses and stopped-target hierarchical distillation. Its hierarchy is a computation tree, **not multiple class hierarchies**. Naming overlap must not be treated as an exact label-partition collision. |
| Spectral ECOC, ICCV2009 DOI10.1109/ICCV.2009.5459355 | Metadata/abstract only | Builds a class-similarity graph and thresholded Laplacian eigenvector codes. The publisher failed in the prior packet; HKUST retrieval loops and the Microsoft page returned404 here. Class-graph coding ancestry is disclosed, but the full method is still unresolved. |

New bounded discovery queries also returned hierarchical multi-label/protein
ensembles, decision-tree methods and unrelated neural code applications. These
are metadata leads, not exact shared neural multi-hierarchy method reads. No
complete-rule literature absence certificate is warranted.

## Graph cut and wrong-rival interpretation

The frozen candidate generator **separates**, rather than groups, class pairs with
large TRAIN class-to-class edge counts. It first improves minimum row-code
separation, then maximizes

`sum_(a<b) W_ab * 1[b_a != b_b] / (1+d_ab)`.

A rival placed opposite the true class receives a positive auxiliary logit
gradient, suppressing that rival under gradient descent. A rival on the same side
can receive a negative auxiliary gradient. The fine CE and other questions must
resolve that rivalry. Thus graph mixing is only a proxy for which pairs deserve
different group treatment. It is not measured classifier confusion, a causal
explanation of common wrong rivals, or proof of useful member diversity.

The graph-free distinct packs, first-pack common risk, all-pack single and
matched untied paths remain frozen required controls. If graph-free packs match,
graph conditioning adds no supported contribution. If the all-pack single
matches, generic composite supervision explains the result. If common packs
match, member-owned risks are unnecessary. If the untied bank wins, sharing has
not supplied the intended accuracy contribution. Whole-population accuracy/NLL,
mean/worst member quality, repairs and introduced errors are required; code rank
and gradient differences are not substitutes.

## Correction: native Coauthor CS split is not stratified60/20/20

Pinned PolyFormer source is commit
`d390f39e88d0eaac80318fdc7704bd3bf3cf8b13`. Its CS launch uses the retained
monomial model recipe. In `training.py:65–70`, TRAIN quota is
`round(.6*N/C)` and VALID size is `round(.2*N)`. With the public acquired
dimensions N18333/C15, these are733 per class and3667 globally. In
`utils.py:49–75`, a class with fewer than733 nodes contributes **every node** to
TRAIN. At exactly733, selecting733 without replacement also exhausts it. VALID
is sampled globally from the remainder and TEST is everything remaining.

Consequently any class of size≤733 has no evaluation examples. Source inspection
does not supply aggregate class counts; no raw labels or held-out targets were
opened. The public acquisition manifest supplies dimensions and aggregate sizes,
but not a per-class histogram, and uses a different project stratified protocol.
It therefore cannot certify native15-class evaluation. The first packet's phrase
“author class-balanced60/20/20” is corrected to **equal TRAIN quota with global
VALID and TEST remainder; complete class coverage unqualified**. Do not claim
representative15-class evaluation or silently change this native split.

The source's selector uses strict VALID-accuracy improvement, earliest ties and
patience250, maximum2000 epochs. Its original routine evaluates TEST each epoch;
a project implementation must preserve clean TEST exclusion and freeze the
validation-only selector before any fits. This is source description, not a
qualified implementation or rerun of any current score.

## Documented alternative, still unadmitted

Shchur et al., *Pitfalls of Graph Neural Network Evaluation*, arXiv1811.05868,
§3 explicitly uses20 TRAIN and30 VALID examples per class, TEST=all remaining
nodes. The pinned author code at
`1e72912a0810cdf27ae54fd589a3b43358a2b161` implements this split in
`config/train.conf.yaml:53–62` and `gnnbench/data/make_dataset.py:57–135`.
For C15 it gives300 TRAIN/450 VALID and17583 TEST, **if** every class has at
least51 nodes and the complete18333-node object is retained.

The paper also uses the largest connected component. Applying its split recipe
to native complete PyG CS would be an explicitly declared protocol adaptation,
not an exact reproduction of the whole Shchur preprocessing or of PolyFormer's
published split. No component or class may be silently dropped. A custodian must
report aggregate class counts, per-split class counts, complete input dimensions,
mask fingerprints and split overlap. Root must freeze fresh split/optimizer
seeds, source and capable single/independent qualification on that protocol.
At least several matched complete splits are needed for a representative
development screen; no final multi-task generalization follows from that screen.

There is an additional construction risk: only edges with **both endpoints in
TRAIN and different classes** enter W. Under20-per-class sampling, their expected
count for a pair is proportional to `400 * E_ab/(n_a*n_b)`. This is an elementary
sampling expression, not a measured dataset statistic. The usable class-boundary
graph may be sparse or zero. Before any fit, inspect TRAIN-only W support, classes
with no incident boundary count, and whether graph-conditioned packs actually
differ from the graph-free packs. All-zero W or identical masks supply no graph
conditioning contrast. Sparse support must be reported and interpreted through
the graph-free control; it cannot be repaired by post-outcome code changes,
held-out labels or silently importing multi-hop affinity.

This standard alternative is documentation for qualification, **not an adopted
split, authorized fit or replacement of sealed current tests**. Existing task
exposure remains development history; fresh optimizer seeds do not make the
dataset untouched.

## Binary and small-class limits

For C2, every nontrivial partition has a singleton true group. Group CE is exactly
fine CE, so `L=(1+.5)*CE`. The rule supplies no new binary supervision and no
automatic binary link-prediction or recommendation extension. Optimizer effects
of rescaling do not create a new label decomposition. Four disjoint14-question
packs are specific to the nominated C15 construction; small-class tasks may not
even contain enough distinct balanced partitions for four full-rank packs.

## Provenance and go/no-go

**No-go now for novelty-led fits.** There is no demonstrated distinct failure
mechanism beyond an established composite-risk geometry, and the split/code
construction are unqualified. A later attributed baseline could be admitted only
after source/data custody, all-class evaluation and TRAIN-only construction are
qualified and the frozen quality comparisons remain meaningful. No extra GPU
request follows from this scout.

Five new bounded primary paper scopes were inspected; zero full-paper reads,
proof audits, author model executions or benchmark replications. One grouped
bounded author protocol-code scope was inspected without running it. New source
retrieval/extraction and this packet are the only writes. Prior sealed packet,
canonical memory, pending Q/K or initializer outcomes, scientific data/models,
servers, manuscript, Desktop, original scores and GENLINK remain untouched.
