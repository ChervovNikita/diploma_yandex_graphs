# Full-class supervision through different graph-conditioned class partitions

## One distinct, attributed hypothesis

Keep the existing full-class graph classifier in every member. In addition to
its ordinary class CE, ask each member a different fixed collection of binary
questions about the same true labels. Each question partitions the classes into
two groups. Its predicted group probability is the sum of that member's existing
class probabilities, so there is no new representation, auxiliary decoder or
coarse-only member. The full-class probability bank serves exactly as before.

Choose the questions once from TRAIN class-to-class edge counts, with explicit
coverage of every class pair. Jointly learn the ordinary shared backbone and
private BE paths. This tests whether different **proper supervised risks** can
teach useful decision alternatives while common weights transfer fine-class
information. It is different from the project's masking/reconstruction,
embedding contrast, initialization, attention-credit and final-routing rules.

This is a proposed complete rule, not an established novel ML principle or a
positive result. ECOC, shared coded classifiers, class-similarity graphs and
hierarchical probability losses are direct ancestry. The particular full-class,
member-owned graph-cut construction was not verified as a complete collision
in the bounded inspected scopes. That is not a global literature clearance.
Positive native/single/untied controls and an exact closest-prior check remain
necessary before a methodological claim.

## Why this targets the observed failure

The permitted complete WikiCS15 development reports show98.123–99.583%
member top-one agreement, zero pooled-only rescues and strict common wrong rivals
for99.896–100% of all-member-wrong nodes. The completed attention-credit report
preserves greater independent-member coverage with relatively close average
individual competence. Those observations motivate learning new full-class
alternatives; they do not establish a cause or TRAIN gradient/graph mechanism.

All members' ordinary CE depends on their correct-class probability. Partition
losses additionally distinguish **where wrong-class probability goes**. For the
same true class y, predictions(.4,.5,.1) and(.4,.1,.5) have identical full-class
CE. If a question groups y with the second class, its losses are−log(.9) and
−log(.5). Thus this auxiliary is not merely the same true-class CE or an own/pool
responsibility multiplier under another name. Different question packs can
produce different private-route gradients without asking competent members to
disagree on truth.

The questions give no new class information. They reorganize the available
TRAIN supervision. A graph boundary between two known classes is an association,
not proof of the fitted model's error cause or a causal neighborhood effect.
The independent graph-free code and all-pack single controls decide whether that
association and the member-owned supervision are useful.

## Exact fixed construction

Nominate the **complete Coauthor CS node-classification graph**, with its15
classes and documented native PolyFormer monomial configuration. The fixed
dimension gives14 class questions per member, M4. No class/difficulty subset,
induced graph or shortened fit is proposed.

1. Root supplies only the complete TRAIN identities/labels and the public graph.
   Class IDs keep their source order. Deduplicate undirected nonself public edges
   for this target builder only; predictor preprocessing remains native.
2. Count edges whose two endpoints are TRAIN. For a≠b, W_ab is the number of
   such unordered edges linking class a with class b. Unlabeled/VALID/TEST targets,
   fitted predictions and current outcomes never enter W.
3. Enumerate the6435 binary class partitions containing exactly7 positive and8
   negative classes. Represent each by its positive-class-ID tuple; a complement
   is not a separate candidate. There is no continuous code optimizer or search
   against quality scores.
4. Build four disjoint packs of14 questions in member order. A candidate must
   increase the exact rank of its pack's class-difference columns
   `[b_a−b_0]_(a=1..14)`. This gives a full class-separating basis by the end.
   Rank and row-code separation are structural checks, not scientific success.
5. At each selection, let d_ab be the current Hamming distance between the two
   classes' rows in that pack. Among rank-admissible unused partitions, first
   maximize the number of separated pairs having the current minimum d_ab.
   Break ties by maximizing `sum_(a<b) W_ab*1[b_a!=b_b]/(1+d_ab)`, then by the
   positive-ID tuple. This prioritizes complete class-pair coverage and uses graph
   class boundaries only for a declared secondary preference.
6. Freeze W, ordered TRAIN fingerprints and all56 question masks before fitting.
   A rank/construction failure leaves the proposal unqualified; do not change the
   class order or replace the graph after outcomes. This generator is proposed
   mathematically and was not implemented or executed in this scout.

The graph-free control uses the same algorithm with W_ab=1 for all a≠b. The
common-pack control gives the first predeclared graph-conditioned pack to every
member; it is never a selected best pack. These controls preserve the number of
questions and full class-pair identification.

## Training and serving

Let p_m(i) be member m's ordinary15-class softmax on the complete factual graph.
For question j, let S_mj(y_i) be the side of its partition containing the true
TRAIN class. Use every TRAIN label in every member's native CE and questions:

```
L_m = mean_TRAIN[-log p_m(y)]
      + 0.5 * mean_(TRAIN,j)[-log sum_(c in S_mj(y)) p_m(c)]
```

Use the exact stable log-sum-exp difference for group probabilities, without
epsilon clamps or a substitute head. The coefficient.5 is one prospectively
fixed exploratory choice; no optimum or published Coauthor recipe for it is
claimed. Shared training averages L_m; an untied matched control sums separable
L_m so each body's own gradients remain unscaled. All live shared/private weights
receive the ordinary scalar gradient. No partial gradient routing, teacher,
confidence pseudo-label or inference adaptation is added.

Keep native dropout, initialization, normalization, optimizer groups, complete
input graph and horizon/selector. The first version uses ordinary unit BE factors
and the existing native classifier. Do not import pending independent-scorer
starts or the masked-context pipeline. Serve mean factual class probabilities,
with no code decoding, code head, group router or changed output calibration.

### What can be shown algebraically

Every target group contains the true class. Consequently
`CE_full <= L_m <= 1.5*CE_full` pointwise, because each group loss lies between0
and the full class loss. Small training objective therefore implies small
training class CE. This gives no finite optimization, validation-accuracy or
member-generalization guarantee.

For a fixed conditional label law q and a fixed frozen question pack, its
population excess score is

```
E_q L_m(p) - E_q L_m(q)
  = KL(q || p) + (0.5/14) * sum_j KL(group_j(q) || group_j(p)).
```

Thus the well-specified posterior remains the unique full-class optimum.
This is elementary proper-scoring algebra, not a graph-independent-data or
novel theorem claim. In particular, it **allows every member to reach the same
Bayes predictor**. It does not force asymptotic diversity. Finite-data optimization
and private path learning are the hypothesized source of any useful difference.
It cannot recover information absent from every predictor's inputs.

Full class-difference rank also means the14 group masses plus total probability
identify the complete15-class distribution. There is no class-probability
direction invisible to every question in that pack. This is an elementary
functional-identifiability statement, not a new theorem or proof of useful
diversity. Its exact scope and the resulting proper-risk curvature are in
`PROBABILITY_SPACE_ANALYSIS.md`.

For one binary partition, full CE +.5 group CE is exactly a depth-two
hierarchical CE: weight1 on the fine conditional and1.5 on the group probability.
That exact prior identity must be disclosed. The proposed increment is a
member-owned collection of overlapping graph-chosen partitions in a complete
shared ensemble, not invention of hierarchical CE or ECOC.

## Representative screen and strongest controls

The retained author Coauthor CS monomial launch uses K2, width128, one block,
8 heads, FFN128, q1, expansion1, dropout0 and native dprate.8; base Adam
lr.001/weight decay1e-7, attention lr.005/weight decay.0001. The author source
uses class-balanced60/20/20 splits, max2000 epochs and patience250. Actual source,
split seed, clean TEST exclusion, selected checkpoints, own-member selectors,
runtime/native competence and resource evidence must be frozen by root first.
Do not silently claim official20-label-per-class Planetoid or untouched-task
confirmation. Task exposure history is pending. No native model/data was run.

| Condition | Role |
|---|---|
| shared4 ordinary full-class CE | exact method anchor |
| shared4 distinct graph-conditioned packs | candidate |
| genuine ordinary independent4 | capable ensemble quality reference |
| shared4 one common graph-conditioned pack | effect of member-specific risks |
| shared4 distinct graph-free packs | graph class-boundary contribution |
| capable native single with all four graph packs | generic composite-supervision reference, matched questions |
| untied copies of the matched complete factorized paths, assigned same packs | accuracy effect of shared learning |
| capable ordinary native single | baseline competence; predeclared independent-member0 reuse only if exact identity is qualified |

Stage1 uses the first three conditions × three fixed optimizer seeds
10101/10203/10307 =9 complete bank records, native full task/horizon. Root
must predeclare served accuracy/NLL and mean/worst member safeguards, preserve
all9, and stop on a failed whole-quality screen. Only passing stage1 permits
the five remaining conditions ×3 =15 records. The complete positive family has
24 comparison records. Possible native-member0 reuse affects fresh fits/costs,
never selects a favorable member. No stage or fit is authorized here.

Before a broad methodological comparison, the strongest directly relevant
published baselines are a source-qualified ECNN/ECOC graph port and an HXE
single/untied ensemble with the same tuning opportunity. Their native task
configuration is not invented in this packet. Existing competent native single,
ordinary independent4 and the same-objective single/untied controls are mandatory
for interpreting the first representative screen.

## Decisive falsification

Reject shared-ensemble utility if the complete candidate does not improve factual
served accuracy and NLL against both competent single and ordinary independent4
while maintaining predeclared mean/worst member quality. Larger question-gradient
differences, code distances, coverage or one seed do not rescue a failed pool.

If the common pack matches, different member risks are unnecessary. If graph-free
packs match, graph conditioning is unsupported. If the all-pack single matches,
this is generic hierarchical supervision. If matched untied paths win, shared
learning did not supply the intended accuracy contribution. Count every repaired
and introduced error, any-correct coverage, common wrong rival and pool harm on
the whole population. Node/member counts are not independent graph replicates.

No masks, codebook order, coefficients, seeds, class subsets or selectors are
changed after these decisions. A complete positive development family would
justify a separately frozen untouched task/split study, not an acceptance claim.

## Scope and resource limits

No new feature views, attack forwards, reconstruction decoder or teacher bank is
needed. Candidate adds group log-sum-exp work to the same four full graph member
paths, plus one deterministic class-edge/code construction. Real graph/model
forwards, gradients, selection, checkpoint and source preparation costs still
count. Coauthor's large feature bank can be substantial; actual memory/time is
unmeasured and root qualification remains necessary. No GPU request is justified
by an unmeasured prototype or presumed positive result.

Two new bounded primary identities were inspected: ECNN's complete main method
and the relevant parameter-sharing paragraph; HXE's §§3.1 and AppendixA.
Deep N-ary retrieval redirected to its abstract; spectral ECOC publisher access
failed. Their metadata/abstract overlap is recorded as unresolved, never as
full method reading or absence evidence. Old scopes and failed alternatives
were reused before these reads. No canonical memory, source worker, fit, data,
checkpoint, held-out score, pending initializer/QK, manuscript, server or Desktop
artifact was touched.
