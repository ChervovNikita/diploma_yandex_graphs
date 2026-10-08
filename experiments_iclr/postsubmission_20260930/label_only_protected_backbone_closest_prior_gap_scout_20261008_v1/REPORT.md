# Four close prior checks for protected label-evidence correctors

8 October 2026. Four new bounded method scopes after reading the saved
`CURRENT_SUPPLEMENT` v26 chain and base index v72. No full-paper, author-code or
published-body-equivalence credit is claimed. Existing scientific outcomes remain
closed. No datasets, experiment checkpoints or allocated servers were accessed.
Training sources and canonical memory/status files remain unchanged.

## Closest checks

| Source/version | Closest overlap | Remaining distinction in the inspected method |
|---|---|---|
| **Echoless-LP**, [2511.11081v1](https://arxiv.org/html/2511.11081v1) | Partition-wide removal of query labels before propagation; feature and label evidence processed together; retained-label normalization. | One joint trainable encoder over precomputed feature/label tensors. A protected feature-only own-CE predictor, detached additive residual bank and BE ensemble are not prescribed in the inspected Method. |
| **GOODIE**, [2508.01209v1](https://arxiv.org/html/2508.01209v1) | Learned label decoder plus feature branch and node-dependent attention; label-aware single-predictor ancestry. | Joint feature/label/classifier learning, propagated pseudo-labels and contrastive loss. Target-query exclusion is not specified in the inspected §4; code/Appendix E may resolve further roles. |
| **LoMETab**, [2605.14365v1](https://arxiv.org/html/2605.14365v1) | Shared/private multiplicative maps, mean member task loss and mean-probability classification serving. | Rank-r identity-residual Hadamard maps in a jointly trained tabular MLP, without a separate protected feature predictor or graph-label operator. |
| **cov-LLE**, [2607.23856v2](https://arxiv.org/html/2607.23856v2) | Several CE-trained heads on one frozen feature map; direct activation covariance objective. | OOD/calibration setting; hidden-embedding covariance does not guarantee complementary class decisions or graph-label utility. |

These are attributable ingredients and close comparison questions. They do not
establish an exact complete-rule collision, absence certificate, novelty,
superiority or a source-qualified new scientific recipe.

## Echoless-LP: exact label and gradient boundary

Read: Preliminary and complete Method, HTML blocks29–72, Equations1–7. The
versioned method source is arXiv **v1**. The exact title and six authors match
AAAI's publisher-deposited Crossref record
[10.1609/aaai.v40i17.38507](https://doi.org/10.1609/aaai.v40i17.38507), dated
14 March 2026; arXiv metadata also says accepted by AAAI 2026. The publisher landing
connection failed. The accepted body was not compared with this preprint.

**Permitted label operator.** For partition `P_i`, the paper defines:

`H_i = F_MP(diag(1 − M_i) Y, G)`.

Every label in the current partition is zeroed **before** the entire propagation.
The initial value argument of this label branch is the masked label vector,
rather than raw feature values. `F_MP` is reused from the feature-precomputation
backbone and can span multiple hops. The method does not instantiate the current
feature-only Q/K scores, learned label-value embedding maps or all-nonself-neighbor
softmax denominator.

**Feature encoder protection.** Equations5–7 first collect feature and label
tensors and then predict with:

`Y_hat = Encoder({H_feat^(k)}, {H_label^(k)})`.

The propagation tensors are precomputed. The final encoder is explicitly
learnable and consumes both kinds of tensors. Precomputation fixes the cached
feature tensors; it does not establish protection of the learnable feature
encoder's update from the label-aware classification objective. The inspected
method specifies one combined classification map. It does not prescribe a
separate feature-only own-CE predictor or a stop-gradient boundary preserving its
native trajectory. Author implementation semantics remain unaudited.

**Additive correction.** Equation7 specifies a joint encoder, rather than a fixed
`feature_logits + label_only_delta` operation. A generic encoder could support
many implementations; this scope does not certify or exclude every possible
implementation. The current detached additive bank is a separate recipe question.

**Masks.** APS randomly partitions TRAIN targets into approximately `n/M` groups,
then assigns unlabeled targets to one separate partition. TRAIN groups receive
labels from other groups. Unlabeled targets retain all TRAIN anchors. This is
partition-based label precomputation, rather than a newly drawn common half-TRAIN
query mask before every native update.

**Normalization.** APS appends a TRAIN indicator column to the label matrix,
propagates it through the same operator, and extracts a retained-label vector
`r_i`. Equation4 scales each output row by `max(r)/r_i`. Its scale depends on the
node, propagation operator and partition. The current core instead uses the
fixed conditional TRAIN multiplier `(n−1)/(n−k)` once, scale1 at serving, and an
attention denominator over all nonself neighbors, including zero-value neighbors.
The paper's retained-label normalization is not the same normalization or a
first-moment theorem for the current operator. Handling `r_i=0` is not specified
in the inspected Method, so a port needs an explicit zero-context convention.

## cov-LLE: covariance geometry is not decision complementarity

Read: complete §3, §4.2 objective and neighboring setup/captions, selected Appendix
S2. Version **v2**, updated28 July 2026; no peer-reviewed publication was verified.
Default classifier heads train on frozen features; joint-feature variants appear
in the selected appendix. A shared-backbone-gradient explanation does not, by
itself, explain collapse in the frozen default.

For centered concatenated head embeddings `Z ∈ R^(N×KD)`, Equation7 is:

`Omega_cov(Z) = ||Z^T Z/N − I||_F² / (KD)`.

It is added to CE. Its exact transformation properties are:

- Centering removes constant translations of raw embeddings.
- `Z → ZQ` preserves the objective when `Q` is orthogonal. This includes head
  permutations and orthogonal changes within a head's coordinates.
- General invertible basis changes and rescaling generally change the objective.

The paper separately uses linear CKA. AppendixS2 calls that metric invariant to
invertible linear reparameterization. That general claim is false: with centered
`X^T X=I_2` and `Y=X diag(a,b)`, linear CKA is
`(a²+b²)/(sqrt(2) sqrt(a⁴+b⁴))`, which is below1 when `a²≠b²`.
Orthogonal transforms and isotropic scaling have the usual invariance.

Decorrelated hidden directions can be ignored by classifier readouts, leaving
identical decisions. The covariance criterion therefore is not a sufficient
condition for correct or complementary predictions. For current corrected
logits, class-uniform shifts also leave softmax/CE unchanged; a diagnostic should
remove that gauge before interpreting residual-vector separation.

A direct structural transfer limit is relevant here: at fixed parameters, each
64-dimensional label-only message lies in the span of at most `C` learned class
value vectors. Four concatenated messages have rank at most `4C`. With10 classes,
their rank is at most40, so a256-dimensional identity covariance target is
unattainable. This is restricted algebra about the printed current operator,
not a measurement of data or an audit of all paper proofs.

## One inactive, attributed conditional hypothesis

**Two-hop literal-label-echo-free correction.** A second linear label-message
pass would let a query receive permitted evidence along two-edge walks. It changes
reachable label support while retaining protected native learning. Every common-Q
label must be zeroed before the first pass, and every derived intermediate label
field must be recomputed after masking. Scores stay feature-only; bias-free linear
value paths preserve zero correction without a reachable permitted anchor.

For a mask-independent operator linear in initial label values, apply the
conditional TRAIN scaling once at the source label field. Message first moments
still do not make probabilities, losses or gradients unbiased. Multi-hop
propagation and query-label exclusion are established C&S/GAMLP/UniMP/Echoless-LP
ancestry; no new primitive is claimed.

The utility question is conditional: additional reachable anchors must supply
useful evidence without increasing misleading corrections or cost enough to erase
the benefit. A capable same-operator joint single remains the ensemble falsifier.
This hypothesis is **not implemented, adopted or authorized for execution**.
Existing source, frozen gates and closed outcomes remain intact.

## Scope and custody

Exact raw HTML, metadata, version hashes, paragraph/HTML-line locators and read
passages are retained in `primary/`; accounting is in `READ_SCOPES.json`.
GOODIE's arXiv metadata says KDD2025, but the independent registration query
returned429. LoMETab and cov-LLE remain versioned preprints in this packet.
Reported numbers encountered in abstracts/setup/selected cov-LLE prose were not
adopted. Full-paper reads, code audits, new fits and source mutations are all zero.
No novelty conclusion is inferred from existing C&S helpers/code.
