# Context targets: one information question across three tasks

This is a prospective definition. It does not qualify native integration or
claim a predictive advantage. The context is an auxiliary TRAIN target; model
inputs, operators and serving information remain native.

The question is whether four routes learn useful complementary information
when different routes fit different factual context similarities. Its control
gives every route the same average of those targets. Both use exactly the same
examples, donor opportunities and work. Different targets alone do not imply
different useful predictions.

## Common learning rule

Use the existing live task prehead. Unit-normalize along the native hidden
dimension with epsilon1e-12. For each anchor, exclude the same object and admit
only distinct donors in the same observed TRAIN class (and same source relation
for links). Do not use VALID or TEST labels. Links' sampled negative group is
a source supervision group, not a claim of verified nonexistence.

For each factual context channel, L2-normalize its vector with epsilon1e-12.
Target donor probabilities are softmax of context cosine divided by0.2 over
the admitted donors. A zero vector gives zero cosine and a uniform target;
report that count. No compatible donor means an omitted auxiliary anchor,
reported explicitly. The native task risk still covers that anchor.

Each route predicts a donor distribution using its own existing prehead
cosines at temperature0.2 over exactly those same donors. Minimize target
cross-entropy, class-balanced over supported anchors for nodes/molecules and
with the source separate-positive/negative or mixed convention for links.
All parameters receive the gradient of F+0.05A. No new projector, decoder,
backbone pass or teacher is added.

In **common**, every route fits0.5qA+0.5qB. In **split**, routes0/1 fitqA and
routes2/3 fitqB. This is one specific assignment; there is no assignment search.
Do not count expected-mean-equivalent target shuffles as another information
mechanism. Freeze input signatures and target construction in checkpoints.

## Node classification

- A: the exact native feature vector at each admitted TRAIN node.
- B: one symmetric degree-normalized propagation of those same features over
  the source-preprocessed transductive edge records, with one self contribution
  per node. If native preprocessing already contains self records, remove only
  those records for this target calculation before adding one self contribution.
  Preserve nonself multiplicities. Use target adjacency convention matching
  `edge_index` incoming aggregation and degrees of that target adjacency.
- Only source TRAIN IDs/labels define anchors/donors. Public transductive
  features/topology may contribute to the factual B target.
- At most512 distinct TRAIN nodes, fixed once per fit by seed/ID hash ordering
  with class-preserving round-robin selection. The full native TRAIN CE remains.

## Binary molecular graph classification

- A: concatenate per-field atom-category frequency histograms using the pinned
  native AtomEncoder categorical vocabularies.
- B: concatenate per-field bond-category frequency histograms using the pinned
  native BondEncoder vocabularies. Use the actual source directed bond records;
  an edgeless molecule has a zero B vector and uniform B donor target.
- Use every graph in the original native minibatch, source graph IDs and
  observed labels only. Missing labels are absent from auxiliary targets.
- The target requires neither chemical edits nor an assumption that an
  augmentation preserves the molecular label. It is not a fingerprint model
  reproduction or a new inference feature.

## Sampled-query link ranking

- A: sorted pair of `log1p` endpoint degrees in the source TRAIN simple
  undirected target graph, excluding the queried pair if present.
- B: `log1p` common-neighbor count, Adamic-Adar sum and resource-allocation sum
  from that same target graph. Remove loops and deduplicate records for this
  auxiliary target only; source predictor topology stays unchanged. A candidate
  witness must be distinct from both endpoints. Its target degree is computed
  after query exclusion. Empty witness context yields a zero vector.
- Query exclusion applies equally to observed positives and source-sampled
  negatives; sampling does not certify that the pair is absent from TRAIN.
- Same source binary group and relation are required for donor compatibility.
  Keep at most1024 existing query rows, up to512 per source group, selected by
  the fixed query hash law already used by the link auxiliary. Do not add
  queries or alter the source task reduction to fill donor support.
- Target preprocessing and per-update work must be measured before a full
  campaign. Missing witnesses/support are reported; they do not authorize an
  automatic alternate target or a fabricated relation.

## Attribution and limits

This combines known supervised metric learning, soft similarity targets and
route-specific objectives. It is not a novelty claim for contrastive learning,
graph propagation, chemical histograms, common-neighbor statistics or mixture
learning. A positive comparison would identify a useful training recipe under
the tested native settings. Further primary-literature comparison and heldout
confirmation would still be needed before a methodological claim.
