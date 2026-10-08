# GOODIE label-posterior boundary follow-up

8 October 2026. This is one new appendix scope and one bounded author-repository
read. It resolves four open boundaries left by the saved method scout. It is not
a full-paper read, full code audit, published-body comparison or empirical check.
No verdict, novelty claim, superiority claim or new experiment is proposed.

## Sources and scope

The paper is **Oldie but Goodie: Re-illuminating Label Propagation on Graphs with
Partially Observed Features**, arXiv **2508.01209v1**, dated 2 August 2025.
Only [Appendix E, Algorithm 1](https://arxiv.org/html/2508.01209v1#alg1), numbered
lines 1–45, is newly read. Its saved HTML locator is section A5, source lines
1790–2109. The primary Method is not reread. The saved metadata links the author
repository [SukwonYun/GOODIE](https://github.com/SukwonYun/GOODIE).

The repository's main HEAD was pinned to **fbc714b599564e4f0e843227dc9de9db964bf297**,
commit timestamp **2025-08-04T22:08:22Z**. New semantic scopes are the GOODIE
training loop, constructor and forward loss; its GCN/GCNConv/normalizer; and the
four embedder lines assigning masks and labels. Other model implementations,
data loading, external PyG implementation, hyperparameters, images and repository
result prose remain unread. Exact URLs, raw-byte SHA-256 hashes, Git blob checks,
retrieval times and read locators are saved in `primary/` and `READ_SCOPES.json`.
The public repository was available; no source-unavailability inference is needed.

## Four resolved boundaries

| Boundary | Appendix E and pinned author code | Relation to the proposed inactive recipe |
|---|---|---|
| **TRAIN-query label removal** | Algorithm 1 accepts the TRAIN label matrix, propagates it once before training (lines 1, 4, 21–31), and restores original TRAIN labels after every LP step (line 24). Code calls `LabelPropagation` once with `y=self.labels, mask=self.train_mask` before the epoch loop. The same `self.train_mask` supplies the supervised loss indices. No query-mask draw, removal from all contexts or LP recomputation appears in this path. | The proposed common TRAIN-query labels are removed from every route's initial context before propagation. GOODIE's shown path retains those supervised TRAIN targets as LP anchors. A request to avoid adding decoder self-loops does not remove a target's label from upstream LP or multi-hop paths. |
| **Protection of the feature branch** | Algorithm 1 constructs learned FP and LP embeddings, attention-weighted joint embedding Z, and one final classifier (lines 12–19). Code includes all model parameters in one Adam optimizer and backpropagates the joint loss. The active feature decoder participates in the mixed prediction without a detach/freeze boundary. LP and feature completion are computed before the epoch loop, but the neural FP decoder remains trainable. | Precomputing inputs does not establish a protected feature predictor. The proposed feature-only GNN is selected by its own selector, then frozen, with detached H and native logits. GOODIE's selected validation output belongs to the joint model. |
| **Direct CE on the label branch** | Algorithm 1's CE is on P returned by the combined classifier, plus the pseudo-label contrastive objective (lines 6–9, 17–19). Code computes `F.cross_entropy(output[idx_train], labels[idx_train])`, where `output = classifier3(attention_mix(fp_embed, lp_embed))`. The label decoder returns an embedding; it has no separate CE in the inspected forward. | The proposed routes each train `CE(label_logits, y)` directly and independently of frozen native logits. GOODIE's shown CE supervises a joint nonlinear feature/label classifier; it is neither that direct route loss nor a fixed `CE(base_logits + label_logits, y)` residual loss. |
| **Serving aggregation and fallback** | Algorithm 1 returns one softmax after the final classifier. Code's validation/test path repeats the learned two-branch embedding attention, passes it through `classifier3`, and retains the output selected by validation accuracy. This is the repository's exposed evaluation inference path; a separate production serving API was not inspected. | The proposed rule uses native prediction when no permitted anchor is available, otherwise `.2 native probabilities + .8 mean route probabilities`, represented by a stable log-mixture. GOODIE's shown path has a learned embedding mixture and one joint output, with no fixed probability mixture, route mean or explicit native fallback. |

Code locators for this table:

- LP construction and TRAIN anchors: [GOODIE.py lines 39–55](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/models/GOODIE.py#L39-L55); masks and labels: [embedder.py lines 51–54](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/embedder.py#L51-L54).
- Joint optimizer/update: [GOODIE.py lines 61–77](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/models/GOODIE.py#L61-L77); active feature/label forward and loss: [lines 162–191](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/models/GOODIE.py#L162-L191).
- Joint evaluation and selector: [GOODIE.py lines 79–119](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/models/GOODIE.py#L79-L119).

## Additional operator boundary relevant to this comparison

GOODIE's LP decoder consumes the **propagated LP class-score field**, whereas the
inactive recipe's message values come only from retained TRAIN anchors. GOODIE
then compares LP and FP embeddings through a shared learned attention vector at
each node; this attention is over the two branches. It is not the recipe's
feature-only Q/K attention over all nonself neighbors. The shown decoder uses
GCN graph aggregation and graph-degree coefficients, without that neighbor
softmax denominator. See [GOODIE.py lines 143–170](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/models/GOODIE.py#L143-L170)
and [layers.py lines 35–71, 188–227](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/layers.py#L35-L71).

The internal GCN linear map is bias-free, but GCNConv separately enables and adds
a trainable output bias by default; the GOODIE constructor does not disable it.
Thus the entire LP decoder is not a bias-free zero-preserving value route. A zero
incoming LP tensor can yield a nonzero embedding through the learned bias. This
is a static conditional observation, not an executed measurement. See
[layers.py lines 119–139, 178–186, 208–224](https://github.com/SukwonYun/GOODIE/blob/fbc714b599564e4f0e843227dc9de9db964bf297/layers.py#L119-L139).

## Bounded overlap statement

GOODIE remains close prior for a learned graph-label decoder, a learned feature
branch, node-dependent combination of their evidence and supervised joint
prediction. Appendix E and this pinned code resolve the four implementation
questions toward retained TRAIN anchors, joint feature gradients, joint CE and
joint embedding aggregation. They do not instantiate the proposed complete
training/serving rule in the inspected path. This statement is local to the
listed scopes and commit; it is not an absence certificate for other code,
versions, papers or mechanisms.

The comparison target stays inactive: one source-native feature-only predictor
selected and frozen; four shared BE label-attention routes with detached feature
Q/K inputs, TRAIN-only bias-free values and common query exclusion; direct route
CE; and the specified native/route probability mixture with anchor-free native
fallback. The capable nonlinear joint four-head SINGLE receives the same
information, weighting and selection, alongside an untied four-route control.
These are supplied comparison constraints, not conclusions drawn from GOODIE or
newly designed experiments. Existing ingredients remain attributable to their
literature; this follow-up makes no novelty or utility claim.

No scientific server, dataset, fitted model/checkpoint, scientific outcome file,
PDF compilation, GENLINK, canonical ledger or training-source mutation was used.
All new files are inside this fresh follow-up folder under P.
