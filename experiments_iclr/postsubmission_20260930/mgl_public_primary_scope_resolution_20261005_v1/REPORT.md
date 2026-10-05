# MGL: public author-source resolution

5 October 2026. MGL's paper remains unread, but its metadata-only lead can now be qualified by a bounded public implementation scope. Index68 was checked first: DOI `10.1145/3580305.3599428` has one unresolved lead and no paper record or completed method read. No earlier primary scope was reopened. This packet changes no index, study, gate or experimental authority.

## Public route and reading credit

Crossref identifies *Meta Graph Learning for Long-tail Recommendation* (2023), first author Chunyu Wei. Public repository [weicy15/MGL](https://github.com/weicy15/MGL) uses that exact title and calls itself the official implementation; the owner's public profile names Chunyu Wei. The inspected revision is `c887b11676e7c80f108941aa7ce3cd2d35194f9c`. This supports association with the paper as a claimed author implementation. Publication-to-code algorithm equivalence is not independently established.

Three code files were retrieved at that immutable revision and matched against their Git blob identities: `model.py`, `train.py` and `load_data.py`. Exact selected line ranges and text hashes are retained in `PUBLIC_SOURCE_SCOPES.json`. Nothing was imported or executed. The repository's dataset blobs were listed as tree metadata only and never retrieved. No manuscript appeared in the public repository tree. The prior ACM HTTP403 is inherited evidence of an access failure; this task did not retry ACM.

Accounting is **one bounded public author-repository scope event across three semantic code files**, with zero primary-paper method/full reads and zero new paper-identity reading credit. It is a source-scope upgrade to a known metadata lead, not a newly read paper. The immutable source URL, title/profile association and runtime limits remain explicit.

## What the selected source specifies

| Question | Source evidence and interpretation |
|---|---|
| Support/query construction | `train.py:109–143` derives item co-occurrence from TRAIN user-item interactions, applies a proposed mask, and creates item-pair support. `158–187` pairs support batches with TRAIN user-positive-negative query batches. `model.py:189–207` uses an item-similarity MSE support objective with random false items. `load_data.py:239–248` draws query negatives outside that user's TRAIN positives. No exclusion of both endpoints of positive/negative query pairs, matched-random geometry, or disjoint supervised-fold construction is specified. |
| Shared and fast roles | One persistent model contains item-feature embeddings, an encoder/decoder generator and user/item ID embeddings (`model.py:24–122`). Four encoder tensors from two affine layers become temporary `fast_weights`; they are four tensors, not four predictors. Query graph augmentation consumes those virtual encoder weights (`227–302`). There is no bank of private learner rows. |
| Current update differentiation | `train.py:178–189` takes `autograd.grad(..., create_graph=True)` of support loss with respect to generator-encoder parameters, forms one plain gradient step, and evaluates query ranking loss plus a regularizer. The backward graph retains dependence through the current support gradient. The inner map is not Adam. |
| Optimizer carry/reset | A single `optim.Adam(model.parameters(), ...)` is created before the epoch/batch loops (`151`, `160–193`) and stepped on the outer loss. Its ordinary global state is carried if this path runs; `zero_grad()` clears gradient buffers. No separate private inner moments are created, reset, carried or differentiated. Virtual weights are recreated from current encoder weights for each query batch. |
| Commitment | The source takes one global Adam step after outer backward (`191–193`). It does not assign the virtual support step to a persistent private learner, or recompute that step from unchanged old private weights/moments after a shared commit. |
| Deployment | `model.py:306–333` rebuilds an auxiliary item graph with the ordinary generator, propagates trained user/item embeddings, and returns item scores. The evaluation body calls `predict(user_id)` and masks past TRAIN interactions (`train.py:202–226`). That prediction function performs no support-label adaptation or optimizer step and serves one model, not a raw-logit mean of persistent members. |

The full TRAIN adjacency is constructed at `model.py:126–149`; the inspected query routine does not remove each current target from it. Support and query labels arise from the same training interaction source. Thus this source establishes a training-only graph auxiliary-learning response, without certifying independent tasks, endpoint cross-fitting or the active support/masking convention.

## Derivative and release limitations

The observed virtual base is obtained with `encoder.state_dict().values()` (`train.py:180`). Under standard PyTorch `state_dict()` semantics, those bases are detached unless `keep_vars=True` is requested; this source does not request it. The support-gradient graph is retained, but the virtual base's identity path is not the usual fully connected MAML parameter copy. No exact native-Adam or complete active-operator derivative equivalence follows.

There is also an explicit functional mismatch: normal `Generator.encoder` has ReLUs (`model.py:50`), while `q_link_predict` applies its two fast affine maps without them (`245–249`). The virtual and deployed encoder maps therefore differ in this revision. The repository is not runtime-qualified: `train.py` imports `models` while the tree supplies `model.py`, and its evaluation body refers to `best_checkpoint` without a definition in the inspected script. No source repairs, numerical checks or runtime claims are made here. These limits require treating the reading as evidence of the source's specified/intended construction, rather than an authenticated executed algorithm or publication-equivalent implementation.

## Closest ancestry and one concrete gap

MGL is useful close **component ancestry**: a learned graph generator is virtually updated on auxiliary supervision, and downstream graph recommendation supplies current-response training credit; inference is specified without an optimizer adaptation step. Broad claims that graph-supervised training lookahead or training-only generated graph structure are new should account for this source.

The one concrete active-operator gap supported here is the **persistent private-state and commitment coupling**. The active rule differentiates a current private Adam response conditional on each route's old parameters/moments, updates shared parameters, then recomputes and commits the private response. This MGL source uses temporary plain-gradient encoder weights and one ordinary outer Adam update; it does not implement that private-state/commit coupling. This is a precise configuration difference, not proof of historical novelty, predictive benefit or ensemble necessity. No ready successor or new comparison is supported by this reading.

MGL's publication methodology remains unresolved until a primary manuscript or a verified publication-to-source correspondence is available. The public source nevertheless resolves several previously abstract-only operator questions. The exact scope, association evidence, source defects and zero paper-reading credit must travel together.

## Boundaries

All retrieval was public citation/search/repository material. No allocation, 18.77 or MacLink contact occurred; no run outcomes, datasets, histories or checkpoints were opened. No scientific computation, model import, training, source repair, paper review, canonical change or fixed-study change occurred. Experimental access remains allocation-only unless the human restores 77; this packet grants no access or launch authority.
