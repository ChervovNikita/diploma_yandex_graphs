# Frozen topology-only OGBL-Collab context

This small post hoc study is proposed after the learned OGBL-Collab outcomes were read. It is an exploratory conventional-baseline comparison, not a new holdout or evidence that a learned model is generally better. Review and hash this protocol and both source files, then write `OGBL_COLLAB_TOPOLOGY_BASELINES_FREEZE.json` before any topology scores are computed. Retain both baseline outcomes regardless of sign.

## Input and fixed graph

Use the official OGBL-Collab dataset with `collab.zip` SHA-256 `c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc`. The existing frozen learned runner (`ogbl_collab_frozen.py`, SHA-256 `5ba77c00311e465fd0cadcbe51d0dd66f34690be72a48138a33cb0b502caf4ac`) loads the data on CPU and verifies 235,868 nodes, message-edge years no later than 2017, both directions of every training edge, and equality between the message graph's unordered edges and the official training positives. The new runner additionally requires exact equality of all eight published run fingerprints for node features, message edges/years, and train/validation/test positive/negative pools, as embedded in its source and freeze. It rejects self-edges and collapses any duplicate directed training edges into a binary symmetric adjacency. No validation or test edges are added to this adjacency at any stage.

## Two prespecified scores

For a candidate pair `(u,v)`, let `N(u)` be its neighbors in that fixed binary training adjacency and `d(w)=|N(w)|`. Compute exactly:

- Common Neighbors: `CN(u,v)=|N(u)∩N(v)|`.
- Adamic–Adar: `AA(u,v)=Σ_{w∈N(u)∩N(v)} 1/log(d(w))`.

A common neighbor of two distinct endpoints has degree at least two, so no singular term contributes. No weights, embeddings, thresholds, preprocessing options, or graph augmentations are fitted or selected. Both scores are evaluated on every official validation and test positive and negative pair, retaining raw per-pair scores in the source order.

## Metric, output, and interpretation

For each baseline and each split, OGB's Hits@50 rule uses the 50th largest official negative score as one global threshold and counts a positive hit only when its score is **strictly greater** than that threshold; ties are misses. Retain the threshold, positive tie count, positive/negative pool sizes, and Hits@50. There is no validation selection between the two baselines; both validation and test outcomes must be reported. A separate verifier recomputes all raw scores using an independent neighbor-intersection algorithm and recomputes the strict Hits@50 arithmetic. Save `official_pool_scores.npz`, `results.json`, and the verifier audit in a new result directory inside the repository. Never overwrite existing results.

Run CPU-only with `CUDA_VISIBLE_DEVICES=''`, one PyTorch/OpenMP thread, and low CPU scheduling priority while GPU jobs run. The output is topology-only context for the existing 400-step learned study, which uses node features and trainable encoders/decoders. It is one temporal split and has no optimization-seed variability. Do not make a recommendation-system or general link-prediction superiority claim from it.
