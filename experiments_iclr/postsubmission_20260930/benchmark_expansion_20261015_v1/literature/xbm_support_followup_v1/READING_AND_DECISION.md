# Cross-batch memory and rare-class molecular supervision

## What was read

Wang, Zhang, Huang and Scott, **Cross-Batch Memory for Embedding Learning**,
CVPR 2020, pp. 6388–6397. Primary CVF abstract and paper Sections 3.1–3.3,
Equations 1–6 and Algorithm 1 were read. The related-work continuation on
page 3 and adjacent hard-mining description were also inspected. Pages 4–5
were visually checked. The supplementary proof, experiments and source code
were not audited. This is one scoped primary-method read, not a full-paper
read or a systematic novelty search. The v72 index contained neither this
title nor its arXiv identifier 1912.06798 when checked.

Source: https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html

## The useful prior idea

XBM stores detached embeddings and labels from earlier minibatches in a FIFO
queue. Current examples receive gradients from comparisons with that queue.
The method uses a warmup because old features can be poor approximations early
in training. Its claim about small gradient error depends on small feature
drift and regularity assumptions for a pairwise objective. Those assumptions
are not established for our supervised contrastive loss, trainable route
factors, graph BatchNorm or molecular scaffold shift. The prior paper's image
retrieval improvements do not predict our molecular classification results.

## Why it is relevant to the current data

Our audited MolHIV TRAIN role has 1,232 positives among 32,901 molecules.
With the native batch size 32, 29.47% of uniformly sampled complete batches
have no positive molecule and only 33.81% have at least two positives.
The current class-alignment objective is inactive when a class is absent.
Its positive-class variance term needs two distinct positive molecules.
Four routes of one molecule do not provide four independent molecules.

A queue could supply contrasting class examples while preserving the native
data order, batch size and optimizer. It would not create current positive
anchors in a negative-only batch. Detached past features also do not restore
the live positive-class variance gradient. These are separate limitations.

## A bounded follow-up worth considering

Keep this outside the frozen 6,460-fit roster for now. If the full MolHIV
adapter diagnostics confirm the support problem, consider a fixed queue of
the previous eight native batches (256 distinct molecule IDs), retaining all
four route embeddings per molecule. Compare an unchanged alignment objective
with a queue-augmented alignment objective; retain the native task loss and
current-batch-only variance term. Do not label the queue itself a contribution.

The queue must use TRAIN labels only, store detached tensors, remove stale
copies of a repeated molecule, preserve route identity, exclude same-identity
self pairs explicitly, and be included in recovery state. Update timing and
warmup must be fixed before the pilot. The current-batch class weights and
available candidate classes need separate reductions; a missing anchor class
must not receive an invented gradient. Four-route cross-comparisons have a
larger denominator than a single-route queue and must be costed explicitly.

Before any predictive run, record auxiliary active-batch fraction, distinct
positive support, queue age and observed feature drift on a fixed TRAIN-only
probe. An identical queue with the auxiliary coefficient zero is a lifecycle
check, not a scientific extra baseline. The scientific comparison needs the
same no-queue objective, base GNNM, native single and I4 controls, all with
complete convergence traces and several paired seeds. A successful result
would be evidence about accessible supervision in a shared ensemble; an
image-memory primitive or more distant route embeddings alone would not
establish a new method or a useful ensemble.

## Decision

Retain as a concrete, literature-grounded follow-up to the TRAIN support
finding. First finish the private-BN pilot and production benchmark pipeline.
No queue implementation, training, heldout scoring or roster addition was
performed by this reading. It does not resolve the present deficit to I4.
