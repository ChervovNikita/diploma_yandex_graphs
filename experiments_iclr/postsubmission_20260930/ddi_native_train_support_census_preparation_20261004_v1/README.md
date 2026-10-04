# DDI native TRAIN support census preparation

Status: disabled source preparation. Root selects and authorizes the exact final contract before execution. This packet has not imported numerical libraries, opened the acquired TRAIN tensor, or run the worker.

## Frozen census contract

Use acquired official DDI TRAIN only: 1,067,911 undirected records and 4,267 nodes. Run one disclosed census-only seed, 0. Preserve the author DDI batch size of 24,576, all 43 complete native masks, the discarded tail of 11,143 records, one native default negative draw, and the exact pinned native training permutation. Each population has 1,056,768 queries. The same record IDs index positive and negative queries. The negative draw's additional rows and the native tail remain unqueried and receive identity digests.

Keep cndeg/trndeg at -1 and predictor/encoder adjacency dropout at 0. Each native mask removes all 24,576 TRAIN records in both directions. Consecutive 1,024-query chunks retain that entire mask. The chosen native batch size is fixed; chunking only bounds counting workspace.

No model, node embedding, feature tensor, optimizer, likelihood, logit, score, or VALID/TEST split is constructed or read. The DDI author fit uses learned node-ID embeddings and pretrained NCN weights. This census measures neither their behavior nor any transfer from the Collab representation.

## What is counted

For each query (u,v), let G_full be acquired TRAIN and G_visible be TRAIN after the full native mask. Left support is N_visible(u) minus N_visible(v); right support swaps endpoints. A left candidate w is selected exactly when (v,w) is in G_full; a right candidate is selected when (u,w) is in G_full. Zero denotes an unobserved TRAIN incidence.

For each side, n is support size, k is the number of selected identities, and r=min(k,n-k). The conditional law has unique support at r=0, categorical or complement support at r=1, and a genuine subset choice at r>1. Joint strata report both-variable and both-genuine opportunity directly. Exact histograms, quantiles, sums, maxima, per-batch fractions, and counts digests summarize the entire selected mask stream. Power-of-two cost bins have explicit interval bounds; n*r sums and maxima remain exact.

Boolean rows implement the same adjacency sets as the native coalesced symmetric SparseTensor, given unique loop-free undirected unit-weight TRAIN records. Removing both directions equals masking the unique record list before symmetrization. Unsampled native intersection/difference sets equal Boolean intersection/difference sets. These are source-level arguments; numerical equivalence has not been executed. Raw graph byte order was not acquired. The reused native sparse default sampler is membership/count based; no identity with an unacquired raw edge ordering is asserted.

## Prospective cost and practical scope

A dense Boolean graph occupies 18,207,289 bytes; one 1,024-by-4,267 Boolean workspace occupies 4,369,408 bytes. The complete positive and negative pass has 9,018,458,112 query/node sites across 2,064 chunk passes. These counts establish a bounded storage strategy, not a time forecast. The named core device tensors total about 130 MB under the native requested negative count, before workspace, temporaries, allocator behavior, and library overhead. See `PROSPECTIVE_COST.json` for exact arithmetic.

The geometry pass is not clearly excessive from its declared storage bounds. Execution must still measure sampler time, counting time, CUDA peaks, and worker peak RSS. Actual n*r summaries will inform prospective exact conditional-Bernoulli DP cost. No DP implementation or padded accelerator layout is benchmarked here; this proxy is not an achieved likelihood throughput estimate.

The worker checks 1,800 seconds, 2 GiB allocated CUDA, 4 GiB reserved CUDA, and 4 GiB Linux worker peak RSS. Memory checks occur after full batches, and wall checks occur during chunks and before completion. These observed worker caps do not impose instantaneous system limits. Root owns physical launch, monitoring, and any supervisor limits. An incomplete run is nonqualifying and has no automatic retry.

## Controls and handoff

`PLAN.json` is disabled. `ROOT_RELEASE_TEMPLATE.json` is also disabled and has no authorization reference. The worker accepts only the exact external `ddi_native_train_support_census_execution_root_20261004_v1/ROOT_RELEASE.json`, its supplied hash, root approval, and matching sealed source/plan contract. It requires fresh `run01` output and rechecks source, release, TRAIN, and sampler custody before completion. Existing package versions and sampler source pins are reused; no new runtime qualification or author loader is introduced.

`STATIC_VALIDATION.json` records text/AST and metadata checks only. `SOURCE.diff` compares the final worker with the preceding Collab census source. `INPUTS.json` and `SOURCE_PINS.json` bind the inspected evidence. `MANIFEST.json` seals payloads; `SEAL.json` binds the manifest.

Root receives the separately sealed method-gap and DDI recipe scout through `LITERATURE_HANDOFF.json`. Its four new primary methods were scoped, with zero full papers read. HL-GNN is the preferred additional backbone to qualify; the notes do not grant predictive adoption or novelty clearance. Existing packets and the canonical index remain untouched.

Interpretation is restricted to TRAIN support opportunity and prospective DP cost for this one native stream. It does not establish learned component differences, cross-side association, target ranking gains, independent-graph uncertainty, or auxiliary-transfer value.
