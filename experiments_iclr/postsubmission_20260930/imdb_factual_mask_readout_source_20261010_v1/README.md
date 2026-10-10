# Disabled finite IMDB factual-mask reader

Prepared 2026-10-10 from the sealed historical audit. This packet has not been run. It supplies one CPU resident readout of all 24 original selected factual banks: six shared arms and two genuine independent committees, each on the three original roles. It supplies no training, model construction, inference, new quality scores, threshold selection or raw-array transfer.

## What the counts answer

For each fixed label 0–4 and each role, the reader partitions events by the number of correct members: 0, 1, 2, 3 or 4. An event is one already labelled VALID row × label, using stored correctness masks. Each label has 274 events; all labels have 1,370 events per role.

| Plain-language question | Count definition |
|---|---|
| Was any correct member available? | Stored `any_correct_member_events`. |
| Was the pooled decision wrong although a correct member was available? | `any_correct_member_events & ~full_pool_correct`. This is a lost alternative. |
| Was the pooled decision wrong with no correct member available? | `common_wrong_events & ~full_pool_correct`. These events cannot be repaired by choosing one existing member. |
| Was a correct hard majority discarded? | Wrong-pool events with exactly three correct members. |
| Did pooling help despite an incorrect hard majority? | Correct-pool events with exactly one correct member. |
| Which member supplied a unique alternative? | Correct events for member m with exactly one correct member. |
| When did pooling help or harm an existing member? | Pool-correct/member-wrong and member-correct/pool-wrong event counts, separately for every member. |

Every 0–4 subgroup reports its size, pool-wrong count and pool-correct count. All subgroups, labels, members and roles are reported; none is selected using its result. No target-class, graph-degree, confidence, source-utility, selected-epoch or favourable-role subgroup is introduced. Positive/negative and label3 false-positive counts from the earlier report remain inherited context; this reader does not calculate them again.

The three paired contrasts are fixed: `assigned_source_supply` versus `shared_own_only`, `plain_native`, and `untied_same_six_factors`, on each role. They count actual factual pool repairs/harms, gained/lost any-correct-member coverage and whether a reference lost alternative becomes a correct candidate pool. Repairs and harms are also partitioned by the reference's fixed 0–4 correct-member count. Member indices are never paired across separate committees.

Three-role sums retain every event instance. Each method's all-label sum has 4,110 role-event instances; a label sum has 822. They are not unique-node counts, independent replicates, a micro-F1 denominator, or a new quality criterion. No percentages, F1, BCE, accuracy, calibration score, confidence interval or p value is calculated. The same VALID population was used for original checkpoint selection.

## Inputs, identity and custody

`INPUT_BINDINGS.json` copies the exact 24 diagnostic paths, lengths and hashes, original selected-checkpoint identities, and 77 existing JSON metadata/custody descriptors. The originals remain under the authorized allocation phase directory. The metadata set preserves the original readout release, freeze, complete-family and entry reports, complete cells, checkpoint-origin verification, per-bank RESULT records, and the original launch/terminal/absence evidence. Metadata hashes establish exact historical evidence; they do not create a new observation of old process absence.

Before archive loading, every bound metadata file is checked at its original resident path. Every diagnostic is then size/hash checked and deserialized through the same open file with `weights_only=True`, `map_location='cpu'`. No checkpoint, dataset, role JSON, TRAIN/TEST array or new label file is opened. Checkpoint and role/data descriptors remain origin context only.

Expected paired identities are fully declared for each role: study, role/base seed, role SHA string, full-view binding and original input descriptors. Full-view identities come from both independent specifications and agree. Stored ordered `valid_ids` must be identical across all eight banks in a role, with 274 unique integer IDs. Only a canonical ordered-ID hash leaves the reader. Original member order `[0,1,2,3]` and five-label axis are fixed.

The four accessed tensors are bool `[4,274,5]` `member_correct` and bool `[274,5]` `full_pool_correct`, `common_wrong_events`, `any_correct_member_events`. Shapes are supported by the original source and root schema, not freshly checked against binary payloads during preparation. The saved coverage masks must equal their original Boolean reductions. Unanimously correct members must have a correct binary pool and unanimously wrong members a wrong pool; violations fail instead of being repaired.

Archive deserialization can materialize other fields. The reader never accesses stored targets, probability/observed-event probability arrays, ablation tensors, U/D or logits. The source threshold remains the original strict `sigmoid(logit)>0.5` and FP32 mean probability `>0.5`; exactly 0.5 predicts negative. These are inherited masks, not newly thresholded predictions.

## Separate admission and finite execution

`RELEASE_TEMPLATE_DISABLED.json` has all approval flags false and no runtime, review receipt or budget. A future root-owned release must bind this packet's exact manifest, input bindings and source-review receipt, the resolved existing Python executable and exact Torch version, a fresh output directory and a positive finite wall budget. The template is not enabled by this preparation.

The intended entry is `owner.py --release <separate enabled release> --release-sha256 <exact release digest>`, using that existing runtime. Before resident metadata reads it checks hostname `anogena-2-0` and sole GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. The child runs with CUDA hidden and one CPU thread, and imports Torch only after source/release/custody admission. No runtime installation or retry is supplied.

The owner charges admission against the declared wall budget, owns a fresh child session and records PID/group/session/start ticks/boot ID. A timeout kills only that new owned group; a bounded five-second reap allowance is disclosed. Success additionally requires child exit zero, complete24 terminal and child-group absence. Original jobs are untouched. Root's outer execution accounting must retain process startup/owner admission failures too.

All 24 banks and all paired cohort checks must finish before count tables are written. Failed/timeout receipts retain cost and verified-origin progress without partial comparative tables. Output consists of compact count JSON/CSV, bindings/hashes and finite cost/closure receipts. No arrays, IDs, labels or predictions are exported. Child lifetime RSS and owner/child CPU and wall accounting are reported; nested wall times are not added together.

`CONDITIONAL_REMEDY.md` predeclares one possible later decision-rule comparison. This readout neither calculates that rule nor admits it.
