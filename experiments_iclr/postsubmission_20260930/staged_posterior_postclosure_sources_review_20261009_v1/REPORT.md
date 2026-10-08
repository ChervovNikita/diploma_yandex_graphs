# Staged posterior postclosure source review

9 October2026. Local source/metadata only: no server, numerical import/fixture, dataset/checkpoint/array access, current partial scores or other scientific outputs. Reviewed readout SHA `97648be1bf8b420a44f86200403712e343ff94545ceed445fbb236af0881c647` and C&S analyze SHA `e74d5b459820298b4c158e8a39473502053cb452825163c1c33faabb878edb2d`. Exact inspected dependency hashes are in REVIEW.json.

## Concrete fixes before use

1. **Readout independent launch observation uses incidental process state** (`readout_once.py:100`). `terminal['owner'] == parent['identity']` joins the same captured descriptor and is appropriate. `launch['parent']` was observed independently; its `state` can differ with the same process birth. Use `owner.same(launch['parent'],parent['identity'])` for that join. This avoids rejecting a valid completed family without weakening PID/start/group/session/boot custody. Copied child-handle/EXIT descriptors can retain their existing exact joins.
2. **C&S has an incomplete owner terminal/absence barrier** (`analyze.py:73`). `stage.verify_closed_family` validates all3/all12 bank/cache/work bindings, but the extra two completion booleans do not verify exact release/launch/owner/seed EXIT joins, absence of FAMILY_FAILURE, or current original parent/child group/CUDA absence. Reuse the readout’s pinned `closed_owner` helper before the old CS receipt or any tensors are read. Root has agreed to implement this in a CS v2 source successor.
3. **C&S preflight lies outside its reported180second attempt and failure/cost scope** (`analyze.py:89`). Source/closure hashing and old receipt checks already ran before the stopwatch and timer; failures there leave no routine cost/failure receipt. Include those operations in the same finite attempt, or record a separately charged preflight with a finite bound. A small direct fix is sufficient.

No numerical qualification or new model/data work is requested by this review.

## Verified source behavior

Readout APIs match the sealed stage: exact-file selected reconstruction, `serve_ids`, posterior `readout`, EXPECTED serving-head counters and `verify_closed_family` mapping. It uses cached H/native logits, prohibits further native forwards/updates, restores exactly one learned selected bank, and performs one serving call per bank. All twelve **5274-node** ordered development readouts use original IDs/truth; repairs/harms, actual-member coverage, common wrong class and pool-only rescue masks count the full population. The source has no reselection, maximum search, native fit or mixture tuning. The descriptor fix above is the only observed readout correctness blocker.

The original collector writes a separate FROZEN_GATE using original selected metrics before reconstruction. Numeric discrepancies are retained diagnostically, with2e−5 absolute metric tolerance and exact correct-count/member-semantic comparisons; `all12_match:false` does not rewrite or rescue the original gate. Reconstruction reports/errors are additional selected-development events. The routine reserves one fresh output, one worker,285active+15cleanup seconds, records worker/routine costs and partial failures, and checks actual reaped/absent worker group/CUDA. Historical scientific costs are bound; historical native elapsed/memory joins are explicitly pending rather than presented as free acquisition.

C&S invokes the existing keyword-only TRAIN-only helper correctly: full11701×10 original native probabilities, original nonself graph records,580 TRAIN IDs/labels,C10, exact prechosenCS06 configuration and CPU serving. The helper never receives development truth. The code authenticates the old probability file, joins its authentic native state digest to staged native metadata and fixed capture, allows2e−5 cached-probability difference with unchanged argmax, and checks original count preservation. No native forward, training, new recipe search or original-score replacement is performed.

The secondary map is exactly the declared float64 nonnegative row normalization, uniform at zero-sum rows, followed by0.2native+0.8normalizedCS **on all nodes**. The original CS06 results remain retained; this is a separate serving adaptation. True-zero targets are reported as infinite NLL, not clamped. Repeated propagation is charged with attempted/completed calls, completed passes and an explicit unavailable-partial-work flag on interruption. The missing preflight scope above remains to be fixed.

## Interpretation limits

Matching0.2/0.8 weights does not match one-hop support: the staged posterior falls back to native on unreachable nodes, while C&S propagates over its declared graph everywhere. This is a useful secondary serving control, not an isolated aggregation/label-support causal test. Its CS06 choice and the staged checkpoints used previously consumed development evidence. Three seeds share one graph/split, and the new readout is still on the selecting population. Neither source establishes published-method superiority, unseen confirmation, novelty or independent-four quality. Counts for the single arms correctly refer to one actual served predictor; four attention heads are not four independent models.

AST parsing/source checks passed; runtime/scientific success is not inferred. Root owns the direct fixes and eventual full12 reconstruction.
