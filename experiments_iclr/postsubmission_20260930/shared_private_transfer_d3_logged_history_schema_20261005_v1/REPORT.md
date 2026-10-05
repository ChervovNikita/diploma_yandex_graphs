# D3: what the retained training histories can establish

5 October 2026. Source assessment for the existing five-diagnostic plan. This packet contains no experiment result, numerical analyzer or execution approval. It reads only saved source and plans. The fixed thirty-fit study, nine companions, selectors and promotion rules are unchanged. The six companion fits on 18.77 remain unreleased, and access to that machine remains withdrawn.

## Purpose

The learning-rule comparison changes more than the final loss. Different models can see the same links while paying for different numbers of encoder evaluations and optimizer updates. Endpoint separation also changes which links a member can learn from. D3 will measure these differences using every completed fit's retained histories, so a later quality result can be interpreted against the training that actually occurred.

This assessment makes the source schema and accounting explicit. It does not replace the complete-family custody checks or implement them. Every history must authenticate before its contents are used. No partial family will be analyzed.

## Actual retained records

| Record | What the source writes | Interpretation |
|---|---|---|
| `CYCLE_DRAWS.jsonl.gz` | One JSON object per cycle: cycle number, negative bank, complete outer order, both paired episode constructions, kept-positive IDs, descriptions and exposure summary | The sampler's proposed links and graph support, independent of model quality |
| `EPISODE_HISTORY.jsonl` | Cycle and episode number, elapsed neural time, outer positive rows, inner positive rows per stream, removed-positive count, symmetric support entries and update receipt | The work paid for the selected geometry in that episode |
| `CYCLE_HISTORY.jsonl` | Geometry, neural and whole-cycle times, link exposure, cumulative optimizer/forward counters and cumulative CUDA peaks | Completed cycle accounting, excluding the subsequent VALID evaluation |
| `VALID_HISTORY.jsonl` | Complete VALID scores, validation time and cumulative counters every five completed cycles | Existing selector history; score access stays under the separately reviewed complete-family analysis |

`FREEZE.json` binds the bytes of all four histories. That is a retained reference, not independent proof of pre-fit admission. The exact-nine collector and retrospective history inventory must supply their actual custody first. The D2 preparation currently authenticates `VALID_HISTORY.jsonl`; D3 additionally needs a separately reviewed inventory for all three training histories across all 39 cells. This packet supplies no substitute inventory.

## Checks fixed by the producing source

For each arm, require complete cycle numbers `1..last_cycle`, and complete episode numbers within each cycle. The draw's outer order must contain each TRAIN positive ID once, and its positive/negative outer IDs must retain the tail. In the current Citeseer study, 3,870 TRAIN positives and outer size 64 imply 61 episodes, with 30 rows in the final episode. These are this study's counts, not universal constants for future tasks.

Both endpoint and matched-random descriptions are saved even when only one is trained. Both use the same union mask of outer positives and positives targeted by either inner construction. The retained support is symmetric, with `support_nnz = 2 * (TRAIN_count - mask_positive_count)`. The source rejects self links, duplicate undirected TRAIN facts and an inconsistent support count before fitting. D3 should check the recorded identities/counts and authenticate the corresponding input binding; it should not infer the raw TRAIN graph from an unbound count.

Compare matching seed blocks and cycle indices across arms before summarizing paired geometry. The model-independent draw identity includes the negative bank, outer order, both episode ID sets and kept-positive IDs. Compare full canonical content identities, not Python process hashes. Retain any mismatch as a failure of the intended pairing rather than silently pairing by cycle number. Different trained horizons remain visible; shared-prefix comparisons supplement every arm's full-history summary and do not truncate its recorded costs.

The endpoint descriptions record zero inner queries touching the current outer endpoint union. Matched-random descriptions can have overlap. Pre-mask degree/common-neighbor strata are matched per query class and stream; post-mask strata are separately measured and need not match. D3 can report the recorded post-mask distributions and isolated endpoint counts. It cannot claim that degree matching removes every geometry or supervision difference.

## Exposure and actual optimizer work

The sampler's `exposure_summary.actual_optimizer_updates` is always **zero**. Its scopes explicitly deny an optimizer-supervision claim. Do not use that field to conclude that training performed no updates. Actual work is recorded in the episode update receipts and cumulative cycle counters.

The four exposure routes are four query streams. They are not always four served predictors. The capable single concatenates their inner queries for one inner forward. The matched row0 single retains four separate inner forwards and averages their private gradients. Report the stream count and predictive member count separately.

| Training rule | Actual update receipt | Forward accounting |
|---|---|---|
| Live or detached private learning | One shared Adam update and one private Adam update; virtual private moments are discarded; the private update is recomputed from the old private state/moments | Virtual inner calls + one outer call + recomputed inner calls |
| Stale-private control, if present in a future bound family | Recomputed work is paid and discarded; stale virtual private parameters/moments commit | Same paid calls as the corresponding recomputed rule; not an additional fit in the current fixed 39 |
| Ordinary joint control | Three committed joint Adam updates: inner, outer, repeated inner | Both inner passes and the outer pass are paid; these are not virtual private updates |

`functional_forward_calls` and `model_encoder_forward_count` are different counters. A whole untied-four outer call runs four encoders. Sum the actual episode receipts and compare every cumulative cycle counter, including episodes, shared updates, private updates, joint updates, functional calls and encoder evaluations. Validate the source-defined counter semantics for each arm instead of assuming one functional call equals one encoder evaluation.

Report drawn positive and negative exposure per stream, unseen TRAIN IDs, negative-equivalent-fact repetition and node exposure. The twice-computed inner exposure convention describes two paid inner evaluations; distinguish virtual/recomputed passes from ordinary joint passes. Repetition and broad exposure are not evidence that the private rows learned useful different functions.

## Time, memory and loss summaries

Preserve each arm's completed horizon, inclusive fit time and all cycle times. Geometry and neural times do not exhaust whole-cycle wall time: drawing/log writing, progress records and other overhead occur inside the cycle. `complete_cycle_seconds` is recorded before that cycle's VALID evaluation. Report separately paid VALID time and fit time, with their scope labels; do not silently add cumulative fields as if they were disjoint measurements.

CUDA peaks in the cycle records are cumulative after one initial peak reset. Their maximum is a run peak; their sum is meaningless, and they are not independent per-cycle peaks. They do not give total GPU reservation by concurrent jobs or host RSS. Read actual owned terminal/resource records for those other quantities after custody release.

For private-learning arms, retain every stream's virtual and recomputed inner losses and the virtual outer objective. Ordinary controls instead record the three joint task losses. These are losses at different source-defined states. A virtual-to-recomputed inner-loss difference measures the effect of changing shared weights before recomputation on that same episode's evaluated inner losses. It is not the outer-loss improvement caused by private adaptation, a heldout gain or a measured historical mixed-gradient contribution. Those causal probes remain D5, with its separate source review and discarded TRAIN computation.

Report complete per-arm histories and descriptive summaries without selecting favorable cycles, streams or graph subgroups. Loss summaries and exposure differences cannot replace the unchanged quality gate or rescue a failed contrast.

## Later implementation prerequisites

1. Authenticate all 39 actual metadata/source/job/terminal/artifact records, then all three training-history byte inventories, before parsing retained histories. Keep every unavailable cell explicit.
2. Authenticate all retained FREEZE history references after that global byte-custody pass. Perform no selector-history audit or tensor loading as part of D3.
3. Independently review the eventual analyzer against the bound producer schemas and complete family. This report is not an enabled analyzer.
4. Parse on the server and return compact results. A gzip JSON-lines reader still materializes an entire cycle if it calls `json.loads(line)`; do not claim bounded episode-level memory without a genuinely streaming parser and measured qualification. The Pubmed geometry census already illustrates that final serialization can increase peak RSS.

`INPUT_BINDINGS.json` binds the exact sources and existing analysis plan. `MANIFEST.json` and `SEAL.json` bind this assessment. No target source was imported or executed; no saved history, model, feature, score or server was accessed to create this packet.
