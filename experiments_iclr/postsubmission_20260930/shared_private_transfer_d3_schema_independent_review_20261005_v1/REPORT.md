# Independent source review — D3 logged-history schema

## Scope and result

Reviewed the immutable D3 assessment manifest `4ba8152f14a8feb70ecb5a487863338bf8889c62b57d2542542e7774b75bf870` against its exact bound run, transfer-step, custody and geometry sources. Its three manifest files and thirteen source/plan input hashes and sizes agree. Two supplementary `models.py` files were read to confirm served member counts and the row0 four-stream flag; their exact bindings are included here.

Two material wording/schema corrections and one minor preprocessing clarification follow. The source-only purpose, disabled status and lack of a numerical analyzer are accurately stated. Missing full39 custody/data is explicitly a later prerequisite in the assessment and is not a review defect. No source imports, numerical/function/fixture execution, histories/FREEZE/model/feature/score payload reads, network/server calls, existing-packet edits or extra agents were used.

## Findings

### [P2] Capable-single inner losses are not recorded per stream

[Assessment line 54](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_private_transfer_d3_logged_history_schema_20261005_v1/REPORT.md:54) says to retain every stream's virtual and recomputed inner losses for private-learning arms. The capable single concatenates the four streams into one inner forward and one normalized union loss: [original transfer_step.py:87](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_backbone_private_transfer_training_source_20261005_v2/transfer_step.py:87). The `values` list has one entry, retained as `virtual_inner_losses`/`recomputed_inner_losses` at line 164. There are no four separate historical capable-single loss values to recover. This also follows from the distinction correctly made in the assessment's line 36.

Suggested correction: “Retain each logged virtual/recomputed loss: one union-normalized value per pass for capable_single, and four per-stream values for shared/untied private learning and row0_single.” A later analyzer should validate the source-defined list length rather than expand the union loss into invented stream values.

### [P2] Maximum cycle CUDA peaks need not equal the final fit peak

[Assessment line 52](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_private_transfer_d3_logged_history_schema_20261005_v1/REPORT.md:52) correctly identifies cumulative peaks and rejects summing them, but “their maximum is a run peak” needs a cutoff qualification. [Original run.py:217](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_backbone_private_transfer_training_source_20261005_v2/run.py:217) captures cycle peaks before VALID at line 222. The final VALID evaluation can produce a higher peak after the last cycle snapshot; the producer queries CUDA peaks again for final FREEZE at line 246. Companion source has the same order. Cycle snapshots can include earlier VALID peaks but exclude any new final-evaluation peak. The single reset also occurs after runtime/input/model/setup work (line 163), so these are peaks since that reset rather than process-lifetime maxima.

Suggested correction: “The maximum cycle value is the cumulative peak observed through the recorded cycle boundaries; the authenticated final FREEZE records the producer's final peak since its initial reset and may be higher. Neither is total concurrent GPU use or host RSS.”

### [P3] Distinguish raw self-loop filtering from graph rejection

[Assessment line 26](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_private_transfer_d3_logged_history_schema_20261005_v1/REPORT.md:26) says the source rejects self links before fitting. [custody.py:194](/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/shared_backbone_private_transfer_training_source_20261005_v2/custody.py:194) counts and omits raw self loops, then checks declared raw/self/nonself counts; duplicate nonself undirected facts are rejected. Geometry `neighbors` rejects self loops if present in the graph it receives. Suggested wording: “The loader filters declared raw self loops, rejects duplicate nonself TRAIN facts, and the graph/support checks require unique nonself facts and the symmetric support count.” The stated 3,870 count refers to retained nonself TRAIN facts.

## Other semantics checked

- The four history filenames/producer fields, full-cycle TRAIN coverage, 61 episodes with a 30-row tail, and symmetric union-mask support formula agree with source. Endpoint and random descriptions share retained support; full-TRAIN per-stream/class degree/CN matching does not enforce post-mask matching.
- Exposure routes are four sampled query streams. Capable_single and row0_single each serve one member; shared/untied/native-four serve four. Row0 averages four private gradients and makes one private commit. Source counters distinguish functional calls from encoders: capable single pays 3/3 calls/encoders per episode, shared F4 and row0 pay 9/9, and a four-route untied/native-four outer pass yields 9/12. These are training receipt counts, not VALID forward totals; `validate()` does not increment cumulative training counters.
- Private learning commits one shared plus one recomputed private Adam update; virtual moments are discarded. Ordinary control commits three joint Adam updates and pays both inner passes. The sampler's literal zero optimizer field makes no supervision claim. Stale commit pays and discards recomputation and is absent from the current fixed39 spec.
- Episode neural timing begins after support/query preparation and ends after update synchronization. Geometry and neural times do not exhaust whole-cycle time; cycle time precedes VALID, while inclusive fit time begins before authorization/runtime/input/model setup. Cumulative fields should not be summed as disjoint costs.
- Virtual/recomputed losses use the same episode/support and replayed inner dropout, with the private start state held until shared updating finishes. Their difference is a same-episode response to the shared update, not private-adaptation outer improvement, heldout gain or a logged mixed-gradient causal term. Ordinary task losses are at different committed states and have source-defined scales: four-route ordinary losses sum own inner losses and scale the outer objective by member count. Raw cross-arm loss magnitudes therefore also need normalization/scope labels.
- The proposed later byte-custody → FREEZE-reference checks → training-history parsing order, exclusion of selector/tensor analysis from D3, and caution that gzip JSON-lines still materializes a full cycle are appropriate.

The findings require prose/schema corrections in a separately sealed successor. This review supplies no execution approval or actual full39 result.
