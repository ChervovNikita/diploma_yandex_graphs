# Filtered-Chameleon study timing amendment

This note was added after the completed results were audited to correct an inaccurate sentence in an early post-run README. That sentence said the filtered study was chosen after legacy-Chameleon outcomes were known. The saved study record supports the following narrower chronology (all times UTC on 2026-09-26):

| Event | Recorded time | Evidence |
|---|---:|---|
| Filtered dataset and complete protocol proposal saved | 00:18:17 file modification time | `FILTERED_CHAMELEON_PROTOCOL_PROPOSAL.md`, SHA-256 `a6e193cbbd7674e7592f7268d3a808a285da8eee6a9c0dab2458426079d52ab8`; its text states it was prepared before inspecting the Cora/legacy-Chameleon outcomes. |
| Filtered depth matrix frozen before training | 00:20:23 `created_utc` | `PRETRAIN_FREEZE.json`, SHA-256 `400d4c68a2b6cd408fbec06ddb4ebd762e147836176ff664f888e10e6fccb71e`; `outcome_files_seen=0`. |
| Filtered sharing-position matrix frozen before training | 00:22:20 `created_utc` | `POSITION_PRETRAIN_FREEZE.json`, SHA-256 `cb87a1c3e57d2bd4a08a57b949c8d785a6d00bd0dd04478aff99ea52b290f12c`; `outcome_files_seen=0`. |
| Cora/legacy-Chameleon 48-arm compact stage built | 00:23:38 file modification time | `NEW_GRAPH_COMPACT_STAGE/anonymous_new_graph_stage/evidence_manifest.json` in author evidence. |
| Cora/legacy-Chameleon validation-choice report written | 00:27:55 file modification time | `NEW_GRAPH_VALIDATION_CHOICE.json` in author evidence. |

The filtered graph was motivated by Platonov et al.'s documented duplicate-node and evaluation concerns. Its comparison with legacy Chameleon still changes graph size and the legacy graph's retained 50 self-loops. This timing amendment changes neither frozen study file nor experiment result. It records why the earlier README wording was wrong and identifies the post-run documentation correction; file modification times alone do not prove the exact time at which any human first inspected a result.
