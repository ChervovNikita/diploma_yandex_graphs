# Independent bounded source repair review: V2

5 October 2026. **PASS_SOURCE_REVIEW** for the exact sealed `amazon_polynormer_logits_graph_moment_source_preparation_20261005_v2`, within the completed V1 source audit and this bounded repair audit. V1 F1 is resolved. No additional concrete source defect was found in the delta. Numerical qualification and execution admission remain root-owned; this report does not establish numerical behavior, payload availability or predictive outcomes.

## Custody

Expected and observed V2 manifest SHA256 `af15ac11349ac7d9dbc8608b4362d461195409ca6348fdfc67e7ac7b4a273b21`; seal SHA256 `2ed8fef0d08e420e140ddcddfa2929e00849797715d16e1e0be626e9cb2ef900`. All nine declared payload hashes and byte lengths match. The operative protocol, reference correction and V1 predecessor manifest match. The actual four-module unified diff exactly equals `SOURCE_V1_TO_V2.patch`, SHA256 `66e8d3aa068c30f5e355b18d7042cd4729a56914731d5430ac73037c672839ce`. The sealed V1 independent audit is bound and retained.

## F1 resolution

| Requirement | Exact V2 source correspondence |
|---|---|
| Native single accuracy retains original raw FP32 logit argmax | `numerical.py:51` uses `z[0].argmax(-1).numpy()` when member count is one. `custody.py:222–225` already requires FP32 CPU saved logits; the independent member0 alias remains `study.py:365`. `arrays` converts separate values to FP64 without replacing or mutating the input `z`. Original V6 `evaluate.py:91` uses the same raw argmax. |
| Four-member native accuracy keeps pooled native FP32 argmax | `numerical.py:46,51` uses `torch.softmax(z, dim=-1).mean(0)` and its argmax when member count is four, matching the original evaluator. |
| Graph mask is preserved | `numerical.py:50` retains `native_class = native32.argmax(-1)`, and `study.py:93` still calls `graph(edge, a['native_class'])`. This includes the specified single softmax-based graph class. |
| Both raw native metric call sites use the separate scoring class | Fold `study.py:190` and aggregate `study.py:213` use `a['native_scoring_class']`, indexed by their own scored IDs and sorted VALID IDs. The native predicate remains operator0 for single and operator1 for banks. |
| Native NLL/Brier and corrected metrics remain unchanged | FP64 `native`, `native_log`, all probability objects and the entire `metrics` function are unchanged. Native log remains passed only to the raw native calls; corrected calls retain their own probability argmax. |
| Difference is retained as a diagnostic | `numerical.py:52–53` counts scoring-versus-graph class discrepancies; `study.py:99` records the count in graph metadata without using it to fit, tune, choose gates or alter the graph. |

## Delta boundary

`custody.py` and `run_development.py` are byte-identical to V1. Removing only the added scoring and discrepancy dictionary fields makes V2 `numerical.py` byte-identical to V1. Reverting only the diagnostic metadata addition and the two raw-native metric substitutions makes V2 `study.py` byte-identical to V1. All four files parse as source ASTs without import. README and metadata additions document the same repair and predecessor; the protocol is unchanged. Thus fold exclusions, custody, constants, graph, QP, objective/iterate selection, operators, practical gates, 126-fit budget and sparse-call counts retain the completed V1 static review. Unrelated source checks were not repeated.

## Scope

This audit read the exact diff, repair metadata, native array/scoring/graph passages, the two metric call sites, unchanged metric function, relevant FP32 loading and member0 alias passages, and the original native evaluator reference. It performed source text/AST comparison and safe byte-hash checks only. No inspected-module import, synthetic numerical test, fit, original target payload access, checkpoint replay, SSH, manuscript/source edit or new gate occurred. Root's already planned synthetic qualification is still required before execution under its existing admission workflow.
