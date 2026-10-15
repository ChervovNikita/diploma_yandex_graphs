# Native NCNC ports: actual CPU engineering result

Terminal supervisor status: **passed**. 8 of 8 fixed cases passed.

Each case performed two complete native-sized CPU optimizer updates, three whole
VALID calls and serialized restoration into a freshly constructed native instance.
Every served and same-bank mean VALID output was bit-exact after restoration.
Original native Parameter identities and optimizer options were preserved, and
the resumed native update clock ended at 2.

| Exact method ID | Status | Updates / whole VALID | Wall seconds | Peak RSS GiB |
|---|---|---:|---:|---:|
| `aggregation_conditional_globalJ_v1` | passed | 2 / 3 | 391.15 | 12.43 |
| `link_endpoint_responsibility_private_v2` | passed | 2 / 3 | 432.42 | 12.94 |
| `portable_input_residual_be_v1` | passed | 2 / 3 | 401.64 | 12.98 |
| `aggregation_mean_globalJ_v1` | passed | 2 / 3 | 403.83 | 12.43 |
| `aggregation_global_globalJ_v1` | passed | 2 / 3 | 351.84 | 13.03 |
| `link_query_iid_responsibility_private_v2` | passed | 2 / 3 | 415.83 | 13.12 |
| `link_unit_responsibility_private_v2` | passed | 2 / 3 | 406.03 | 12.45 |
| `portable_input_residual_only_v1` | passed | 2 / 3 | 353.29 | 12.67 |

Actual totals: 16 optimizer transitions, 24 whole VALID calls and 16 native constructions.
Supervisor wall time: 3164.41 seconds; summed case time: 3156.03 seconds.
Maximum observed process RSS: 13.1181 GiB. Each sequential child used two CPU threads,
a 64 GiB address-space cap and an 1,800-second timeout. The cap is not a memory estimate.
Other project CPU work ran during this check, so these are not isolated speed measurements.
Per-case OOM and OOM-kill deltas are recorded in the compact and raw evidence;
the allocation counters already included an earlier OOM event. No file cache was
cleared, no cache settings changed and no unrelated process was stopped.

## Native task scope and risk

The unchanged native source handles negative sampling, source-sized 65,536 positive
and 65,536 negative TRAIN queries, source adjacency masking and four original routes.
The new observer uses the existing `TaskCallScope.depth`: only calls entering at
source depth 0 enter the task objective. Each checked update has eight top-level
task calls; actual recursive dispatch counts remain in the compact evidence.
The source output boundary bypasses all recursive calls. No recursive completion logits become task rows. Whole VALID
uses all 60,084 positive and 100,000 negative queries on every call.

C6 reuses the frozen detached entropy-confidence/JS core with exact uniform
initialization and all-block J = .5F + .5L_actual_served. Mean/global/conditional
readouts add 0/4/20 gate parameters. Both F and L retain the native positive-mean
plus negative-mean reduction. Each C6 update uses one backward collection and
one native optimizer transition. Actual served VALID drives selection; same-bank
mean output is retained as a diagnostic.

Responsibility v2 reuses the frozen corrected sampler and objective modules.
Shared native parameters receive F gradients and existing private BE factors receive W.
Endpoint/query modes collect two gradients at one old state and take one step;
unit mode uses one reverse collection. Query control uses two independent Exp
halves for nonself queries and one Exp draw for self-loops. Weights normalize
within the native positive/negative groups. No normalized-marginal matching or
unbiased-original-risk claim follows.

Both portable input modes use the unchanged `portable_private_input_v2/link_callback.py`.
For residual-only, inherited output-scope names such as `task_applied` describe
task eligibility and do not assert that an output BE factor exists. The raw checker
records this distinction explicitly.

## Retained evidence and limits

`CPU_RESULTS.json` retains the actual job descriptors, callback and dependency hashes,
both update receipts, whole VALID diagnostics, resource counters and checkpoint receipts.
One- and two-update checkpoint binaries remain in the allocation.
`PORT_METHOD_BINDINGS.json` is built from the actual successful job callback/SHA/settings.
`NCNC_CPU_QUALIFICATION.json` records compact actual evidence. Frozen source files
and its source manifest remain unchanged.

This qualification is limited to NCNC/Collab split 0, seed 0, CPU and two updates.
No full TRAIN epoch, convergence, predictive-quality comparison, scientific fit,
CUDA admission, pair-resource qualification, TEST scoring or final TEST wrapper
reconstruction was performed. NCN remains separately qualified. BUDDY v2 is
source prepared and still requires authentic year2007 context materialization and
actual execution. No all-link or campaign readiness and no server77 access is claimed.
