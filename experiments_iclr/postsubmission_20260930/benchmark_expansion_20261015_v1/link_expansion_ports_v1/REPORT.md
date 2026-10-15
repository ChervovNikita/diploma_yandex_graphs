# Native NCN ports: actual CPU engineering result

All eight fixed NCN/Collab cases passed. Each used two complete native-sized
optimizer updates (65,536 positive and65,536 negative TRAIN queries per update),
three whole VALID calls (60,084 positive plus100,000 negative queries), and
serialized restoration into a freshly constructed native source instance.
Served and same-bank mean VALID outputs were bit-exact after restoration for
every case. Original native Parameter identities and optimizer options stayed
intact. The resumed native update clock ended at2.

| Exact method ID | Source | Added gate parameters | Updates / full VALID | Wall seconds | Peak RSS GiB |
|---|---|---:|---:|---:|---:|
| `aggregation_mean_globalJ_v1` | New NCN callback | 0 | 2 / 3 | 34.53 | 8.65 |
| `aggregation_global_globalJ_v1` | New NCN callback | 4 | 2 / 3 | 35.70 | 8.88 |
| `aggregation_conditional_globalJ_v1` | New NCN callback | 20 | 2 / 3 | 36.86 | 8.91 |
| `link_unit_responsibility_private_v2` | New NCN callback | 0 | 2 / 3 | 39.86 | 8.73 |
| `link_query_iid_responsibility_private_v2` | New NCN callback | 0 | 2 / 3 | 52.40 | 9.12 |
| `link_endpoint_responsibility_private_v2` | New NCN callback | 0 | 2 / 3 | 49.12 | 8.95 |
| `portable_input_residual_be_v1` | Existing portable v2 callback, unchanged | — | 2 / 3 | 35.89 | 8.98 |
| `portable_input_residual_only_v1` | Existing portable v2 callback, unchanged | — | 2 / 3 | 34.53 | 8.13 |

The total was16 actual optimizer transitions,24 whole VALID calls and16 native
constructions. Supervisor wall time was326.10 seconds. Checks ran sequentially
with two CPU threads, a32GiB address-space cap per child and a1,200-second
timeout per case. Peak observed process RSS was9.1193GiB. Per-case OOM and
OOM-kill counter deltas were zero; the allocation's pre-existing OOM counters
are preserved in the launch and raw results. File-cache reclamation was not
forced, no cache settings were changed and no unrelated job was stopped.

## Source and semantics

`callback.py` and `session.py` add the missing NCN C6/responsibility orchestration
around unchanged `link_native_integration_v1/session.py` and the frozen native
NCN callback. Native negative sampling, shuffled minibatches, adjacency masking,
encoder/predictor forwards, per-route dropout/RNG and source reductions still
execute in the native session. Actual predictor query endpoints and route
outputs are observed and checked against the native sampler order.

C6 uses the unchanged `conditional_aggregation_v1/core.py`: detached entropy
confidence and JS features, exact uniform initialization, and all-block
J=.5F+.5L_actual_served. The unchanged native positive-mean plus negative-mean
reduction defines both F and L. All three readouts use one backward collection
and one source optimizer transition. VALID selection receives the actual served
readout; same-bank mean is diagnostic and uses the same native forwards.

Responsibility uses unchanged `structured_responsibility_control_v2` sampler
and objectives. Four authenticated actual route signatures match. Shared native
parameters receive F gradients; existing private BE factors receive W gradients.
Endpoint/query modes collect two gradients at one old state, then perform one
source step. Unit uses one reverse collection. The corrected query control has
two Exp halves for nonself queries and one Exp draw for self-loops. Weights
normalize within the source's positive/negative groups. This does not match
normalized marginals or establish unbiased original risk.

Portable input v2 already had an actual link callback, so it was reused without
duplicating or editing that source. Both BE-plus-residual and residual-only
variants passed these exact NCN checks.

`PORT_METHOD_BINDINGS.json` records exact outer callback paths, SHA256 values,
manifest settings and base procedure for all eight methods. The new frozen
source manifest is `c80f6e736e6cb96683f5f429ab9a3c3f78c19c8ed241e074a741f54e9d18c8db`.
`CPU_RESULTS.json` preserves actual outcomes, source metadata, steps, checkpoint
receipts and costs. `NCN_CPU_QUALIFICATION.json` is the compact evidence summary.
Checkpoint binaries and source-check logs remain in the allocation qualification
folder; only JSON evidence was collected locally.

## Limits

These results cover only NCN/Collab split0/seed0 on CPU. No complete TRAIN epoch,
stopping trajectory, scientific fit, predictive-quality comparison, GPU run,
pair-resource admission, TEST scoring or final TEST wrapper reconstruction was
performed. The new C6/responsibility NCNC and BUDDY ports remain explicitly
pending; portable NCNC/BUDDY execution is also unqualified here. No all-link or
full-campaign readiness follows. Frozen runtime, native bridge, source callbacks,
prior pilot and original paper scores are unchanged. No server77 access occurred.
