# Native BUDDY ports: actual CPU engineering result

Terminal supervisor status: **passed**. 8 of 8 fixed cases passed.

| Exact method ID | Status | Updates / whole VALID | Wall seconds | Peak RSS GiB |
|---|---|---:|---:|---:|
| `aggregation_conditional_globalJ_v1` | passed | 2 / 3 | 96.76 | 5.56 |
| `link_endpoint_responsibility_private_v2` | passed | 2 / 3 | 94.61 | 5.54 |
| `portable_input_residual_be_v1` | passed | 2 / 3 | 110.65 | 5.67 |
| `aggregation_mean_globalJ_v1` | passed | 2 / 3 | 92.96 | 5.53 |
| `aggregation_global_globalJ_v1` | passed | 2 / 3 | 95.42 | 5.54 |
| `link_query_iid_responsibility_private_v2` | passed | 2 / 3 | 100.10 | 5.56 |
| `link_unit_responsibility_private_v2` | passed | 2 / 3 | 98.36 | 5.55 |
| `portable_input_residual_only_v1` | passed | 2 / 3 | 100.03 | 5.46 |

Actual totals: 16 optimizer transitions, 24 whole VALID calls and 16 native constructions.
Predictive-check supervisor wall time: 795.97 seconds.
Maximum process RSS in predictive checks: 5.6734 GiB.
Context materialization separately took 26.54 seconds and peaked at 3.7122 GiB,
with one native construction, zero predictions and zero optimizer transitions.
Each predictive child used two CPU threads, a 64 GiB address-space cap and a
1,800-second timeout. These costs include concurrent project CPU activity and
are not isolated performance measurements. The cap is not a memory estimate.
OOM counters are allocation-global; per-case deltas remain in JSON evidence.

## Authentic context and native semantics

The original smoke native BUDDY authority explicitly uses installed datasketch
1.5.9. Its installed distribution RECORD hashes were checked and retained. The
older retained year0 cache uses datasketch1.6.5, fixed negatives and different
batch ordering; none of its cached values or dependency files were substituted.

The materialized context was constructed through the unchanged native source,
with the original year2007 filter, weighted graph, SIGN-k0 features, hash banks,
degrees, negative sampling and1024-row joint minibatches. The first two exact
TRAIN sampler rows and hydrated inputs, and both complete source VALID roles,
were authenticated against materialization. Context tensors were validated,
not injected. Every whole VALID uses60,084 positive and100,000 negative queries.

All three C6 modes use frozen all-block J=.5F+.5L_actual_served, detached
entropy-confidence/JS gate features and0/4/20 gate parameters. BUDDY retains
one original joint BCE mixed-batch mean for F and L. Responsibility retains
shared F/private W, with one normalization group over the original joint batch.
No NCN positive-mean plus negative-mean training reduction was transferred.
Endpoint/query controls collect two gradients at one old state and take one
source step; unit uses one reverse collection. Source Parameter identities,
shared BatchNorm modules and original optimizer options remain intact.

Portable input modes reuse their unchanged callback. Six new methods report
joint BCE VALID loss diagnostics; inherited portable split-group VALID loss is
an explicitly separate diagnostic and does not change native joint-BCE training.

Successful cases retain bit-exact served and same-bank mean VALID replay in a
fresh native instance, original native Parameter identities and resumed clock2.
Actual callback paths, SHA256 pins and settings are recorded from successful
jobs in `PORT_METHOD_BINDINGS.json`. Checkpoint binaries stay in the allocation.

## Limits

This is BUDDY/Collab split0/seed0 CPU two-update qualification only. No complete
TRAIN epoch, convergence, predictive-quality comparison, scientific fit, CUDA
admission, pair-resource qualification, TEST scoring or final TEST reconstruction
was performed. Other seeds and full-campaign readiness remain unqualified.
NCN and NCNC evidence stay in separate immutable packages. No server77 access
occurred. The gate was configured to stop on its first failure; no case failed
or was retried.
