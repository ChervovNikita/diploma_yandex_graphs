# Final v29 resource status

The finite plan and this resource report are prepared. Scientific launch remains disabled: no GO, new fits or predictive scores. Complete trained-state, CUDA peak memory, whole-host disk and timing qualification remains unresolved. Current 18.77 admission is unverified.

## Exact proposal

The immutable plan is `massive_practical_master_20261013_v29/PLAN.json`: canonical SHA-256 `0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b`; raw SHA-256 `722c5234cfc50def2b3de334f4326a7df777290eea23e5f530f946b17fce246f`; 44,212,006 bytes. Summary raw SHA-256: `3c1bb09e9b0a3e4f4c26aa3b7dbc0d76137916a462ab321de5d60768947dde0b`. Index raw SHA-256: `6d10effd59b1922627b3a7717e38b4fbf11e496fc9959e4a7fea8d7aae5efadc`. Controller: `master_controller_source_20261013_v29/runtime`.

The plan registers 98 studies, 62 logical factories, 80,850 atomic model-fit groups, 1,082 selectors, 43,473 banks/evaluations, 43,362 CAL jobs and 88 author CLI groups. Its 18 distinct dataset names include aliases and composite author rosters; this is not a deduplicated provider-dataset count. Original-paper experiments are excluded. Original GNNM has 4,516 comparison fits on new data.

| Worker | Registered fit groups | Metadata HPO parent frontier | Separate protected source fits |
|---|---:|---:|---:|
| Allocation / GPU 0 | 34,083 | 144 | 12 |
| 18.77 / GPU 0 | 39,859 | 144 | 62 |
| 18.77 / GPU 1 | 6,908 | 36 | 2 |

Fit groups are not equal amounts of work. Registered costs include 83,637 paid optimizer starts, 83,040 physical slow bodies, 312,057,212 full-budget optimizer updates and 385,770,612 known optimizer step calls, with 124,350,000 native update caps. Complete step count is bounded rather than determined. These are planned costs, not execution observations or an ETA.

## What was checked

47 of the old 55 source families pass initialized CPU construction and capture. Import-only namespace repairs and two narrow plain-state/delegated-clock repairs corrected actual defects; their original failures remain. These witnesses do not certify mature optimizer state, all branches or CUDA memory.

Eight old families still require their actual paid context or device admission: label stage (918 fits), landmark memory (3,140), Polyformer recompute (96), shared link replay (249), preaggregation growth (147), curvature pipeline (762), SeHGNN (1,230) and typed context (45): 6,587 fits. They are resource/context gates, not assumed loader faults. Correct live execution supplies registered parents; preparation must not fabricate them.

Seven additions are separately unpriced at maximum mature state: configured relation (12), completed private hop (42), structural interaction (33), cardinality (3), conditional pattern (9), cardinality controls (15) and postfit (1,251): 1,365 fit groups. Source/engineering evidence is not physical qualification. The 30 typed scaler dependencies are correctly carried by plan barriers; the suspected dispatch issue was cleared and no repair was made.

## Known bytes and retention

| Item | Allocation | 18.77 |
|---|---:|---:|
| Banks | 19,158 | 24,315 |
| Raw banks plus IDs metadata bound | 200,169,515,658 B | 279,138,743,322 B |
| Largest single raw bank bound | 441,557,662 B | 93,185,502 B |
| Last guarded free disk | 1,147,005,698,048 B | Unverified |
| Last guarded available RAM | 1,881,615,679,488 B | Unverified |

Bank bounds assume Float64 values, 16 views (20 for Bayesian 4+16 controls), long IDs and uncompressed registered input/query tensors. All registered input metadata was found; no target arrays were decoded. Source-schema confirmation remains necessary. Checkpoints, caches, traces, headers, failures and atomic writes are additional.

The integrated prospective policy bounds 892 metadata HPO cohorts, 14,360 releasable search fits and 76 protected paid prerequisites. At most two open cohorts per GPU are allowed; source-pinned selectors must finish, owners must be terminal, and exact deletion intent/completion/tombstones and absent binaries must agree. Failures remain reserved. Protected76 was rebound to final v29. These are conditional count bounds, not complete byte bounds; no old history was deleted.

Recovery embeds selected and distinct prefix bytes. A conservative active frontier is `(5+3p)S + 2H(K) + TRACE`, with mature full state `S`, prefix count `p` and growing header `H(K)`. Completion adds final-training and serving authorities plus serialization coexistence. Nested local models/Adam survive in Polynormer-family serving; actual cloned tensor bytes must include alias duplication. Curvature retains paired warm states and five trial snapshots, while paid parents remain live in RAM. Initialized NSD does not price populated native L. Postfit needs its fixed feature/neighbor caches; its DirectHead uses LBFGS history, not Adam's two moments. Remaining trace/header and learned-cache growth must be budgeted.

## Launch limits

Whole-host storage sufficiency, complete mature state, CUDA admission, reliable ETA and efficient host balance have not been established. The allocation's free disk cannot be declared sufficient from banks alone. 18.77 was not contacted under the allocation-only preparation instruction. Two jobs per GPU is the maximum concurrency: their combined VRAM, RAM and disk reservations must fit before both start. Human verification and explicit GO remain required.
