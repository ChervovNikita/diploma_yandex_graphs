# Independent engineering source review — collab TRAIN resource preparation v1

## Scope and conclusion

Reviewed preparation: `graph_ncNC_collab_resource_epoch_preparation_20261003_v1`, manifest SHA256 `8661011fe87a9980acf806678483e7ed6f903628024813dff07c8f00b1eec5f8`.

This is a source and custody review. I read all seven Python modules, their preparation/admission/accounting documents, the relevant pinned portable and native sources, the bound CPU engineering evidence and oracle source, and root's exact copied installed PyG sampler source. I used file reads and SHA256 checks. I did not execute or import the package, prototype, native reference, sampler, data loader or numerical code; access data, labels, caches, checkpoint payloads or study outcomes; run remote commands; or change any original source. The two files in this child directory are new review artifacts.

**One substantive source blocker remains before GPU parity release:** both stages can retain a successful qualification status after their final accounting receipt fails, including a failed final CUDA synchronization. A preserved successor should make successful accounting collection part of the pass gate. Source inspection otherwise supports the declared TRAIN-only native stream route, graph/batch semantics, native one-member gradient correspondence plan and four-member own-versus-post-clamp-pool construction. It does not certify GPU arithmetic, resource feasibility, a scientific result or redistribution clearance.

### Prior context and independence

I inherited prior NCNC paper/repository context and earlier literature comparisons involving completion and ensemble association, including PENCIL, IECNC and E-GAE. Earlier contextual outcome summaries were available in the surrounding session; I claim no total historical outcome blindness. This review independently traced the supplied source flows without reopening study outcomes or treating earlier reviewer verdicts, static source checks or CPU PASS metadata as a substitute for the trace. The CPU receipts read here are explicitly engineering certificates. This is not a fresh paper review or a novelty assessment.

## Finding requiring a successor

### F1 — final accounting failure can leave qualification successful (P1)

**Sites:** `resource_epoch.py:136–147`, `run.py:66–77`; relevant receipt operation `runtime.py:108–118`.

The resource loop sets a twin's status to `COMPLETE_NATIVE_EPOCH_FINITE` at line 136 before collecting `meter.receipt()` at line 142. Receipt collection first synchronizes CUDA and then retrieves memory/RSS/counter data. If that operation raises, lines 143–144 retain an error string but preserve the COMPLETE status. The family check at line 146 only tests twin status and batch count. Thus two completed batch lists can result in `BOTH_COMPLETE_NATIVE_RESOURCE_EPOCHS_FINITE` and `all_required_checks_passed=True` despite missing accounting or a failed final synchronization.

The GPU stage has the same top-level issue. After `gpu_parity.run()` reports PASS and sets its checks Boolean, `run.py:69–73` catches final `meter.receipt()` failure without invalidating either value. The exit code at line 77 only checks that Boolean. `guards.py:123–130` can subsequently accept that successful GPU receipt; it does not reject an accounting error. This is a concrete reachable exception branch in the source, not an observed runtime incident.

The preparation promises retained concrete failures and measured resource accounting (`ACCOUNTING_AND_FAILURES.md:30–42`, `README.md:31–37`). Recording an accounting error while continuing to certify the stage is insufficient for that purpose.

**Repair:** retain accounting errors and any available partial counters, but force a non-pass stage/twin/family status and `all_required_checks_passed=False` whenever a required receipt or synchronization fails. Complete a twin only after its required accounting is collected. Successful exit should require the final admitted pass status as well as the checks Boolean. Apply equivalent invalidation to caught runtime errors so a previously set checks Boolean cannot survive a later failure. Preserve v1 and review/reseal the successor; a later failure-injection check should establish that receipt exceptions retain diagnostics and produce failure, without a smaller graph or a scope rescue.

## Native binding and semantic trace

The prototype manifest is `ac04bc7f6f6f37b86c50d38436aee111e987195c5f992816c4a5ef9d175887ae`. Its portable `prototype.py`, `graph_ops.py`, `native_reference.py`, `selected_state.py` and `qualify.py` hashes match the pinned bytes. The native custody pin identifies `GraphPKU/NeuralCommonNeighbor@11d597013750da17ce7468e344bec756a7af39a4`; preserved native model, utility, training and loader sources match their declared hashes. `guards.py:52–71` verifies the entire prototype manifest and the bound CPU evidence, and `guards.py:139–148` verifies the imported portable module directories. `native_reference.py:24–46` checks preserved native bytes and imports the exact model against the corresponding utilities. The reviewer did not perform these imports.

| Operation | Source correspondence and assessment |
| --- | --- |
| Full TRAIN graph and mask | Native `NeighborOverlap.py:42–63` masks supervision record IDs before constructing/symmetrizing a SparseTensor. Portable `graph_ops.py:19–39` removes the same records before reciprocal construction and uniqueness. Unmasked duplicate records may retain a supervised edge. This is native record masking, not guaranteed removal of every edge identity represented by the batch. |
| Encoder | Native `model.py:82–100,104–195` uses input dropout, independently dropped directed coalesced adjacency entries with rescaling, a GCN, LayerNorm/dropout/ReLU, and a residual only when input/output dimensions match. Portable `prototype.py:48–76` and `graph_ops.py:41–53` reproduce those mathematical operations, including diagonal replacement with ones and row-sum normalization. At 128→64 the residual is inactive. Decoder topology remains the original masked graph, independent of encoder edge dropout. Exact GPU accumulation is still prospective. |
| Decoder feature maps | Native `model.py:525–592` and portable `prototype.py:143–173` preserve xlin/xcnlin/xijlin/lin module indices, capture endpoint products before xlin, use xijlin before xcnlin, and retain native fixed-pt ptlin state. Shared W/bias plus private rank-one r/s, private normalization affine rows and beta are the declared four-member composition; they are not a claim that native NCNC itself shares members. |
| Candidate identity | Portable CSR enumeration (`graph_ops.py:55–89`) returns full common and left/right exclusive neighbor lists in query/node order. Native `utils.py:135–254` supplies corresponding overlap matrices with unit values when degree sampling is disabled. GPU parity compares exact COO row/column arrays for both outer positive and negative passes (`gpu_parity.py:101–117`). |
| Recursive completion | Native `model.py:717–779` passes already outer-transformed features to separate left/right depth-zero forwards under no_grad. Portable `prototype.py:168–205` retains the second full-node xlin path, native query orientation and left-before-right recursive order, with training dropout active. At splitsize=-1 even an empty query follows the full path. The portable depth-zero helper computes exclusive sets as additional enumeration work; only its common set enters base scoring. |
| Surviving gradients | Recursive score outputs and clamp weights are detached. The outer transformed feature tensor is retained, and weighted feature sums remain differentiable (`prototype.py:199–215`, `graph_ops.py:92–100`). No gradient through completion scores is claimed. Bound CPU test `qualify.py:132–153` separately checks surviving feature pullbacks. |
| Twin switch | `prototype.py:130–140,193–215` clamps each member's left/right score first, then either retains those weights or takes their arithmetic mean across members. Each member keeps its own transformed features and nonlinear decoder. Both modes execute the same member/scorer schedule. Outer decodes occur after all members' completion weights have been computed; this disclosed cross-member ordering is part of the composition. |
| Native objective/update | Native `NeighborOverlap.py:63–76,280–281`, portable `prototype.py:241–255`, and resource driver `resource_epoch.py:78–97` use one shared encoder pass, distinct positive then negative decoding, the sum of two mean negative-log-sigmoid terms, and two native-rate Adam groups. This loss is twice balanced mean BCE over equally sized positive/negative arrays and all members. No scheduler or optimizer step exists in GPU parity. |

One arithmetic-order difference is visible: the native completion feature sum is common + right + left, while the portable sum is common + left + right. They are mathematically equal, but floating accumulation need not be bitwise equal. The retained tolerance gate, rather than a source assertion, decides whether that implementation difference is acceptable on the admitted GPU. Native helper behavior in structurally exceptional empty cases is not repaired; if a full-scale native reference fails, that is a concrete failed qualification, not license to skip the case.

## TRAIN loader and installed sampler audit

`train_only_data.py:28–66` opens only exact admitted official TRAIN `train.pt`, raw features and raw edges. It uses `weights_only=True`, CPU placement and exact expected TRAIN keys; discards native weight/year after the authority check; retains all 235,868×128 float32 raw features; and checks compressed/decompressed custody hashes, acquired tensor hashes, endpoint bounds, finite features and the duplicate-preserving undirected TRAIN/raw record multiset. The complete coalesced graph is separately counted and hashed (`run.py:43–50`). The OGB source conversion is matched at `read_graph_raw.py:33–46,100–106` and `read_graph_pyg.py:25,34–36`. Using pinned counts avoids opening extra OGB count/accessor files. The tensor digest convention matches `cache_builder.py:17–21`.

I read root's copied installed sampler `graph_ncNC_77_runtime_qualification_root_20261003_v1/INSTALLED_NEGATIVE_SAMPLING.py`; its full-file SHA is `c04beecc5331144a2e10fdc3c66fbd8b4ac495f9dbdeb1649ae3cf047237b2f9`. The live function SHA reported by root is `0f26dd305f8c4443d0231a9ae09f591bd741d671c3e9ddb68f0debc07218112b`. The file hash was independently checked; I did not import the function to reproduce its inspect-source digest.

| Native default | Exact installed-source meaning |
| --- | --- |
| Two-argument call | Driver `train_only_data.py:79` matches native `NeighborOverlap.py:48`: raw ordered reciprocal `data.edge_index`, then declared node count. It supplies no count/method/undirected override. |
| Count | Sampler lines 71–77 use raw `edge_index.size(1)` as the requested count: 2,358,104 here, including duplicate entries. Input is not pre-coalesced. Duplicate nonself input entries also contribute to the rejection probability estimate. This is a requested/approximate count, not a guaranteed output count. |
| Pair universe | Lines 334–345 and 367–374 represent directed pairs from the n(n−1) nonself universe. `force_undirected=False` gives no reciprocal coupling. |
| Rejection and RNG | Lines 94–109 transfer positive indices to CPU, use Python `random.sample` through helper lines 301–305, reject supplied TRAIN entries via NumPy isin, and reject previously selected negatives on later tries. At most three tries occur. No future positives are consulted. NumPy supplies membership computation, not a sampling RNG draw on this route. |
| Stream pairing | The full negative draw is fresh each epoch. The native CUDA `PermIterator` uses a GPU randperm (`utils.py:8–32`). Both arrays are indexed by TRAIN record IDs, so the surplus negative draw is native overhead. The incomplete shuffled tail is dropped. The driver hashes the entire negative draw, full permutation and tail, checks enough negatives to index every TRAIN record, and validates self/TRAIN/bounds exclusion (`train_only_data.py:80–97`). |

Python, NumPy, Torch CPU and single-device CUDA RNG state are captured/restored (`runtime.py:53–79`). This includes the actual Python sampler RNG and CUDA permutation/dropout RNG. Both twins restore the same initial state and epoch RNG, draw fresh streams, compare complete identities with parity and each other, and compare final RNG identities (`resource_epoch.py:22–60,118–120`). I found no sampler-default substitution or future-positive filter.

## GPU arithmetic gate

`gpu_parity.py:73–82` uses the exact native collab constructor and a one-member portable recipe, then copies native weights, normalization, beta and fixed buffers through `prototype.copy_native_unit_member`. It builds both full masked graphs, checks exact topology and outer candidate order, and uses the first complete shuffled native 65,536-record batch. All nodes/features remain; both positive and native negative supervision are scored from one encoder pass per model/profile.

For evaluation and training separately, the driver zeros gradients, restores identical captured RNG, runs complete native and portable forwards/backward, and compares positive logits, negative logits, TRAIN loss and all-node input gradients. The mapping has exactly 31 corresponding native parameter tensors: encoder four, beta one, xlin six, xcnlin six, xijlin four, lin six, and ptlin four. Four native fixed-pt gradients remain absent; every active corresponding gradient is numerically compared. Private normalization selects its sole member row for the one-member mapping. Additional portable factor gradients are checked for presence/finiteness; unused ptlin factors remain absent. There are 70 required entries: two profiles × (four tensor/loss comparisons + 31 parameter comparisons). The stage also requires exact post-forward/backward RNG identity and no optimizer updates (`gpu_parity.py:122–191`).

The unchanged gate is atol=rtol=128×float32 epsilon (about 1.526×10^-5), finite float32 tensors and matching shapes. It tests maximum elementwise discrepancy on a fixed full native batch, not statistical equivalence over all batches. No tolerance search appears. The all31 mapping is stronger than checking only selected decoder gradients; it does not supply native counterparts for the new r/s factors, a native four-member ensemble equivalence claim or a mathematical GPU error bound. The pinned separate float64 factor derivative oracle and twice-balanced-BCE gradient oracle cover their stated synthetic claims (`supplemental_oracles.py:35–103`); the CPU certificate explicitly does not claim native float64 encoder parity. Numerical GPU qualification remains unexecuted by this reviewer.

## Complete epoch work and measurement

The native iterator retains 17×65,536=1,114,112 supervised records and drops 64,940 of 1,179,052. This is explicitly not all-positive coverage. Each mode begins from an identical hashed four-member state and a newly created empty Adam optimizer. The explicit complete path is 17 batches per mode; the family requires both. There is no branch for graph/query reduction, candidate caps, degree sampling, split adaptation, skipped batches, changed precision or tolerated nonfinite state.

The expected per-batch work map (`resource_epoch.py:12–18`) matches the traced four-member execution:

| Work per batch | Required count |
| --- | ---: |
| Shared full-node encoder | 1 |
| Outer / recursive enumerations | 2 / 16 |
| Outer / recursive full-node xlin paths | 8 / 16 |
| Full-node xlin learned linear maps | 16 + 32 |
| Recursive scores / native clamps | 16 / 16 |
| Outer / recursive feature aggregates | 24 / 16 |
| Outer / recursive nonlinear decodes | 8 / 16 |
| Backward / Adam update | 1 / 1 |

The runtime temporarily wraps actual prototype enumeration, aggregation and clamp globals and instance recursive/xlin/decode methods, restores those hooks in finally, and records candidate counts without arrays (`runtime.py:121–197`). The encoder counter is incremented at the explicit encoder call. Each xlin wrapper requires 235,868 input rows. Per-batch deltas, 18 enumeration records, 131,072 outer queries and four times the combined outer-exclusive recursive query population are required (`resource_epoch.py:98–112`). Finiteness covers intermediates, active gradients, all parameters and Adam tensors; fixed-pt gradients must stay absent. These checks expose work, but their success is still a runtime fact.

Timers synchronize at boundaries and are nested inclusive. Model setup, data/sampler/permutation guards, graph rebuild, full decoder work, loss, serving, backward/update, finite guards, hashes and internal state serialization/writes are charged in enclosing wall times. Explicit observable transfers and synchronizations are counted; sampler/sparse/dynamic-shape/serialization library internals are latency-charged, not claimed as exact counts. CUDA peaks reset per twin and include resident data/live state; Linux host ru_maxrss remains the cumulative process high-water mark. There is no warm-up exclusion or cross-method speedup claim.

The mean raw-logit serving rule is unchanged (`prototype.py:235–238`, `resource_epoch.py:89–92`). Resource state files contain mode-bound model/Adam/RNG engineering artifacts with restricted file permissions and hashes. They are not selected predictor donors. No receipt writes labels, query arrays, score vectors, raw loss values or quality metrics; arithmetic discrepancy metadata are engineering comparisons only.

## Limits and root release conditions

1. **Repair F1 in a preserved sealed successor before release.** This is the source blocker identified by this review.
2. Root's live metadata receipt, SHA `5bf8d386bbcbfdfbad75e4a71e5b9619204fa2cf48880113f913708e1feb1668`, reports actual imports on 18.77: Python 3.12.11, Torch 2.7.1/CUDA 12.6, PyG 2.4.0, torch-sparse 0.6.18+pt27cu126, torch-scatter 2.1.2+pt27cu126, NumPy 1.26.4 and pandas 2.2.3, one A100 80GB. It explicitly reports no sampler/data/model execution and no GPU numerical qualification. I read this metadata; I did not independently access that host.
3. The driver binds interpreter, versions, selected imported Python sources and sampler file/function hashes, but its runtime identity does not bind the shared-library hashes now listed in root's `runtime_binary_files` metadata. Exact sparse/scatter binary continuity between stages therefore needs root's immutable-environment/custody enforcement or a successor admission check. This is a provenance limit, not evidence that a binary changed or a kernel is wrong.
4. Preflight (`run.py:18`, `guards.py:75–136`) precedes both output-directory creation and the runtime receipt try. A refused/malformed/source-invalid admission can exit without a QUALIFICATION.json; root must retain its concrete launcher exception/output separately. The documentation accurately calls these preflight failures but must not be read as promising a runtime receipt for them.
5. `total_process_seconds_including_stdlib_admission` is taken before the final QUALIFICATION.json write (`run.py:74–75`). Internal state writes are timed, but the final reporting write is outside that number. Serialization's hidden device copies are within elapsed time, not enumerated exactly by explicit transfer counters. These measurement limits should accompany any engineering resource report.
6. Full candidate populations, GPU arithmetic/memory behavior, complete finite epochs and serialization completion remain unknown until root's admitted runs. A full-scale failure is the result to retain; source/CPU preparation cannot replace it. NCNC redistribution permission remains unresolved and was not adjudicated here.

The own-versus-pool comparison means preserving or breaking each member's completion-weight association with its own features/decoder under identical scoring work. It pools already clamped probabilities, not raw scores, node embeddings or final served logits. Native independent NCNC already keeps its own completion/decoder association. These sources establish a declared shared/factorized composition and control; they do not establish a new completion primitive, novelty, calibration, advantage or predictive quality.

## Evidence inventory

Exact hashes and source-read coordinates are recorded in `REVIEW.json`. All seven module SHA256 values were independently checked against the sealed manifest. Supporting prototype/native/OGB/oracle hashes and root's sampler-copy and live-metadata hashes were independently checked. No preparation source checker or historical verifier was rerun. No original file was modified.
