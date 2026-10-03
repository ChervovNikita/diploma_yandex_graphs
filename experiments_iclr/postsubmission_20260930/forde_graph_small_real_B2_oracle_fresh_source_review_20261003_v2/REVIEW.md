# Independent source review of sealed Q03 V2

The V2 source is accepted for the fixed full-Amazon B2 comparison design, pending a separate root release, current availability approval and actual numerical execution. No remaining source blocker was found. V1’s rejection is preserved; its identified comparison-diagnostic failure-custody defect is remedied by the only executable change in V2.

This is an independent source-only recheck. No numerical library was imported, numerical execution performed, GPU accessed, data/checkpoint or runtime binary opened or hashed, remote operation used, author source changed, subagent spawned, isolation used, sudo used, or PDF compiled. Standard-library text/JSON/AST and source-byte hashes were used. Root authored the V2 successor; the reviewer inspected it independently. No manuscript, predictive, novelty or scientific-performance verdict is made.

## Packet and diff custody

Reviewed packet: `forde_graph_small_real_B2_oracle_preparation_20261003_v2`.

Manifest SHA256: `6efd04c8913ae7bfcc9fded06493f3842ddc25e832fe57209428a744d598183c`.

Executable SHA256: `6d1e009f8520752d23321e0342d217d4dc1cf3bfe6a2005b5f1f0be159685fcb`.

All nine packet payload descriptors and 104 source/metadata descriptors match; inventory and SEAL match. The independent full-executable AST comparison is identical to reviewed V1 except for `metric`. `PLAN.json`, `READ_SCOPES.json`, `RELEASE_TEMPLATE.json` and `SOURCE_BINDINGS.json` are byte identical to V1. The executable diff agrees with the sealed amendment/patch and adds only derived finite/positive comparison guards. V1’s executable still matches its preserved manifest; its separate rejected REVIEW.md/JSON are unchanged.

The exact source dependencies previously traced remain unchanged and were hash rechecked: Q02 profiler helper, warm common runtime helper, native adapter, native PolyFormer bodies, token/preprocessing source, TRAIN roles/core imports, pinned PyG graph-normalization conversion/loop source, materialized graph reference and direct streamer. Q01 actual source/threshold/process metadata and Q02 resource-only receipt/exit metadata remain bound. Declared runtime binaries and data descriptors were read as metadata only.

## Q03-F01 is resolved

V1 checked finite inputs but could store an overflowing subtraction/scaled maximum. The strict JSON writer would then reject that record, and the retained nonfinite diagnostic could also prevent final failure publication.

V2 `run_small_real_oracle.py:194-209` verifies finite absolute error, finite scale, strictly positive scale and finite scaled error before constructing or returning any metric record. Thus finite input subtraction overflow and division overflow raise before a caller can attach a contaminated row to the case or receipt. A normal exception then follows the existing stage/main catch/finally path with a JSON-safe receipt. All metric callers use this same guard. After the guards, tensor maxima and their Python floats are finite. The arithmetic expression remains `abs(a-b)/(atol+rtol*abs(reference))`; no threshold, dtype, reference or acceptance criterion changes.

This resolution was checked by source/AST/diff inspection, not by executing a numerical failure fixture. Ordinary filesystem failures or native termination remain subject to root external custody; the source is not a crash-proof external supervisor.

## Independent full-X chain and comparison coverage

`independent_full_x` (234-256) makes four separate complete CPU X leaves of shape `[24492,300]`, rebuilds the native sparse monomial tokens from each live leaf, checks complete operator/token bytes, selects the two fixed target rows and transfers them differentiably to the CUDA predictor. Each selected raw-score scalar is differentiated back to the full CPU feature matrix with `create_graph=True`; all eight resulting full-X derivatives per recipe transfer differentiably to CUDA. The token builder retains X through every CPU sparse recurrence. Hashes inspect detached copies without replacing the live chain.

The pinned predictor is row local, so selecting token rows after complete graph recurrences retains the intended score dependency on every original X feature. The separate reference does not construct CUDA sparse recurrences or all-node CUDA predictor activations. The full-X materialized objective (259-272) uses the pinned normalization, stopped reference directions, detached column median bandwidth and R. The coefficient direct reference uses the same live q as the unchanged streamer; dR/dq is correctly qualified against that source. The live-X pathway independently supplies complete-X and private-force comparisons.

The fixed value checks in `one_recipe` cover every selected logit, every complete `[4,2,24492,300]` derivative coordinate, every normalized coordinate, and every norm2/similarity/D/h/kernel/R element. Streamer directions are inspected for both complete target slices. CE and CE+R values are also compared. Force checks cover dR/dq versus materialized coefficient direct, every private derivative of R versus both references, and every private derivative of CE+R versus both references. The six private endpoint calls use `allow_unused=False`; missing paths stop. Finite zeros are permitted. The custom backward's `once_differentiable` restriction fits the requested first parameter derivatives through the live q mixed-derivative chain; further differentiation of that custom backward is not admitted.

## Fixed states, thresholds and access scope

Both source_defaults and roman_mono run in fixed order, with M4, B2, native float32 and the exact Q02 seed17 common/sinusoidal R/S state. Initial state hashes and private-name order must equal Q02, and complete CPU operator/token hashes must equal actual Q02. CPU descriptor/model transfer to CUDA is checked byte exactly. All common parameters are frozen and every module is eval. Every R/S tensor is included: 58 in source_defaults and 210 in roman_mono, including both factors at lin1, lin2 and lin3. No common bias override, checkpoint, optimizer or donor is used.

Only the allowed public graph and compact split0 TRAIN files are opened. The full native TRAIN role/order and all Q02 selected IDs/channels are reconstructed and compared before selecting `[10052,2874]`, channels `[2,3]`. VAL/TEST mask key names are checked; their arrays and heldout label files are not decoded. Preservation hashes only source/metadata and the same allowed two payloads. No raw labels, other TRAIN payload, VAL/TEST labels or checkpoint payload is opened for preservation.

The review accepts the exact Q01 float32 thresholds prospectively for these fixed endpoints: value absolute `3e-6`, relative `3e-4`; force absolute `1e-4`, relative `3e-3`. Every element must satisfy `abs(a-b)/(atol+rtol*abs(reference)) <= 1`. The full-X derivative is treated as a value endpoint as in Q01. Source admission checks exact Q01 threshold metadata. No real-graph outcome was used by this review, no numerical passage is asserted, and no widening is permitted.

## Runtime, mutation guards and costs

The exact source/metadata inventory, separate release, Q01 adoption/status, resource-only Q02/exit0, positive root free-memory precondition and availability approval are admission gates. Numerical imports require a fresh ordinary interpreter; retained Python3.12/package source/binary file metadata, deterministic settings, native float32, autocast/TF32 off and physical device UUID are checked. The source uses ordinary in-process library settings and no subprocess, transport or isolation. It does not change other jobs/devices.

Fixed finite loops, a fresh output, and no retry/fallback code keep the prospective recipe/target schedule intact. Native member forwards refuse reentrancy and restore selections in `finally`. Immutable descriptor validation checks complete cache bytes during forward/backward and at final custody. Final checks cover state bytes, eval modes, exact trainable R/S set, unwritten parameter/X gradients, idle member selections and forward state, X features and reference operator/token bytes.

The schedule per recipe remains four complete live-X builds, eight score-to-X derivatives, three streamed backward calls and six complete private-gradient endpoints. Branch receipts require two forward target reconstructions and three backward calls with two target reconstructions and two normalization/A VJPs each, and zero self-direction cotangents. Ordered-pair/global-bandwidth work is recorded by the unchanged streamer.

Synchronized stages charge CPU cache/reference construction, differentiable transfers, materialized arrays, retained mixed graphs, full comparisons/hashes/diagnostics and reference/allocator cleanup. Receipts record CUDA allocated/reserved peaks, current free/total device memory, wall time and cumulative process peak RSS; preservation has explicit time and whole-body time includes runtime/preservation. Primitive array bytes are correct but do not bound total process memory. Root current availability approval and external startup/admission/exit custody remain required.

## Conditions and limits for root adoption

The source guard verifies an external REVIEW.json descriptor and root adoption booleans but does not parse the review’s exact packet binding, accepting disposition or freshness. Root must bind and adopt this actual V2 review, with this exact manifest. Admission/output creation occur before the internal receipt try/finally and therefore require root external failure custody, as the packet states. Runtime binding remains a declared-file/settings/device scope rather than a full linked/transitive runtime closure. This review did not inspect current hardware availability or execute deterministic kernels.

All three graph pathways consistently use normalization epsilon `1e-24`. The referenced upstream forde_train_forde.py default is `1e-12` (line68, applied at line290). Source acceptance here concerns equality to the pinned graph adaptation; unchanged upstream default-epsilon equivalence is not established.

Any later numerical passage is restricted to the fixed full-graph B2 CPU-cache/CUDA endpoints for these two source-fixed states under the unchanged thresholds. B128 GPU equality, CUDA sparse recurrence equality and trained-donor/predictive/scientific outcomes remain open. This review supplies source acceptance only and does not authorize launch by itself.
