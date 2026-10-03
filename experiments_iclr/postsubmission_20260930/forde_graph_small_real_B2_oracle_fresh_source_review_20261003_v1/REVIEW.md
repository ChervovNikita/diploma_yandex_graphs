# Independent source review of Q03 full-Amazon B2 oracle

The comparison design covers the fixed requested endpoint, but this source review does **not** accept the packet for root numerical release until Q03-F01 is remedied. One failure-custody defect is present: nonfinite diagnostic arithmetic can prevent publication of the final FAILED receipt. No core mathematical or private-derivative coverage defect was found in the inspected comparison paths.

This is a source-only review. No numerical libraries were imported, numerical execution performed, GPU accessed, data/checkpoint or runtime binary opened or hashed, remote operation used, or author source changed. No subagent, isolation, sudo, or PDF compilation was used. The reviewer read text, JSON metadata and source ASTs and verified text-byte descriptors using the standard library. Prior reviews were metadata context and did not supply this verdict.

## Exact reviewed packet

Packet: `forde_graph_small_real_B2_oracle_preparation_20261003_v1`.

Manifest SHA256: `4e37a950283cab600caf9004060c51269bdcaeee2ba41cc7bd446d109bc44d19`.

Oracle SHA256: `3087439473770b768e108cc2b0d7f0e6f2c914eb671c052e5b99f88116970a77`.

All seven packet payload descriptors and all 104 declared source/metadata descriptors match. The seven-file inventory and SEAL manifest hash match. Exact invoked helpers and native source paths were traced: Q02 profiler helper, warm common runtime/configuration helper, native adapter, token builder, preprocessing, native PolyFormer bodies, row-role functions, their local core imports, pinned PyG normalization/conversion/loop source, direct materialized graph reference and direct streamer. The Q01 runner/metric, actual Q01 receipt and process metadata, actual resource-only Q02 receipt/exit metadata, and upstream FoRDE formula/protocol text were also inspected. Runtime binaries and NPZ descriptors were read only as metadata.

## Genuine independent full-X reference

`run_small_real_oracle.py:229-251` makes a separate `[24492,300]` CPU X leaf for each of four members, calls the native monomial builder from that leaf and stacks the complete token bank. It independently checks operator and complete token bytes against the bound native cache. It selects the two fixed target rows only after complete CPU recurrences, transfers those rows through live `.to(device)`, runs the same CUDA member predictor, and differentiates each selected raw score back to every original CPU feature with `create_graph=True`. All eight complete derivatives per recipe transfer to CUDA through live `.to(device)` before the materialized objective.

The native token builder (`native_source/tokens.py:87-112`) normalizes only graph data through SciPy and retains X in every sparse recurrence. No detach/no_grad is introduced in the reference X-to-score or mixed-derivative chain. Hashing uses detached inspection copies; those copies do not replace live tensors. The predictor is row local in its pinned bodies: attention, LayerNorm, FFN and classifier operations do not pool nodes. Selecting token rows therefore keeps the intended full-X score dependency. Actual kernel and second-order support remain numerical execution gates, with no fallback in this source.

The materialized reference (`254-267`) retains member, target, node and feature axes and applies the pinned graph normalization, stopped reference directions, column median bandwidth and R. The coefficient direct reference uses the same live q as the unchanged streamer. That reference establishes dR/dq against the pinned materialized coefficient formula; it is not an independent q parameterization. The separate live-X reference supplies the independent complete-X and private-force endpoints.

## Fixed state, precision, targets and derivative coverage

The recipes run in fixed order: `source_defaults` then `roman_mono`. Seed17 common state and the sinusoidal every-R/S initialization reproduce Q02 without a common bias override. Initial complete state hashes and the ordered private-name sets must equal actual Q02. Complete native operator/token hashes must also equal actual Q02. The B2 row powers are rebuilt on CPU and transferred with the immutable descriptor hashes; they are not claimed to have the B128 row-power hash.

The compact official split0 TRAIN projection reconstructs the native TRAIN column, fixed stratified fit/control roles and complete ordered Q02 target/channel list. Only after that comparison does Q03 take `[10052,2874]` and channels `[2,3]` (`450-460`). Neither heldout labels nor checkpoint payloads are opened. `arrays.files` checks VAL/TEST mask key names; the allowed public graph is hashed as a whole file, and only features, edges and TRAIN mask are decoded. Preservation hashes the same allowed graph and split0 TRAIN files, plus source and metadata.

Every existing affine R/S tensor is enabled, including `lin1`, `lin2` and `lin3`. There are 58 private tensors in source_defaults and 210 in roman_mono. All common parameters remain frozen, all modules are eval, and parameter/X `.grad` fields remain unwritten. Parameter gradients use `allow_unused=False`; a missing path is an error. Finite zero gradients are permitted and no artificial nonzero-force requirement is introduced.

The source compares all 40 selected logits, every `[4,2,24492,300]` full-X derivative coordinate, every normalized coordinate (including the streamer’s complete directions one target at a time), full norm2/similarity/D/h/kernel/R arrays, CE and CE+R. It compares dR/dq with the same-q direct source, then every element of every private derivative of R and separately CE+R against both the coefficient and full-X references (`336-385`). The streamed custom backward is used only for first private derivatives through the live q graph; its `once_differentiable` restriction is consistent with that endpoint.

Value thresholds remain absolute `3e-6`, relative `3e-4`; force thresholds remain absolute `1e-4`, relative `3e-3`. The full-X pullback uses Q01's value-endpoint classification. The criterion is the exact every-element `abs(a-b)/(atol+rtol*abs(reference)) <= 1`, with the reference defining the relative scale. The Q01 float32 threshold metadata must match. These thresholds are prospectively adopted for real B2 and CE+R; this source review neither validates passage nor widens them.

## Q03-F01 — failure receipt can become unserializable (P2)

At `194-204`, `metric` verifies finite alternative/reference inputs but does not verify the derived subtraction or scaled error before storing their maxima as Python floats. Finite float32 operands can produce an overflowing subtraction. An Infinity diagnostic can therefore be returned in a comparison record even though every input is finite.

At `359-363` (and similarly `379-385`), comparison records are attached to the case before `profile.save`. That writer uses `json.dumps(..., allow_nan=False)` (`144-147`). If a derived diagnostic is nonfinite, the save raises. The error is caught by the main body and status is changed to FAILED, but the contaminated case still belongs to the receipt. The final write (`493`) also uses `allow_nan=False` and raises again, before the explicit return path. A previous local receipt may remain RUNNING/ENTERING and no complete FAILED receipt is published.

This is a statically identified failure branch, not an observed numerical outcome, and does not show that the fixed endpoint will overflow. It nevertheless contradicts the promised ordinary failure-receipt custody. Reject nonfinite derived error/scaled values before storing records, or represent failed diagnostics in a JSON-safe finite/null form while rejecting the comparison. Keep the thresholds and endpoint fixed, seal a successor source packet and obtain a fresh review. The reviewed author source was not edited.

## Runtime, mutations and costs

Admission precedes numerical imports and requires exact packet/source descriptors, separate root release, Q01 adoption/status, resource-only Q02 and exit0, current availability approval and a positive free-device-memory minimum. The helper rejects already loaded conflicting numerical/native modules, binds retained Python3.12/package files and deterministic float32 settings, disables autocast/TF32 and verifies the physical GPU UUID. This is ordinary in-process execution with no source subprocess, transport, isolation or changes to other jobs/devices.

Fixed loops and a fresh output prevent automatic retry or adaptive batch/device/precision/backend changes. Native member forwards are nonreentrant and restore member selections in `finally`. Bound cache validation checks byte custody during streamer forward/backward and at final custody. The final model, frozen/trainable sets, modes, member selections, X leaves and reference operator/token bytes are checked. The source does not implement candidate updates or optimizer steps.

The per-recipe source schedule is four complete reference builds, eight score-to-X derivatives, three streamed backward calls and six complete private-gradient endpoint calls. Streamer accounting records two forward target reconstructions and three backward calls with two target reconstructions/normalization VJPs each; every self-direction cotangent must be zero. Pinned source loops also account for complete ordered pairs and global median/repulsion work. These are meaningful branch counts for this fixed endpoint.

Synchronized stage records cover native cache reconstruction, live reference builds/transfers, retained materialized and mixed graphs, complete tensor diagnostics/hashes/comparisons and cleanup. CUDA allocated/reserved peaks and device free/total memory, wall time and cumulative process peak RSS are recorded. Primitive array sizes are correct but are not a total memory ceiling; CPU token stacks and several graphs coexist. Whole-body time includes runtime and preservation. Root external elapsed/exit custody is still required for startup/admission and native termination.

## Root responsibilities and scope limits

The review descriptor guard (`104-109`) only checks an external file named REVIEW.json and root booleans; it does not parse the review’s packet binding, accepting disposition or freshness. Root must adopt the actual fresh review of the exact packet, and cannot infer those semantics from descriptor verification alone. Admission/output creation (`429-430`) precede the local receipt try/finally; root external custody must cover those failures, as the packet explicitly states.

Current availability was not inspected. Declared runtime file/settings/device admission is not a complete linked or transitive dependency closure. Numerical passage, exact actual resource peaks and kernel support remain unobserved. Any future success is confined to the two fixed full-graph B2 CPU-cache/CUDA endpoints; B128 equality and CUDA sparse recurrence equality remain open.

The pinned graph materialized reference, streamer and Q03 reference consistently use normalization epsilon `1e-24`. The referenced upstream `forde_train_forde.py` defaults to `eps=1e-12` (line68) and uses eps in normalization (line290). Thus equality here would establish the pinned graph adaptation's behavior, not unchanged upstream default-epsilon equivalence.

No trained-donor, predictive, novelty, scientific-performance or manuscript verdict is made. The independent source disposition is **remedy Q03-F01 and re-review before root numerical release**.
