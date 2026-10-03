# Work and failure accounting

Every native full minibatch removes **record IDs** from the complete TRAIN supervision list before symmetrization/coalescing. A duplicate record that remains can retain an edge. All 235,868 nodes and raw 128-dimensional features remain. The decoder sees this original masked graph, while the all-node encoder separately performs native directed stochastic adjacency dropout and GCN normalization. Graph masks, reconstruction, uniqueness/sorting/CSR, transfers, and implicit synchronization latency are paid by enclosing measured elapsed times.

The positive and negative decoder passes are sequential and distinct. Each pass enumerates all common and exclusive candidates. For each of four members, it captures the outer endpoint product before xlin, transforms all nodes, invokes left and right depth-zero recursive scorers on the already transformed features, applies each native clamp, routes private weights or the mean of clamped weights, then performs weighted aggregates and nonlinear decoding. Recursion runs under `no_grad` while training dropout stays active. Empty recursive query lists still perform the full-node path because split size is -1.

## Exact expected work per full minibatch

| Work | Count |
|---|---:|
| Shared all-node encoder | 1 |
| Outer query enumeration | 2 |
| Outer full-node xlin paths with gradients | 8 |
| Recursive full-node xlin paths under no_grad | 16 |
| xlin full-node learned linear maps | 48 |
| Recursive candidate-query enumeration/base scoring | 16 |
| Native member completion clamps | 16 |
| Outer common/left/right feature aggregations | 24 |
| Recursive common-feature aggregations | 16 |
| Outer member nonlinear decodes | 8 |
| Recursive base nonlinear decodes | 16 |
| Native backward and two-group Adam update | 1 each |

The driver checks actual wrapped execution against this work map. It checks that recursive scored queries equal four times the combined exclusive-candidate population of both outer passes. It records all 18 query enumerations per minibatch, each with query/common/left/right counts. Full-node xlin row work is 24 × 235,868 per batch; xlin linear-map row work is 48 × 235,868, plus the one encoder projection. No candidate arrays or tensor outputs are stored in receipts.

Per complete twin epoch: 17 encoders, 34 outer enumerations, 136 outer full-node xlin paths, 272 recursive full-node xlin paths, 272 recursive scorer/enumeration paths, 408 outer and 272 recursive aggregations, and 17 backward/Adam steps. Full candidate populations are runtime facts and remain unknown until root-admitted execution.

## Measurement boundaries

The process total starts before standard-library source/evidence and custody hashing. Separate setup times cover compressed/uncompressed file hashing, cold TRAIN/CSV reads, raw reciprocal reconstruction, tensor hashing, duplicate-preserving multiset validation, full data transfers, and complete graph validation. GPU parity setup and native/portable all-node/decoder/backward passes are measured separately.

Each twin starts a fresh meter and GPU peak counters after device selection, restores the same initial state, then times the native default negative draw, guards, permutation and hashing. Per-batch timers cover graph rebuild, encoder, complete positive/negative decoders, loss construction, serving pool, backward, gradient/parameter/Adam finite guards, and optimizer step. Wrapped xlin, enumeration, recursive scorer, clamp, aggregation, and decode timers expose inclusive internal work. Final model/Adam/RNG serialization and restricted writes are also timed. Overall twin and process wall time includes instrumentation, state/stream hashes, guard checks, and synchronization overhead; inclusive sub-times must **not** be summed as disjoint phases.

CUDA synchronization occurs at each timer boundary. Explicit timing/finite scalar synchronizations, known graph-row `.item()` synchronizations, and source graph validation synchronizations are counted. Explicit full data/model/gradient/stream host-device transfers are counted where their sizes are observable. Native sparse/dynamic-shape/sampler/serialization libraries may perform additional internal transfers or synchronizations; their latency is charged in enclosing times but their opaque counts are not claimed exact. Internal serialization bytes and total internal artifact size are reported. Linux process `ru_maxrss` is a cumulative high-water mark (KiB converted to bytes), not a per-twin isolated host-memory peak. CUDA allocated/reserved peaks are reset per twin but include the resident full dataset and live state at that boundary.

No warm-up excludes inconvenient work. No measured loss or predictive comparison selects a recipe, batch, candidate scope, initialization, or subsequent run. These measurements qualify the actual portable path and its native correspondence; they do not establish an optimized implementation or cross-method speedup.

## Failure retention

Wrong/admitted source, interpreter, runtime, GPU, sampler, data or CPU evidence fails preflight or qualification. Native negative shortage (fewer rows than TRAIN record indexability requires), graph/candidate/RNG/gradient arithmetic mismatch, nonfinite tensors/state, allocation failure, illegal CUDA access, reduced node/query/work coverage, incomplete epoch, or serialization failure prevents a resource pass. Runtime receipts retain the concrete exception type/condition, attempted phase/batch, finished batches, and accounting when available. A CUDA failure that prevents synchronizing is recorded as an accounting failure as well. No smaller graph/batch, candidate cap, sampling, split-size adaptation, skip, precision change, or tolerance relaxation is attempted.

A full resource pass requires both entire native epochs and finite state. A failed or unavailable full-scale run must be reported as that concrete condition; source preparation and CPU QA do not substitute for it. Any future source-defined batching/split adaptation requires a separately named, reviewed, sealed successor and renewed correspondence. Root owns that decision.
