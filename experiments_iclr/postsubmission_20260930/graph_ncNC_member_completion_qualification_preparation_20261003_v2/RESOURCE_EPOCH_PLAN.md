# Prospective complete-collab resource qualification

**Plan only; root must separately authorize execution after independent engineering QA.** No data payload, model runtime, epoch, resource measurement or final seed is admitted by this file. A full experiment driver/selector/heldout barrier has not been prepared or frozen. Current active studies are unchanged.

## Prerequisites

1. Review the source/semantic mapping, sealed input pins and private-source handling. Run all eight engineering checks in a qualified runtime. A missing native dependency or parity failure blocks data execution; no skip-based pass. Root owns any environment change and preserved successor package.
2. Bind the exact collab TRAIN-only acquisition route before opening it. Use provided 128-dimensional node features, all 235,868 declared nodes and all official TRAIN records, with no LCC/year reduction. Read TRAIN and permitted feature/graph payloads only. Do not call an aggregate split accessor that hydrates TEST. Bind node order, features, raw TRAIN records, graph construction, source, dependencies and sampler implementation hashes. No validation/test payload or future-positive identity may enter qualification.
3. Qualify the source-specific unweighted symmetric adjacency and native one-layer GCN normalization on the intended device. Source-based CPU parity does not substitute for GPU/full-graph parity, finite checks or memory feasibility. Root selects an engineering RNG identity separately; it is not a final study seed.

## One representative native resource epoch per twin

Start each twin from the same source-bound cloned encoder/decoder/Adam/global RNG state. Unit-factor engineering identity and a future scientifically chosen factor initialization are distinct. Do not tune either using predictive outcomes. Reset corresponding data-order/negative/dropout streams in the two jobs, and verify their call schedules from source/engineering receipts.

Retain the author collab batch size65536 for the first memory qualification and the native profile in `Recipe`. One **complete native epoch** shuffles all TRAIN supervision records and iterates the native full batches, dropping the shuffled incomplete tail exactly as `PermIterator` does. Complete topology and native per-epoch supervision coverage are different: report the dropped tail; do not call it full-positive coverage or silently yield it.

At the epoch start, use the pinned native PyG negative_sampling route against the complete original TRAIN edge_index and all declared nodes, with its native default requested negative count. Only TRAIN edges and self-links are forbidden. Verify the native indexability of negatives by every shuffled supervision record. A shortage blocks qualification; it does not trigger a silent replacement sampler or future-positive rejection. Bind positive order and negative draw identity. Negative sampling/adquisition work is timed and charged.

For each full positive batch:

1. Remove that batch's positive **record indices** from the complete TRAIN supervision list, then construct the unweighted symmetric graph. Preserve remaining duplicates and all nodes. Charge mask creation, full adjacency reconstruction/coalescing, CSR construction and transfers.
2. Run one shared learned encoder on all node features, with native input dropout and directed encoder edge dropout/normalization. It is rebuilt per minibatch and cannot be cached across optimizer updates. The decoder receives the masked original graph, rather than the encoder's stochastic adjacency.
3. Run the positive decoder and then the paired negative decoder separately, as native code does. Enumerate all common and exclusive neighborhoods; no fanout or positive subsampling. For each query pass, each of four members runs outer full-node xlin, two no-grad recursive full-node xlin/base-scorer paths, clamp, weighted aggregate and its nonlinear final decoder. Both twins compute every member's weights before the private/pool switch. Splitsize=-1 initially retains the native full candidate pass.
4. Use native loss scaling: mean negative-log-sigmoid on positives plus mean negative-log-sigmoid on negatives, averaged across member scores. Run the native two-group Adam update, with source learning rates and default betas/epsilon/zero decay. No validation selection, objective variant, LR sweep or convergence claim is introduced.
5. Keep necessary finite/shape/coverage and provenance guards. Accumulate resource counters and timings, not predictive comparisons, tuning losses, labels or candidate arrays in the public receipt.

An epoch is representative only if **all native full minibatches complete**, including the actual high-degree/candidate workload. A few convenient batches cannot establish full-epoch feasibility. Memory failure is retained. Root may prepare a source-defined positive scorer splitsize or outer batching successor if necessary, repeat native/dropout correspondence and resource qualification, and name it as a recipe adaptation. No shape-dependent truncation, candidate cap, smaller graph or skip is permitted as an automatic rescue.

## Mandatory accounting

Record GPU device/memory and runtime versions, source manifest/QA identities, model dtype, parameter/state bytes (including unused ptlin), optimizer/gradient bytes, global full graph/node dimensions, each masked graph nnz, native batch count/tail size, query count and complete common/left/right candidate counts. Record recursive candidate-score query counts and calls, recursive full-node transformations, encoder calls and full-node transformations, sparse query-enumeration/aggregation calls, transfers, synchronizations, peak allocated/reserved device memory and peak host memory.

With splitsize=-1, one training minibatch has **one encoder pass**, two outer query-neighborhood enumerations, **eight outer full-node xlin paths with gradients**, and **sixteen recursive full-node xlin paths under no_grad**. It also has sixteen recursive candidate-query enumeration/base-decoder passes. Empty candidate lists do not automatically erase the recursive full-node work. Charge all of these rather than only terminal scoring or parameter storage. Actual counts must be verified from the executable path; graph construction and native sampler work are additional.

Time cold acquisition/read/graph construction, forward parts, completion enumeration/scoring/clamp/aggregation, backward, Adam, state serialization and total epoch. Set the selected CUDA device before resetting memory counters; synchronize at declared timing boundaries and charge synchronization/profiling overhead. The private and pooled causal twins pay for the same candidate scorer bank; a later optimized common scorer is a separately named deployment comparison. Trace only metadata/work counters; retain no public training labels or score vectors.

After the full epoch, a serving-only TRAIN probe can use complete TRAIN topology and a prebound TRAIN query/negative list, with eval mode, all candidate paths and the unchanged mean-raw-logit pool. It provides measured throughput/memory/cold-vs-warm setup for this operation, **not a validation or test result**. Bind selected-state source/mode/pool and exercise replay without a quality selector. Do not use resource epochs as predictor-selection donors unless a later separately reviewed study explicitly authorizes that role.

## Root decision after resource qualification

Use measured complete-epoch and serving work to forecast all fits, validation forwards, checkpoints, preprocessing and final once-only serving for any later prospective comparison. No current cost or affordable-resource claim exists. Availability determines scheduling, not scientific rejection.

Any representative quality pilot still needs prebound initialization, seeds, budget, negative streams, selection/ties, complete-family closure and an independent heldout wrapper. It must compare the exact private/pool twins with a capable native NCNC single and a capacity/cost-capable single, a competent independent completion ensemble, the frozen BUDDY references and recent PENCIL for broader LP claims. Native and complete-TRAIN/TRAIN-only adapted settings remain separately named; native VAL-at-TEST access is not silently reintroduced. Passing resource or synthetic engineering checks establishes no quality, novelty, speedup, uncertainty or manuscript acceptance result.
