# Prospective TRAIN-only paired replay provider

4 October 2026. **Implementation and qualification preparation only.** No provider, numerical audit, real-data reader, model, fit, result, checkpoint, VALID or TEST traversal was executed. Current sealed V5 J/F source and scientific runs remain untouched. This packet prepares new paired replications under a separate future release; it cannot release fits.

## Compact durable inputs

Each prospective epoch stores two uncompressed, non-pickle int64 `.npy` arrays and one externally hash-bound JSON receipt:

| Payload | Exact content |
| --- | --- |
| `PERMUTATION.npy` | All 1,179,052 ordered native record IDs. The first 17 consecutive blocks of 65,536 are the actual positive record masks; the final 64,940 IDs are the dropped tail. |
| `NEGATIVE_PAIRS.npy` | The entire returned native negative draw in `[N,2]` order, including unused rows. Positive and negative query rows both use the same selected record IDs, exactly as frozen V5. |
| `EPOCH.json` | Source/runtime/TRAIN identities, root-bound master seed and epoch, actual array shape/bytes/hashes, original native `[2,N]` negative identity, permutation/tail identities, and all 17 graph plus positive/negative support receipts. |

No mask, negative query or tail is inferred from an old hash. This provider makes fresh prospective arrays. The disabled `ROOT_RELEASE_EXAMPLE.json` must be completed and independently authorized before its CLI can import Torch or read data. A future fit source must bind the actual `EPOCH.json` SHA separately; `EpochReplay` requires that external pin before trusting the array pins inside it.

## Why exact support regeneration is available

The provider pins the complete immutable TRAIN record array, the directed raw graph used only by the native negative sampler, and the exact frozen graph/support source. The preserved `PermIterator` constructor creates `randperm`; its iterator returns consecutive full slices and drops the last partial slice. There is no second stochastic step inside mask iteration.

For a batch, `Graph.mask_train_batch` deletes the selected **record IDs before symmetrization and unique-edge coalescing**. Surviving duplicate records can retain an edge. Sorted adjacency rows and the fixed ordered query table determine native common/left/right neighbor arrays completely. Left/right residual coordinates have the source order **all left then all right**:

`(query_row, left0_right1, query_u, query_v, candidate, counterpart)`.

Common coordinates are `(query_row, query_u, query_v, candidate)`. The receipt binds common supports as well as the residual auxiliary supports. Every coordinate is visited in bounded blocks; no degree cap, neighbor sampling, top-k, deduplication beyond the native graph operation, support reordering or slot truncation is introduced. A hash records what regeneration produced. Qualification must compare the **actual complete integer arrays**, including every streamed coordinate block, before that hash can support a matching claim.

The implementation's compact form therefore has a source-qualified reconstruction path. Its runtime equality remains **unverified**. It currently requires exact source/runtime identity, including the admitted device visibility. Byte format portability does not qualify a different device or runtime; separately establish full integer equality before widening that contract.

## Model-RNG independence

The native sampler and permutation run under an epoch seed determined solely by a prospectively bound master seed and epoch. Python, NumPy, Torch CPU and all visible CUDA RNG states are saved and restored exactly, with CUDA synchronization and a full typed-state digest check, including exception paths. The 100 derived uint32 epoch seeds are distinct. Generation never reads a model, its RNG digest as a seed, a density draw, labels or model outcomes.

The synchronous isolation helper requires exclusive access to global RNG. The CLI runs in a fresh standalone process; simultaneous RNG-consuming threads in the same process are outside its contract. Count-control latent draws and all model dropout must have their separately bound RNG rules.

Reading the saved arrays and regenerating supports consumes no model RNG. Different model sizes or dropout histories can thus share literal masks and queries. This does not reconstruct current runs. Moving future J/F sampling to a replay provider also separates its RNG consumption from dropout: it is a new prospectively admitted stream adapter, not exact stochastic-program parity with the current run. Preserve the frozen objective, architecture, labels, budgets and selector; freeze and qualify the future adapter separately.

## Boundary between provider, predictor and teacher

The provider's internal complete TRAIN record pool is necessary to delete records and enumerate the permitted masked graph. The complete directed raw graph is used to draw native negatives and check forbidden edges. Neither is an additional predictor adjacency. Its returned context contains record IDs, the **masked graph**, actual positive/negative query arrays and native neighbor arrays. Models receive the masked graph, query endpoints, native supports and their separately admitted raw TRAIN node features. Record IDs, receipt keys and pool-role bookkeeping stay in the adapter and never become predictor features.

The provider stores no Z, count target, teacher membership keys, feature table, query class flag or predictive output. A training adapter calls the label-only complete-TRAIN teacher **after** the support is fixed. Z=0 means unobserved in TRAIN, not a verified latent nonlink. Only the fabricated audit imports the existing teacher, after contexts exist, to check alignment and removed-observed/source-unobserved semantics. The real trace audit has no label or teacher operation.

`replay_runtime.load_train_only` reads only authenticated `split/time/train.pt` and `raw/edge.csv.gz`. The retained authority JSON contains TRAIN/VALID metadata, and the inspected V5 loader contains VALID source code; those references do not open VALID arrays. This new reader never calls V5 `load_data`, a dataset constructor, a split accessor, an evaluator or a checkpoint reader. Features are unnecessary for model-free stream construction and are not opened.

## API and future integration

- `draw_epoch` uses the native default sampler call followed by native `PermIterator`, with all returned negatives retained.
- `write_epoch` stores the two arrays and computes complete support receipts.
- `EpochReplay(..., expected_epoch_sha256)` authenticates the receipt, arrays and immutable TRAIN identity. `query_batch` supplies the selected record IDs, borrowed full negative array and expected receipt. Consumers must not mutate that array.
- `regenerate` supplies and checks a complete visible batch context. `assert_actual` checks actual decoder contexts directly, allowing a qualified future adapter to reuse native enumeration and avoid a second support traversal.

There is no fit launcher. Future J/F/count adapters must be separately source-frozen and qualified against these exact arrays and native support coordinates; this provider alone does not certify their feature, decoder, dropout, gradient, update or selector parity. Future model interfaces must not receive provider internals or teacher state as context.

## Prepared qualification, not completed qualification

`replay_driver.py` has only `fabricated`, `full_batch` and `generate_epoch` stages. Its stdlib gate requires an exact root-authorized invocation, own sealed manifest, pinned dependencies and a fresh output outside sealed source. The example has execution disabled and no authorized invocation or seed. Runtime reuse authenticates the ordinary single A100 environment before the explicit unchanged-RNG V5 deterministic-profile transition. No installation, sudo, server configuration, external job, PDF or other source modification is performed.

**Fabricated common core.** Prepared code uses a small multiset with duplicate records, full native draw/permutation operations, partial tail, ordered asymmetric supports and isolated empty-query cases. It checks all saved integers against their originals, every native support against an independent Python oracle, ambient Python/NumPy/CPU/CUDA RNG perturbation, exception restoration, explicit source-zero/removed-positive teacher bits, and rejection of wrong identities/receipt pins, corrupt arrays, duplicate permutations, self-links and forbidden negatives. The same writer, reader, mask and support core is used by full-size work. No such test was invoked here.

**Ordinary complete real batches.** A pinned passing fabricated receipt is required before opening TRAIN arrays. Prepared code compares the provider against frozen V5 `epoch_stream` under the same isolated provider seed, using TRAIN arrays and a node-count sentinel because the frozen stream only uses `len(x)`. It requires equality of the entire returned negative array, full permutation and dropped tail, all 17 actual masks, every graph/query/common/left/right integer and every coordinate block. This is one full epoch of data/support work, with zero model forward, backward, optimizer, metric or held-out traversal. It does not qualify count/J/F learning code.

`generate_epoch` requires separately pinned passing fabricated and full-batch receipts for the same provider/source/runtime and exact TRAIN identity. Failed attempts remain as failures and cannot supply admission. Successful future generation records the epoch receipt SHA and measured file bytes/time/CUDA peaks. The CLI supports one epoch per standalone invocation; 100 invocations repeat startup and TRAIN loading. A multi-epoch wrapper would require its own frozen release to amortize those costs.

## Storage and paid work

Let P=1,179,052 and N be the **actual** native returned negative count. Integer payload bytes per epoch are `8P+16N`; NumPy headers and JSON receipts are additional. Native default `num_neg_samples=None` uses the directed raw input draw-size convention; 2P=2,358,104 is the planning request size, not a guaranteed returned count. Source accepts N≥P, and the provider records N without silently shortening it.

| Planning case | Integer bytes per epoch | For 100 epochs |
| --- | ---: | ---: |
| N=P, smallest admitted complete draw | 28,297,248 ≈26.99 MiB | 2,829,724,800 ≈2.64 GiB |
| N=2P, native planning request size | 47,162,080 ≈44.98 MiB | 4,716,208,000 ≈4.39 GiB |

There are 2,228,224 supervised positive/negative query rows per epoch. If r is the average **total** residual support per query, saving all six-column residual coordinates would cost `48×2,228,224×r` bytes per epoch. At r=10,50,100 this is approximately 1.00,4.98,9.96 GiB per epoch, or 99.61,498.05,996.09 GiB for 100 epochs. Common coordinates add `32×2,228,224×c` bytes for average common support c. These are byte sensitivities from fixed source counts, not measured degrees or dataset statistics. Storing all coordinates is not justified as a default without actual slot counts and disk admission. Compact arrays plus complete qualified support checks are the proposed default.

One generation epoch pays one native sampler draw, one permutation, two array writes and two round-trip array reads, 17 masked-graph rebuilds and 34 full 65,536-query support enumerations plus complete graph/coordinate hashing, three full-array validations and exact persisted-array comparisons. The prepared full audit pays two draws/permutations, 51 graph rebuilds and 102 full support enumerations, plus serialization/readback and streamed exact comparisons. No work is credited as free because it is diagnostic.

The graph input remaining after each mask has 1,113,516 records; symmetrization feeds at most 2,227,032 entries before native coalescing. Graph row/col/rowptr storage is at most 37,519,464 bytes before temporary sorting, copies and supports. Streaming six-column coordinate blocks limits one block to 6 MiB; it does not bound total native support memory or eliminate graph/support computation. Two equality streams can hold two such blocks. Actual CPU maximum RSS and process-wide CUDA allocated/reserved peaks are recorded by the prepared driver and remain unmeasured here. The compressed TRAIN/raw files total 46,064,177 bytes per process; verification and loading each read them, with additional decompression, hashing, CPU/device copies and ordinary runtime authentication costs. `BYTE_AND_WORK_BUDGET.json` carries these formulas and unmeasured limits.

## Custody

`SOURCE_BINDINGS.json` pins the sealed control proposal, frozen V5 source/plan, graph functions, native permutation source and retained metadata authorities. References stay in their original trees. `QUALIFICATION_PLAN.json` distinguishes prepared gates from unexecuted outcomes. Static AST/compile/JSON/hash checks are recorded separately; they are not numerical parity, replay equality, feasibility, predictive evidence or execution authority.
