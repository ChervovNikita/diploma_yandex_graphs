# Final TEST population preparation

**Source prepared; builder not executed.** No remote access, native dataset
load, TEST metadata enumeration, model construction, prediction or scoring has
been performed for this package. Counts and TEST-file hashes in `INPUT_PLAN.json`
remain null until actual authorized execution. Source preparation does not
qualify runtime behavior, RAM use, numerical parity or a campaign launch.

This separate package prepares compatible role contracts for the unchanged
`final_selected_scoring_v2` interface. It preserves all 27 adopted presets:

| Family | Presets | Populations per preset | Unique dataset populations |
|---|---:|---:|---:|
| PolyFormer, four datasets × mono/cheb | 8 | 10 literal splits | 40 |
| MolHIV/MolBACE, four architectures × two dropouts | 16 | 1 official scaffold | 2 |
| Collab NCN, NCNC, BUDDY | 3 | 1 official temporal query population | 1 |
| Total | 27 | 99 preset-specific proofs | 43 |

The frozen roster, native sources, scoring v2 and prior publications are not
edited. The 43 unique populations share their ordered-ID artifacts across
presets. Every preset still receives its own complete evidence and contract.

## Exact identity and native source laws

`SOURCE_BINDINGS.json` pins the actual roster, admitted data-custody bindings,
original native source, current serving interface and qualified interpreter
metadata. `INPUT_PLAN.json` lists each exact admitted processed/role file.
It is a readable plan; the runtime derives its authority from the pinned
roster and `CELL_BINDINGS.json` and checks all shared-preset identities.

- **Node:** instantiate the same PyG `Planetoid`/`Coauthor` providers with the
  original names and `NormalizeFeatures`. Load existing processed stores only.
  Execute the exact AST definitions of `index_to_mask`, `take_rest` and
  `random_splits` from original PolyFormer `utils.py`. Derive the ten literal
  seeds from the roster; use the original rounded 60% per-class and 20% VALID
  requests. Verify complete disjoint TRAIN/VALID/TEST masks. TEST IDs are
  `torch.nonzero(test_mask, as_tuple=False).flatten()` in native ascending order.
  The graph and its full labels are deserialized; labels are used for original
  stratification. No label vector is exported. Polynomial bases are neither
  loaded nor computed. No sparse inference graph is rebuilt. The split functions
  depend on labels/node count, so basis and edge preprocessing are unnecessary
  to enumerate their mask; their forward behavior is not qualified here.
- **Molecules:** instantiate original `PygGraphPropPredDataset`, retaining each
  dataset's exact root: MolHIV uses `massive_molecular_packet_20261012_v1`, and
  MolBACE uses `massive_molecular_packet_20261012_v3`. Pin/read the original sibling
  `split/scaffold/test.csv.gz`. Pin all original TRAIN/VALID CSVs. If the native
  session's preferred TRAIN/VALID PT exists, pin it and require identical full
  ordered IDs to the admitted CSV. Verify the complete disjoint scaffold
  partition of the provider. Reuse the exact `MoleculeRoleView` definition from
  scoring v2 with native PyG collation, batch size 32, shuffle false, workers 0.
  The original skip law is **packed `batch.x.shape[0] == 1`**. It does not remove
  every one-node molecule. Preserve all official IDs, eligible IDs, skipped IDs
  and every batch's IDs/packed node count. `records` counts the full official
  role; `ids_sha256` covers only the eligible evaluator sequence. Original
  labels are incidentally deserialized/collated but never inspected or exported.
- **Collab:** pin the separate admitted graph/TRAIN/VALID paths, then pin/read
  original `split/time/test.pt`. Validate every positive and negative endpoint
  tensor as native int64 `[N,2]` within the original 235,868-node domain. Use
  original BUDDY `get_ogb_pos_edges`/`make_obg_supervision_edges` AST definitions
  to obtain positives followed by negatives in original file order. Export all
  ordered endpoints and IDs `arange(positive+negative)`. No sorting, subsampling,
  deduplication or negative generation occurs. Membership in positive/negative
  roles is accessed and acknowledged. Graph/TRAIN/VALID files are byte-hashed
  without tensor deserialization. BUDDY's `filter_by_year` affects TRAIN alone;
  it does not filter official TEST queries. Contracts retain unweighted
  TRAIN+VALID for NCN/NCNC and weighted year-2007 TRAIN+weighted VALID for BUDDY.
  No adjacency, HashDataset, hash/SIGN features or model is constructed.

New TEST-file receipts establish the bytes of the complete existing official
role files at their native admitted sibling paths. They are not a fresh upstream
download or an independent reauthentication of the original dataset release.
Missing files or conflicting native roles fail; there is no replacement source.

All identity arrays use raw contiguous little-endian native int64 bytes, with
no file header. Their file SHA256 equals scoring v2's `tensor_sha` for nonempty
IDs. Array metadata gives dtype, shape, byte order, file byte count and hash.
An empty skipped-ID artifact is a zero-byte file with the SHA256 of empty bytes.
For independent inspection, `numpy.fromfile(path, dtype='<i8').reshape(shape)`
reconstructs the exact sequence. These are population metadata, not predictions.

## Prospective reconstruction tolerance

`TOLERANCES.json` freezes `rtol=1e-4, atol=1e-5` for all presets, following root's
15 October direction and the pinned prior engineering criteria. It applies to
future float32 VALID reconstruction, and is not an effect-size or accuracy
criterion. Future reconstruction must record maximum absolute and RMS residuals
and preserve exact IDs/order/labels/checkpoint identity. No bit-identity or
exact-metric numerical gate, tolerance search or tuning is added. This builder
does not execute floating reconstruction checks. Scoring v2 remains unchanged;
recording residual diagnostics belongs to the reviewed execution integration.

## Access disclosure

Future execution **accesses TEST population metadata**. It cannot truthfully
report `TEST_accessed: false`. Each worker appends an `ACCESS.jsonl` event before
TEST-related loads/hashes, recording node-label stratification, molecular
incidental label deserialization, or link positive/negative query membership.
Complete outputs explicitly set `TEST_accessed: true`, `TEST_scored: false` and
no prediction/model/checkpoint/training work. Partial failures preserve their
access ledger; a failed process never implies that TEST was unopened.

The package does not inspect model scores, choose checkpoints/architectures,
calibrate, change scientific criteria, or perform TEST-driven selection. It
does not create a campaign GO or authorize final inference.

## Future allocation command and dependencies

Root owns review, publication and execution. No execution is requested by
source preparation. After publishing this exact directory, use the literal
allocation SSH destination documented in the project `AGENTS.md`, port 2222,
the approved private key and `-tt`. Verify hostname `anogena-2-0` and the sole
GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` before any remote project
operation. The builder repeats this guard before all project reads/writes.

The future invocation, after the external route check, is:

```text
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python -B /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/final_role_population_preparation_v1/prepare.py --source-manifest-sha <exact MANIFEST.json SHA256 from the source handoff> --output-name allocation_population_v1
```

The final source handoff supplies the literal manifest hash. It cannot appear
inside this manifest-covered README without creating a circular hash. The
script requires an external hash and refuses an existing output directory.
Use the qualified lexical `.venv/bin/python` (CPython 3.11.14); its binary,
resolved location, prefix and `pyvenv.cfg` are checked against the published
environment record. Do not use the resolved base interpreter directly.

Required existing environment: PyTorch, NumPy, PyG, OGB and their installed
SciPy/pandas/scikit-learn dependencies. No installation or environment update
is attempted. Each worker records actual versions, package RECORD receipts,
torch/PyG entry source and the installed source files for native providers,
normalization, batching and loaders. Installed versions are observed at future
execution, not invented here. The package pins project/native source bytes;
package RECORD receipts document installation metadata and are not a claim
that every installed dependency binary was independently rehashed.

## CPU, RAM and failure limits

- Seven fresh workers run sequentially, one per unique dataset. Processes exit
  between datasets to release provider memory. The supervisor imports stdlib
  and the original pure metadata contract module only.
- Both supervisor and workers use the same two allowed CPU cores. CUDA is hidden
  before worker Python starts; native work uses CPU. Tensor/BLAS threads are set
  to 2, interop to 1 and DataLoader workers to 0. Actual affinity/thread counts
  are recorded. Auxiliary parked library threads may exist; affinity limits CPU
  execution to two cores. No GPU kernel or tensor allocation is requested.
- Supervisor virtual address-space soft limit: 2 GiB; each worker's hard virtual
  address-space limit: 12 GiB. Supervisor stops a worker above sampled 8 GiB RSS.
  Initial conservative available memory must be at least 12 GiB; live available
  memory must stay at least 4 GiB. These are prospective ceilings/admission rules,
  **not measured requirements**. Native package mappings or Physics normalization
  may fail within them. Such failure is preserved and cannot trigger a larger
  automatic rerun.
- Admission checks host `MemAvailable` and visible cgroup v2 limits/ancestors.
  One-second samples retain worker RSS/high water/threads and shared cgroup
  current/peak/events. `ru_maxrss` is saved at completion. Shared cgroup data
  includes peers/page cache; it is not attributed to this builder. Sampling can
  miss a transient RSS spike; `RLIMIT_AS` provides the independent hard bound.
- Require 1 GiB free output disk. Each worker has a 900-second wall limit; the
  supervisor has a 3,600-second alarm covering all work and final verification.
  Stop violations terminate only its own worker process group. No peer process
  is killed and no GPU reservation is claimed.
- Use normal host execution and ordinary incidental runtime caches. There is no
  filesystem audit hook, blanket network/process blocking, prompt interception,
  cache redirection or operating-system sandbox. Deliberate preparation outputs
  remain under the fresh repository output directory. The original native
  provider's `download` and `process` entry points are temporarily replaced with
  an explicit missing-input error during construction only. Its original class,
  data access and normal runtime behavior remain in use. Molecular construction
  first pins the installed OGB `master.csv` version and corresponding existing
  dataset `RELEASE_v*.txt` marker to avoid the native cache-update prompt. Missing
  input/release markers stop preparation; this package does not obtain new data.
- Missing/changed source or data, absent dependency, provider/schema mismatch,
  resource excess and native failures stop the preparation. Logs, partial arrays,
  access history and `FAILURE.json` remain. There is no retry, overwrite, repair
  download or fallback population. Only a complete output manifest qualifies
  the resulting role package.

Route/source preflight errors occur before an output directory is created; the
calling execution record must retain their stderr/exit status. Once the fresh
output exists, subsequent failures retain their own `FAILURE.json` and partial
worker records. A rejected route never writes a project artifact on that host.

Input bytes are checked before and after native enumeration. Final metadata
checks stream each unique source once and require unchanged inode/size/mtime/
ctime on reuse, avoiding dozens of redundant reads of the Physics graph.
The original v2 authority function itself is reused via its exact AST definition.

## Byte-pinned output contract

Everything stays in the fresh `allocation_population_v1` output child:

| Output | Meaning |
|---|---|
| `ROLE_CONTRACT_INDEX.json` | Exact receipt map for all 27 contracts and complete inventory counts |
| `contracts/<preset>.json` | v2 role contract, native topology and 10 node or 1 molecular/link population |
| `provider_evidence/<preset>__split<i>.json` | 99 v2 provider proofs, actual ordered arrays, official input/source receipts |
| `<dataset>/*.i64` | Full TEST IDs; molecular official/eligible/skipped sequences; Collab ordered endpoints |
| `<molecular>/BATCHES.json` | Exact packed batch identities, node counts and original skip decisions |
| `<dataset>/RESULT.json` | Native population metadata, input receipts and measured worker outcome |
| `<dataset>/DEPENDENCIES.json`, `INSTALLED_PROVIDER_SOURCES.json` | Actual interpreter/dependency/provider source evidence |
| `<dataset>/ACCESS.jsonl`, `ADMISSION.json` | Honest TEST metadata access and CPU/RAM admission evidence |
| Root worker logs, resource samples and exit receipts | Actual process and shared-allocation costs and failures |
| `RESULT.json` | Complete preparation status; zero models/updates/predictions/scores |
| `OUTPUT_MANIFEST.json` | Receipt of every prior deliberate output file, including logs/evidence; excludes itself |

The builder prints the exact `OUTPUT_MANIFEST.json` receipt for an external
handoff. A role proof binds the builder/source manifests, original inputs,
installed provider source and ordered output arrays. All receipts have exactly
`path`, `bytes`, `sha256`. Do not edit/move completed files while retaining their
old receipts. Root can hand the 27 loaded contracts/receipts to the existing v2
roster compiler; the post-training selection seal and scoring runtime admission
remain separate, later work under the campaign's authorization.

`STATIC_REVIEW.json` records local syntax/source/coverage inspection only. It
does not import this builder or any scientific library, execute native source,
load a dataset, access remote files, or establish runtime success.
