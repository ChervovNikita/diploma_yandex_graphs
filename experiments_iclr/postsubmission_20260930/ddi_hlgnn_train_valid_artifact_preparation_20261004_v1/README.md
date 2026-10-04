# Official DDI TRAIN+VALID artifact preparation

Status: disabled source preparation. Root owns the exact release, CPU launch, and monitoring. This packet has not imported numerical libraries, opened archive/tensor/model/score payloads, connected to the server, or executed the worker.

## Selected inputs and outputs

Reuse the existing official DDI v1 archive, SHA256 `0d0371d3dbd2d30a6801af245b226c6b25bf0dcebc6f26f162a6057d8d78d5f4`, and the prior TRAIN acquisition's central-directory metadata. No download is needed. Exactly five literal archive members are selected from the hash-bound archive's central directory:

- `ddi/split/target/train.pt`
- `ddi/split/target/valid.pt`
- `ddi/raw/num-node-list.csv.gz`
- `ddi/raw/num-edge-list.csv.gz`
- `ddi/raw/edge.csv.gz`

Root explicitly authorized the tiny edge-count member because the native OGB loader uses it to slice the graph. Selection rejects duplicate/ambiguous names, combined split storage, unexpected attributes/weights, and missing literal members. TEST bytes remain opaque; no combined OGB split accessor or dataset loader is called. The full archive's binary hash includes its opaque bytes.

Fresh execution output is `ddi_hlgnn_train_valid_artifact_execution_root_20261004_v1/run01`. Successful execution produces the five original selected member files, strict `HLGNN_DDI_TRAIN_VALID.pt`, and `QUALIFICATION.json`. Failure preserves selected evidence and `QUALIFICATION_FAILURE.json`; it is nonqualifying, with no automatic retry. Only a COMPLETE qualification report authorizes use of the artifact.

## Native graph and weight semantics

The retained OGB loader sources and official master metadata are pinned in `SOURCE_PINS.json` and copied as evidence. They are never imported. Metadata selects a homogeneous nonbinary DDI graph, inverse edges enabled, no node/edge attributes, and no additional files. Count files must describe exactly one graph: 4,267 nodes and 1,067,911 raw edge rows.

Bounded two-column integer CSV parsing preserves the exact native row values/order in the accepted domain. The worker reproduces the pinned loader's `np.repeat(..., 2, axis=1)` and odd-column reversal assignments. Every raw row is immediately followed by its reverse. The artifact preserves this native graph order without sorting, coalescing, normalization, target masking, or VALID insertion. The HL-GNN adapter subsequently performs the author's sparse conversion itself.

The raw graph's canonical unique undirected membership must equal official TRAIN, with zero loops and exactly two directed entries per TRAIN pair. A mismatch fails before artifact serialization and retains its observed counts/hashes. The worker cannot substitute a graph reconstructed from TRAIN.

The original official TRAIN dictionary must be exactly `{'edge'}`. Graph weights are `None`; TRAIN has no weight field. No synthetic ones or loss margins are created. Consequently, the preserved HL-GNN loss branch is **AUC**, despite the recipe's `WeightedHingeAUC` label. Native node indexing is unchanged: identity of zero-based integer IDs with the original node count, including isolated nodes.

## Fixed VALID contract

Original VALID must contain exactly `edge` and `edge_neg`. Original int64 dtype, pair order, orientation, and candidate multiplicity are retained through the native NumPy-to-Torch conversion and a checked weights-only serialization roundtrip. No candidate resampling, deduplication, grouping, or reordering occurs. Canonical keys are separate temporary qualification arrays.

The worker verifies unique loop-free TRAIN/VALID positives, TRAIN/VALID positive disjointness, negative exclusion from TRAIN and VALID positives, and zero candidate self-loops. It reports negative multiplicity without changing candidates. The native `[M,2]` negative field remains the fixed global pool shared across positive queries for Hits@20/50/100, as used by the pinned HL-GNN VALID adapter. At least 100 negatives are required. TEST disjointness is not numerically checked; graph exclusion is bound to verified TRAIN equality and official graph provenance.

The saved artifact has exactly `schema`, `num_nodes`, `graph_edge_index`, `graph_edge_weight`, `train`, and `valid`, conforming to the existing `hlgnn-ddi-train-valid-v1` contract. Original TRAIN row bytes must match the previous acquisition's tensor digest. `TRAIN_ONLY.pt` is hashed for custody without decoding it. Legacy official NumPy split files use the same trusted official-payload decoding format as the prior acquisition; the resulting artifact loads with `weights_only=True` and is checked for exact keys, dtypes, shapes, and values.

## Bounds and release

Use the existing pinned RAPIDS Python 3.12 interpreter, Torch 2.7.1, NumPy 1.26.4, two Torch CPU threads, one interop thread, and hidden GPUs. The worker makes no GPU call, constructs no model, and produces no score. Recommended thread environment variables are recorded in `PLAN.json`.

Named native graph storage is 34,173,152 bytes; TRAIN is 17,086,576 bytes. With at most one million positive and one million negative VALID rows, named artifact tensors are bounded by 83,259,728 bytes. Selected member bytes are capped at 320 MiB, decompressed edge CSV at 64 MiB, and serialized artifact at 128 MiB. Original selected files plus the artifact have a named disk bound of 448 MiB, excluding small evidence/control JSON. Actual VALID sizes and runtime peaks remain unmeasured.

Worker checks allow 600 seconds and observed Linux worker peak RSS of 4 GiB. Checks occur while streaming, during CSV parsing, between major operations, and before completion. Legacy payload allocations and library temporaries are not instantaneous physical limits; root owns supervision. No training runtime, GPU resource feasibility, or full 500-epoch budget is qualified by this artifact worker.

`PLAN.json` and `ROOT_RELEASE_TEMPLATE.json` remain disabled. The worker requires the exact external execution-directory `ROOT_RELEASE.json`, its supplied hash, root authorization, and sealed manifest/plan hashes. It rechecks source controls, archive, acquisition central metadata, TRAIN_ONLY, and all selected member bytes before completion. No author loader, existing source packet, or canonical index is changed.

`STATIC_VALIDATION.json` records stdlib-only source/AST/metadata checks. They verify the proposed worker and pins; numerical equivalence, data qualification, elapsed time, and artifact production require the later root execution.
