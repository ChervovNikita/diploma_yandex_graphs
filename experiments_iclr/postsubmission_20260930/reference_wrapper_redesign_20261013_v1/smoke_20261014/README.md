# Real-data smoke qualification, 14 October 2026

The user explicitly authorized these checks after asking whether short runtime
tests had been performed. Full scientific campaign approval remains pending.

## Coverage and execution

The manifest contains 167 distinct cases: 56 node, 96 molecular and 15 link
configuration/procedure combinations. Each source setting covers its native
single, eligible boundary GNNM procedures and a genuine independently owned
four-model ensemble. Seed-zero checks cover the common execution path; they
do not execute all 2,480 proposed cold fits or every trial seed.

Each case uses real TRAIN data, the fixed source architecture/optimizer/batch
settings, two initial optimizer updates, complete native VALID inputs, and a
checkpoint. A third update is run uninterrupted and then replayed after reload.
Both executions and all four actual callbacks are charged. An independent
ensemble charges four separately owned optimizer transitions per callback.
TEST scores are never computed. Short losses are engineering diagnostics.

Checks include finite losses, outputs, gradients and optimizer states; retained
native parameter identities; explicit source inactive-gradient exceptions;
four independent weight/optimizer/RNG owners and actual mean-logit pooling;
saved model/optimizer/sampler/private and global RNG state; restored modes and
training clocks; and checkpoint/replay agreement under the declared engineering
tolerance. The amended checker first requires exact restored state before any
new work, then compares floating parameter/optimizer tensors using per-tensor
RMS error. Prediction and loss tolerances remain unchanged. Original coordinate
checks and every failed report are retained; repairs use fresh attempt folders.
No automatic retry or rerun-until-pass policy is used.

## Source ports

- PolyFormer uses its literal source class/function definitions, native normalized
  inputs and Mono/Cheb bases, ten-source-split policy (split zero for smoke), and
  original ordinary/attention Adam groups. Factors bind native `lin1`/`lin3`.
- Molecular models load the pinned original GCN/GIN, virtual-node, categorical
  AtomEncoder/BondEncoder and pooling definitions. All atom fields participate
  in the logical input boundary; integer IDs and native modules are retained.
- NCN/NCNC retain the original encoder/predictor, input masking, native batches,
  loss reductions and recursion. Factors bind the native encoder input projection
  and final task scorer; internal NCNC completion remains source-owned.
- BUDDY retains native hashing, year-2007 policy, structural/feature branches,
  shared normalization and source batches. Its parallel canonical input maps
  and final scorer are explicitly bound.

PolyFormer route activation checkpointing and blocked full-TRAIN contrastive
evaluation reduce memory without changing the source model or auxiliary
objective. They add recomputation; no inference-speed claim is made. Native
BatchNorm modules in molecular/BUDDY sources remain shared and retain the
declared sequential route semantics.

The allocation runtime is PyTorch 2.1.2+cu118, PyG 2.7.0 and OGB 1.3.6.
The missing BUDDY dependency datasketch 1.5.9 was added to the project environment.
Existing source models and original paper scores remain untouched. Raw datasets,
checkpoints and logs stay on the allocation; the Mac stores source and small
summary artifacts.

## Evidence and limits

`results/` on the allocation contains fresh-process case reports, checkpoint
files, launch receipts and costs. The compact terminal report is published under
`publication/`, with every configuration and attempt represented. Strict replay
failures remain failures even if updates and exact restoration succeed. Missing
legacy telemetry remains unknown. Raw tensors and logs stay on the server.

Source review found and corrected two link-wrapper errors: native bias is now
scaled by the declared output factor, and NCNC applies final factors only to
top-level task calls. Its native recursive completion bypasses these factors
in both TRAIN and VALID. All 15 link cases were rerun after the repairs.

The two-job probe runs Physics Mono/private ensemble credit alongside NCNC/I4
on the allocation. Its separate memory receipt reports sampled concurrent
usage; it cannot certify all possible pairs or the 18.77 runtime.

Passing short checks establishes initialization, update, validation and resume
compatibility. It does not establish accuracy improvements, convergence,
the complete full-horizon training controller, every possible job pairing, or
the unavailable 18.77 runtime. Scientific launch remains disabled.
