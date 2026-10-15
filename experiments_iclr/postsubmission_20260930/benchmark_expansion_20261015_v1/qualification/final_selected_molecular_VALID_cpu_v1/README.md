# Genuine saved MolBACE CPU VALID qualification

This frozen source bundle reuses three authentic existing selected states:
MolBACE native M1 at epoch 98, original GNNM mean at epoch 2, and private BN at
epoch 92, all seed 101. It reconstructs through the exact original runtime and
callback/extension/source dependencies, then calls scorer v4's molecular
`predict(..., 'VALID', None, ...)` once per state. It compares with the existing
own selected VALID export at fixed `rtol=1e-4`, `atol=1e-5`, with exact ordered
IDs/labels and unchanged parameters, buffers, private banks and optimizer state.
No training, selection, new state creation, TEST role or campaign authority is
created. This source preparation executes no model/data/checkpoint/tensor work.

## Explicit CPU scope

The original JOB/hash and checkpoint/export are authenticated and unchanged.
A separately recorded construction-only device override sends the authentic
factory to CPU. The checkpoint's model/optimizer/private/loader state is loaded
without schema or dtype conversion. Saved Python, NumPy and Torch CPU RNG are
restored exactly; saved CUDA RNG bytes are retained and hash-recorded without
applying them. CUDA visibility is empty and CUDA initialization is a failure.
This is CPU engineering replay under the genuine producer, not exact original
CUDA-device replay or current portable restoration qualification. No reseeding,
tolerance adjustment, checkpoint rewrite or selection change follows a mismatch.

The original source constructor reads the existing processed molecular store
and authenticated scaffold TRAIN/VALID role indices. Automatic download is
already disabled by that producer. The harness authenticates the original role
files, rejects changed PT/CSV precedence, and never opens a TEST role index/view
or performs a TRAIN forward. Original saved state and RNG payloads are checked
unchanged after serving. Ordinary VALID loader RNG consumption is allowed;
the immutable saved RNG and checkpoint are not changed.

## Finite work and failure retention

The fixed order is M1, original GNNM, then private BN. Three separate serial
workers avoid runtime import collisions. Total maximum work is three native
constructions, three selected-state loads and three whole-VALID passes: 15
batches and 45 native GNN forwards. Each case evaluates all 151 official graphs,
with the native batch 32/single-node skip law unchanged. No optimizer transition,
TRAIN callback, extra source validation pass, geometry analysis, alternate seed
or automatic retry occurs. Passive hooks and append-only events retain forward
attempt/completion and batch progress, including interrupted work.

Root supplies CPU/peer/RAM/interpreter admission. The worker applies the finite
address-space limit and wall alarm; the supervisor monitors only its owned
worker's RSS/wall and can terminate that worker. Numerical mismatch preserves
the CPU export/RESULT and continues the other fixed independent cases. Source,
identity or resource failure stops the remaining bundle for root review. A
bundle passes only when all three RESULTs pass, worker exit codes are zero and
no supervisor resource reason occurred. Previous attempts are never overwritten.

## Command handoff

`ROOT_ADMISSION_TEMPLATE.json` is not permission. Root creates an external
immutable admitted record binding this manifest, INPUTS, qualified project
Python, current peer/resource review and a fresh output root. CPU/RAM values
remain unset for root's actual reservation. The source ceilings are 600 seconds
per case, 12 GiB RSS reservation and 32 GiB address space; root may set lower bounds.
Use the exact argument array/environment in `COMMAND.json` only after admission.
No CUDA worker is added. `supervise.py` validates host/sole physical UUID before
project/model/data reads, then runs each source-bound case at most once.

`check_source.py` performs AST, free-name and finite-input checks only. Native
CPU execution, numerical outcomes and measured resources remain pending root
admission. All failures, costs and the original producer/device scope must be
reported without turning this engineering result into an accuracy conclusion.
