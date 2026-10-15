# Retained node and NCNC selected VALID replay

Source preparation only. No replay was executed and no numerical result is claimed.
Root owns review, publication, resource admission and execution. This package adds
one entry point and reuses `tree_identity`, `no_training`, and stdlib receipt helpers
from the passed molecular harness. It does not run the three passed molecular cases.

## Fixed scope

`INPUTS.json` pins the original `final_selected_VALID_qualification_plan_v2/MAPPING.json`.
Its `node_C4` and `NCNC_C4` rows are the sole checkpoint/export/descriptor authority.
Each invocation authenticates the retained lifecycle metadata and its last completed
fit, imports the genuine original runtime, constructs once, restores once, restores
the full saved RNG, and calls current `final_selected_scoring_v4.role_serving.predict`
once with literal `VALID`. The original JOB and checkpoint are unchanged.

| Case | Original producer / device | Whole VALID | Counted work | Resource reservation / internal wall bound |
| --- | --- | --- | --- | --- |
| node_C4 | runtime_v4 / CPU | 665 nodes, epoch 2 | 4 native backbone forwards | 4 GiB RSS; 32 GiB address space; 1 GiB disk; 180 s |
| NCNC_C4 | runtime_link_v6 / CUDA:0 | 60,084 positive + 100,000 negative queries, epoch 1 | 4 encoder forwards; 8 top-level predictor calls | 32 GiB GPU; 8 GiB RSS; 64 GiB address space; 1 GiB disk; 600 s |

Both use 2 CPU threads and 1 interop thread. Reservations are conservative proposals,
not measured replay costs. The retained node lifecycle used 2,803,452 KiB peak RSS
and 19.93 s for three constructions/two TRAIN/four VALID calls. The NCNC lifecycle
used 1,847,004 KiB RSS, 12,673,890,304 bytes peak CUDA allocated and 26,698,842,112
bytes peak CUDA reserved over 180.20 s with training. These observations motivate
the finite ceilings; no scaling-derived timing guarantee is claimed.

NCNC retains its full original graph and query batching. Recursive predictor work
is recorded through the producer's task scope and compared with the retained
VALID trace: 8 task dispatches and 16 recursive dispatches, with only the 8 task
output corrections applied. A recursive call is not counted as another encoder
or another top-level query batch. The replay never imposes a four-predictor-call cap.

The learned-readout logits and same-bank mean each use the unchanged
`rtol=1e-4, atol=1e-5`. IDs, labels, query endpoints and their order are exact.
Complete adapter restoration, saved RNG, model parameter/buffer values and objects,
optimizer state, and TRAIN update clock are checked. Backward, gradient collection,
TRAIN callbacks, optimizer steps and extra validation callbacks are blocked during
the serving call. A numerical mismatch is retained with the fresh `VALID.pt`; no
retry, tolerance adjustment, reselection or new fit is performed.

This closes only genuine retained producer/current serving coverage if it passes.
It does not turn old checkpoints into current portable runtime inputs, authorize
TEST, or constitute scientific accuracy evidence. The NCNC role contract is a
minimal VALID view of the exact native graph/TRAIN/VALID receipts in its retained
producer metadata; it has no TEST entry. Node retains its original transductive
graph and masks and scores only the VALID mask.

## Root handoff

1. Review and publish the frozen package and its pinned source dependencies using
   the existing root workflow. Verify only the dedicated allocation route before
   project reads: hostname `anogena-2-0`, sole physical GPU
   `GPU-44039938-fd82-41d2-fefd-de71514e2fac`.
2. Make a separate admitted copy of the appropriate template. Fill the exact
   manifest and INPUTS SHA256, qualified interpreter path/review receipt, a fresh
   peer/resource review receipt, and a fresh output path under `qualification/`.
   Set `status` to `ADMITTED` only after root admission. Never edit the templates.
3. Account for existing owners and growth envelopes before each case. For NCNC,
   root must hold the established shared GPU slot
   `analysis_outputs/selected_VALID_collection_gpu_slot.lock` throughout the child.
   This entry point neither acquires the slot nor launches or queues other owners.
4. Root's existing child supervisor must enforce the RSS/GPU/disk reservation,
   TERM/KILL timeout and cleanup of this own process group. The worker enforces
   address space, native operation counts and internal wall timeout; CUDA allocator
   fraction limits allocator memory, not all driver memory. Post-run peak RSS is
   also gated. Reserve the full listed envelope until child exit.

Execute each case in a fresh process after its own admission. Set `REPLAY_PYTHON`
to the exact interpreter recorded in that admission, and `REPLAY_ADMISSION` to the
admitted JSON path. These are allocation-local child commands for root's existing
supervisor, not remote launch instructions:

```sh
REPLAY_ROOT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
REPLAY_PACKAGE="$REPLAY_ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/qualification/final_selected_node_NCNC_VALID_v1"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=10s 190s "$REPLAY_PYTHON" "$REPLAY_PACKAGE/replay.py" --root "$REPLAY_ROOT" --admission "$REPLAY_ADMISSION" --case node_C4
```

For the separately admitted NCNC child, with its GPU slot held:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=10s 610s "$REPLAY_PYTHON" "$REPLAY_PACKAGE/replay.py" --root "$REPLAY_ROOT" --admission "$REPLAY_ADMISSION" --case NCNC_C4
```

Preserve `START.json`, `EVENTS.jsonl`, `RESULT.json`, `VALID.pt` when emitted,
supervisor logs and incomplete attempts. An absent terminal result is an incomplete
attempt, never permission to retry. No TRAIN/TEST/score-fit API is called here.
