# Serialized checkpoint inference replay: local source preparation

Status: **source prepared; execution has not been authorized or performed**.

This packet covers exactly the 30 retained cfg0/K4 cases: Photo and Squirrel,
seeds 17/29/43, and graph/common_only/random_tangent/topology_permuted/warm_copy.
The fixed cohort comes from the closed source comparison and the physically
completed V10 saved-array source-VALID audit. The canonical registry, original
results, old failures, and V10 audit are read-only inputs.

## Native inference construction

The entrypoint imports the unchanged, hash-pinned original driver only after
root release and source review. It reuses `source_guard`, `load_runtime`,
`canonical_graph`, `sync`, `clone_boundary`, `raw_arguments`, `alias_inventory`,
`storage_receipt`, and `difference` from the admitted source.

For **each case**, it performs the following construction:

1. Reopen the exact context, fit/selection metadata, matched warm freeze/receipt,
   graph manifest, role freeze, and qualified preprocessing certificate.
2. Recompute the full native graph preprocessing with the original adapter.
   Require the entire preprocessing receipt to equal the retained qualified
   receipt and the fit receipt. The raw feature/edge payloads, dimensions,
   source, and frozen runtime remain bound by their hashes and identities.
3. Load the matched warm checkpoint with the original safe loader arguments,
   `torch.load(..., map_location=rt.device, weights_only=True)`. Construct
   `TeacherFamily(warm['specification']).to(rt.device)`, strictly restore its
   complete native state, and restore its mode and stage. Require its
   specification to equal both the warm receipt and original cfg0 single-author
   specification for that exact seed/backbone.
4. Call the original `clone_boundary(native, boundary_api, 4)`. Then reopen that
   arm's **selected** checkpoint with the same safe loader and strictly install
   its complete state. Check every state key, shape, dtype, finite value, and
   restored tensor bytes. All R/S/B factors are included for every arm. The
   warm identity factors created by cloning are overwritten by the selected
   factors; no factor operation follows selected-state loading.
5. Validate serialized schema, arm, context, selected epoch/update/NLL metadata,
   ordinary named Adam coverage, alias inventory, and storage ownership. Both
   inactive and active Photo heads remain registered and restored. Explicitly
   enable the selected Photo global branch.
6. Put every native module in `eval()` and run the original complete forward
   under `torch.no_grad()`. Require finite float32 `[4,N,C]` logits, compare the
   complete tensor, save the independently inferred array in the new output,
   and verify the selected state remains byte exact after inference.

Optimizer metadata is checked without creating an optimizer or restoring its
state. Training RNG is unnecessary for this deterministic eval: the pinned
native bodies use dropout only when `training=True`, and all complete selected
tensors are installed after constructor RNG consumption. No train, validation,
or held-out label pack is opened. The already completed V10 source-VALID score
audit remains a separate immutable evidence link; this replay does not
recompute a score or select a checkpoint.

## Comparison contract

Record both complete tensor payload byte equality and `.npy` file byte equality.
The prospective acceptance comparison is the original integration's complete
logit comparison: `torch.allclose(inferred, saved, atol=1e-6, rtol=1e-5)`, with the
saved selected logits as reference. These values are the existing declared
`TOLERANCES` and are checked at runtime. There is no outcome-driven tolerance
widening. Finite float64 error diagnostics are also required. A finite logit
mismatch is retained and the remaining cells are compared; the child returns
failure after the full cohort. A construction/runtime error preserves all
completed rows and the next case in the failed child result and stops that run.

## Root release and physical closure

`ADMISSION_TEMPLATE.json` is deliberately disabled. Root must issue a separate
admission outside this source directory, bind the exact manifest and an
independent review, release the resource bounds, and dispatch the exact child
through root's existing physical custody mechanism. The source includes no
process launcher, supervisor, SSH client, isolation requirement, or ledger
operation. It requires a fresh dedicated replay output.

The root review JSON must contain
`approved_for_checkpoint_replay_execution: true` and `source_manifest` equal
to the admission's exact descriptor. Other review fields are unrestricted.

The child writes `STARTED.json`, append-only `CASE_RESULTS.jsonl`, all 30 inferred
arrays on a successful run, and `INFERENCE_RESULT.json`. Even a successful child
result has `accepted_checkpoint_verification=false` and `physical_closure=false`.
After root has actually reaped its own child, root supplies the measured
physical receipt described in `PHYSICAL_CLOSURE_CONTRACT.json`. `close` then
rechecks source/dependency/output hashes, all 30 outcomes, actual route probes,
exit, timestamps, and released wall/memory bounds. Only this operation creates
the one final `SERIALIZED_CHECKPOINT_REPLAY_RECEIPT.json` with physical closure.
This is checkpoint verification; its receipt sets `manuscript_acceptance=false`.

The physical source route is the retained sole GPU
`GPU-44039938-fd82-41d2-fefd-de71514e2fac` at
`anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru`, port 2222, and the exact
repository Python 3.11.14 environment. Frozen settings include
`cudnn_allow_tf32=true`, `cuda_matmul_allow_tf32=false`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. `load_runtime` requires the entire original
environment dictionary; it does not replace these settings with new values.

## Exact future child commands

These are commands for root after review/release. They have not been run. Root
must place the eight manifested payloads and `MANIFEST.json` at the corresponding
remote phase directory, preserving their bytes. Use the existing authenticated
SSH route and root custody; the private-key path is
`/Users/alex/.ssh/mlspace__private_key_anogena.txt`. The authentication-only
options are `IdentitiesOnly=yes`, `BatchMode=yes`, `UpdateHostKeys=no`, and
`StrictHostKeyChecking=yes`.

Run this exact child under root's bounded physical process custody:

```sh
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_init_serialized_checkpoint_replay_preparation_20261003_v1/replay_selected_checkpoints.py run --admission /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_init_serialized_checkpoint_replay_release_20261003_v1/ROOT_REPLAY_ADMISSION.json
```

After root has written its real `PHYSICAL_CLOSURE.json` and measured its SHA256,
run this stdlib-only closure command with that literal digest substituted:

```sh
/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_init_serialized_checkpoint_replay_preparation_20261003_v1/replay_selected_checkpoints.py close --admission /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_init_serialized_checkpoint_replay_release_20261003_v1/ROOT_REPLAY_ADMISSION.json --physical-receipt /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_init_serialized_checkpoint_replay_supervision_root_20261003_v1/run_v1/PHYSICAL_CLOSURE.json --physical-sha256 ACTUAL_PHYSICAL_CLOSURE_SHA256
```

## Dependency and preparation evidence

`DEPENDENCIES.json` identifies the 16 unchanged source bindings, 90 metadata
text bindings, 30 selected checkpoints, 30 saved logit arrays, six matched warm
checkpoints, and four raw graph payloads. Binary dependencies are descriptors
only in this local preparation: none were opened, hashed, or loaded locally.
The canonical comparison descriptor was checked against its local byte-identical
metadata mirror because the canonical comparison directory is remote.

`PROSPECTIVE_RESOURCES.json` gives declared bounds and existing metadata evidence;
it is not a measurement of this replay. `SOURCE_PREPARATION_CHECK.json` records
the local AST and metadata checks. No source module, numerical library, science
process, GPU process, or SSH command was executed during preparation.
