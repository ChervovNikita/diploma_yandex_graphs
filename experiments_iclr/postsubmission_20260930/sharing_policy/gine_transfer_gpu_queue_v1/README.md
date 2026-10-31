# Serialized GINE selector probes on the authorized GPU

This packet schedules the unchanged v6 scientific probe on physical GPU0 of the original one-GPU allocation. Every complete cell is mandatory: seeds0/1/2 at10% labels, then seeds0/1/2 at100% labels, each25 epochs with checkpoints0/10/25. Every probe receives a full checkpoint-autograd and selector-decision replay. No predictive validation or test scoring is scheduled here.

The queue calls the byte-identical parent `probe.py`; its runs and replays remain in a fresh `authorized_gpu_probes_v1` subtree of the parent v6 folder. Queue locks, qualification, launcher logs and supervisors remain here. `scientific_reference/` contains exact code copies only. No wrong-allocation runtime/export evidence is copied into this packet or accepted as a prerequisite. The ready request names only the fresh authorized1gpu outputs supplied by root; runtime qualification remains mandatory.

## Required qualification

The sealed ready request `QUALIFICATION_REQUEST_authorized1gpu_run01.json` binds root's newly completed correct-allocation outputs and their exact bundle/protocol paths and digests. The template documents the schema only; this sealed version accepts the ready request and exact PLAN bindings. A different corrected-allocation preparation requires a new source packet. Paths are research-phase-relative and must identify fresh `_authorized1gpu_` outputs. Required completed jobs are chemistry, mapping, synthetic autograd, real128 training autograd, complete training-only export, the parent scientific freeze, and the real128 CUDA/CPU-oracle/one-AdamW-step engineering gate.

Every prerequisite must have the exact `run_authorized_v2.py` authorization record, correct SSH destination, one A10080GB, consistent GPU UUID, successful inner supervisor and immutable source/command/log hashes. Export/protocol/report identity and output-command bindings are checked. Missing qualification, an old unguarded job, mismatched source, missing full export, or a wrong route stops execution. Bundle and protocol digests can match earlier bytes exactly; only new authorization-bound jobs qualify their use. The qualifier hashes the training-only bundle but does not deserialize its target tensor.

Queue admission binds all seven role-specific supervisor/report paths to the sealed request, checks the request and qualifier source hashes, then reruns the complete semantic qualification and compares every stable field. Freeze, real128 and CUDA commands must name the exact newly authorized training bundle; the CUDA command must also bind its exact expected digest and fresh exporter supervisor. A successful supervisor cannot substitute for another prerequisite role.

Each prerequisite command must match its complete exact argv, including the repository interpreter and official provider paths for export. Extra arguments, `--help`, abbreviated option names, alternate option spellings and changed argument order are rejected.

Root launches both qualification and the queue through the exact authorized wrapper, from the repository cwd, with the repository environment sourced and `GNNM_SSH_DESTINATION` set to the corrected supplied destination. All commands below are prepared examples; correct-allocation inputs have not yet been qualified by this source preparer.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments_iclr/postsubmission_20260930/protocols/run_authorized_v2.py \
  --output experiments_iclr/postsubmission_20260930/protocols/logged_runs/gine_GPU_qualification_authorized1gpu_run01 \
  --require-idle-gpu -- .venv/bin/python \
  experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_gpu_queue_v1/qualify_inputs.py \
  --request experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_gpu_queue_v1/QUALIFICATION_REQUEST_authorized1gpu_run01.json \
  --output experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_gpu_queue_v1/qualifications/authorized1gpu_run01.json
```

After qualification passes, root launches the complete serialized queue:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments_iclr/postsubmission_20260930/protocols/run_authorized_v2.py \
  --output experiments_iclr/postsubmission_20260930/protocols/logged_runs/gine_GPU_queue_authorized1gpu_run01 \
  --require-idle-gpu -- .venv/bin/python \
  experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_gpu_queue_v1/gpu_queue.py \
  --python .venv/bin/python \
  --qualification experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_gpu_queue_v1/qualifications/authorized1gpu_run01.json \
  --output experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_gpu_queue_v1/queues/all_six_authorized1gpu_run01
```

Every child uses the exact authorized wrapper and inner supervisor. Before and after each stage, the queue requires one physical GPU0 with the qualified UUID, at most100MiB used, zero utilization and zero compute contexts. A bounded30-second wait permits stale utilization to clear; occupation or identity mismatch stops the next child. It never kills or inspects other processes. These are snapshots, not an exclusive GPU reservation or OS-level syscall confinement.

Technical failures preserve their logs and partial artifacts and halt the queue; there are no automatic retries or favorable-seed subsets. Root must audit all six complete locks and replays before any predictive validation comparison. The selector can abstain. MolHIV remains development, and train-only selection is not a predictive gain.

The unchanged parent replay checks artifact/checkpoint digests, source and grouped input identities, recomputes checkpoint C/N gradients and raw/centered Gram matrices, and recomputes the selector decision from the saved probe records. It does not independently recompute saved nonlinear virtual Adam gains or reconstruct and compare the seeded epoch-0 initialization/checkpoint metadata. "Full replay" here means the full inherited replay procedure; these additional checks remain outside its coverage.

Record full supervised probe and replay process times and their sum separately from queue wall time. Fresh export, preparation, autograd gates and freeze costs remain separate required overhead. CPU replay time and GPU probe time provide no inference or GPU-speed comparison. The inherited exporter materializes existing all-target storage and may internally materialize public CSV fields; it exports official training tensors only. Those limitations remain explicit.
