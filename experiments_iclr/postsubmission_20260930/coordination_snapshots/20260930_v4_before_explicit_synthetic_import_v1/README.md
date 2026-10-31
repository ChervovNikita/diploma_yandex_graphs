# Single final-replay technical recovery bridge, v4

This separately versioned source accepts one narrowly defined technical
recovery of the original queue: all six probe fits succeeded and five complete
checkpoint replays succeeded; the final seed-2/full-label replay was refused
by the allocation guard before its child or inner supervisor was created.
The original failed queue, missing original replay outputs and v3 source remain
unchanged. No source is retrained and no subset is chosen from outcomes.

The actual coordinator recovery reason preceded the single manual replay.
That reason did **not** separately pin the final artifact-manifest digest.
V4 records this limitation explicitly. It checks the current complete original
fit manifest, its checkpoint/record identities, the original authorized probe
chain, and the completed unchanged replay/autograd chain. It never describes
the missing separate prelaunch digest as existing.

## Unchanged method and interface

The complete K8 screen still comprises six source cells × eight arms = 48
trajectories. Own/pooled losses, tied/first/middle/last complete-block copies,
seed-704 grouped Q split, all screening graphs, actual same-family tied-after-K
comparisons and the fixed selection rule are unchanged. `admitted_core.py` is
byte-identical to v3. Execution saves 480 full initial/step/final snapshots;
independent replay compares tensors, gradients, optimizer histories, logits,
RNG and traces exactly. No validation/test graph or predictive score enters
this package. All P25, gate, failure, recovery, K8 and replay costs remain.

The method output interface retains schema
`gnnm-molhiv-qualified-lookahead-mechanism-lock-v3` because its mathematical
semantics are unchanged. The lock explicitly adds `admission_generation: 4`
and `technical_recovery` provenance; its exact source mapping binds v4.
`admission.qualify_completed_lock(lock_path)` returns `Admission` with
`.descriptors`, `.qualified_gpu_uuid`, `.evidence['bundle']`,
`.evidence['protocol']` and `.evidence['completed_mechanism_lock']`. It repeats
all-six input admission, execute/replay/finalizer completed-chain checks and
all-48 selection replay without importing torch/provider or loading tensors.

## Source preparation and exact stage order

Every executed stage uses the original corrected `anogena-2` account, one
authorized A10080GB, the unchanged repository environment wrapper and
`protocols/run_authorized_v2.py`. Current authorization is created after child
exit; later stages must wait for each completed guard chain. The seven-GPU
account is forbidden. The source preparer launches no remote jobs.

1. `runner.py prepare --output <new preparation JSON>` emits source hashes and
   incomplete request/binding templates, using stdlib only.
2. Run `runner.py synthetic-gate --output <new GPU report> --supervisor
   <current gate supervisor>`. It compares the v4 native engine with v2 and
   exactly replays all synthetic snapshots. Require its completed guard.
3. The coordinator supplies a recovery request with **exactly** these fields:

   ```json
   {
     "schema": "gnnm-molhiv-single-replay-technical-recovery-request-v4",
     "source_sha256": "<complete preparation source dictionary>",
     "queue_qualification": "<path/sha256 reference>",
     "coordinator_reason": "<actual pre-execution reason reference>",
     "original_queue_failure": "<original failure.json reference>",
     "original_queue_events": "<original events.jsonl reference>",
     "original_final_replay_launcher": "<original seed2_label100_replay_launcher.log reference>",
     "original_queue_supervisor_directory": "<phase-relative original failed supervisor>",
     "new_final_replay": {
       "report": "<fresh completed recovery report reference>",
       "supervisor_directory": "<fresh completed phase-relative recovery supervisor>"
     }
   }
   ```

   References contain exactly `{path: <phase-relative path>, sha256: <digest>}`.
   The single recovery replay is the original `probe.py replay`, using the
   original TRAIN bundle and final immutable run, with separate new output and
   supervisor paths. There is no general retry or fitting API in the assembler.
4. Run `runner.py recovery-assemble --request <request JSON> --output <new
   RECOVERED_QUEUE_RESULT_v4.json> --supervisor <current assembler supervisor>`.
   It checks the failed queue's completed exit-1 chain, exact 36-event schedule,
   guard-refusal log, genuinely earlier reason, still-missing original replay
   outputs, six original fit chains and six complete replay chains. It
   publishes a separate recovered result and retains the original failure.
5. After the assembler completes, supply a real binding with exactly:

   `schema: gnnm-molhiv-lookahead-real-binding-v4`, `source_sha256`,
   `coordinator_approval`, `queue_qualification`, `recovered_queue_result`,
   `recovery_assembler_supervisor_directory`, `v2_gpu_gate`, `v4_gpu_gate`.
   Each gate is `{report: <reference>, supervisor_directory: <phase-relative>}`.
   The coordinator approval is exactly:

   ```json
   {
     "schema": "gnnm-molhiv-lookahead-coordinator-approval-v4",
     "approved_source_sha256": "<complete source dictionary>",
     "recovered_queue_result_sha256": "<recovered result digest>",
     "queue_qualification_sha256": "<qualification digest>",
     "v2_gpu_gate_sha256": "<v2 gate report digest>",
     "v4_gpu_gate_sha256": "<v4 gate report digest>",
     "stages": ["execute", "replay", "finalize"],
     "training_only": true,
     "maximum_trajectories": 48,
     "optimizer_steps_per_trajectory": 8,
     "single_final_replay_technical_recovery_allowed": true,
     "no_source_retraining": true,
     "automatic_retry": false,
     "previous_wrong_account_evidence_admissible": false
   }
   ```

6. Run `runner.py execute --binding <binding> --output <new run directory>
   --supervisor <current execution supervisor>`. All recovery/source/gate
   semantics and complete six-cell admission pass before tensor loading.
7. After execution's guard completes, run `runner.py replay --binding
   <binding> --run <run directory> --execute-supervisor <completed execution
   supervisor> --output <new replay JSON> --supervisor <current replay
   supervisor>`. All 48 trajectories and 480 snapshots must replay exactly.
8. After replay's guard completes, run `runner.py finalize --binding <binding>
   --run <run directory> --execute-supervisor <completed execution supervisor>
   --replay <replay JSON> --replay-supervisor <completed replay supervisor>
   --output <new MECHANISM_LOCK_v4.json> --supervisor <current finalizer
   supervisor>`. It emits the unchanged decision plus explicit recovery lineage.
9. Before any downstream validation/predictive use, wait for the finalizer's
   completed guard and call `qualify_completed_lock`. Premature files, source
   changes, incomplete replays or altered recovery lineage are rejected.

All output paths must be new and inside this v4 packet; input references stay
inside the authorized project phase. No output is overwritten or resumed.
The shared scientific model, original scores and failed spectral criterion
remain unchanged. Source-readiness checks are not real/GPU execution evidence.
