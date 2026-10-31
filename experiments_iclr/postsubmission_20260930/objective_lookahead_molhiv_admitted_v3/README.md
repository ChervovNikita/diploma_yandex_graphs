# Separately admitted GPU lookahead bridge, v3

This is executable source for the fixed 48-trajectory MolHIV mechanism screen.
Source preparation is complete only after the stdlib readiness check passes.
No real input has been admitted or training launched by this source preparer.
The earlier v1/v2 source-only entry points remain unchanged and permanently
refuse real inputs in their own generations.

## Scientific budget

All six seeds/fractions use their freshly qualified, same-cell GPU P-only
epoch-25 tied checkpoint. Each cell makes eight fresh AdamW continuations:
own/pooled objectives crossed with tied/first/middle/last complete-block
placements. Every continuation performs exactly eight scheduled minibatch
updates. The seed-704 split of all Q scaffold/molecule groups is frozen before
the updates. Every Q-screen graph is scored. Gains compare with the actual
same-family tied-after-eight trajectory. The original numerical/practical
margin, three-seed requirement, full-label harm guard, abstention and fixed
first/middle/last tie rule are unchanged. This screen never accesses validation
or test graphs and never establishes predictive superiority or novelty.

The native update/scoring arithmetic and order mirror the sealed v2 GPU code.
New capture records every initial, step-1 through step-8, and final state, full
AdamW state, gradients, member logits, CPU/GPU RNG and loss/schedule trace.
Real execution retains 480 snapshots across 48 trajectories. The new synthetic
gate compares the sibling against v2 and repeats all its snapshots exactly.
Actual file sizes and supervised execution/replay times count as cost.

P25 checkpoints were produced with the parent probe's declared runtime flags.
K8 continuation deliberately uses the separately gated deterministic recipe:
float32, one A10080GB, no AMP/TF32, one intra/inter-op thread and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each cell records both sets of flags. The
continuation has fresh optimizer history; it does not claim to resume P25 AdamW.

## Closed boundary and stage order

1. `runner.py prepare --output <new JSON>` writes hashes and an incomplete
   binding template, with no torch/provider import or tensor access.
2. Run `runner.py synthetic-gate --output <new report> --supervisor <current
   supervisor>` through the exact corrected-route `run_authorized_v2.py` and
   repository environment wrapper. The synthetic fixture has no data input.
   Wait for its completed authorization before constructing a real binding.
3. The coordinator supplies a new real binding and exact approval JSON. The
   binding includes the complete fresh six-probe/full-replay queue result,
   queue qualification and completed queue supervisor, plus the existing v2
   and new v3 GPU gate reports and completed supervisors. Every reference is
   `{path: <phase-relative path>, sha256: <exact digest>}`. Binding schema:
   `gnnm-molhiv-lookahead-real-binding-v3`; `source_sha256` is the complete
   preparation manifest's exact source dictionary. Missing or old inputs fail.
4. `runner.py execute --binding <binding> --output <new run directory>
   --supervisor <current supervisor>` repeats source/runtime/export/prerequisite
   semantics and qualifies all six P25 contracts, masks, grouped folds,
   immutable artifacts, checkpoint records and full replay chains **before
   importing the target runtime or loading tensors**. It then fits all 48,
   saving an immutable run manifest. The summary remains provisional and emits
   no mechanism choice while its current authorization is pending.
5. After the execution guard exits successfully, run `runner.py replay
   --binding <binding> --run <run directory> --execute-supervisor <completed
   execution supervisor> --output <new replay JSON> --supervisor <current
   replay supervisor>`. It independently reconstructs the qualified source
   and inputs, refits all 48 K8 trajectories and compares all 480 full tensor
   snapshots plus every screen forward/loss/logit/trace exactly. A drift is a
   recorded failure; tolerances are never relaxed after outcomes.
6. After the replay guard exits successfully, run `runner.py finalize
   --binding <binding> --run <run directory> --execute-supervisor <completed
   execution supervisor> --replay <replay JSON> --replay-supervisor <completed
   replay supervisor> --output <new MECHANISM_LOCK_v3.json> --supervisor
   <current finalizer supervisor>`. This stdlib stage checks both completed
   chains and replays the fixed selection on all 48 gains. Its lock includes
   both objective choices and the exact provenance/cost references.
7. Wait for the finalizer's guard to exit. Before a downstream exporter or
   predictive runner uses the lock, call `admission.qualify_completed_lock`
   to check the **finalizer's completed chain too** and replay all selection
   semantics. All source, binding, execute, replay and finalizer hashes must
   match. This API imports/deserializes no torch/provider/tensors and returns
   an `Admission` object with `.descriptors` (six cells), `.evidence['bundle']`,
   `.evidence['protocol']` and `.evidence['completed_mechanism_lock']`.

No stage overwrites or resumes outputs, retries failures, schedules from
outcomes, accepts wrong-account artifacts, alters masks or loads validation.
Admission repeats the existing correct-route semantic qualifier and may require
an idle authorized GPU. A trusted coordinator's SSH-route attestation plus
hardware/source/supervisor checks is the scope of this authorization; it is
not an OS-level isolation proof or exclusive GPU reservation.

## Exact coordinator approval JSON

The `coordinator_approval` reference must resolve to exactly these fields:

```json
{
  "schema": "gnnm-molhiv-lookahead-coordinator-approval-v3",
  "approved_source_sha256": "<replace with complete source dictionary>",
  "queue_result_sha256": "<binding queue_result digest>",
  "queue_qualification_sha256": "<binding queue_qualification digest>",
  "v2_gpu_gate_sha256": "<binding v2_gpu_gate.report digest>",
  "v3_gpu_gate_sha256": "<binding v3_gpu_gate.report digest>",
  "stages": ["execute", "replay", "finalize"],
  "training_only": true,
  "maximum_trajectories": 48,
  "optimizer_steps_per_trajectory": 8,
  "automatic_retry": false,
  "previous_wrong_account_evidence_admissible": false
}
```

This template is illustrative and never itself qualifies an execution. The
coordinator must provide all fresh evidence and the exact source dictionary.
All current-job authorization files are created by the guard after the child
terminates, so premature output files cannot qualify downstream work.

## Scope and downstream use

The final lock maps abstention to tied and a single chosen index to its fixed
placement. It licenses only a separate predictive study under the integration
plan. The full 30-fit primary matrix, later mandatory controls, uncertainty and
independent confirmation requirements remain. Existing paper scores and the
failed spectral rule are unchanged. This source package is not a reported
research result or a new-method claim.
