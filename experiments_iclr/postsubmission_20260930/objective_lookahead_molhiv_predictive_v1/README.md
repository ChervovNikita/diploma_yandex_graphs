# Primary MolHIV predictive runner — source-only v1

This new sibling prepares the **unchanged 30-fit primary matrix** in
`objective_lookahead_molhiv_v1/INTEGRATION_PLAN_v1.json`: seeds0/1/2 × own/pooled
training objectives × tied/first-private/middle-private/last-private/declared
low-rank, at10% labels,100epochs. It does not launch a job, load real data, open
checkpoints, change an original source, or produce a predictive result.

The public `preflight128`, `train` and `replay` commands refuse before importing
torch or opening their binding. There is no mutable source-only switch or CLI
override. Root must supply qualified inputs and approve a **separate sealed
admission entry point**. That entry point can use the engine functions below;
the source-only runner is not silently promoted or edited to admit data.

## What is implemented

- Original `molhiv_pilot.py` mask construction is called unchanged: largest
  fractional-remainder class quotas, class RNG101/102, sorted official training
  positions. The entire selected official-training subset is fitted.
- Canonical fresh tied initialization for each optimizer seed. Private copies
  and zero-delta low-rank factors are constructed before a fresh AdamW optimizer.
  P25 weights never enter a predictive model.
- Original width128/L3/M4/multiplier2/dropout0.1, AdamW0.001/decay0,
  betas(0.9,0.999)/epsilon1e-8, batch128,100epochs. Float32, deterministic
  algorithms, no AMP/TF32, one thread/inter-op thread, no workers.
- Original DataLoader-generator ordering `19709+1000*seed+epoch`; actual batch
  IDs are saved. A continuous device dropout stream is seeded once by the
  optimizer seed; model construction consumes the canonical CPU stream and the
  low-rank adapter uses its private generator. There is no per-batch reseeding.
- Exact original own-member finite-target/equal-graph BCE and direct pooled-logit
  BCE. Both training losses are recorded for every arm. Separate objective
  families remain separate; no favorable objective is selected from validation.
- Full official validation after each epoch, maximum pooled ROC-AUC, minimum
  pooled BCE, earliest-epoch ties. All100 checkpoints and member logits,
  targets/IDs are retained. Selected and epoch100 metrics are explicit.
- Per-fit original/new source archives, runtime class source, full environment,
  data/recipe/admission hashes, initial state/RNG, optimizer states and complete
  artifact coverage. A fit stays provisional until its completed authorized
  supervision and independent replay are qualified.
- Replay audits every epoch's retained metric and full trace/selection, then
  loads both the selected and epoch100 checkpoint states and performs full
  same-device official-validation forwards with absolute tolerance1e-6. The
  selected forward is reused when selection is epoch100. Both endpoint metrics,
  hashes and maximum member-logit errors are retained. No test evaluator is
  implemented.

The declared low-rank source is the sealed **v2 compatibility repair**, retaining
the v1 method and private initialization distribution. It enumerates actual
rank0–128 counts without data, matches the full copied-private middle block
within1%, and uses factor seed`29001+1000*optimizer_seed`. It is a declared
TabLoRA-inspired GINE adaptation. Pooled training is an explicitly different
objective control, not an exact reproduction of the original TabLoRA recipe.

## Admission interface for root

`contract.audit_future_binding(value)` verifies references and completed
correct-route receipt chains. It is a **format/provenance audit, not real-data
admission**. A separately qualified semantic report must prove P25
checkpoint/record/replay identity, all six method locks before validation, full
train/validation ID/tensor identity, exact masks101/102, and mechanism replay.

The future binding schema is `gnnm-primary-molhiv-future-admission-v1`. Its
required fields are:

| Field | Required contents |
|---|---|
| `source_sha256` | Exact `contract.sources()` mapping |
| `coordinator_approval`, `admission_entrypoint`, `semantic_admission_report` | Root-supplied immutable `{path,sha256}` references; no approval or report is fabricated here |
| `six_P25_cells` | Six distinct seed/fraction cells (seeds0–2; fractions0.1/1.0), epoch25, artifact references and completed correct-route supervisors |
| `six_training_method_locks` | Same six distinct cells with artifact references, no validation/test use, completed correct-route supervisors |
| `mechanism_lock` | Hash reference to qualified v3 lock supplied by the adaptive agent |
| `qualified_gates` | Report references and completed correct-route supervisors for fresh model/low-rank gates |
| `train_export`, `validation_export` | `{bundle:{path,sha256}, supervisor_directory:...}` for split-only exports |
| `data_freeze` | Hash reference to actual full split-ID/tensor hashes and unchanged `train_budget_manifest` |
| `runtime_freeze` | New runtime/structural freeze emitted by this engine and subsequently authorized |

P25 and method-lock rows have `seed`, `label_fraction`, `artifacts` mapping of
`{path,sha256}` references, `supervisor_directory`, and
`validation_or_test_used:false`. P25 rows additionally have `epoch:25`.
Every path is phase-relative and hash checked before deserialization.

The exact v3 mechanism lock is coordinated with the adaptive agent. The parser
requires its qualified status, all six cells, all48 replayed trajectories, both
own/pooled families, and hash-matched **execute and replay** authorization/
completion chains. It also requires the finalizer's completed authorized chain:
`finalizer_live_supervisor` must retain its exact command/environment hashes and
pending declaration, and the command's single `--output` must resolve to this
referenced lock. A still-running finalizer is not qualified. Root's semantic
admission must use the adaptive packet's `qualify_completed_lock` API to replay
all48 gains and the selector choice. Abstention maps to tied; a singleton index
maps to its fixed placement. Multiple or inconsistent indices are rejected. No
adaptive alias is chosen from validation, and this runner never recalculates the
mechanism choice.

The existing v6 training-only export is supported directly: its graph dictionaries
are converted to PyG `Data` objects only after authorization/hash checks. A new
validation-only export must use schema
`gnnm-molhiv-official-validation-only-bundle-v1`, `dataset:ogbg-molhiv`,
`official_validation_indices`, `graphs`, and an `export_scope` with
`graph_targets_exported:"official validation only"`,
`test_target_indexing_or_summary:false`, `scores_read:false`. Its semantic
qualification must match the full official validation split. **No general OGB
dataset provider is constructed by the engine.**

The new data freeze records `official_train_indices_sha256`,
`official_validation_indices_sha256`, `full_official_train_tensor_sha256`,
`full_official_validation_tensor_sha256`, and `train_budget_manifest` using the
engine's `tensor_hash`/`fingerprints` and the unchanged inherited masks. Root's
admission must compare these actual fingerprints with the qualified exporter/
official-provider evidence before accepting this derivative. A hash alone does
not restore qualification to a wrong-allocation artifact.

After separate admission, call `configure()` once in a fresh process, then:

```python
real128_preflight(binding, output_json, current_supervisor_directory)
run_fit(binding, seed, objective, layout, output_directory, current_supervisor_directory)
replay(run_directory, binding, output_json, completed_fit_supervisor_directory,
       current_replay_supervisor_directory)
```

The actual128-graph preflight uses the first128 selected observed official
training positions under the exact10% mask. It checks all five initial member
functions and performs one real AdamW update under each objective/layout. It
opens no validation graph and is engineering evidence only.

Every operation must use `protocols/run_authorized_v2.py` on the supplied
`anogena-2` destination and one A10080GB. The seven-GPU account is forbidden.
The running child's `authorization.json` does not exist until its wrapper exits;
the engine therefore retains live command/environment hashes, marks the fit
provisional, and requires the matching completed authorization before replay/
qualification. It cannot mistakenly accept an unrelated authorized job.

## Verification status and remaining work

Local checks cover AST parsing, the exact30-cell design, source preservation and
pre-input real-command refusals. Local Python has no torch/PyG/OGB, so numerical
gates were **not executed here**. `verify_synthetic.py` is ready for the qualified
runtime and checks independent analytic objective gradients, complete/private
state and storage, optimizer identity, actual learning in all10 cells, paired
dropout, live A/B factors, rankzero update identity, checkpoint replay and
selection/order oracles. Source-only checks are not predictive evidence.

Root still supplies/qualifies the validation export, all six P25/recurrence locks,
the completed v3 mechanism lock, semantic admission and new execution entry
point. `COMMANDS.md` gives bounded preparation/gate commands. The root's separate
primary analysis handles uncertainty and Holm correction; the engine provides
full member evidence for later descriptive AP/Brier without changing selection.

## Runtime and storage accounting

No predictive throughput measurement is available from this source packet.
Each completed fit records observed end-to-end seconds, training graph visits,
effective graph visits per second, checkpoint bytes, validation-evidence bytes,
and CUDA peak allocated/reserved bytes. Runtime includes all100 full validation
passes and evidence serialization. It must not be presented as pure training
throughput or extrapolated from the P25 training-only probe without measurement.

For planning, each epoch checkpoint's raw parameter/AdamW payload is about
`12*P` bytes for `P` float32 trainable parameters, plus buffers, scalar optimizer
state, RNG, and serialization overhead. Each validation-evidence payload has
four float32 member logits and one float32 target plus one int64 graph ID per
validation graph: about `28*N_validation` bytes. The100 checkpoints dominate
storage. Using the source-defined counts, the30-fit matrix has approximately
14.7GB of parameter/AdamW payload alone; root must use the qualified runtime's
actual counts and the saved file sizes for resource decisions. Selected/initial
copies, sources, buffers and serialization add further storage. This is a
structural storage estimate, not an observed resource result or training ETA.
