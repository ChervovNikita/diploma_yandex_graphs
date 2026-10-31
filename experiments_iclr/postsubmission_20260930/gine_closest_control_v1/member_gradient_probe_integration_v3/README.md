# Training-only member-gradient integration

Status: source preparation only. AST and metadata checks are complete; **Torch numerical qualification, checkpoint loading, and GPU execution have not run**. This packet does not admit or launch an operation. It reuses `member_gradient_probe_v1/probe.py` unchanged and copies no scientific source or data.

## Scope and existing inputs

`EXISTING_INPUT_IDENTITIES.json` lists exactly 48 states: Reddit2 and ogbn-products; seeds 17/29/43; exact-member, shared-message, D64 single and parameter-matched single; INITIAL epoch 0 and FINAL epoch 300. Checkpoint path/hash references and source fit metadata come from the completed `shared_message_predictive_v3/science_authorized1gpu_run01` fits. Their numerical contents were not opened during preparation. Selected checkpoints are not used.

The existing scientific binding, train-only acquisition exports, complete CSR preparation, source generation, precision policy, package versions, checkpoint flags, and GPU UUID are retained. The runner uses the existing train-only loader. It neither calls validation-freeze/target loaders nor opens logits or public provider storage. Public validation/test split IDs may be present in the acquired structure; their target values are absent from that export and unused here.

The separate caller policy adds only this runner and the unchanged helper to the inherited source hashes. It preserves the inherited package roots and scientific source files. Output contains scalar training losses, gradient Gram/cosine summaries, factor-route norms, operation counts, state hashes, source/data/runtime provenance, and timing/memory receipts. It exports no logits or gradient tensors and constructs no optimizer.

## Three ordered supervised stages

1. **`qualification`** uses a deterministic 11-node ring and real source model classes. The synthetic graph is only a numerical fixture. It checks all four source architectures. It checks dropout-free checkpointed train-mode forward against ordinary eval/no-grad; exact serial pooled cotangent/VJP against native packed pooled-CE autograd; and repeated fixed DropoutKey forwards plus preservation of parameter state, mixed module flags, dropout probability, RNG states and empty `.grad` fields. It additionally compares helper Gram summaries against native autograd under p0 and one fixed key. The qualified helper remains unchanged.
2. **`products_memory`** loads the existing complete Products training export and shared seed 17 FINAL checkpoint. It executes one full helper probe with the source's activation checkpointing retained, recording preparation and probe GPU peaks. This checks the largest established architecture/graph shape; it does not establish that every state has the same peak. The existing 60 GiB reserved-memory ceiling must hold. Any failure stops continuation and stays in the cost record.
3. **`primary48`** runs every fixed state in the listed order with p0/train mode, retaining each source checkpoint flag. Graph preparation is reused within a task. GPU model state is released between states, and graph/data state is released between tasks. Results are written per state before the next one. This stage requires successful completed terminals and qualified result files from both earlier stages. No secondary actual-data family is included.

The primary reruns the memory-gate state as part of all48. This preserves the complete-family order and avoids treating a separate gate as a selectively substituted result. That one duplicate diagnostic is included in the estimate.

## Root integration and launch environment

Root must review the source and create a separate approved request from `REQUEST_TEMPLATE.json`. It must provide current accounting and an approval receipt binding the canonical JSON request payload, excluding its `root_approval` field. The template intentionally has `root_authorized=false`, `completed_development_family_qualified=false`, and no accounting/approval reference.

Current accounting must include every previous success, failed attempt, preparation and the current upload. It must not reset the existing combined phase budget. All three whole-child caps must fit before stage 1; each later stage charges actual prior outer wall, including an over-cap failure. A failed or terminal attempt is not automatically retried, resumed or promoted. A partial primary is incomplete evidence.

Use the original repository interpreter and `protocols/repo_env.sh`, with:

```sh
CUDA_VISIBLE_DEVICES=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONDONTWRITEBYTECODE=1
GNNM_SSH_DESTINATION=<the existing authorized destination supplied by root>
```

Each stage must use the existing `bounded_run_v1.py` outer wrapper and `run_authorized_v2.py --require-idle-gpu` inner supervisor. `LAUNCH_ARGUMENT_TEMPLATES.json` gives their argument structure; it is not a launcher. Replace its three tokens with the remote repository root, remote phase root, and separately approved request. The stage output/supervisor/outer directory names are fixed in the request template. The original repository allocation and repo environment remain authoritative; no SSH occurs in this packet's preparation.

The runner's `RESULT.json` is provisional until its outer terminal and supervisor are complete. Root should inspect each successful gate and its receipts before invoking the next stage. `FAILURE.json`, partial per-state outputs, stdout/stderr, inner receipts and outer terminal remain available for failed-work accounting.

## Resources and qualification limits

The existing fit-based proxy gives 99.34 seconds of numerical work for the48-state primary, with a twofold allowance of198.68 seconds. The extra Products memory-gate probe adds about10.01 proxy seconds, or20.01 seconds with that allowance. Tiny-fixture qualification, interpreter/bootstrap, complete graph/data/checkpoint hashing, loading, scalar reduction, serialization and supervisor teardown are additional and **unmeasured**. A provisional end-to-end estimate is roughly6–18 minutes for all stages after deployment; the actual Products gate will refine it. This is not a guarantee that primary preparation plus all48 operations fits its600-second child cap.

Frozen ceilings: three600-second whole-child envelopes with five-second termination grace; a2400-second diagnostic allowance including its prior preparation/failures;64 MiB additional diagnostic artifacts; and the inherited combined21600-second phase,200 GiB phase disk and60 GiB reserved GPU limits. CPU threads, precision and installed package versions come from the existing binding. The present packet has no authority to increase these limits. If the first measurements cannot fit the remaining cap, root must stop and assess resources rather than drop states or widen a cap silently.

These are instantaneous gradients of full training-member CE and pooled CE at two endpoints. Negative member dots identify first-order opposition for the specified shared block; they do not show that the actual AdamW step or final predictor was harmed. A small coherence ratio alone is not proof of conflict. FINAL exact and shared states followed different trajectories. The diagnostic has no same-weight/same-forward full-versus-own-only adjoint comparison and measures no actual AdamW directional response. Thus gradient geometry and connected-path existence cannot identify harm caused by peer feedback. Those limitations are explicit in the request's fixed policy. No validation/test score is recalculated.

The optional own-only message adjoint described in the earlier helper README is not implemented here. Removing cross-member adjoints would change gradients of earlier shared weights as well as member factors; it would not isolate exclusively private-factor learning. Such a derivative attribution control requires measured motivation and a separate plan, and would still not establish a novel learning method or an effective training remedy.

## Version 2 qualification repair

Version 1 failed on the 11-node numerical fixture because its native oracle reused a retained checkpointed graph for several VJPs. The pinned checkpoint ledger context is a one-shot generator. Version 2 uses a fresh packed forward for each native own-loss VJP and another fresh packed pooled-loss VJP, keeping parameters, mode, dropout probability and DropoutKey fixed. The original helper is unchanged. Checkpointing, tolerances, memory gate, all48 states, caps and interpretation are retained. Version 1 outputs and requests remain terminal; this is a fresh source/request/output generation. Version 2 reached the fresh native/serial checks, then failed because provenance snapshots supplied CUDA model tensors to a CPU-only fingerprint. No actual graph/checkpoint probe ran.

## Version 3 fingerprint repair

Both model fingerprint call sites now pass detached contiguous CPU copies to the inherited CPU-only fingerprint. The CUDA model state is unchanged. This is a provenance snapshot repair, preserving the numerical checks, helper, data, modes, gates, caps and tolerances. v1/v2 requests and outputs remain terminal. v3 is a fresh source/output generation; its numerical qualification has not run.
