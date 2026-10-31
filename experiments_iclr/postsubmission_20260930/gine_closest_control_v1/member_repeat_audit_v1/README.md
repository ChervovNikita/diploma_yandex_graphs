# Frozen full-graph forward-repeat audit

Status: **source preparation only; no execution authorized or performed**. Root must audit the source, supply fresh accounting, approve a separately hashed request, upload, and execute through the existing supervisors. Preparation opens metadata and source text only, not checkpoint or dataset tensors.

## Exact state and purpose

This audit is restricted to the failed first primary state: **Reddit2, seed 17, `exact_member`, INITIAL epoch 0**. Its original complete graph, training-only acquisition, prepared CSR, checkpoint, package runtime, float32 policy, sole GPU UUID, serial source path, and `checkpoint_blocks=false` are retained. All v1/v2/v3 outputs and failures remain unchanged. The original helper's threshold remains `1e-6`.

The original helper created pooled cotangents from no-grad serial forwards and later rejected a grad-enabled serial repeat. Its exception did not serialize the error. This audit measures forward-repeat behavior independently of losses and VJPs. It does not identify a CUDA cause, justify tolerance relaxation, repair the model, establish gradient behavior, or supply positive methodological evidence.

## Measurements

At frozen weights, `model.train(True)`, `p=0`, `colors=(0,2)`, and no DropoutKey, run four ordered passes:

1. `no_grad_A`: members 0–3 under no-grad.
2. `no_grad_B`: members 0–3 under no-grad.
3. `grad_A`: members 0–3 under enabled grad, without backward.
4. `grad_B`: members 0–3 under enabled grad, without backward.

Each call traverses the **complete official graph**; only its **complete official training-node logit slice** is retained temporarily in CPU memory. Each grad graph is discarded immediately. No training labels are selected or used. The inherited training-only acquisition loader may deserialize its stored training labels; validation/test target values are absent and no held-out target loader is called. No logits or gradient tensors are serialized.

Compare all six pass pairs for the same member: one no-grad/no-grad pair, one grad/grad pair, and four cross-mode pairs. “Per-pair” never means comparing different members' predictions. Record per-member and pooled-across-members maximum absolute difference, RMS difference, original-threshold exceedance count/fraction, source logit max/RMS scales, and relative errors. Difference subtraction and the `>1e-6` test retain the original float32 operation. Scalar sum-of-squares uses float64, with fixed 8,192-node chunks. Zero-scale ratios are null. No new acceptance tolerance is defined.

Record actual grad flags, output `requires_grad` and grad-function type, train mode, dropout probability, checkpoint flag, per-forward sparse/dense ledger rows, source operation counts, synchronized forward seconds, preparation/measurement peak GPU allocation and reservation, and process CPU peak RSS. The ledger counts the source's logged sparse and dense operations, not every scalar-reduction instruction. State checks cover all parameter/buffer values, all module training flags, `p`, Torch CPU RNG, all Torch CUDA RNG states, and empty parameter `.grad` fields. Those checks are recorded per forward; the original p/modes are restored and checked at completion. Python/NumPy RNG invariance is not claimed.

The model's source has two serial sparse forward calls and two logged dense calls per member forward. Sixteen forwards imply 32 such sparse calls and 32 logged dense calls, with **zero backward/VJP, loss, score, or optimizer calls**. Unexpected execution or state changes fail the audit and preserve partial scalar evidence. Differences above the old threshold are measured and reported; they do not cause an automatic retry or a widened threshold.

## Provenance and outputs

The new caller extends the inherited import policy with exactly three hashed text files: this runner, the unchanged v3 utility runner, and the original unchanged helper. The helper is retained as evidence and is not called. Existing scientific sources and package roots remain sealed. The v3 bootstrap cannot be used directly because its request/stage policy belongs to the old family; this caller uses the same guarded import and live-context mechanisms with its own separate request.

`EXISTING_STATE_AND_PRIOR.json` pins the exact input-table row, scientific binding, failed-primary receipt, previous requests and terminals, and complete v3 coordinator records. Runtime checks compare the row back to the unchanged original input table and checkpoint metadata. The output directories are prescribed in `REQUEST_TEMPLATE.json` and lie within this packet. Files use exclusive creation. A terminal or existing numerical directory cannot be retried or overwritten.

Per-forward files contain scalar scales, invariance, mode, timing, and ledger evidence. Per-pair files contain scalar errors. `RESULT.json` is provisional until root verifies the completed inner and outer supervision chain. `FAILURE.json`, partial scalar files, and all supervisor output remain available for failed-work accounting. A result says `audit_completed`, not that the old diagnostic is qualified to continue.

## Frozen budgets and root accounting

Use **one 120-second complete child envelope with five-second termination grace**. The ordinary child budget is 115 seconds. Keep the inherited combined phase limits of **21,600 seconds / 200 GiB**, diagnostic limits of **2,400 seconds / 64 MiB**, and GPU reserved-memory ceiling of **60 GiB**. Preparation, hashing, import, data/checkpoint loading, forwards, CPU reductions, scalar serialization, supervision, failure, and teardown all remain chargeable. This packet provides no measured runtime forecast.

`CURRENT_ACCOUNTING_TEMPLATE.json` requires new root accounting. It retains the v3 prior, all v1/v2 terminal references, **both successful v3 stages and the failed primary**, and the complete coordinator records. The caller validates those old chains and requires totals at least as large as the old prior plus each v3 stage's maximum of outer and coordinator wall time. Existing v3 runtime artifacts are also retained in the minimum disk charge. Root must additionally account for intervening work, preparation and this upload; these lower-bound checks cannot establish all current activity by themselves. No stale v3 accounting is accepted as this audit's accounting.

## Exact launch template

`LAUNCH_ARGUMENT_TEMPLATE.json` supplies the ordered argv through the original repository interpreter, `bounded_run_v1.py`, and `run_authorized_v2.py --require-idle-gpu`. Root supplies the approved request and original allocation environment. Use the original `protocols/repo_env.sh`, `CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONDONTWRITEBYTECODE=1`, and the existing authorized `GNNM_SSH_DESTINATION`. There is no SSH or launcher execution in this source packet.

Root approval must bind canonical JSON of the request excluding its `root_approval` field. Source/request/output identities are separate from v3. Root must inspect the measured differences and invariance before deciding whether a separately proposed repair or numerical-policy amendment is warranted. This audit itself authorizes neither.
