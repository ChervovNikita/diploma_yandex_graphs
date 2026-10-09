# Native15 saved state predictive materiality diagnostic

Inactive source only. Root supplies the exact checkpoint hashes and source, role, runtime, closure and normal owner-release receipts. No role archive, checkpoint, model or outcome was opened while preparing this packet.

## Fixed scope

| Owned failed attempt | Seed | Selected epoch | Fresh full graph evaluations |
| --- | --- | --- | --- |
| native_single d4_f16_L2 | 2207 | 405 | 1 |
| native_single d4_f16_L4 | 2207 | 372 | 1 |

Root reports both attempts completed 500 epochs, exactly restored parameters and failed the original reconstructed-logp gate. The reported original TRAIN/VALID max logp differences are 0.0220413/0.00543356 and 0.00358820/0.00234985 respectively. These are root-supplied rounded metadata, not newly read values or diagnostic thresholds. All15 attempts are terminal/reaped; original failed statuses and `incomplete_no_freeze` persist.

The diagnostic does not train, call backward, instantiate or step an optimizer, reselect a checkpoint, retry an evaluation, rank architectures, choose a surviving winner or change the old family gate. It adds no acceptance threshold. It provides one materiality report and retains failures.

## Exact reused implementation

`materiality.py` loads the original V2 `common.py` and `native_placement.py` after root activation. It uses the unchanged `read_roles`, owned `torch.load(..., map_location="cpu", weights_only=True)`, original native placement factory, strict state load and exact parameter/buffer equality. It selects unchanged AST definitions of the V2 RNG, metrics and `evaluate` helpers; the fit loop, architecture selector and original main are excluded. Each checkpoint gets one constructor and one call to the original `evaluate` at its saved epoch. The helper owns the original `seed+2000003+epoch` evaluation stream and `no_grad` context. It does not restore or use the saved optimizer state.

Before any numerical imports, activation verifies this packet, original source pins, two fixed checkpoint/result hashes, original failed result metadata, closed-family custody, original screen release and a fresh output path on the released server hostname. It checks original roles/device/deterministic/TF32 policy and actual provider/Python/CUDA versions. Full original runner/native source seal verification is also required before construction. Source/runtime mismatches refuse; no environment or source repair is performed.

## Per role report

TRAIN and VALID each retain:

- Maximum and mean absolute native float32 logp changes, across all role rows and both class columns.
- Maximum and mean absolute probability changes, from `exp(logp)` on those same columns.
- Original metrics recomputed from the saved role logp using the exact original metrics helper, recorded original checkpoint scores and replayed scores.
- Signed replayed-minus-original AUROC, NLL, accuracy and Brier differences. Accuracy differences are fractions, not percentage points; Brier is the original binary positive-class squared-error mean.
- Signed recomputed-original-minus-recorded score differences, so original metric arithmetic/provenance is visible separately.
- Decision-change counts using the original `argmax(-1)` convention, including its tie handling.

The saved outputs are the original selected epoch's stored role logp, not a second original-model forward. No epsilon, floor, clip, double-precision rewrite, alternative solver, extra draw or prediction threshold is added.

## Costs and custody

Each fresh server-only attempt records load, constructor, strict restore, evaluation/metrics, comparison and raw storage times; CPU/wall cost, cumulative RSS, CUDA allocated/reserved peaks, parameter and topology storage, checkpoint/result/source/role identities and every attempted/completed evaluation. Total cost includes activation hashing and setup. Per-attempt CUDA peaks include resident role tensors; CPU RSS is the cumulative process high-water mark. Cleanup/query failures remain visible. Neither is a measured isolated inference-efficiency comparison.

Each `ROLE_OUTPUTS.pt` remains on the released server and contains only original/replayed TRAIN/VALID logp and identity, without labels, TEST truth, model or optimizer states. Scalar role reports and a summary also remain in the fresh output. Enabled stdout prints only diagnostic status, the unchanged family status and attempted-forward count. No raw output transfer is implemented.

Before the sole evaluation call, the attempted counter is saved. Setup failures record both fixed slots as unavailable/not executed; per-checkpoint failures are retained and are not rerun. Existing failed results, checkpoints, source packets and protocol files are read only. `materiality_reported` describes a completed diagnostic, never a family pass or architecture admission.

## Entry point

Default invocation prints an inactive plan and returns before numerical imports or role/checkpoint access. Root may activate only with an exact enabled release and its normal external owner/supervisor:

`python materiality.py --execute --release ABSOLUTE_ROOT_RELEASE.json`

The disabled template is not an activation or a runnable data input. Source/hash/AST/JSON checks passed; numerical behavior remains unexecuted and unqualified in this preparation.
