# Runtime v4: selected same-call VALID mean and epoch reporting

Status: frozen source after static review and bounded saved-tensor regression.
**Native v4 stop/resume/finalization lifecycle qualification is pending.** This
generation has not been staged or launched by this task. No existing pilot or
launcher was changed. V3 evidence remains frozen and is not transferred to v4.

## Changes from v3

`PARENT_SOURCE.json` pins all six v3 producer files and the separately reviewed
reporting proposal. `DELTA.diff.txt` is the complete v3 to v4 producer diff.
Only `adapters.py` and `run_fit.py` changed. The other four producer files are
byte-identical to v3.

Node and molecular validators retain the optional native same-bank arithmetic
mean returned alongside the actual serving output from their single existing
VALID call. The mean is diagnostic. Selection still receives only the actual
serving result. No extra forward, optimizer step, RNG operation, or averaging
reconstruction is added.

The mean provider must declare one recognized source metadata record. Its
method and extension manifest must match the immutable JOB. Complete VALID
scope, closed TEST access, exact native IDs/order, optional raw labels, finite
mean logits, matching shape/dtype, finite loss and native metric keys are
checked. Original ungated callbacks with no mean fields or provider keep their
original path. A declared provider that omits the mean is refused.

The adapter caches a CPU clone of that one transaction with job/source/tensor
receipts and generation/validation counters. TRAIN, TRAIN diagnostic and load
entry invalidate the cache; VALID clears it before entering the callback, so
a failing callback cannot leave an older diagnostic available. Node split
construction keeps its original JOB authority before the existing split
binding copy is made.

After the finalizer loads its own selected checkpoint and performs its normal
actual VALID call, it exports the mean under `selected_VALID.pt` key
`same_bank_mean_VALID`, schema `own-selected-same-bank-mean-VALID-v1`. The export
binds job ID/hash, selected epoch, actual checkpoint byte receipt, runtime and
native source binding, exact IDs/labels, mean logits/loss/metrics, and same-call
authority. Changed identities, stale transactions and altered authority are
refused. Runtime binding includes all six current producer hashes. Native
callback/extension/dependency hashes are the authenticated construction
receipts used by the existing fit runtime. Link adapters without the optional
exporter continue through their existing path.

The reporting repair moves the exact existing four-field `result.update`
statement immediately after selector epoch commit and before checkpoint,
CURVE and PROGRESS I/O. A later I/O failure can therefore retain the completed
epoch count in RESULT. The computation and arguments are unchanged. Work before
selector commit still requires retained optimizer/work receipts; the repair
does not establish native failure recovery correctness by itself.

## Evidence and remaining gate

`check_static_contract.py` and `LOCAL_CHECKS.json` establish that all remaining
producer AST is identical after removing the declared additions and move.
Each Node/OGB validator still contains exactly one native validation call;
TRAIN and diagnostic code match v3 after leading cache invalidation. The new
helpers contain no native session/model/optimizer or RNG calls.

`../qualification/runtime_v4_saved_tensor_regression_v1` preserves genuine
previously saved native/C6/interaction VALID tensors, original receipts, both
harness failures and costs. Five cases pass, with 76 mutation refusals and
41 recorded tensor callbacks in the passing attempt. All attempts used zero
native constructions, native forwards, optimizer steps or TEST access.
Provider views and selected context are simulated: this is not a native v4
selected-checkpoint or lifecycle qualification.

Native v4 full-epoch selection, own selected export persistence, interrupted
resume, stopped finalization resume, and native reporting fault qualification
remain pending. There is no checkpoint migration from earlier generations,
scientific outcome, production readiness claim, or full-campaign GO here.
