# Explicit owned BUDDY private BN inference

This immutable source successor adds `inference_operation('VALID'|'TEST')`
and an authenticated compatibility overlay for genuine v2 producer sessions.
TRAIN/VALID training algorithms, state schema and saved payload layout remain
unchanged. Inference uses the literal role, eval mode, original sequential
native routes and private manager. Native and all active/inactive private bank
objects, pointers, shape, dtype, device and values must remain unchanged.
Original model modes are restored on normal or exceptional exit.

`bind_inference_owner(session, job, root, manifest_sha256)` verifies either a
new genuine v3 producer or attaches the separately bound API to an already
restored genuine v2 session. The v2 path authenticates its exact class, source
manifest, callback, descriptor and saved producer source metadata, then
replaces only the producer-owned context guards. It preserves producer method
metadata, checkpoint contents, native/optimizer/manager objects and runtime
binding. TRAIN/VALID logical counters stay unchanged during final inference;
call diagnostics live only in the returned context audit. State access,
sampling, hydration and nested training operations remain forbidden while the
inference owner is active. A failed attachment invalidates the worker.

The overlay source identity is recorded separately in `inference_metadata` and
in the scorer's source authority. Existing producer metadata is never
relabeled as v3. No retraining or conversion of existing checkpoints is
required merely to add this inference ownership API. A future training job
that genuinely chooses the v3 callback has its own honest v3 source identity;
it is not an old state relabeled as new.

The overlay does not authorize TEST, load role data, select states or launch
work. Final scoring still needs its existing campaign/role/resource authority.
Source checks use ordinary scalar fixtures and AST comparisons only. No model,
tensor library, dataset, checkpoint, GPU, remote or TEST payload was executed.
Whole selected VALID and actual-role/resource qualification remain pending.
The prospective registry/source rebind is recorded with scorer v4 for root
review; current portable sources are preserved.
