# Proposed NCNC runtime link v6 lifecycle gate

Status: source and plan frozen; no native v6 worker launched. Root publishes
this source first. Prefer one later CUDA attempt after a slot frees and its
resource envelope is admitted. The CPU JOB is an explicit alternative, not a
second required training run. Neither choice authorizes the giant campaign.

## Fixed native question and work

Use the frozen Collab/NCNC conditional-J interaction, seed0/split0, original
encoder/predictor/recursion/Adam groups and full native source batches. Preserve
17 updates per epoch: 1,114,112 positive and the same number of sampled-negative
records, with the native64,940positive dropped tail. The separate engineering
policy declares maximum=minimum=2epochs, patience1, actual whole-VALID Hits@50,
strict improvement and first ties. It is not the production1000/100/50 policy.

1. Fresh actual worker executes all17updates of epoch1 and one complete VALID.
   Inject one declared exception only after recovery, CURVE and PROGRESS are
   durable. Preserve the failed invocation and exact17attempts/completions.
2. Fresh native construction resumes that same JOB/hash/epoch1 recovery and
   executes only epoch2's17updates. The real selector stops, loads its own
   selected checkpoint, binds its exact payload and exports the actual plus
   same-call mean from the existing selected VALID call. Never force epoch1
   or epoch2 to win; never select from the diagnostic mean.
3. Preserve selected/recovery/export receipts, then resume the stopped epoch2
   recovery in another fresh construction. Require0TRAIN/optimizer work.
   The real finalizer reloads its own selected checkpoint and re-exports actual
   and mean. Authenticate JOB, runtime, explicit policy, callback/extension,
   all native/ingredient sources, selected epoch and exact checkpoint bytes.

Total hard bounds: 2completeTRAINepochs,34Adam attempts/completions,
34native scalar-J backwards,0extra backwards,3native constructions,
4fullVALID calls,2selected exports and0extra mean inference. Each VALID has
all60,084positive+100,000negative records in native order. Record native query,
TaskCallScope/output-bank routing and epoch/tail receipts independently. The
controlled completed-epoch exception is expected; stop on any other failure
or timeout and preserve actual costs. No automatic repair/retry or duplicate
uninterrupted trajectory is included. Abrupt partial-epoch recovery remains
outside this gate.

## Exact state versus repeated floating computation

Saved native model/Adam/sampler/TaskCallScope/private-global RNG/mode/epoch state,
source bindings, IDs, labels, endpoint order, selected checkpoint bytes and
same-call tensor authority must match exactly. On CUDA, a fresh forward may
use nondeterministic kernels even after exact restore. Record bit equality,
max absolute/relative errors and the existing fixed rtol1e-4/atol1e-5 check for
repeated floating outputs and summaries. Do not rerun to find a pass or tune a
tolerance. Keep both observations; small errors establish only numerical
engineering consistency. CPU repeat exactness is recorded separately.

Selection remains the first best actual validation of the original trajectory;
replayed inference cannot revise that saved decision. The inherited result
field convergence_established reflects a patience stop only; this two-epoch
engineering gate must not interpret it as scientific convergence.

## Measured budget and admission

The completed native conditional-J CPU check took408.68s for2updates,
3fullVALID and2constructions, with12.67GiB peakRSS. Recorded phases were at
most88.45s for first update plus construction,77.00s for second update,81.63s
for restored VALID and77.34s for final VALID. A34update/4VALID lifecycle is
roughly50–60CPU minutes before contention, not a measured full-epoch result.
Its fallback watchdog is90minutes,2threads,64GiB AS,20GiB RSS reservation,
>=24GiB cache-aware headroom after peer growth, and2GiB free-disk reserve.

Prefer CUDA after current owned workers release a slot. The finite CUDA wall
budget is30minutes; no CUDA duration or memory claim is inferred from CPU RSS.
Before admission, pin a suitable current NCNC CUDA memory receipt or an
explicitly admitted measurement, reserve other active growth, and verify the
literal allocation/sole GPU UUID. Never add a third own GPU model worker.

The native qualifier/launcher must be frozen against these sources and exact
JOB bytes before execution. It must count attempted/completed work and retain
checkpoint/export/log receipts. This plan does not itself launch or make that
unwritten qualifier ready. No TEST, original-five rerun, HPO, scientific score,
full campaign GO, or source-checkpoint migration is admitted.
