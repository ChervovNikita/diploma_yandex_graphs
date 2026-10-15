# Minimal runtime v4 native selected-mean lifecycle gate

One CPU engineering attempt on the guarded authorized allocation. Root admitted
this exact bound after review, subject to an immediate conservative available
memory check of at least 6 GiB. Account for concurrently owned NCN PID834717
and BUDDY PID834307 and their reserved growth. No existing process is changed.

Use frozen runtime v4 and the already-qualified interaction I3
`interaction_input_conditionalJ` source. Keep the actual CiteSeer-Cheb native
callback, graph/basis, full TRAIN1929, VALID665, source Adam/LR/WD, boundary
factors, residual, loss and reductions. Seed101, split0. This is an explicitly
finite engineering descriptor: maximum=minimum=2 epochs, patience1,
accuracy/strict improvement/first ties; no TRAIN diagnostic. No scientific
score, paper result or original-paper experiment is inferred.

## Fixed sequence

1. Fresh actual `run_fit.run`: execute full epoch1. Inject one declared
   engineering exception only after actual recovery, CURVE and PROGRESS are
   durable. Preserve the failed invocation. Require RESULT now records
   `training_epochs_this_invocation=1`, completed epoch1, and one Adam step.
2. Resume the same JOB/hash/policy and epoch1 recovery in a fresh native
   construction. Execute only full epoch2, stop at the finite cap, load its own
   selected checkpoint, perform the existing selected VALID call and persist
   actual plus same-call mean under `selected_VALID.pt`. Require two cumulative
   Adam steps and one step in this invocation. The selector may choose epoch1
   or epoch2 from the actual metric; do not force a winner.
3. Preserve all checkpoint/export bytes before resuming the same stopped
   epoch2 recovery in another fresh construction. Require zero TRAIN epochs
   and zero optimizer attempts/completions. Let the actual finalizer reload
   its own selected checkpoint and re-export actual plus mean. Compare both
   persisted logits, IDs, labels, loss/native metrics, selected epoch,
   job/source/runtime binding and selected checkpoint receipt exactly.
   Generation/validation transaction counters may differ between invocations;
   each must authenticate its own current call and exact tensor hashes.

No duplicate uninterrupted training is added. V3 already qualified exact
uninterrupted versus resumed native trajectories, and the v4 AST review shows
unchanged training/RNG/selector logic after cache invalidation. This gate
executes v4's actual native finalizer, persisted mean, completed-epoch resume,
stopped finalization resume and the reporting repair. It does not claim every
family/device, abrupt partial-epoch kill or scientific convergence is covered.

## Exact work and resources

At most two full TRAIN epochs, two Adam attempts/completions, three native
constructions, four complete native VALID calls and four same-call means.
There are two selected exports, with no extra mean inference. Native callback
calls are counted independently from adapter calls. Stop on unexpected failure;
retain all attempted/completed work, snapshots, terminal receipts and costs.
No source repair, retry, seed/batch search or tolerance change within this
attempt. No TEST, CUDA context, GPU worker change or pilot/launcher migration.

One process; two Torch/OMP/MKL/OpenBLAS/NumExpr threads and one Torch interop
thread. Reserve 4 GiB peak RSS and 512 MiB disk. Immediate launch admission
requires at least 6 GiB conservative available memory and two CPU cores.
The prior eight-case interaction check peaked 3.34 GiB; this smaller sequence
is expected to take under 45 seconds without contention, not a guarantee.
Enforce a 180-second wall watchdog, preserving any watchdog termination as a
failed engineering attempt. Freeze source/descriptor/plan before staging or
execution; verify every input/data/source receipt before construction and
again after work. No earlier checkpoint is imported or migrated.
