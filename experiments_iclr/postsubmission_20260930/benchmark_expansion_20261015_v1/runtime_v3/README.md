# Runtime v3: explicit CPU engineering execution

This separate generation copies the six producer files from immutable runtime
v2. It changes three files only. `DELTA.diff.txt` contains the complete runtime
delta; `PARENT_SOURCE.json` pins v2. Existing pilots, launchers, queues, frozen
v2 and all native sources are unchanged. No launcher is migrated to v3.

Default jobs still target `cuda:0`, use the original CUDA RNG API sequence and
CUDA peak-memory collection, and retain their native training and selector
logic. This source generation has its own runtime hash binding; v2 checkpoints
are not migrated. The frozen selected-VALID replayer registry does not silently
recognize v3.

## CPU admission

A CPU engineering job requires all of these immutable fields and environment:

```json
{
  "device": "cpu",
  "execution_mode": "cpu_engineering_v1",
  "engineering_only": true,
  "TEST_access": false
}
```

`CUDA_VISIBLE_DEVICES` must be empty. CPU mode supports node/OGB sessions; link
CPU mode refuses. The same literal allocation hostname/sole-UUID guard applies
before metadata reads. CPU source and RNG states are saved/restored without
CUDA RNG or CUDA resource calls. CUDA resource fields are null with an explicit
collection-disabled marker, not measurements. CPU engineering results identify
their execution device and do not imply scientific fits or accuracy evidence.

BaseAdapter binds a construction-only recipe copy's `_repository_root` to the
actual repository root, rejecting a conflicting declaration. The original JOB
and canonical job hash remain untouched. Native model, source callback,
optimizer, data loading, epoch traversal, objectives, selection, stopping,
checkpoint contents and completed-epoch recovery/finalization flow retain v2
logic. Bookkeeping, controllers and source RNG emulation are byte-identical.

## Qualification scope

Local admission and RNG checks use no native model/data or CUDA. They passed;
the actual allocation qualification is separate and must retain full epochs,
source batches, exact job hashes on resume, own selected exports and all costs.
CPU checks do not certify GPU execution or alter any active pilot.
