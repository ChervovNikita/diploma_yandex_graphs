# Bounded runtime v3 full-epoch CPU qualification

Freeze this source and the independent runtime v3 before execution. Two CPU
threads, CUDA hidden, literal allocation guard, no TEST or GPU worker changes.
The native MolBACE GIN/dropout.5 model, full categorical features, native Adam,
source losses and exact TRAIN1210/VALID151 scaffold roles remain unchanged.
These are new engineering descriptors with finite explicit horizons, not
scientific fits, paper-reference horizons or modified pilot jobs.

Fixed sequence, no automatic retry:

1. Native M1: uninterrupted two full source epochs, stop and own selected export.
2. Same stopped job: resume its epoch2 recovery and finalize; zero TRAIN epochs
   and optimizer steps, exact same selected export.
3. Identical native M1 job/hash/policy in a separate directory: a controlled
   exception immediately after actual epoch1 recovery, curve and progress are
   durable. Preserve the expected failed invocation and all checkpoint copies.
4. Resume that identical job from epoch1 recovery for its one remaining epoch;
   compare whole native/Adam/loader/Python/NumPy/Torch CPU RNG/selector state
   and own selected export exactly with the uninterrupted trajectory. Never
   shorten a horizon to emulate interruption or change a job hash on resume.
5. Boundary bank: one full native epoch, own selected export, then a stopped
   finalization resume with zero extra epochs/steps and exact selected export.

Maximum five additional full TRAIN epochs (under the authorized cap of six),
six source constructions and ten complete VALID calls. Source batch_size32
gives38 raw batches per epoch; preserve skip-law outcomes. If all source
batches are eligible,190 observed Adam steps total. The measurement wrapper
checks actual complete traversal, repository-root binding and work counts
without changing source sessions, optimizer, loss or selection.

Use seed14005101, a new engineering identity. Native two-epoch descriptors
are byte-identical across interrupted/uninterrupted paths. Runtime v2, source
callbacks, native model code, active pilots and launchers remain frozen.
Record actual costs, all terminal invocation receipts, expected controlled
fault, prior checkpoint/export copies, hash checks and any unexpected failure.
Stop on an unexpected failure; do not tune tolerances or retry automatically.
Exact CPU trajectory comparison is an engineering correctness gate, not an
effect-size criterion or scientific improvement claim.
