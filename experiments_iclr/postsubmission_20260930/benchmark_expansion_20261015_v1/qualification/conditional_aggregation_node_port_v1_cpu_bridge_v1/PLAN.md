# Exact CiteSeer-Cheb CPU C6 bridge qualification

Parent authorized this node support task and a bounded actual CPU check for the
three frozen C6 controls. Frozen node bridge manifest:
`aa90be4c1f989e6b4f81250eb3f30bb6a56ad75d9b242e445d539821be956a0c`.
Frozen original C6 package remains
`edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f`.

Use the literal guarded dedicated allocation, CUDA hidden, two CPU threads,
and the existing `polyformer-citeseer-cheb/gnnm_boundary_native` case from the
native smoke MANIFEST. Preserve exact source data root, split, model preset,
class order, fullTRAIN1929, fullVALID665, graph3327 and feature width3703.
The recorded resource preflight found256CPUs,1.833TB available and no active
qualification processes; no GPU workers are changed.

For each mode, construct the real source model, compare the actual original
mean VALID with the bridge's initial actual/mean outputs, and verify isolated
gate construction RNG. Perform one fullTRAIN update through the exact native
`_routes(True)` activation-checkpointed path. Evaluate complete VALID, save
the wrapped native/optimizer/gate/mode/source-data state and global RNG.
Construct a fresh session, restore, require bit-exact actual/mean VALID and
original Parameter identity. Restore RNG and perform a second fullTRAIN
update, followed by full VALID. Maximum six optimizer steps, two per mode.
There are nine post-update/restore VALID calls plus six initial parity VALID
calls, all complete. Six native constructions if all pass.

One backward of the same all-block J and one original source Adam transition
per update. Preserve source parameter groups/options; gate inherits the
existing lin3.weight group's LR and WD. No scientific fit/convergence/quality
or selection claim, Physics/link task, TEST scoring, full-fit/GPU operation,
capacity fit, recipe grid or automatic scientific retry. Preserve each attempt,
failure, checkpoint, count, wall/CPU cost and peak RSS. Source freezes stay
byte-identical; later evidence belongs only in this qualification directory.
