# Runtime v3 CPU full-epoch readiness

The frozen runtime passed bounded actual MolBACE GIN/dropout.5 qualification.
Runtime manifest: `d04193717372bb08eb85b780822338ad25b0205e1fd5a1ef72680d293506fd93`.
Qualification manifest: `e1a7869fe9a4c0de75651a586525ac94108e5b930e218d650ba97d6016c5e440`.

The native model/configuration, source categorical inputs, Adam, source batch/skip laws, objectives, complete TRAIN1210 and VALID151 roles were retained. Construction recipes received the actual repository root. Runtime v2, native sources, existing pilots and launchers were unchanged.

## Actual run paths

| Invocation | Outcome | New Adam steps |
| --- | --- | ---: |
| native_uninterrupted2 | complete | 76 |
| native_stopped_finalize_resume | complete | 0 |
| native_controlled_stop_after_epoch1 | expected controlled failure after durable epoch1 | 38 |
| native_resume_remaining_epoch2 | complete | 38 |
| boundary_one_full_epoch | complete | 38 |
| boundary_stopped_finalize_resume | complete | 0 |

The uninterrupted two-epoch native trajectory and the identical job interrupted after completed epoch1 then resumed matched exactly: full native parameter/buffer state, Adam, loader states, Python/NumPy/Torch CPU RNG, selector, runtime binding, saved recovery VALID and own selected export. The policy, horizon and canonical job hash did not change on resume.

Native and boundary stopped finalization resumes added zero TRAIN epochs and zero steps, then reconstructed exactly the same own selected VALID exports. Three snapshot sets preserve prior checkpoint/export/curve/result bytes, including the declared failed invocation.

## Accounting caveat exposed by the controlled fault

The injected exception occurs after recovery, CURVE and PROGRESS are durable but before runtime v2/v3 assigns `RESULT.training_epochs_this_invocation`. That one failed result retains0 for this summary field despite correct completed_epochs1, durable epoch1 receipts and38 optimizer steps. Charge work from actual epoch receipts and optimizer deltas; total5epochs/190steps is explicit. This pre-existing summary-field timing is preserved and disclosed. Any correction belongs to a separate frozen source generation; v3 was not modified or retried.

## Actual cost and bounds

Five complete TRAIN epochs, each1210 records/38 raw batches/38 updates/0 skips;6 constructions;10 complete VALID151 calls;190 optimizer attempts and190 completions. Measured qualification loop 45.213803s; launcher-to-log wall 48.960845s; CPU user/system 78.018391/3.263446s; peak RSS 1,305,780 KiB.

Two CPU threads, CUDA hidden; no CUDA context, TEST, scientific fits, GPU worker or host-setting changes. Costs overlapped the NCNC CPU qualifier and a parent-reported NCN C1 CPU check. They are observed shared-resource costs. Cgroup16core quota and cache-aware available memory were recorded. Two conservative admission deferrals performed no native/model/update work and are preserved separately.

## Limits

This qualifies finite CPU engineering paths for native MolBACE GIN/dropout.5 and one boundary bank. It is not a convergence, accuracy, paper-reproduction or scientific effect-size result. GPU, node, link and new extension training paths were not executed. Interruption is a declared exception at a durable completed-epoch boundary, not a partial-epoch process kill.

The six-file runtime is a new source generation with no v2 checkpoint migration or launcher migration. The selected-VALID replay registry remains frozen and will refuse unregistered v3 runtime bindings. Default GPU RNG calls were traced against v2 without executing GPU work.
