# Disabled V3 sequential-view memory repair

This separate packet preserves V2 and its failed qualification. `blocks.py`, `lockstep_forward.py`, and the package export are byte-identical to V2. The fresh native-scorer constructor, model, route exchange/separable parameters, graph, labels, member contexts, initialization, serving and snapshot primitives are retained. V3 adds only `accumulated_own.py`, its callable method binding/counters, a new snapshot identity, and source-bound update semantics. No numerical source was imported or executed during preparation.

## Concrete failure and repair

The actual V2 whole-TRAIN qualifier reached its baseline local update after completing initialization and two factual/parity forwards. Its owner observed68,847,403,008 bytes of owned GPU memory against64 GiB, sent SIGKILL only to its verified owned child group, obtained direct wait exit-9, and verified child/process-CUDA absence. No Adam step or snapshot completed. Immutable terminal/progress bindings are recorded in SOURCE_BINDINGS.json and MEMORY_REPAIR_NOTE.json. This is an engineering memory failure; it does not reject the exchange mechanism or establish any quality result.

The pinned public `Session.train_step` and `core/objectives.py` establish that plain shared `be_unit` F is the mean over four members of .5*(CE_A+CE_B), with auxiliary zero. Its cross-view alignment/residual terms are conditional on `model.contrastive`; V3 explicitly rejects contrastive and independent sessions. The old unused contrastive index is omitted because this exact plain-F path uses every TRAIN target and has no auxiliary selection.

V3 performs zero_grad once, then forward A, pinned own_supervision, .5*mean own CE backward, and deletes every A output/loss reference. It then performs forward B at the same unstepped model, pinned own_supervision, .5*mean own CE backward, and deletes B references. One original Adam step follows both backward completions and the finite accumulated-gradient check. Original finite prediction/representation/loss/state checks and `loss`, `own_mean`, `auxiliary` return names are retained. The returned auxiliary is zero.

This is the same scalar objective in real arithmetic. Reduction and accumulation order change, so bitwise gradient or trajectory equivalence is not claimed. Backward is checked not to consume default CPU/CUDA RNG, and neither weights nor Adam history advance between the two views. Existing persistent member streams therefore resume the same realized dropout views for this pinned path. Subsequent trajectories may differ through floating-point accumulation. This is a memory repair, not a new method or novelty claim.

## Coupled graph and accounting

Every exchange/control view still retains all four live prefixes and all source/recipient paths through the single exchange, six continuing local blocks, and native global path. No member detach, one-member gradient replay, recomputation, frozen teacher, graph support change or extra label opportunity is introduced. Both baseline and candidate/control use exactly the same sequential-view accumulation.

Completion counters now include two backwards per update. `live_last_update_events` records the actual forward/backward/release/Adam order. An optional read-only qualification observer flushes this execution evidence at each stage so interrupted work is not inferred from a successful final return. Counters/events are excluded from training snapshots. There is no per-view optimizer, gradient reset between views, AMP, clipping or scaling change.

The same native initializer and block start remain identical across arms. The closed initializer screen remains failed, with worsened served NLL; its result is not promoted or confirmed. The attributed exchange hypothesis, capable same-task M1/genuine-I4 references, Cross-stitch/separable/objective controls, retention diagnostics and compute limits remain those in V2's sealed HYPOTHESIS_AND_COST.md, bound by this packet.

## Disabled admission

New continuation schema/source identity rejects V2 snapshots. The separate V3 qualifier retains six updates, six constructors/saves/restores and42 joint forwards, and asserts12 backwards plus actual live-view order. Root must separately review and activate it with64 GiB allocator,68 GiB owned-tree GPU cap and72 GiB fresh free memory. The total budget includes CUDA context/workspace overhead that allocator telemetry does not capture. No new qualification has run; memory success, speedup, numerical parity, quality and fit admission remain unclaimed. No automatic retry or full trainer/owner framework is supplied.
