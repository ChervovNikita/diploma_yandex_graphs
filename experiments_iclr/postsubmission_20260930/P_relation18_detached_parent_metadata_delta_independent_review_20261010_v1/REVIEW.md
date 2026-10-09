# Independent relation18 detached-parent metadata delta review

**Ready for root adoption of this narrow source delta: no new blocking source defect found.** This is source review only, not runtime custody, resource admission or a scientific result. Root reports launch ppid3713401 and later detached ppid1 with the same PID/start/group/session/argv; actual receipts were not opened in this review.

## Exact reviewed packets

| Packet | Seal SHA256 |
|---|---|
| graph_relation18_selected_readout_source_20261009_v3 | `5bfa8c56f7e219dc8eabc7d8b8810516635f6b11a1dba07f00d16a3e297329aa` |
| relation18_root_activation_assembler_source_20261010_v2 | `f7ca9a93948aa2e928772604caf1a920c2e387aaf747effdf7eb2bc30a770f1a` |
| relation18_normal77_read_only_observation_source_20261010_v2 | `b2064d2ca35cd2dea01b3446efcf9f0dfd71ded76009226a50bb9d5a3bcfaf42` |

The three NARROW_SOURCE.diff files and prior reader-V2/activation-observation reviews were consulted. Actual source comparisons independently confirm the declared delta.

## Identity repair and full snapshot preservation

`readout_gate.py:313–319` compares launch and saved identity using PID, start_ticks, pgid, sid and argv, and requires `observation_complete is True` in both. It still binds the exact original LAUNCH path, root release and expected pinned controller argv; detached launch, null parent poll, no retry and no scores read remain required. PID plus birth ticks distinguishes ordinary PID reuse on the admitted host; group/session/argv and release bind the same recorded controller birth. Mutable ppid and state are no longer demanded equal across observations.

Full saved dictionaries remain intact. `:303–305` still requires complete PARENT_OWNER identity equality with FAMILY_CLOSURE, and `:320–325` requires the terminal owner identity to equal that full saved identity. Actual current parent/process-group absence and no CUDA rows remain required. Detached direct wait stays **false** and OS exit stays **null**, with no inferred exit0.

`OBSERVE.py:117–127` applies the same stable launch comparison and complete-observation requirement, retains full parent/family equality, and additionally binds launch argv to the saved identity argv. Handles still carry the original full family identity. No receipt is edited, filtered, normalized or regenerated to remove ppid/state. The assembler's full snapshot construction is unchanged. Original child wait/reap/exit0 checks, process-birth/group absence logic, historical original-route handling and prior failed-attempt behavior are unchanged.

This verifies **source preservation** of immutable full snapshots. It does not independently verify current receipt bytes or current process absence; those remain bound runtime facts for the authorized root path.

## Numerical behavior, criteria and dependencies

Mechanical source reversal proves reader `readout_gate.py` differs from V2 only in the one launch-equality predicate. Observer source differs only in that predicate and dependent packet names. Assembler source differs only in reader/adoption names and associated messages. All other definitions retain identical ASTs; no entry was executed.

`collect_relation18.py`, `collection_hooks.py` and `readout.py` are byte-identical to reader V2. Existing reader SOURCE_BINDINGS fields are exactly equal after removing the two new descriptive/supersession fields. Thus scientific/runtime pins, inference ordering/counts, restoration, readout arithmetic, comparison criteria, costs and resource limits remain unchanged. Disabled release, terminal, reference, stage and original-route custody templates are byte-preserved. Assembler runtime/readiness/observation templates are byte-preserved; fresh default output names and exact reader-V3 adoption bindings are the declared metadata changes.

Verified all **32 new packet payloads**, their three manifest/seal chains and superseded seals; reader's **44 source bindings** and **14 source manifests / 164 payloads**; and observer → assembler-V2 → reader-V3 exact dependency bindings. All seven Python sources were parsed as AST without importing them. Manifest sidecar hashes are checked separately in the verification record.

## Retained obligations and scope

The prior fresh-sibling placement instruction remains necessary because the earlier generic input-directory exclusion caveat is unchanged. Root must retain actual full receipts and the failed attempt, supply exact metadata/staging/adoption/resource evidence, and use the separately authorized finite launch. This review grants no automatic approval, retry, numerical opening or launch.

No entry, consume gate, numerical provider, actual outcome/checkpoint/archive, runtime process/GPU observation or server was accessed. No previous packet or seal was changed. The review is limited to the detached-parent metadata repair and dependent rebinding.
