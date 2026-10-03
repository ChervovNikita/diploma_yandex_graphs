# Independent root release contract

The sealed driver accepts a root-authored JSON release. The release is hashed into its immutable pre-execution marker. It must contain:

- `graph`: `Photo`.
- `execution_authorized`: `true` for native execution; it may be `false` for resource-only preflight.
- `run_name`: the exact fresh run name passed on the command line.
- `prepared_manifest_sha256`: this packet's sealed manifest hash.
- `mode_donor_freeze_sha256`: `ab9d154ca62f033bfdc99a20af2789a1b802b5a3b1131a5a70ba47dec90a9113`.
- `expected_gpu_uuid`: one fixed `GPU-...` identity, selected by root before the resource probe.
- `Photo_resource_evidence`: exactly `required_free_bytes`, `planned_wall_seconds`, and `wall_budget_seconds`; each is a positive integer authored by root from a Photo prospective plan.
- `Squirrel_gate_assertion`: `complete`, `all3_blocks_complete`, `all12_arm_terminals`, `all12_selected_fits`, and `development_trigger_satisfied` all `true`; `is_quality_evidence` is `false`.

The gate assertion is root's release statement. The Photo author did not open Squirrel outcome files to reconstruct it. It preserves the staged and outcome-aware nature of the follow-up. No measured effect is claimed in this packet.

The release does not change the immutable donor choice or algorithm. The driver compares all six donor identities to the original root cutoff, matches every Photo donor path/hash, and compares exact arm order, continuation, selector, training loss and RNG semantics. Native execution cannot start with a resource-only release. The root scheduling choices are retained separately from the frozen scientific protocol.

The source validates resource evidence and probes the fixed UUID before loading the Torch runtime. Insufficient resources produce a retained deferred terminal. The source has no scheduler, remote command, GPU choice search, automatic retry, replacement donor, partial comparison or final-label release operation.
