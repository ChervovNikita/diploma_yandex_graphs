# Peptide complete-block launcher port

Review scope is the host adapter of the existing paired-pilot v2 queue. b1 binds physical GPU0 and b2 physical GPU1. Both use the new qualified repository-local CPython3.11/Torch2.1.2+cu118 environment, exact source7f274c09..., and original raw qualifier b7eae293.... No accuracy fit has started. Separate root block release remains required.

## Scientific differences

None. COHORT_PLAN.json is the original 21,535-byte canonical plan (ae4b0f5c...), including original source-manifest scientific lineage. EXTERNAL_ANCHORS.json, STUDY_SPEC.json and SCIENCE_CONTRACT.json are byte-identical originals. Each queue contains its exact fixed ten-cell whole block and order. Sixty cycles, eval every5, eleven misses, outer64/inner256, seeds, all cell soft/hard bounds, resource limits, and analysis policy stay frozen.

## Operational differences

Three source files are adapted. pilot_common.py binds peptide paths, actual qualified source/raw gate, the ordered two-GPU inventory, one prespecified block UUID, and the previously reviewed e715 ownership helper file. run_queue.py changes selected-GPU preflight and adds the root provider-release/admission/registration checks. Its run_fit and output_bytes functions are AST-identical to reviewed pilotv2. freeze_queue.py copies canonical plan and anchors as raw bytes, emits only the selected ten jobs, applies each block's physical UUID and root evidence, and emits provider admission/donor registration before any child. The original numeric/cost/budget/analysis guards remain. Host provider/source/qualifier bindings are separate from canonical scientific lineage.

Concrete disabled releases and twenty disabled job previews are included. The launcher commands require root-supplied reviewed releases. The emitted jobs authorize TRAIN/VALID only after that release; TEST and retry remain false. No resource-driven method skipping, reorder, horizon shortening, or partial-block migration is introduced.

## Verification

Syntax checks passed. run_fit/output_bytes AST equality passed. A local metadata-only freeze exercise produced both exact ordered ten-cell blocks with identical plan/anchor bytes, identical scientific job fields and schedules, exact qualified gate/source/physical UUID, and donor bindings before any start. No models were imported, no child was spawned, and no score was read. The first local harness used macOS's /var alias for its temporary directory; resolving the temporary root fixed that harness path comparison without changing ported source.

METADATA_STAGE_PLAN.json lists the exact original numeric decision/cost/root-review metadata required by the retained freeze guards. They must be staged byte-identically before the remote freeze. Root must review HOST_ONLY_DIFF.patch, SOURCE_MANIFEST.json and the disabled releases before any fit. Any further source change requires a successor packet.
