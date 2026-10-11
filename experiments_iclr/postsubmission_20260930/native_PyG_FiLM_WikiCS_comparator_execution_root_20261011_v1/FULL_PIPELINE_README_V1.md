# Native FiLM full execution pipeline

Status: SOURCE ONLY, awaiting root seal, actual admission, publication and dispatch. These files add one fixed acquisition and readout pipeline to the existing execution root. They perform no SSH or deployment. The four readiness scripts and the sealed scientific source packet remain byte unchanged. The actual full-WikiCS shared4 readiness already passed; this pipeline does not execute another check.

## Root entry point

Root seals the three new Python files and this integration material, then materializes `ROOT_PIPELINE_ADMISSION_V1.json` from `ROOT_PIPELINE_ADMISSION_TEMPLATE_V1.json`, setting its status to `ROOT_FIXED_FiLM_PIPELINE_ADMITTED` and recording the actual decision time. The template contains exact scientific-manifest and execution-source receipts. It is not an admission.

On the verified allocation, from the existing repository root, root dispatches `.venv/bin/python -B experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1/queue_full_pipeline_v1.py --admission experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1/ROOT_PIPELINE_ADMISSION_V1.json` through its existing detached-owner transport. No local invocation or SSH was performed during preparation.

Every project operation follows the hostname `anogena-2-0` and global sole-GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` guard. The acquisition child explicitly receives that UUID as `CUDA_VISIBLE_DEVICES`; the CPU readout child receives an empty CUDA route. Inherited Python path overrides are removed and native threads are fixed at two, as in the existing endpoint owner.

## One finite sequence

1. Wait at most 24 hours for physical terminal closure of either Penn94's `native_acquisition_v1_owner/OWNER_END.json` or Physics' `external_cohort_owner_v1/OWNER_END.json`. Read only owner/process metadata. Require direct-child wait, an actual terminal exit code, and current process/CUDA absence. Penn also binds its start/child/boot identity; Physics checks the controller and both recorded core/author acquisition children. Any exit outcome qualifies when physical closure is proven. No scientific-success flag, score or favorable result is required. Missing physical proof retains the wait until its deadline; expiry writes a failure and starts no fit.
2. Bind once from the exact successful readiness config, existing full WikiCS role receipts and installed runtime/source receipts. Validate the scientific manifest, bind the root admission and physical gate, recheck relevant process absence, and record fresh GPU/process/system resources immediately before full acquisition. Write a new scientific runtime fingerprint, admission and config. Preserve all readiness artifacts.
3. Run the original complete M1/shared4/genuine-I4 roster: 9 banks, 18 independent fit/selected-state records and 9 selected VALID archives, seeds 7301/7403/7507 in their fixed order. No seed-subset argument is supplied. The finite owner derives the existing endpoint owner and emits the exact 9/18 custody fields required by the sealed reader. It waits/reaps its direct child and checks current CUDA absence before permitting readout.
4. Materialize the sealed `READER_INPUT_TEMPLATE.json` with the actual config, owner and complete-family receipts, then run `analysis.py` once. All 45 fixed five-fold calibration attempts terminate before contrasts. Individual calibration failures remain reportable; successful full readout does not require favorable metrics or every scalar fit to succeed. There is no TEST, retry, grid, final refit, partial readout or hidden survivor selection.

The queue records `PIPELINE_START_V1.json`, the physical gate, fit/readout owner sidecars, and `PIPELINE_END_V1.json`. Stage logs and every scientific failure remain. Exclusive creation prevents a previous attempt from being overwritten or silently rerun. Terminal pipeline counts come from the actual stage receipts; expected counts are labeled separately.

## Finite caps and readiness compatibility

Full acquisition: 6 hours, 16 GiB CUDA, 16 GiB process RSS. The unchanged family checks finite wall/RSS/allocated CUDA at its existing boundaries; the external owner samples process RSS/high-water memory and owned GPU process memory every two seconds, terminating on excess or monitoring failure. Direct-child cleanup has bounded 15-second TERM and 15-second KILL waits. Readout: 15 minutes, CPU only, 16 GiB RSS, the same cleanup. Queue wait: 24 hours; no automatic retry after expiry or failure.

The actual readiness peaks were 7,300,246,528 bytes allocated CUDA, 7,532,969,984 bytes reserved CUDA and 902,176,768 bytes RSS. All are below the proposed 17,179,869,184-byte caps. The 7.709-second readiness owner observation is not a full-family ETA or an isolated speed result. Full acquisition may still fail a cap, and the remaining current cohort may overlap; actual resources and costs are retained.

## Preparation verification

Only stdlib AST parsing/compilation and byte/hash comparisons were used. No numerical imports, acquisition, readiness check, calibration, predictions, current cohort scores or SSH were executed/read. `PIPELINE_STATIC_CHECKS_V1.json` records unchanged readiness hashes and the sealed scientific manifest verification. Root owns the final seal and scientific admission.
