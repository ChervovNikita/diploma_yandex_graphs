# HGT35 CPU scheduler v2: runtime repair

Prepared source amendment only. No staging, process dispatch, resource qualification or training was performed by this packet's author.

The closed v1 cohort has exactly five selected native-HGT terminals and thirty resource-deferred terminals. Its 8 GiB address-space qualification exercised one update and did not establish repeated-training allocator headroom. The v1 source packet, admission, all 35 terminals, failure records and selected artifacts must remain intact. Selected checkpoints are best states, not interruption states, and are not eligible restart inputs.

## Runtime change

- Five isolated seed workers, one CPU thread each.
- 16 GiB preimport RLIMIT_AS per worker.
- 14 GiB separately monitored RSS limit per worker.
- At least 96 GiB currently available host/cgroup headroom before dispatch.
- The root must set a positive per-worker wall budget in a fresh release.

The exact sealed v2 fit, models, optimizer, scheduler, selection, split/seeds, arm order and scientific freeze remain bound to their original hashes. Every worker builds fresh initialized models. A separately released complete 35-case rerun occupies this v2 packet's new run directory. There is one admitted v2 study; no automatic repeat, resume, subset replacement or merging of v1 selected rows.

## Admission remains closed

The template is unapproved. Before completing a new execution release, root must observe a successful disposable CPU qualification covering six consecutive TRAIN updates for every one of the seven families, one score-free eval per update, exact initialized geometry/AdamW/OneCycle/member RNG, full graph and state/RNG custody. Qualification must mirror the lifetimes of TRAIN outputs/loss through eval and previous eval/checkpoint objects through the next TRAIN forward, including the all-seven resident build and arm transitions. It must record per-update VmSize/VmPeak/RSS. No validation/test scoring or model selection is permitted in that qualification. A short successful qualification is evidence for admission, not a guaranteed full-run bound.

The actual wrapper transport must prove preimport 16 GiB AS, 14 GiB RSS, one CPU thread and hidden CUDA. Its exact qualification-result object must equal the bound result file. The release supplies absolute path/hash/byte descriptors for the new qualification and transport receipt, wrapper and remote code. This allows the amendment to remain sealed while root qualification is pending.

The release also binds a root-observed `HGT35_closed_failure_inventory_v1` JSON for the exact remote v1 run `root_parallel_cpu_run01`. It lists every original run file with an absolute path, hash and byte count; no symlinks or omitted files are admitted. It declares the run closed/incomplete, no outcome metrics requested, and no selected-checkpoint reuse. Admission verifies the 35 ordered terminal statuses and incomplete/no-subset summary. It hashes all original files and repeats original custody checks at the existing worker/controller boundaries. Original tensors are never deserialized by this amendment.

Use the exact fields in EXECUTION_RELEASE_TEMPLATE.json. Root's original-run inventory must have `schema`, `root_observed`, `original_run_directory`, `status`, `outcome_metrics_requested`, `prior_selected_checkpoints_reused`, and `files`. Inventory status is `closed_incomplete`. Qualification rows retain the existing schema and add `optimizer_updates: 6` and six `per_update_memory` records.

## Narrow verification

SOURCE_DIFF.patch covers the scheduler and deployment helper. The scheduler's worker, collection, closure and signal handling are byte-identical. `main` changes only the runtime-cap call; `run_workers` changes only the optional separate RSS-limit argument and RSS comparison. Its process ownership, signal masking, termination, kill escalation and final reaping are unchanged. The deployment helper changes only the v2 destination and defers qualification verification to the new release-bound admission path.

The existing v1 cancellation suite and independent SIGTERM/KeyboardInterrupt receipts are reused, with their source identities recorded. `narrow_guards.py` checks this source identity and synthetic admission failures for legacy limits, insufficient headroom, unqualified/one-step evidence, checkpoint resume, incomplete original custody and subset scoring. It does not repeat cancellation or execute models.

All 35 new terminals must close before the unchanged development-comparison gate can open. Heldout scoring, original-paper scores and model tuning are outside this runtime amendment. Root owns qualification, source review, inventory creation and any subsequent release.
