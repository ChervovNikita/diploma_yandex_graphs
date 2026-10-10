# Static audit and versioned watcher repair for the 18.77 F readout

## Status

Both F training shards have successful normalized owner closure receipts. Shard 0 closed at 2026-10-10 21:11:57.186553 UTC; shard 1 closed at 21:10:43.101100 UTC. Their terminal records match the saved monitor transport receipt and retain complete-family, direct-child-wait, child-absence and owned-CUDA-absence flags. These are execution closure records. The joined complete reader has not yet admitted the full artifact roster or produced the scientific analysis.

The original watcher failed at 21:12:12.019976 UTC with `Current safe role differs: train`, reader_launches = 0, no reader child PID and no reader child wait. The failed compact fetch retains that exact watcher terminal record. This is a pre-reader metadata failure, separate from the successful fits. No retraining is justified by it.

## Cause and observed facts

`ready_command.py` writes file descriptor paths as `str(q)`, preserving the submitted role alias. The v1 watcher's `descriptor` uses `Path.resolve()`, then compares the entire resulting descriptor with that preserved alias and also compares its canonical path with the un-resolved CONFIG string. The frozen readiness and CONFIG identify the same alias; descriptor construction has already checked its exact admitted SHA before the failing comparison.

Root supplied the actual remote resolution receipt `F_ROLE_PATHS_ACTUAL_V1.json`. TRAIN and VALID are symlinks from the frozen `label_correction_full_WikiCS_role_projection_allocation_20261008_v1` paths to the corresponding `wikics_unit_contrastive_attribution_gpu77_data_20261007_v1` paths within the same authorized repository. TRAIN remains 21,137,972 bytes with SHA256 dce6c4a982027604dad198533352d397c5d1b7a0a6d6fa4868468c872457f9e2; VALID remains 84,878 bytes with SHA256 99114cb1d50b876e9cdb2d4a1c116e217f0a8aa2583a2561d0491461c0c0ec48. The alias/canonical path mismatch is therefore established without a data-content change.

The sealed complete reader already resolves safe-role receipt paths and submitted CONFIG paths before comparing them. Its source and fixed protocol need no repair.

## Prepared successor

`reader_after_closure_v2.py` is a separate local source successor for later deployment into the unchanged pilot-root directory. It has not been executed, deployed or launched.

The bounded role repair keeps the exact readiness alias equal to the frozen CONFIG path. It checks the same admitted SHA and byte count, requires the current canonical descriptor to equal the exact target recorded in the remote resolution facts, and retains repository scope. It passes the canonical path/bytes/SHA descriptor to the unchanged reader. `ROLE_PATH_RESOLUTION_V2.json` records both the admitted alias and current resolved descriptor, with readiness, CONFIG and prior-failure custody. No alternate data, re-exported array or refreshed role admission is introduced.

The successor requires the exact observed v1 failure object and an existing original watcher-start receipt. It also requires absence of v1 reader-execution-start, input-spec and complete-analysis artifacts. Original v1 receipts remain intact. The new source has one exclusive v2 claim and fresh v2 input/output/log/start/end names, so a failed v2 attempt cannot be silently repeated. Its records identify one versioned metadata recovery, zero scientific reruns and zero automatic retries.

Both successful original owner closures, source/config/ready/freeze hashes, submitted shard configurations and owner complete-file hashes are still required before one CPU reader child launches. The original fit-owner command, training source, data, selectors and results are untouched. The reviewed reader's 15-bank / 24-optimizer-bundle / 33-native-body closure, all 75 terminal calibration attempts before interpretation, raw authority, fixed accuracy/NLL rules and failure retention are unchanged.

## Static verification and limits

The successor parses as Python AST. `require`, `stamp`, `digest`, `descriptor`, `read`, `write_once` and `stopped` remain AST-identical to v1. The subprocess Popen, child waits and killpg calls are AST-identical. The normal admitted Python, CPU reader environment, 3,600-second bound, 60-second cleanup reserve, direct-child ownership, bounded cleanup and complete-summary checks remain preserved. The reader and protocol pins are unchanged.

This audit used local source, JSON metadata and byte hashes. It made no remote connection, source execution, numerical import, array/model/checkpoint read, fit, retraining or criterion change. No quality conclusion is drawn from the training monitor payload. All observed failures and successful fits remain in their original records.

The packet is ready for root source review and separate deployment/launch admission. Deployment must add the v2 watcher file to the existing pilot-root directory while preserving every v1 artifact; the source explicitly checks that directory and authorized host/GPU roster. The analysis itself remains unlaunched. Its future success or failure must receive the new v2 terminal record, and only its completed output can support scientific interpretation.
