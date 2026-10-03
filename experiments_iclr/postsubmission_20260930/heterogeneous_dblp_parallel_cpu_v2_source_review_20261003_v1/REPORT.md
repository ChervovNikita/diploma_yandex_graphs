# Independent HGT35 scheduler v2 source review

**GO for source scheduling admission** against manifest `632a8ff944c5318e1e038069f3b51d2af7a5be24f9f43bc33cd101cbc5fa37ed`.

Scheduler SHA-256: `9b9429df839e9aedbc748ed413e9c18c5a21f73fae4797eb5bce1f9d5740f4ea`.

## Narrow source verification

All 12 payloads, 39 external source records and the exact scientific freeze match bound hashes/bytes. Fourteen existing functions, including worker, collection, closure, resource screen, signal installation/restoration and source preservation, are byte-identical to the audited v1 source. Main changes only the 16 GiB AS / separate 14 GiB sampled RSS worker call. run_workers changes only its optional RSS argument/threshold; PID ownership, signal masking, termination, kill escalation and final reaping are unchanged. The deployment helper changes only the v2 destination and defers qualification verification to release-bound admission. Existing cancellation evidence matches this reviewer's original receipt. No cancellation/algebra/fixture suite was rerun.

## Admission guards

- Full seven-arm qualification is required in exact frozen order, with six consecutive observed updates per arm, score-free complete state/RNG custody, original preservation and successful exact 16 GiB AS / 14 GiB RSS / one-thread / 1200-second transport. Its result object must equal the descriptor-bound qualification file. Partial qualification cannot admit execution.
- The release must match the bound original-failure inventory SHA-256 `af906d64d2a06396aa3c9fe3b38696e2d4c9bfeb68637b9aef169d520609ae4f`. Admission checks the exact all-file set, absolute paths, no symlinks and all hashes/bytes. It requires the ordered 35 original statuses, five native selections and 30 resource deferments, an incomplete summary and no subset comparison. Original inventory/file records are added to existing before/after preservation boundaries. Root reports 178 bound files; these remote original payloads were not opened locally in this source review.
- A fresh complete 35 rerun, no selected-checkpoint resume, exact root release and at least 96 GiB current host/cgroup headroom are mandatory. Fresh model construction, fit and OneCycle calls remain unchanged; no original selected row enters the new cohort. The existing no-repeat marker still admits only one v2 study.
- Original fit/selection/summary hashes, all 35 closure, artifact custody, canonical STUDY interface and closed TEST policy remain unchanged.

## Launch conditions

Root must wait for successful completion of all seven qualification arms and bind the actual final result/transport in the release. Root should also verify the inner process peak RSS against 14 GiB alongside the wrapper's sampled peak: the admission guard checks phase VmRSS/VmSize/VmPeak snapshots but not process_peak_RSS_bytes. Current 96 GiB headroom and a positive worker wall budget remain required at dispatch. This is a source GO; it does not certify the pending qualification result or guarantee 300-epoch resource sufficiency.

`CHECKS.json` records independent source/hash comparisons. Only source and frozen/custody metadata were read. No real labels, checkpoints, logits or quality outputs were opened; no models, training, GPU or remote scientific work were run.
