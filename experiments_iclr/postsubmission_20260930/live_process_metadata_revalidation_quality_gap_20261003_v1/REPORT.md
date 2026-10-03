# Read-only initializer and mixed40 revalidation

Observed 3 October 2026 at **09:53:57.335080 UTC**. SSH used the existing authorized `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222` route and `/usr/bin/python3 -I -S -B`. The repository path and single GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` matched the existing allocation receipt.

## Initializer

Current study: `graph_init_cfg0_outcome_aware_precision_v2`, immutable registry SHA256 `715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f`. Current coordinator is `graph_init_precision_continuation_v3_cap_binding/coordinator_run_v4_occupied_recovery`; its preserved v3 predecessor failed on a pre-child GPU-occupancy refusal, and the root recovery decision admitted only the eight untouched fits.

| Phase | Successful canonical terminals | Expected |
| --- | ---: | ---: |
| qualify | 6 | 6 |
| warm | 6 | 6 |
| initialize | 30 | 30 |
| fit | 28 | 30 |
| Total | 70 | 72 |

There are 71 canonical claims and 70 terminals; no unknown keys or terminals without claims. All 70 successful terminal/claim/FREEZE descriptors and canonical phase identities were consistent in this metadata snapshot, and all 70 terminals retained `final_labels_read=false`. This check did not run the strict source guard, traverse scientific payloads, or audit numerical results.

Completed fits comprise all five arms for Squirrel seeds17/29/43 and Photo seeds17/29, plus Photo43 `graph`, `common_only`, and `random_tangent`.

- **Running:** Photo43 `topology_permuted`, key `929bff78202b8b37e0524ff349489184ba4160800dd2d40d63a65566e52cfecd`; canonical claim present, terminal absent. Root START is 09:47:50.663176 UTC; outer START is 09:47:42.656436 UTC, with the registered 15,000-second cap.
- **Unexecuted:** Photo43 `warm_copy`, key `5b66833ca4b56853db345872727cb6c3fddf6f3ee61637ebd90bcede1fd6d4e6`; neither claim nor terminal present.

Local recovery coordinator PID73484 (`Ss`) and launcher PID24536 (`S`) were present. The corresponding remote process chain was PID381275 →381277 →381279 →381281 →381283, with the first four sleeping and the leaf running. Process scope matched the current precision initializer/coordinator paths; no unrelated process commands were emitted.

Latest completed recovery fit: Photo43 `random_tangent`, key `cb8e89094e304def76844ff1be973865aa9bc225a5a7916132675aa3ea8889f1`.

- Canonical terminal SHA256: `b01bf502ef76b6b0ee1941e762a19d689059081ab1e4c097d88e9a9bef7d33f2`.
- FREEZE SHA256: `8ba7c905f93f7832a7dbbf818eda2eb521b440504203383bab8fdb215ae8f5dc`.
- Root terminal: 09:47:05.323505 UTC, completed, no automatic retry, final labels unread; SHA256 `c3a8a6a95ec2cac828643e0147ac87c443bfeba54bacbc89dcf7d855c5f2f1dd`.
- Whole terminal: 09:47:05.420904 UTC, complete, within cap, no timeout, unchanged root request; 5,330.15554753691 whole-process seconds; SHA256 `436d328217f17219662a5a40fe017aee92025ccea0527f8dd795325970949095`.

The current local coordinator has neither `COMPLETED.json` nor `FAILED.json`. Canonical comparison and final-report directories are absent. **The cohort is not closed.** Missing `START.json`/`TERMINAL.json` at guessed inner-supervisor names are not interpreted as failures; canonical root/outer terminals govern the status reported here.

## Mixed40

Run: `graph_mixed_block_training_preparation_20261003_v2/runs/root_full40_run01`, admitted fresh-all40 source manifest `ee2cc8623c8752e1fc5a56dde5968c624d2e9e9b735f9534672fae6549c592d4`.

- All20 DBLP cases have `selected` terminal status: seeds131/137/139/149/151 × `own/own`, `pool/pool`, `pool/own`, `own/pool`.
- All20 ACM cases have no terminal yet. No failure/resource-deferred terminal was observed among the fixed40 cases.
- Every completed case retains `final_labels_closed=true`.
- Supervisor PID379192 is present (`S`, PPID1); training child PID379193 is present (`R`, PPID379192).
- `STUDY_STARTED.json` exists. `STUDY.json` and `SUPERVISOR_TERMINAL.json` are absent. **The full40 family is not closed or reaped.**

Transport STARTED SHA256: `587bad6eb5520991a8b78a347cf600c0155198b9043c50efb0fb9278c28a7099`; CHILD SHA256: `2bc796a0f7d5e9b2de2a80b1536d20d0f5cd4ecd88ecf248748dc8e45ade952b`.

## Audit entry points after closure

No audit/evaluation entry point was executed or released.

- Initializer strict canonical metadata checker: `graph_init_precision_continuation_v3_cap_binding/continuation_support.py:completed_phase` and `scan_state`. Root must retain the exact current registry, all72 canonical terminals, both coordinator namespaces' per-key root/outer supervision evidence, and the preserved failed/pre-child evidence before separate comparison/report admission. `graph_init_analysis_companion_v4/analyze_complete_report.py` is downstream of a separately admitted completed final report and does not replace the original selected-tensor/source audit.
- Mixed40 independent selected-state replay and development evaluation: `graph_mixed_block_closed_family_evaluation_preparation_20261003_v2/evaluate.py`, using fresh external evaluation freeze/release binding the final STUDY hash/length, exact final evaluator manifest/source critic, all40 selected/unreused/binding-verified terminals, preservation, closure and child reaping. It remains unadmitted while training is incomplete.

## Scope and receipt custody

The remote operation made no writes. It read only source-free canonical identity/status/custody JSON fields, relevant process metadata and aggregate GPU identity. Terminal JSON was parsed to extract an explicit metadata allowlist; score fields were not accessed or emitted. No heldout labels, checkpoint/logit arrays, selections, selected descriptives, training traces or metric logs were opened. No evaluator, restart, process termination, environment edit or process-isolation change occurred. Snapshot observations are time-specific and do not establish scientific quality.

Files retained locally:

- `OBSERVATION.json`: SHA256 `0f479d4e0cd7e28b73e96cb786afeadde2780d603c0db45078c6462235b5114a`, 57,778 bytes.
- `TRANSPORT_RECEIPT.json`: SHA256 `e1ebaed88718ef21f947b7c6f18e98ec23d4f4aede15f206aad64877db94e26b`, 823 bytes.
- `REMOTE_READ_ONLY_CODE.py.txt`: SHA256 `f016e7775437cd9e9e53cf2f7946b4e163fd8cf48d46ad69b14e0b22afa44b0e`, 9,562 bytes.
- `revalidate_metadata.py`: SHA256 `627aad35cefba6c947ebe0fc8bba2d205baf09a17bdf45022a247e5bbc2ff51d`, 13,959 bytes.

## Native source handback remains v4

`graph_conditional_response_native_source_preparation_20261003_v4` remains sealed and unchanged; manifest SHA256 `cdece1f2a5e317de54d509d6b961ca41235cab880c3becb959b792aa75e03d72`, dispatcher SHA256 `4aac90f7622ab42d42ee7cbfa397f9d8897c3f59d16ca1ef2e249c13e0ac3c88`.

Its independent recheck is `graph_conditional_response_native_source_recheck_20261003_v4`: REPORT SHA256 `3ac05c6c5bd6f86d17f29f9d7b41d6ed66e79594ce16725ef2994535d0a006aa`, REVIEW SHA256 `68d895ffc60f796c90ec1c73eb59aea6c8544fd0a885c18da657fa89865bfc82`, MANIFEST SHA256 `296054ba18695e6dabe0a1941035c6f09a09d489aca43474f16f5ef2fc78aed2`.

All37 sealed payload hashes passed the independent recheck. RC-01 execution receipt/cost-event preservation repair passed source review; numerical passage is still unverified. Declared runtime-file binding is not a complete linked/transitive dependency closure. The minor persisted-versus-returned elapsed-time receipt-I/O wording remains disclosed and does not justify another preparation version.

Authorized CPU entry remains `prepared_native_checks.run_authorized_native_checks(authorization_file=<absolute root-issued JSON>, receipt_file=<absolute fresh project execution JSON>)`. The root-issued authorization must carry the explicit CPU-engineering flag, root reference, exact runtime versions/files and actual deterministic CPU settings. The sealed source is not changed for the one-GPU host's Torch2.1.2 missing `get_default_device`; root owns the existing77/Torch2.7.1 qualification plan.
