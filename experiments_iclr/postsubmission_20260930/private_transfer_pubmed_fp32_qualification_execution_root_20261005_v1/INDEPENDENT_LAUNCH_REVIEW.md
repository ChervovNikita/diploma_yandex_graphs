# Independent focused static repair review of Pubmed FP32 launcher v2

5 October 2026. The preserved v1 P2 finding F1 is resolved at source level. I found no additional concrete source defect in this precise repair recheck. The v1 complete workflow review and its finding remain intact. This review creates no execution, numerical, scientific, fitting or release admission.

## Exact reviewed bytes

Target: `private_transfer_pubmed_fp32_qualification_launch_preparation_20261005_v2`.

| Artifact | SHA256 |
| --- | --- |
| MANIFEST.json | `ca79163892d1ea1af7d0b99b770b1029836eca06e6b0b9ee7beedef306db08c4` |
| SEAL.json | `8b7b1c361e4f168fd6de30d55f163fea44944e13e3068bea79900edb58c9fb8c` |
| execute_once.py | `899e8f608126352fe0387b532eb1768f59c3fcba440bc96d8b5e3da52459a535` |
| owned_supervisor.py.txt | `e97ce9c0ff10de762565f7c02950c312c97452b6d1141ca7d7767d02ae9f48be` |
| Unchanged numerical SOURCE_MANIFEST | `ba897c6588c0ac5d7d2e8473e1582b34c67986c23f265f0588f61c55c747aa6f` |
| Unchanged qualify_training_step.py | `38fcec87c05d744135797546c819eb4e203e4a58c0273796ef7f60676194198e` |

The preserved complete v1 independent review is `private_transfer_pubmed_fp32_qualification_launcher_independent_review_20261005_v1`: REPORT `b0a7a57715a81fab7833d67f7884cec865286ee988e67e6e63d95a720113f85e`, MANIFEST `ac31cb30b056db9816a6257e425bccbc2ec6aa91da758bbfc4a86f431cedc4e1`, SEAL `edd1fd6f4a18305dfa2f0fdfb0e9270ce26ef3edb8fc17a69a938136cfcf3fd2`. These exact review pins match.

All supplied target pins, all 12 v2 packet entries, all 11 v1 preparation entries, all five preserved complete-review members and applicable seal links match. All 74 saved permitted v1-review input paths reauthenticated unchanged. Of 57 v2 declared inputs, 55 permitted source/metadata/review/compact-receipt bindings match; the same two unnecessary prior measured/numerical reports remain unaccessed. All 22 numerical source program bytes reauthenticated unchanged. INPUT_BINDINGS records 95 unique saved permitted files.

## F1 repair and honest completed-text handling

V1 checked the inclusive 3600-second bound before serializing/flushing the final completed=True response. V2 retains that response at `owned_supervisor.py.txt:215` and adds one assertion at217 afterward:

`output_bytes() <= caps['output_bytes'] and time.monotonic() - workflow_started <= SOFT`

The output scan runs first; the final elapsed-time read therefore includes that scan and the already completed final serialization/print/flush. The statement lies inside the existing try/except BaseException region. A crossing raises, preserves EXECUTION_FAILURE with partial files/no retry, and reraises to a nonzero remote exit.

The unchanged client subprocess.run waits for full completion. Its returncode check at59 raises before success JSON parsing at60. Thus a completed=True line already emitted before the new late-bound failure is not accepted. An early-written QUALIFICATION_RECEIPT or EXECUTION_RECEIPT likewise does not establish workflow admission. This resolves the exact static normal-completion counterexample. No timing path, fixture, model or target code was executed.

## Exact preservation

Independent complete-module AST comparison proves that removing only the new final assertion makes the entire actual v2 supervisor AST identical to v1. The actual client text is byte-identical after replacing only the updated remote-source pin with its v1 value. All staging whitelists, authority refs, single-attempt dispatch, full-exit rejection, direct-child ownership, sole wait4/raw signaling, exception cleanup, CUDA accounting, final wait4 RSS and existing receipt/resource checks are therefore preserved.

QUALIFICATION_JOB_DISABLED, QUALIFICATION_RELEASE_DISABLED, BOUNDS_PROPOSAL and SUPERVISION_REUSE.diff are byte-identical to v1. Disabled admission changes only client/remote pins. The numerical manifest/qualifier and all 22 member bytes remain exact. Tolerances, three architectures, seed/factor seed, full589 sampler cycle, first TRAIN episode, two reachable recomputed histories plus the discarded stale step, and states-discarded policy are unchanged.

The v1-reviewed actual CPU package/input, feature-value and TRAIN-geometry authorities remain present and narrowly scoped. TRAIN/features alone are loaded; no VALID positives/pool or TEST values are opened by this qualification path. Prior GPU/model qualification is the operation's intended result, and historical VALID-generator authority is unnecessary for these loaded values. No extra receipt, quality gate, fitting permission or scientific definition is introduced. The unchanged job evidence template has no circular binding to an admission containing its job hash.

Caps remain soft3600/hard4200 seconds, RSS16 GiB, owned and each reported allocated/reserved CUDA peak20 GiB, total remote owned files64 MiB and SSH4500 seconds. The repair adds the final required response boundary. It does not change sampling/grace into instantaneous kernel quotas. Source preparation v2 retains the fresh v1 execution identity; it represents a preserved source repair, with no attempt made. Root and independent-client approvals remain false, review references absent, and the job remains disabled pending actual root/resource release.

## Scope and disposition

This was the requested focused source/AST/hash/JSON recheck, building on the sealed complete v1 review. Only the two changed control source bodies were newly AST-compared; the frozen numerical closure was rehashed for preservation. No target/model/numerical import or execution, test, checker, fixture, dataset/history/FREEZE/checkpoint/feature/model/logit/score/prediction/raw outcome read or credential access occurred. No allocation/18.77/MacLink contact, staging, admission, prior/source/schema/canonical/publisher edit or agent spawn occurred. Only this separate v2 review directory was written. Withdrawn access remains unchanged.

FINDINGS records F1 resolved_at_source_level and zero new concrete findings. CHECKS records independent exact-change fingerprints and final-bound/exit-control facts. Successful actual new qualification and terminal/resource evidence remain unobserved. Execution, fit and scientific admission remain false; no release or clearance verdict is supplied.
