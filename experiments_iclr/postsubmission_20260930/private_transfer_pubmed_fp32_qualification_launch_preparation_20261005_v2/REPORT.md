# Disabled Pubmed FP32 launch v2: final flushed-response bound repair

5 October 2026. This minimal successor preserves the sealed v1 packet and repairs the single final-response bound gap supplied by root from independent review. V1 MANIFEST is `344f387ac75716cc16b1623b221c0267f5a49611bdd327b6fd7e423ceba0de08`, SEAL `730efd03973754071c599cb248b17299275f4296e25db42b1c5691cb93f6d4b5`, and REPORT `b069af469227189529defdb020b60ef0d671a4dbbb060ba765cd2c74004d6dfb`. The v1 independent report was still being completed; no report pin or review approval is invented here. One focused source repair check remains necessary.

## Exact change and source identity

At `owned_supervisor.py.txt:215`, the completed=True response is serialized, printed and flushed. V2 adds one assertion at line217 inside the existing BaseException failure path: scan final owned output against the unchanged64MiB cap, then read workflow elapsed time against the unchanged3600-second soft bound. The output scan precedes the final time read, so that scan is covered by the time check. If either condition fails, the existing handler preserves EXECUTION_FAILURE and reraises to nonzero process exit.

The existing client waits for subprocess completion. Its nonzero transport check at `execute_once.py:59` raises before response parsing at line60. A completed=True line that was already printed before a late bound failure therefore remains unadmitted. No new guard or admission concept is introduced; the existing caps are checked at the final required boundary.

| Source | SHA256 |
| --- | --- |
| Client v2 | `899e8f608126352fe0387b532eb1768f59c3fcba440bc96d8b5e3da52459a535` |
| Actual remote supervisor v2 | `e97ce9c0ff10de762565f7c02950c312c97452b6d1141ca7d7767d02ae9f48be` |
| Exact unchanged numerical qualifier | `38fcec87c05d744135797546c819eb4e203e4a58c0273796ef7f60676194198e` |
| Unchanged complete22-program numerical SOURCE_MANIFEST | `ba897c6588c0ac5d7d2e8473e1582b34c67986c23f265f0588f61c55c747aa6f` |

The client changes only its pinned remote SHA. Disabled admission changes only client/remote SHA fields. QUALIFICATION_JOB_DISABLED, QUALIFICATION_RELEASE_DISABLED, BOUNDS_PROPOSAL and SUPERVISION_REUSE.diff are byte identical to v1. SOURCE_MANIFEST refreshes only the changed control-source rows and identifies this v2 packet. V1_TO_V2.diff is the concise complete source/admission diff.

## Preserved operation and disabled status

All numerical bodies, tolerances, three architectures, seed/factor seed, first prospective TRAIN episode after the full589 sampler cycle, two reachable recomputed histories plus the extra stale commit, and discarded-state policy are unchanged. The actual input/root feature adoption, CPU package/input runtime identity and root-adopted TRAIN sampler geometry remain bound. The qualifier loads TRAIN/features only; VALID positives and negative VALID pool are not loaded. Full historical VALID-generator authority and preexisting GPU/model qualification are not asserted. This operation is the fresh GPU/model/Adam/derivative qualification.

Caps remain3600s soft,4200s hard,16GiB RSS,20GiB owned and per-arm allocated/reserved CUDA,64MiB total owned output and4500s SSH transport. Final authoritative wait4 RSS, actual per-arm CUDA peaks, source/input/runtime/job identity and receipt checks are retained. Normal direct owned-child execution, exact repository/hostname/sole GPU UUID/port2222 route, exclusive one-attempt intent, exception cleanup and unknown-terminal no-relaunch behavior are unchanged. No host/namespace changes, alternate routes or signals against other processes are introduced.

The same still-prospective execution identity `private_transfer_pubmed_fp32_qualification_execution_root_20261005_v1` is retained to preserve job scope bytes. This is a v2 source preparation, not a second attempt. Root admission and independent launcher approval remain false; actual review references remain absent. The fixed21 fit proposal remains disabled and original30 promotion/full39 descriptive gates are unchanged.

## Static verification and remaining work

AST comparison proves the supervisor differs from v1 by exactly one appended assertion in its existing try body. The entire client source differs only at the remote hash. Existing job/release/caps/scope bytes match v1. The client rejects nonzero status before success JSON parsing. All57 bindings match: the44 original saved source/custody/cost references plus all13 exact v1 files. Original v1 manifest entries/seal were reauthenticated and preserved.

No target module/function, qualification, test or fixture was imported or executed; checks used local source, JSON, SHA256 and AST only. No data/history/science result/checkpoint/score/prediction payload or credentials were opened. No allocation/18.77/MacLink contact, staging, admission, canonical/index/publisher or predecessor edit occurred. Only this new v2 directory was written. The source remains disabled pending the focused repair check and actual root/resource admission. No new measured costs, fit, scientific result or authority is created.
