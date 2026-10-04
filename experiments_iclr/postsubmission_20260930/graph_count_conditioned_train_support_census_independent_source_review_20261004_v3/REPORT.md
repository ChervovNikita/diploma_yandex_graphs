# TRAIN census-v3 independent F06 delta review

Disposition: **PASS, source only**, candidate manifest `e6c7f6bc64e9d3454f2a7881acb16f493ec00b2126de0397b62c7b64a8b064e2`.

Verified all 8 payloads (59,849 bytes), 20 declared source/input pins and 5 producer pins. V2 and its independent findings are preserved. Worker census.py, PLAN and disabled root template are byte-identical v2.

F06 closes in source. Every recorded ownership-handler SIGTERM/SIGINT immediately assigns a nonqualifying stop reason. A final drain follows handler restoration before the physical snapshot. The received signal list is retained in the physical receipt and inherited terminal, while adoption still requires stop=None. This addresses the earlier successful-exit/terminalization gap without signal injection.

Removing exactly four expected AST changes reproduces the whole v2 supervisor: handler nonlocal and stop assignment, final received drain, and received_supervisor_signals metadata. All cleanup/reap, guarded physical-before-collection, exact three-seed/17-batch/two-population output and data/count custody checks remain unchanged. Prior F01/F02/F03/F05 repairs and F04 narrowed runtime-profile scope are inherited from the sealed v2 review.

The numerical worker remains unchanged from v2, which was independently proved to preserve the original v1 numerical body bytes and AST. The census covers each complete native 65536-row mask, positive and native negative populations, 17 batches per seed 0/1/2, exactly two TRAIN files and no features, heldout, models, optimizer or fit. Digests and histograms check receipt consistency; they do not independently recompute counts.

Runtime limits remain material: sampled owned-session RSS, worker batch-boundary CUDA peaks, no instantaneous OS ceiling or escape sandbox, partial seed histograms in RAM, and no guarantee against uninterruptible kernel/filesystem or storage failure. Physical wall/RSS measure the child observation/cleanup window; later parent collection/hash/write and parent RSS are explicitly excluded. Full default-dtype/CUDA/autocast profile fields are not newly measured.

Root must authenticate actual source/runtime/route/TRAIN authority and separately admit at most one exact fresh census. Only linked complete child and supervisor terminal/custody may be adopted. No scientific qualification, predictive quality, novelty, independent graph sampling or fresh post-TEST confirmation follows.

Review used local source/JSON, stdlib AST/hashes only. No target/numerical import, compilation or execution, model/data/state/array payload access, runtime binary/interpreter inspection, process/signal probe or remote/server access occurred. No execution is authorized by this source PASS.
