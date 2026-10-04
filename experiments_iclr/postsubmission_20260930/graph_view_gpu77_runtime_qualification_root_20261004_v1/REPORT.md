# 18.77 deterministic full-shape graph-view component qualification

**Both authorized tied-persistent component cases pass exact CUDA function and next-update replay.** The cases ran sequentially on physical GPU0 of `peptide`, co-resident with the root-reported DDI queue. They are TRAIN-only engineering checks, not scientific fits, donors or predictive results.

| Component | First training step (s) | Native evaluation (s) | Snapshot (s) | Peak allocated (GB, decimal) | Peak reserved (GB, decimal) |
|---|---:|---:|---:|---:|---:|
| Fresh local | 13.462 | 5.707 | 0.162 | 17.366 | 18.923 |
| Fresh global | 13.087 | 5.630 | 0.169 | 17.377 | 18.944 |

The two-case supervisor wall time was 116.823s. Each case executes three native/view training updates (24 forwards/backwards), two pooled native evaluations (8 forwards), snapshot/restore, and next-update state/RNG comparisons. Raw logits and fitted states remain in memory only; no checkpoints, donors or predictive scores were saved or fetched. The global case starts freshly initialized with global stage enabled, without a200-update warm handoff.

## Exact custody and physical closure

The original bank package manifest remains `eb34fcb8bc406219007ee39ee9789f1de9bc56acf3cb16164aeab95b4493752f`; its protocol remains `55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46`.35 byte-bound source/context/coverage payloads were staged under this packet's remote `staged_phase`, with no edits to the scientific kernels, constants, initialization recipe or loss. The host-only checker diff records the expected physical UUID pair, fixed selected GPU0, pinned Python, source/coverage relocation and runtime provenance.

The existing remote graph/features artifact, official TRAIN-label projection and data manifest matched the original input hashes before execution. Coverage was reconstructed using the unchanged source and checked against the exact saved split0 receipt. Official VALIDATION/TEST membership masks were used only as role metadata; no VALIDATION/TEST target artifact or predictive score was opened.

Supervisor PID3279686, start ticks1729128889, launched once. Both children exited0 and were closed/reaped by the supervisor. At2026-10-04T19:20:04.693233+00:00, the supervisor and both exact child identities no longer existed, terminal receipts were present, and GPU0 free memory had returned to66,913MiB. No job signal or retry was sent.

Runtime is Python3.12.11, Torch2.7.1/CUDA12.6, NumPy1.26.4 and PyG2.4.0. Package initializer file paths/hashes and executed neural source descriptors are in each case result. These package hashes identify inspected files, not every linked runtime binary. The deterministic policy is unchanged: strict deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled. Replay is exact within this runtime; bitwise cross-runtime initialized weights or trajectories were not compared or claimed.

## Bounded resource implication

The measured first-step/evaluation timings imply an **illustrative projection** of about14.06GPU-hours for one tied augmented fit with200local+2500global updates and one native committee evaluation after every update. Charging a snapshot after every update adds about0.13hours, before preparation, checkpoint I/O, reporting and queue interference. This is arithmetic from two fresh components, not a measured full training schedule or a runtime upper bound. Trained-state performance, other seeds/splits, untied banks and other controls remain unmeasured on this host.

Applying the same tied estimate to all12augmented-bank fits would be about168.8GPU-hours before those omissions; the untied3fits can cost more. Two fully available GPUs would therefore need roughly3.52days for that portion alone, before the18reference fits. Do not advertise completion in hours or dispatch multiple large fits on one GPU simply because memory permits. A bounded schedule should use one owned fit per GPU, balance split/seed blocks and all competing arms across the same host/runtime, and keep every frozen comparison complete before inspecting development scores. Existing DDI/Collab queues retain their ownership and budgets.

Root's next authorized native-GNNM component qualification can give concrete four-native-pass throughput before a30-fit resource decision. This packet launches no predictive job and changes no canonical ledger, status or manuscript. A quality benefit of graph-view sharing remains an empirical hypothesis, not an established methodological contribution.
