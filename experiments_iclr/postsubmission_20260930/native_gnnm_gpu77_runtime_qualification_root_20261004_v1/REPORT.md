# 18.77 native GNNM component qualification

**Both native GNNM component cases pass exact CUDA function and next-update replay.** They use fresh local and global components, full Amazon Ratings graph/features and all 12,246 official TRAIN labels on split 0. They establish correct replay and a bounded resource estimate; predictive quality remains untested by this packet.

| Component | First training update (s) | Native committee evaluation (s) | Snapshot (s) | Peak allocated (GB, decimal) | Peak reserved (GB, decimal) |
|---|---:|---:|---:|---:|---:|
| Fresh local | 6.351 | 5.667 | 0.214 | 17.351 | 18.952 |
| Fresh global | 6.422 | 5.637 | 0.234 | 17.364 | 18.973 |

The two-case supervisor wall time was 77.343 seconds. Each case calls the unchanged native v2 training kernel for three engineering updates: four streaming member cross-entropies/backwards per update, divided by four, followed by one Adam update. Two native committee evaluations perform eight forwards. Snapshot/restore and next-update comparison cover model parameters, Adam state, gradients and RNG. Function replay and next-update replay have zero maximum absolute error; the local and global next-update trees contain 524 and 576 tensors, respectively. The global component starts freshly initialized at update 201 with the global flag enabled; a local-to-global trained-state handoff is not measured.

## Custody and closure

The native v2 source manifest is `fbe1aec5eafb029fce9c9d42ebf9a5b94b98cfbc2ce9881502f0e3fc8e192ca6` and its protocol is `9da2f5ebed5660d09b77ef4a620caa50bda81e4dc8ee999444aefc8f87e58524`. The checker invokes `native_driver.native_train_step` and never invokes the full scientific driver. All 55 source/context/checker payloads (440,188 bytes) were staged under this packet's owned remote `staged_phase`. The packaging verification rechecks each local source byte against the staged inventory and all returned monitor receipts. It makes no edit to source, loss or initialization recipes.

Existing remote input artifacts matched their frozen hashes before execution: public graph/features `19757299bcfd9e493e9ceae9e73248753ab1e773ccc1f6b57c6fadf8c843310f`, split 0 TRAIN labels `9d700d1897bbf73ec70a4c3158e07d6b42909603e28ac54716800f3d4883f749`, and data manifest `8d565c20c6d669c2577e9fb8c5405575a6ed97a9964f91818b052e43cd0379bf`. Coverage equals the byte-bound official TRAIN coverage receipt. VALIDATION/TEST memberships are role metadata only; no heldout targets or scores are opened. Engineering states and raw logits remain in memory only. No donor, scientific checkpoint or predictive result is produced.

The runtime descriptors equal the completed bank qualification on `peptide`: Python 3.12.11, Torch 2.7.1/CUDA 12.6, NumPy 1.26.4, PyG 2.4.0, strict deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled. Package initializer file hashes and executed model source hashes are recorded. These descriptors do not fingerprint every linked binary or imply equality with another runtime's initialized weights/trajectories.

Supervisor PID 3281320/start ticks 1729210361 launched once. Children PID 3281326/start ticks 1729210372 and PID 3281366/start ticks 1729214206 each exited 0 and were closed/reaped. The final monitor at 2026-10-04T19:30:26.517175+00:00 confirms the exact supervisor and child identities are absent and GPU0 free memory is 66,913 MiB. Only authorized physical GPU0 `GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998` executed these cases, co-resident with the root-reported DDI queue. No retry, timeout or signal was used.

## Resource implication for the frozen comparison

Using the measured fresh-component timings, 200 local + 2,500 global updates with a native committee evaluation after every update gives **an illustrative 9.04 GPU-hours per native GNNM fit**. Snapshotting after every update would add about 0.17 hours before preparation, checkpoint I/O, queue interference and other omissions. This is arithmetic from short engineering components, not a measured full schedule, trained-state throughput or upper bound.

The planned 30-fit accounting is 12 augmented-bank fits + 15 ordinary reference fits + 3 native GNNM fits. The earlier tied-bank estimate was 14.06 GPU-hours per augmented fit. Applying that rate to all 12 augmented fits plus this native rate to three fits gives about 195.9 illustrative GPU-hours **before the 15 ordinary references and omissions**; untied bank throughput remains unmeasured and can be slower. A total 30-fit completion ETA is unsupported until the remaining kernels are measured. The bank-only 12-fit arithmetic was about 168.8 GPU-hours, or 3.52 days on two fully available GPUs.

Use one owned fit per GPU, balance competing arms and split/seed blocks across the same 18.77 runtime, and complete frozen paired comparisons before inspecting their development outcomes. Source/data custody and comparator competence requirements remain the parent's responsibility before a scientific launch. Accuracy is the selection objective; time and memory are measured costs. This packet changes no canonical ledger, scientific queue, manuscript or original paper scores and launches no predictive training.
