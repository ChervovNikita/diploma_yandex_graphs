# Native exact-CB bucket diagnostic evidence adoption

Decision: **PASS_FIXED_ORIGINAL_RULE**. This adopts authenticated local run evidence; it is not a fresh scientific or manuscript review.

One seed0 initialization and the first full native TRAIN batch (65,536 positive and 65,536 negative queries, F4 bank, 43,790 model parameters) shared the same retained neural tensors between direct and bucket implementations. Four reverses evaluated direct/bucket J_K and direct/bucket J_K_sep. Zero optimizer updates, fits, target-loss reverses or VALID/TEST/score reads.

All complete loss arrays/scalars, all 49 named parameters per objective (41 compared and 8 both unused), and every positive/negative left/right member-slot gradient passed the unchanged atol=rtol=1.52587890625e-05. Forced r0 gradients were exact zeros. There were 96 numeric comparison reports plus 16 both-unused records; zero violations. Worst fractions of allowed error: loss 0.00404850254, named gradient 3.04453388e-05, member-slot gradient 3.25962830e-08. Exact RNG was unchanged by every reverse; initial/final state digest matched.

| Observed synchronized component | Direct seconds | Bucket seconds |
| --- | ---: | ---: |
| Loss forward, both populations | 8.704110708 | 0.871523365 |
| J_K reverse | 29.064976133 | 4.559713278 |
| J_K_sep reverse | 29.012600757 | 4.650273565 |

Python ESP groups: 1,505 direct / 115 bucket. Recurrence slot loops: 132,447 / 13,609. One encoder forward, two population forwards, four conditional-side calls per implementation, four bucket-plan calls and four reverse evaluations. These are Python dispatch/loop observations, not CUDA kernel profiling.

The complete diagnostic used 92.871474598 seconds to its result and 93.768560797 seconds through child final custody; the physical supervisor receipt records 96.367632069 seconds and the last completed parent budget check 97.465449981 seconds. Sampled owned-session RSS peaked at 3,223,498,752 bytes. Cumulative CUDA peaks were 28,810,767,872 allocated / 44,929,384,448 reserved bytes (28.81 / 44.93 decimal GB). Direct/bucket auxiliary graphs coexisted and four reverses retained shared activations; these peaks do not represent isolated implementation or ordinary training memory.

The source/review seals and payloads, root release/admission, fetched transport descriptors, full 974,562-byte DIAGNOSTIC JSON, child final custody, supervisor custody and physical terminal all authenticated. Child exit0, reaped and session closed; no stop, errors or unresolved cleanup. Existing server runtime/TRAIN descriptors matched prior authority metadata without local binary/data reads. LAUNCH.json, STATUS.json and STDOUT.txt remain represented only by supervisor custody descriptors; the prior direct CPU raw qualification is also descriptor-only under existing admission.

Scope: a single fixed-order diagnostic provides no repeated benchmark, end-to-end speed, fit-memory qualification, kernel causality, predictive result, novelty or donor admission. Sampled RSS and the original unmeasured parent publication/stdio/exit tails retain their stated limits. Existing source, evidence and canonical state were preserved; this packet authorizes no new execution.
