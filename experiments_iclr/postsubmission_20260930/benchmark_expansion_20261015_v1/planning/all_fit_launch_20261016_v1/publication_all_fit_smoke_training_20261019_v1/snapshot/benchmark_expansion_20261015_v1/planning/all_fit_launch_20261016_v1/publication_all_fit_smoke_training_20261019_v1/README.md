# All-fit smoke admission repairs and receipt audit, 19 October 2026

Source successor to `68988d18ad4b9b7d28fb1afc7ba0b273e8e4bc3f` on
`codex/all-fit-smoke-training-20261016`. This snapshot preserves the clean
file-cache memory admission wrapper, isolated CUDA OOM retry wrapper, compact
diagnosis and launch evidence, independent receipt audit, and current observer
and continuation. The scientific design, 6,460 fit IDs, 270 exports and original
three-GPU assignments remain unchanged.

At the recorded recovery observation around 15:53 Moscow on 19 October, both
server51 workers had all 2,153 assigned smokes PASS and were waiting for the
global barrier. The exact failed GPU0 fit passed explicit attempt0002 with two
optimizer updates, complete finite VALID, exact checkpoint reload, no TEST,
exit zero, 83.0 seconds wall time and 55.7 GB peak reserved CUDA memory. Its
zero-update OOM attempt0001 remains preserved. The allocation had 755 PASS
and two active smokes after the admission repair; its backlog was progressing.
Full training had not started. All 6,460 exact PASS records remain required.

The earlier independent audit verified all 5,050 then-current PASS receipt
chains, frozen sources, exactly 10,100 optimizer updates, finite TRAIN and
complete VALID checks, checkpoint reload, and no TEST. It found 1,409 allocation
fits pending and the one server51 GPU0 OOM failure, subsequently repaired above.
Assignments cover the canonical fits exactly once without gaps or overlaps.
The audit and its nonpass list are retained as a dated observation, not rewritten
as newer progress. PID 1495596 named in the OOM report remains unidentified;
no saved execution1–7 native result matched it.

The wrappers use the unchanged execution7 plans, receipts and global training
barrier. Their recorded source checks total 28 PASS: 12 memory-admission checks
and 16 isolated-retry fixture checks. The admission repair changes conservative
RAM accounting only. The isolated retry authorizes only the preserved failed
fit's second attempt, without repeating successful fits or changing native
settings. These checks establish engineering execution and receipt integrity;
they provide no convergence or scientific accuracy conclusion.

Publication uses the existing immutable snapshot and separate Git index,
preserving live HEAD `48329b19941ea869c439e5a220a3df0e30aa1237`. Publication
performs no native work. Compact sources and provenance are selected; raw audit
inventories, logs, data, checkpoints and generated runtime payloads remain in
their existing evidence locations. Six prior publication snapshots are retained.
