# Independent parallel DBLP CPU source review

## Verdict

**GO for source scheduling admission** against scheduler manifest `4ccd3b3469893e078a545eb8797daea197047df4303d84f63fc9be2dbd3e5551`. No unresolved source blocker remains. The root owns the separate exact execution release and launch.

Reviewed packet: `graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v1`.

- Scheduler source SHA-256: `923d107b03b8411466a691649e9585a84c2998b890d9c23aa279a5e6e531671f` (27,485 bytes).
- Seal SHA-256: `bd0b802b3b853be7f1e8b1fee0751479701affd5836ca72a87b2c49c6f5a3795`.
- Frozen study SHA-256: `29885a100527226e9d182c54567e748f17f384254d23e009d9089fb18352d352`.
- Original v2 preparation manifest: `c94153106d6df85f1d825bf834f190d2808350d3dea4a16892c0294c3b4dc633`.

## Findings

1. **Unchanged fit science.** Each isolated worker uses one frozen seed and all seven frozen arms in order. It imports the exact sealed v2 driver and calls `fit` directly with the original OneCycle helper. Inputs, family construction, private HGT implementation, initialization seed rule, fitting, selection, state replay, and complete-cohort paired summary retain their bound v2 source identities. The controller imports no Torch. No new numerical function was qualified in this review.
2. **Exact 35-case custody.** Admission enforces seeds 131, 137, 139, 149, 151 and the native_HGT/global_BE/shared_relation/CP/unrestricted/untied_HGT/wider_BE order. Collection reconciles failed, deferred, missing, malformed, duplicate, wrong-order and partial worker journals into seven slots per seed. Raw evidence stays available. Attempted-arm flags use ARM_STARTED. Closure verifies exact ordered35 slots, replay/binding flags, and the hash/bytes/absolute exact-slot paths of all four selected artifacts. Terminal metadata must match the bound original SELECTION.json. Only complete selected fits, successful workers, and verified preservation can invoke the original paired summary. Canonical STUDY.json otherwise remains incomplete and unscored.
3. **Controller cancellation blocker repaired.** The earlier reproduction retained here escaped KeyboardInterrupt and wrote zero terminal rows after STUDY_STARTED. The sealed source catches KeyboardInterrupt; handles SIGINT/SIGTERM as controlled exceptions; ignores subsequent cancellation during child cleanup; blocks cancellation through child PID registration; restores the child's default signal handlers/mask; terminates/reaps owned children; and recovers exit receipts before reconciling journals. The independent final-source mock interruption after a fabricated arm start returned1 with controller_failed, exactly35 ordered rows, one attempted arm, recovered exit custody, restored handlers and no comparison. The final author's actual SIGTERM synthetic-process fixture was independently rerun and passed with its owned child reaped.
4. **Resource transport custody closed.** All11 scheduler payloads and17 external source records, including the three qualification transport copies, match bound SHA-256 and byte counts. The actual saved REMOTE_CODE is byte-identical to the wrapper's literal. That wrapper applies the 8GiB RLIMIT_AS before Python/Torch import. Successful saved transport metadata records the same cap, and its canonical qualification result reconstructs to the bound SHA-256 `a8776e2687cc7a29b4e25bca63bb50850e1435d58ce9360f9ad7c0f272239c42` (12,656 bytes). All seven resource statuses are qualified and originals preserved.
5. **Five-worker admission is bounded.** Separate one-thread workers receive an 8GiB address-space cap and an explicit positive root wall budget. The controller requires at least40GiB current host/cgroup headroom and acceptable current CPU availability/load. The summed RSS estimate and elapsed-time forecasts are projections from the prior single-process qualification. They establish a resource plan, not an observed concurrent training result or a scientific merit decision.

## Evidence and scope

`FINAL_SOURCE_CHECKS.json` records exact payload/source/freeze checks, qualification transport identity, independent fabricated KeyboardInterrupt recovery, and the selected final-author SIGTERM fixture. `verify_final.py` is the reproducible stdlib-only verifier. `SELECTED_SIGNAL_FIXTURE.txt` records the selected fixture result. The author-sealed SCHEDULER_FIXTURE_RECEIPT reports13 passing final-source checks, source hashes matching this seal, no Torch import and no real labels/tensors.

This audit read source, frozen metadata and resource/transport custody. It ran only fabricated stdlib scheduling tests. It did not read original labels, model checkpoints, logits, validation/test outcomes or execute models, training, GPU work, SSH staging or remote scientific work. It did not repeat the prior model/algebra qualification. The root-adopted development gate and closed TEST policy remain unchanged.
