# Independent execution and analysis source audit

Reviewed packet: `shared_private_transfer_paired_pilot_preparation_20261005_v1`.

The root supplied immutable manifest SHA-256 `e609add68a00d45f5c580d3da50e0fd99d3294406abc32a2e355a7ea3f5879c4`. It matches the local manifest. All 11 listed packet files match their manifest hashes and sizes. No separate packet SEAL file is present; the supplied manifest identity is the audit binding. `INPUT_VERIFICATION.json` preserves the checks. This audit concerns source and protocol, not a scientific result or manuscript verdict.

## Material findings

### E1 — A rejected initial identity can still authorize a later signal

**Repair before singleton execution.** `run_queue.py:56` assigns the raw identity to `owner`. The check at `run_queue.py:58–61` rejects an initial argv/session/process-group/cwd mismatch or observation error, but does not revoke that owner. `run_queue.py:103` then passes it to the pinned helper's `observe_terminal`.

The pinned `shared_backbone_private_transfer_complete_cost_queue_preparation_20261005_v3/queue.py:122–135` invokes `kill_owned` at the hard deadline. Its `queue.py:109–117` compares the later live identity with the recorded owner, including the recorded argv, rather than the intended argv and admitted cwd. A persistent initial argv/cwd mismatch can therefore receive SIGKILL after the pilot explicitly recorded “no signal authorized.” This is a control-flow defect in the new wrapper; the helper's pinned hash is correct.

Keep the raw observation for evidence and set a signal-authorized owner only after every initial check succeeds. Pass only that admitted owner to resource/signal/terminal helpers. Rejected identity must retain authoritative Popen termination observation without acquiring signal authority merely because its rejected observation remains stable.

### E2 — The declared mixed-provider family has no authenticated assembly path

**Repair the full-family completion and analysis path before comparative analysis.** The packet prospectively assigns b0 to the singleton and b1/b2 to 77 (`STUDY_SPEC.json:151–183`; `README.md:35`). Its only complete-family producer is `run_queue.py:167–172`, which writes `COHORT_FREEZE.json` only when one queue completes 30 fits. Default b0 produces a ten-fit `BLOCK_FREEZE.json`.

`analyze_pilot.py:52–58` requires one `COHORT_FREEZE.json`; `analyze_pilot.py:74–75` looks up every job under the same execution root; and `analyze_pilot.py:72` requires the singleton training-source manifest for every fit. No collector in this sealed packet authenticates separate whole-block freezes, provider/source receipts, jobs and outputs into this analysis admission. Merely joining the three `completed` lists would omit the necessary custody checks.

Freeze a metadata-only collection contract that binds the common prospective plan and anchors, exact disjoint complete block cell sets, each block queue/release, each job and FREEZE, and each authoritative terminal receipt. Require terminal wait/poll authority, observed exit zero, no failure reason or signal, one attempt and no retry. Preserve the correct provider/source/runtime evidence per fit and reject incomplete or duplicate families. Analysis must follow those authenticated job/output references and provider source identities before loading logits.

The qualified singleton `custody.py:70–75` correctly rejects 77. The separately reviewed 77 portability source and exact-provider numerical qualification remain a distinct admission dependency. This audit does not recommend loosening singleton custody. E2 does not itself prevent a repaired, approved whole b0 block from launching; it prevents this v1 packet from being a complete mixed-provider execution-to-analysis packet.

## Contracts confirmed by inspection

- The generator creates all 30 scientific cells before returning, at 60 maximum complete cycles, full VALID every five cycles, eleven misses, first maximum rounded-four-decimal selection, seeds 0/1/2 and the fixed per-block factor seeds (`freeze_queue.py:49–52,84–120`; `STUDY_SPEC.json:4–41,147,190,194`). Only selected whole blocks are executable; other generated jobs remain disabled (`freeze_queue.py:103–105`).
- Default execution contains the complete b0 block. Singleton b1/b2 execution needs explicit whole-block fallback evidence (`freeze_queue.py:53–64`). Cell hard bounds, finite resource windows and the queue overhead are frozen (`freeze_queue.py:65–81`). The queue rejects changed paths/order, existing outputs and implicit resumption, and stops on operational failure (`run_queue.py:125–159`).
- Qualified training manifest `db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f` and all 13 listed training files match. `run.py` is `6d7e75f9bae93ef88b2873f55f4f449ae52b9a0b6768fa808fae391da50ed524`. The helper matches `7b195b600ad3cc57c4cf51a1da927164173ca2b0e80e96cf2b6c6449ce7ad468`.
- The local raw gate copy `shared_backbone_private_transfer_fp32_execution_root_20261005_v1/RESULT.json` matches `8601b1138c3f7ab65cb1525d5d8274a8e14e22731cd060801eec437f6169806d`. The packet's remote reference includes `/result/RESULT.json`; that is a retrieval-layout distinction, not a mismatched raw gate. No host was contacted.
- Qualified training enforces exact cell seed/geometry/rule/episode-size and schedule identity (`run.py:64–85`), authenticated source/authority/runtime/input contracts (`custody.py:61–109,140–213`), paired cycle/route construction and common positive-target union masking (`custody.py:230–263`; `run.py:107–108,127–129,171–172`). This audit did not reopen numerical qualification.
- Apart from E1, terminal status comes from the direct Popen wait/poll rather than filesystem disappearance (`run_queue.py:103,108–115`; pinned helper `queue.py:122–144`). Resource helpers traverse owned PID descendants and filter GPU process telemetry to those PIDs (`queue.py:76–102,166–185`). The wrapper traverses only its new fit output and own logs, refuses symlinks and applies declared caps (`run_queue.py:11–23,77–85,104–107`).
- All nine original external raw MRR values equal the authenticated original compact `STRATIFIED_MRR.json` summary, whose hash is `3b17afba59d414d6a9fd3f80d52c18b8b36096fd5ecc691f380abc5f7913ff6d`. They are immutable copied inputs; analysis directly copies them and does not load original predictions/checkpoints or reselect them (`analyze_pilot.py:96–99`). The packet explicitly distinguishes original native-training anchors, new episodic joint controls and the jointly trained native-four control.
- New selected logits and checkpoints are hash-bound and checked against the selected cycle and rounded selector (`analyze_pilot.py:76–87`). All fixed contrasts and three blocks are retained. Quality and mechanism flags are separate; horizon sufficiency is separately reported without claiming recent improvement, novelty, confirmed superiority or acceptance (`analyze_pilot.py:9–29,89–121`).
- Training input roles exclude TEST, and retry is rejected (`custody.py:104–105,161–171`). The analysis release also requires closed TEST and no fits/original-score recalculation (`analyze_pilot.py:48–49`).

## Audit scope and disposition

Two material source/protocol findings are recorded. E1 blocks the v1 singleton wrapper's launch authorization; E2 blocks full mixed-provider comparative analysis. I found no additional material blocker in the reviewed fixed scientific family, immutable anchors, seed/support contracts or exploratory analysis definitions.

Only relevant local sources, protocol metadata, hashes and the original score summary were read. No prepared source module was imported; no model, tensor, statistical calculation, fit, server operation, TEST read, paper read, PDF compilation or canonical edit was performed. Static custody checks are not runtime or scientific qualification. The sealed v1 packet is preserved; any repair should be reviewed as a successor.
