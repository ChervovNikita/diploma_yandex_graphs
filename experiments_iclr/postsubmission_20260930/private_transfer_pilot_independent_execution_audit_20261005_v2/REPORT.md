# Independent v2 execution follow-up audit

Reviewed `shared_private_transfer_paired_pilot_preparation_20261005_v2` against supplied immutable manifest SHA-256 `1b52c20c4228cfb34a7667de056333a60c3e30e766a96a2949b3659382a63579`. The manifest and all 16 listed file hashes/sizes match. No separate packet SEAL file exists; the supplied manifest identity binds this review.

The separately reviewed final staging client is `shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/stage_and_launch.py`, SHA-256 `ad9b71832e3da8891bec7720547a9bf50dc87b55bc3d207f4fb8b914522c892a`. `INPUT_VERIFICATION.json` preserves both identities and the static checks.

## E1 resolution: initial signal admission

Resolved in the reviewed source. `run_queue.py:50` initializes separate `owner` and `raw_owner` values. `run_queue.py:56–60` preserves the raw observation and assigns signal-authorized ownership only after argv, process group, session and observed cwd all match. The observation exception path leaves `owner` unset. Raw evidence and admission status are retained at `run_queue.py:63–68,111–118`.

The resource/signal path uses that admitted owner; the exception handler explicitly guards an unset owner (`run_queue.py:75,94,102`). Terminal observation passes the admitted owner at `run_queue.py:106`. The unchanged pinned helper checks for an unset owner before deadline signaling (`shared_backbone_private_transfer_complete_cost_queue_preparation_20261005_v3/queue.py:132`). A rejected initial raw observation therefore cannot regain signal authority merely by remaining stable. Authoritative exit status still comes from Popen wait/poll.

## E2 resolution: prospective full-family collection and analysis

The missing engine and protocol are now present. `README.md:48–60` specifies provider admission and exact queue/release registration before each block's first fit, retention of every registered donor, byte-identical full plan/anchors, complete-family metadata collection, and a separate analysis release afterward.

`collect_cohort.py:30–60` binds the actual provider source, unchanged numerical files/dependencies, qualification/review evidence, runtime and authorized checkout. `collect_cohort.py:90–99,118–136` authenticates the prospective science, plan, anchors and all three exact whole-block populations in fixed order. `collect_cohort.py:138–176` checks disjoint fit IDs, authoritative observed Popen zero exits, absence of failure/signals/retry, owned session/argv/cwd, exact jobs and selected-state freezes. `collect_cohort.py:177–198` retains common input identities and provider-specific artifact provenance and requires all 30 distinct fits. The collector checks selected artifact existence without opening tensor content and emits `COLLECTION_FREEZE.json` only after the complete family passes (`collect_cohort.py:180–210`).

Analysis now requires that collection, retains each actual provider's source/runtime, follows authenticated per-fit job/output paths and checks selected checkpoint/logit hashes before scoring (`analyze_pilot.py:52–85,114–116`). It no longer treats all providers as the singleton source or assumes every job belongs beneath one original execution root.

These changes address E2 without changing qualified singleton training. Actual 77 admission and qualification remain separate prerequisites before b1/b2 fits. Collection and analysis must use enabled, hash-bound releases and the actual completed artifacts; this static review does not claim those future operations have occurred.

## Final staging client

No concrete execution error was found in the reviewed client. Its local payload checks exact source bytes and syntax, packages existing matching cost/numeric receipts, and binds root review plus this independent review (`stage_and_launch.py:91–110,125–140`). The embedded remote code checks the authorized singleton host/GPU, refuses differing existing staged bytes and existing execution/launch state, and runs the metadata generator before launch (`stage_and_launch.py:22–58`).

Provider admission is created before transport (`stage_and_launch.py:112–124`). Donor registration binds the exact frozen queue, root release and provider admission before queue Popen (`stage_and_launch.py:59–66`). The detached queue launch records PID/start ticks/argv/group/session/cwd and source/release/plan hashes (`stage_and_launch.py:67–78`). Returned canonical plan and registration bytes are hash-checked before local preservation (`stage_and_launch.py:150–158`). The fixed launch remains b0 only and creates all 30 prospective scientific jobs first. Root release templates remain disabled; the staging client's enabled copies are gated by root review and its explicit invocation.

## Unchanged science and disposition

The spec, anchors, base job and common custody helper are byte-identical to v1. The new science contract matches every original cell, block seed, factor seed, episode size, fixed order and 60-cycle/every-five/eleven-miss schedule. All numerical science hashes, all 13 qualified training-source files, the qualified training manifest, raw FP32 gate and pinned process-helper hash still match. The staging cost/numeric inputs exist and match their declared hashes. All inspected Python sources and the embedded remote script parse successfully; no prepared source was executed.

Both v1 material findings are resolved at the reviewed source/protocol level. No further material b0 launch blocker or concrete staging execution error was identified. Full-family comparative analysis remains gated on all 30 authenticated completions and separate provider/collection/analysis releases. No scientific or manuscript acceptance verdict is given.

Only relevant local source and metadata were inspected. No server action, source-science change, model/tensor/statistical execution, TEST read, paper read, PDF compilation or canonical edit was performed. Verification outputs and this report are confined to this fresh v2 audit directory; v1 remains preserved.
