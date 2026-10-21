# Launch safeguards and readable coordinator review — 21 October 2026

Bounded local source review only: no remote call, training, source mutation or new test run. The root reports the user's later authorization to start training as GPUs become available; that supersedes the defer language in the earlier REVIEW.md. This addendum evaluates the new implementation and does not itself activate anything.

**No new concrete correctness blocker found in the reviewed paths.** The source is suitable for the root's remaining freeze/deployment verification and explicit activation sequence. Successful smoke records remain attached to the original scientific descriptors, port6, smoke6, execution7 plans and state roots.

- `campaign_launch_safeguards_20261021_v1/queue_worker.py` closes the I4 defect from REVIEW.md: it confines the receipt to the resolved own `output/I4_selected_VALID.pt` and verifies actual size/hash. Both collection and existing-success reload call this override. Failed validation preserves historical attempt evidence and does not silently rerun it.
- The wrapper adds `20 GiB + candidate disk reservation + all active host disk reservations` under the inherited ledger lock. GPU, RAM, CPU, profile pairing and remaining-growth decisions are retained. It does not pretend to enforce a lifetime retained-data cap, transfer allowance or checkpoint-boundary check. Those limitations remain explicit.
- Full/smoke state loading, exact all-fit barrier, native argv, smoke/full separation and child ownership are inherited. OWNER history is preserved before replacement after the supervisor lock is acquired. The wrapper records its own producer while retaining the original execution/source pins. No failed-fit retry switch is exposed.
- The coordinator defaults to `verify`; the backend defaults to `probe`. Verification fetches only JSON metadata, runs the frozen exact-6,460 validator locally, writes a separately named local candidate, and rechecks remote metadata. It makes no remote marker write. `stage` only installs the named backend/config files. The canonical barrier write is confined to explicit `release`, which requires the candidate SHA256, checks the proof again, uses execution7's actual barrier checker and refuses a conflicting existing marker.
- Release is host-by-host, not a two-host transaction. A transport failure after the first marker can leave that host released and the second waiting. Exact existing marker bytes are accepted idempotently and receipts are retained; root must inspect delivery state before a retry. This is compatible with the authorized start-as-available behavior.
- Plaintext route review follows `readable_transport.py` through local `reverse_maclink/maclink.py`: the `run` command sends a JSON argument vector, relays it unchanged and calls `asyncio.create_subprocess_exec`. Allocation uses the literal authorized `-tt` SSH destination; server51 uses MacLink plus the existing SSH control socket. Hostname/GPU guards precede project operations. Source staging uses ordinary SCP or visible quoted heredocs, received SHA256 checks and text readback. No encoded executable payload was found in this path. This is a local source trace, not independent certification of the currently running remote MacLink agent.

The root's report of server51 GPU1 being externally busy is a capacity wait, not a failed fit or reason to change assignment. Its inherited free-VRAM test should prevent admission while only about 198 MiB is free; work can remain owned in tmux until capacity returns.

Existing safeguards SOURCE_CHECKS reports eight passing local fixture tests, including actual missing/changed I4 files, restart acceptance, disk boundaries, no-child low-disk handling and PASS reuse. Coordinator tests were inspected and cover default no-release, all-6,460 validation, stale evidence refusal, explicit confirmation and idempotent canonical marker handling. They were not rerun here. At this review the coordinator had not yet acquired MANIFEST.json/SOURCE_CHECKS.json; final publication must bind the exact reviewed source or review any subsequent delta.

Reviewed identities:

| File | SHA256 |
| --- | --- |
| safeguards MANIFEST.json | `adf6710ccf68808d21d8f4aeb6a67b665f17b9ee06de18051886caf11fd7bfdc` |
| safeguards queue_worker.py | `daeace816c98a9c11c1f44ee97e731678a6267ada50b18d978ff9605b79fd555` |
| coordinator coordinator.py | `60d9824a63bae0af7c0980b527940c3837a51ba1d3fb9a2b1734a33ee24f32f5` |
| coordinator remote_backend.py | `0343060a2bda59ef9084cf08bb47602104450e9448918fc23786c9e4139598cd` |
| coordinator readable_transport.py | `73e7a204fbabdb1352bfab4718d6f4d873ad6b53ca69bd886a8bfe021f09cf7c` |
| coordinator BINDINGS.json | `b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96` |
