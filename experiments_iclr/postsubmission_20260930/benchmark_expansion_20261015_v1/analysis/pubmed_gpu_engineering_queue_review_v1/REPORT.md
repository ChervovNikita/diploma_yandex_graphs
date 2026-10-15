# PubMed GPU engineering queue: bounded source review

Reviewed source: `pubmed_gpu_engineering_queue_24.py`
SHA256: `76088335aad3f8d879f1377f51b081f6410150e16b1985a029524212545f0293`
Date:2026-10-15. Source review only; no execution, tests, remote/Git operations or source edits by reviewer.

## Conclusion

No remaining concrete blocker found in the requested queue, cleanup, resource and count scope. This review supports root publication of this engineering queue; it does not admit the24-fit pilot.

## Resolved finding

**P2 — Successful child exit could be treated as a GPU-authentication failure.** The retained `gpu_state()` helper can observe a process through NVIDIA's query after that process has exited from `/proc`. The monitor's earlier `child.poll()` does not prevent that race. Root fixed queue lines80–83: only the exact `GPU process could not be authenticated` ValueError is caught, and only when a fresh child poll is terminal. The queue then processes the ordinary child RESULT; other exceptions still stop the queue. The next child receives a fresh resource preflight. The final source above includes this resolution.

## Checked behavior

- The owner imports stdlib bookkeeping only and waits for the exact existing six-fit terminal gate before taking the existing GPU slot (lines35–53). It does not construct a model or initialize CUDA while waiting.
- The source creates exactly seven serial children: three residual lifecycle qualifiers and four production cost probes (lines54–58). Assuming every child passes its own frozen checks, these comprise10 constructors,14 optimizer transitions,17 whole VALID calls and36 reverse calls including12 lifecycle diagnostic reverses. No24-fit driver is invoked.
- Each launch checks24GiB host headroom after the pinned older peer's host growth allowance. It separately checks32GiB GPU headroom after growth to the older peer's32GiB envelope (lines59–69). It authenticates peer PID/start time, rejects an unadmitted GPU peer and checks the child's24GiB RSS/32GiB GPU limits while running.
- Cleanup targets only the Popen child's process group created with `start_new_session=True` (lines18–22,74). The existing scientific owner and older GPU worker are never signal targets. A watchdog initiates child termination after900seconds, then escalates after10seconds of TERM cleanup. Monitoring calls/polling add their ordinary latency; these are not exact wall-time measurements.
- A nonzero exit, failed/missing child result, timeout, resource violation or other exception stops later invocations without retry. A successful child must exit zero and have `RESULT.status=passed`; the next child is started only after that check (lines90–97).

Retained source inspected only where necessary: `qualification/portablev4_link_queued_parent_authority_repair_v1/supervise.py` (`process`, `gpu_state`, `preflight`) and `pilot_20261015_structural_private_risk_v1/run.py` (imports, terminal gate, slot ownership and final END publication). Current live process identities and actual resource availability were not independently queried; those remain runtime root/queue checks.
