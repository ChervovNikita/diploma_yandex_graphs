Same-reviewer successor delta review — source-v3, 2026-10-04

This is a narrow follow-up to the sealed fresh v1 review and v2 delta review, not a new independent reviewer. Source-v3 manifest SHA-256: `cd360a31fc6feead174f1da3ea9478655aa3bd78f68aab548601cc9fb37090e5`.

No remaining launch blocker was identified in the reviewed v3 source. The bounded preflight and elapsed-budget guards introduced in v2 remain present. The v3 cleanup change closes both reported v2 P2 failure-accounting edges.

At [supervise.py:134](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pubmed_native_predictive_program_source_20261004_v3/supervise.py:134>), TERM is attempted only after identity was acquired. Reaping the owned Popen child proceeds independently of that signalling attempt. KILL and the second wait4 reap have separate exception guards, and collected errors are written before the main failure receipt. A missing or changed physical identity continues to prohibit signalling.

Exact cleanup-AST execution with inert stubs verified:

- Missing initial identity: no signal attempt; wait4 still reaps, and SUPERVISOR_FAILURE retains the child exit code and kernel RSS/CPU usage.
- Refused signal after start-time mismatch: zero real or stub signal calls; cleanup still reaps and records usage plus the guarded signalling error.
- Both bounded reaps time out: CLEANUP_FAILURE records the second-reap error; SUPERVISOR_FAILURE is still written with usage explicitly unavailable, and the original failure is re-raised. No COMPLETE classification is produced by this failure path.

The acquired-identity disappearance helper returns False without signalling; the reused-PID case refuses signalling. The exact preflight AST retains FAILED receipts for simulated timeouts and uses 15 seconds or the smaller remaining wall budget. No real subprocess, GPU query, process signal or scientific fit was executed in these probes.

Hashes confirm `common.py`, `fit_cohort.py`, `replay_selected.py`, `COHORT.json`, `SOURCE_BINDING.json` and `ROOT_RELEASE_TEMPLATE.json` are byte-identical to v1. The fresh review's native policy, first rounded VALID maximum, state capture and selected replay assessment therefore carry forward without requiring additional scientific parity tests for this supervisor-only correction.

What can run: after root completion of the already required exact release/prerequisite fields, the fresh six-fit TRAIN/VALID cohort can launch through this supervisor with the existing fixed caps and no retry/resume. Replay requires its separate release after fit/supervisor COMPLETE and owned selected-artifact custody. Direct driver execution does not enforce supervisor host RSS/wall caps.

Limits remain: no empirical scientific cadence/seed1/seed2 parity, six-fit cap feasibility, or selected scientific replay success was established here. A child that cannot be reaped within both cleanup waits is recorded as failure with kernel usage unavailable; this receipt is not proof that the process physically exited. The terminal receipt-write tail remains explicitly unmeasured. This source review does not approve a scientific conclusion or TEST release. Candidate code was not imported or edited, servers and checkpoints were not accessed, and earlier external review/outcome payloads were not read.
