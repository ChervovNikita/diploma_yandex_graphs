Same-reviewer successor delta review — source-v2, 2026-10-04

This is a follow-up to the sealed fresh source-v1 review, not a new independent reviewer. Source-v2 manifest SHA-256: `80789b0bb7e0034a079911034429c6891c0c142f6e352a9ca5adb1ab7e0c7c7b`.

The v1 launch blocker is corrected. The GPU query has `timeout=min(15.0, remaining)`, preflight failure writes a FAILED receipt with `child_launched=False`, and elapsed-budget checks prevent an overdue child launch. Inert execution of the exact preflight AST verified 15-second and remaining-budget timeouts and retained failure receipts.

The original acquired-identity disappearance race is corrected: missing `/proc` identity returns False without signalling, a changed start-time identity still refuses signalling, and the caller can reap its own child with `wait4` while retaining usage.

Two P2 cleanup edges remain in [supervise.py:134](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/pubmed_native_predictive_program_source_20261004_v2/supervise.py:134>):

- If initial `physical(child.pid)` fails before `identity` is assigned, `signal_owned` rejects None before reaping. An exact cleanup-AST probe wrote CLEANUP_FAILURE and SUPERVISOR_FAILURE with no kernel usage and no child exit code.
- If the second signal or bounded reap raises inside the `except TimeoutExpired` handler, that exception escapes the sibling cleanup handler. An exact cleanup-AST probe reproduced a second TimeoutExpired with neither failure receipt written.

Guard signalling and reaping independently, and catch escalation/reaping exceptions before writing the failure receipts. These findings concern failure custody/accounting; neither permits an unrelated process to be signalled or introduces an identified native fit/replay algorithm blocker.

`common.py`, `fit_cohort.py`, `replay_selected.py`, `COHORT.json`, `SOURCE_BINDING.json` and the release template are byte-identical to v1. The fresh review's policy, selector, state/replay assessment and empirical limitations therefore carry forward. No candidate imports or edits, scientific fits, server access, checkpoints, real subprocesses or real signals were used in these probes. Runtime cap feasibility and selected scientific replay remain unobserved.
