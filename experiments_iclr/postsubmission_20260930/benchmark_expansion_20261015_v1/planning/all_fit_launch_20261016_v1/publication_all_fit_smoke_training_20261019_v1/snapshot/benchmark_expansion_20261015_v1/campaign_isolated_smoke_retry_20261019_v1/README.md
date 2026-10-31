# Isolated server51/gpu0 CUDA OOM retry

This local source package permits one explicit second smoke attempt for
`polyformer-coauthor-physics-mono/long_convergence/split-9/seed-9/wrapper-structural_class_quartile_permuted_private_v1`.
It implements the user's authorization to fix technical failures while retaining
the frozen scientific design and original failed evidence.

The authorization fixes server51/gpu0, the original plan hash, original attempt
and native receipt hashes, inventory, port6 and smoke6 sources. The original OOM
occurred before any optimizer update. Every other assigned smoke must already
be PASS. The inherited worker still obtains the original supervisor, slot,
physical GPU and worker leases and applies its original live capacity checks.

`load_states` first calls the original implementation. After verifying exact
original custody, it copies only the authorized attempt-1 failed state and marks
that copy pending in memory. Disk `STATE.json` remains failed until inherited
`start` launches attempt `0002` and writes running state. Successful and failed
attempt evidence stays intact. A PASS at attempt 2 stays PASS; a failure at
attempt 2 stays failed and cannot receive another smoke retry, including when
`--retry-failed` is supplied. That flag retains its original full-phase meaning.

The wrapper inherits `run`, `run_phase`, `start`, native validation, collection,
summary generation and the exact 6,460-fit global barrier without changes. Full
commands and full recovery remain unchanged. The two-update smoke command,
1,800-second smoke wall policy, native sources, descriptors, optimizer settings,
scientific selection rules and TEST prohibition remain unchanged. No source
or original execution manifest is rewritten.

Before the inherited constructor replaces `OWNER.json`, the wrapper preserves
its exact previous bytes in a content-addressed
`OWNER-before-isolated-smoke-retry-<sha256>.json` file while holding the original
supervisor lease. OWNER, STATUS and new INVOCATION records identify the actual
wrapper manifest and explicit authorization alongside the original parent
manifest. The retry invocation also pins the original failed receipts.

## Deployment command

Use a fresh interpreter inside the authorized server51/gpu0 tmux owner. The
deployment operator must supply the frozen package manifest hash and
authorization hash returned by the local freeze; the wrapper checks both.
`RETRY_PACKAGE`, `ORIGINAL_PLAN`, `PROJECT_PYTHON`, `WRAPPER_SHA256` and
`AUTHORIZATION_SHA256` below are deployment placeholders.

```sh
"$PROJECT_PYTHON" -B -u "$RETRY_PACKAGE/retry_worker.py" \
  --worker server51/gpu0 \
  --plan "$ORIGINAL_PLAN" \
  --plan-sha256 b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345 \
  --source-sha256 "$WRAPPER_SHA256" \
  --authorization "$RETRY_PACKAGE/AUTHORIZATION.json" \
  --authorization-sha256 "$AUTHORIZATION_SHA256" \
  --poll-seconds 10 --smoke-wall-seconds 1800
```

This package was prepared and verified locally. No native fit or remote launch
was performed during preparation. It does not incorporate the separate memory
admission wrapper; original execution7 admission applies.

## Engineering verification

Run `python3 -B test_retry.py` from this directory. The standard-library fixtures
exercise original state loading, PASS byte preservation, original OOM receipt
checks, wrong hash/ID refusals, isolated coverage, attempt-2 scheduling with a
mock child, no attempt-3 retry, full-phase compatibility, source verification,
and exact OWNER archival under the supervisor lease. Native execution and GPU
access are mocked or absent. `SOURCE_CHECKS.json` records the local outcome.
