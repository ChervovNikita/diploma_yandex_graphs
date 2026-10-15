# Portable dispatcher v3 — complete runtime-path schema repair

This coherent source successor preserves published portable_dispatcher_v2 and
the partial v2/generated_v3_config_pin metadata from the second failed attempt.
It remains unadmitted. Both77 configurations remain disabled. No native job,
dataset/tensor read,77 connection or Git operation occurred in this repair.

## Complete actual-inventory audit

`FULL_SCHEMA_AUDIT.json` records one read-only scan of all6460 actual compiler4
JSON descriptors on the authorized allocation, after verifying its hostname
and sole GPU UUID. Every descriptor file SHA, canonical job hash and scientific
contract hash verified. It took12.08 seconds with44712KiB peakRSS and wrote no
projection. `AUDIT_TRANSPORT.json` retains the transport receipt.

The complete scan found exactly one unclassified runtime pointer:
`/recipe/_native_callback`, in700 descriptors. There were zero runtime path
normalization or confinement violations. Examples and every observed absolute
runtime/provenance pointer are retained in the audit. Source inspection confirms
this field names the inner native callback, paired with its existing source
hash; it is not immutable historical provenance.

## Narrow, coherent fix

- The shared `descriptor_binding.py` adds only that exact typed pointer.
  Unclassified fields still reject. Native callback source hashes, scientific
  settings and provenance subtrees remain unchanged.
- The metadata preparer and actual native runtime guard both use this same
  v3 classifier. The runtime files are byte-identical to v2; their shared
  host guard resolves the v3 package. The backend's fixed launcher path also
  resolves v3. No v2-runtime/v3-preparer mixture is used.
- Each host configuration metadata read supplies its own manifest hash,
  preserving the previous config-loop repair without changing execution guards.
- Every canonical descriptor is audited and rebound in memory before the
  first projection directory/file write. All unclassified pointers and any
  binding errors are collected across the full inventory. A failed audit
  blocks output; successful output retains `PREWRITE_DESCRIPTOR_AUDIT.json`.
- Fresh output is `portable_dispatcher_v3/generated_v3`. The generated source
  closure contains the complete v3 source and manifest. Canonical campaign
  ownership/claim/output identity remains the existing stable v2 identity;
  no scientific tasks or claimed states are reassigned.

Root's guarded metadata command, from the allocation repository, is:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/portable_dispatcher_v3/prepare_projection.py
```

This successor projection has not been executed. Reporting must explicitly
accept the v3 projection directory and bind v3 native entrypoint paths; an old
v2 reporting registry is not execution authority.

## Reused verification and execution limits

`CHECKS.json` records seven narrow schema/config/source checks. All16 native
runtime modules and eight core dispatcher/ledger/ownership modules are
byte-identical to v2. Its30 metadata/source checks and prior durable-engine
evidence are reused; broad worker/scientific tests were not rerun. `DELTA.diff.txt`
is the complete source delta against v2.

The existing one-command interface remains:

```sh
nohup .venv/bin/python -B -u ABSOLUTE_V3_PACKAGE/run_gpu.py --worker WORKER --run-spec ABSOLUTE_RUN_SPEC --run-spec-sha256 REVIEWED_SHA256 >> ABSOLUTE_WORKER_LOG 2>&1 < /dev/null &
```

Actual runtime/resource/source/identity qualification and explicit campaign GO
remain required.77 additionally needs explicit user reauthorization before any
access. Training has no online coordinator/heartbeat dependency. Global STOP
requires per-host delivery; undelivered STOP cannot affect a disconnected host.
Received acknowledgments do not claim child drain completion. The full ledger,
failure policy, STOP semantics and remaining real-host gates are inherited from
the preserved v2 handoff.
