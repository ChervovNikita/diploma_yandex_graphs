# Continue pending smokes on an unchanged inventory

This separate scheduler uses the exact frozen execution5 plan, inventory, state
directory and helper code. Existing PASS attempts are authenticated and reused
by the original loader. Existing failed attempts remain failed and are never
retried. A newly failed fit does not stop other pending fit IDs. Native commands,
resources, two-child limits, physical/worker/attempt leases, guardian accounting,
STOP handling, VALID/reload checks and scientific descriptors are unchanged.

The actual outer scheduler source is recorded alongside the unchanged parent
helper in OWNER, STATUS, a retained continuation-owner record and each new
INVOCATION. Prior owner bytes are archived by their SHA before replacement.
Old PASS evidence is not relabelled or edited. No source is patched at runtime.

This entrypoint runs in an existing root-controlled tmux session. Root stages
the manifest-listed files, then substitutes actual absolute paths and hashes:

```sh
"$PROJECT/.venv/bin/python" -B -u "$CONTINUATION/continue_smokes.py" \
  --worker server51/gpu0 --plan "$EXISTING_EXECUTION5_PLAN" \
  --plan-sha256 "$UNCHANGED_PLAN_SHA" --source-sha256 "$CONTINUATION_MANIFEST_SHA"
```

Use the corresponding unchanged plan for GPU1. It refuses a live supervisor;
physical leases remain held by already-running children. Root owns tmux starts,
stops and any replacement of the ordinary owner. This package launches nothing
merely by being imported or staged. It does not expose a failed-fit retry flag.

Any failed or missing ID prevents the original SMOKE_SUMMARY. The wrapper never
creates GLOBAL_SMOKE_PASS and has no full-training phase, even if every assigned
smoke passes. After a genuine all-PASS, root can restart the original owner to
consume the unchanged summary/global barrier and perform its ordinary full phase.

This does not make an old PASS applicable to a newly projected port5 inventory:
native receipts bind their original inventory and runtime hashes. A cross-port
applicability decision needs a separate explicit provenance argument. This
wrapper neither implements that bridge nor changes the frozen campaign.

Each unique pending ID gets at most its first attempt; confirmed failed IDs are
skipped. No inferred failure-family classifier or speculative hold group is
implemented. Thus other seeds sharing a defect can each fail once until root
chooses to stop or supplies a separately reviewed scheduling hold policy.
