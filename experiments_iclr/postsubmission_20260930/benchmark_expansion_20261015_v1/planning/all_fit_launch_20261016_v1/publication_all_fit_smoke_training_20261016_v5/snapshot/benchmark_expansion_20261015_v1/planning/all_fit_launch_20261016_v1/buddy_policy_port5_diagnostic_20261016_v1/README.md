# One-cell port5 repair diagnostic

This checks the exact seed-2 BUDDY private-half-pool configuration that failed
before any update because helper metadata expected `policy_receipt`. Its original
descriptor bytes, seed, split, policy, data, optimizer and native training code
stay fixed. The actual outer native runtime is the pinned port5 composed runtime.
Original failure evidence remains intact. The fresh one-row inventory records
the actual new producer and is explicitly ineligible for global smoke coverage.

Root first gracefully stops the ordinary GPU1 owner and waits for its children.
This diagnostic never signals processes or changes the ordinary queue. It takes
the exact physical, worker and slot leases nonblockingly, uses the original
host-wide resource admission and guardian, runs at most two optimizer updates,
checks complete VALID and exact model/optimizer reload, and retains actual costs.
The child wall limit is 1,800 seconds. No TEST, scientific score, global marker,
SMOKE_SUMMARY or full-training phase is requested. A failed run is retained and
cannot be silently retried in the same `run` directory.

After staging all pinned sources and the frozen original-cell metadata, use a
root-owned tmux session with the original port4 GPU1 host configuration:

```sh
"$PROJECT/.venv/bin/python" -B -u "$DIAGNOSTIC/run_diagnostic.py" \
  --parent-plan "$EXISTING_EXECUTION5_GPU1_PLAN" \
  --parent-plan-sha256 "$UNCHANGED_PLAN_SHA" \
  --source-sha256 "$DIAGNOSTIC_MANIFEST_SHA"
```

The parent helper authenticates the old plan/config and retains custody. The
native invocation explicitly passes the new port5 host config in its environment,
so its host guard and source identity are those of the actual producer. Outputs
are confined to this package's fresh `run` directory. Root resumes the ordinary
queue after this bounded diagnostic ends. Source preparation launches nothing.
