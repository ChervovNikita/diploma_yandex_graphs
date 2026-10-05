# Native PENCIL Citeseer resource execution

This packet owns one complete native epoch-0 TRAIN forward/backward traversal and complete VALID traversal on the authorized one-GPU host. The reviewed probe source is unchanged. It performs zero optimizer updates, computes no ranking metrics, saves no scores or checkpoints, and never opens TEST. It does not open NCN comparative outcomes.

The complete 36-fit Citeseer cohort freeze is required before child dispatch. A final hostname, singleton GPU UUID and free-memory check is performed immediately before child launch. Dispatch requires at least 34 GiB free GPU memory. Resource ceilings are 24 GiB allocated CUDA memory, 28 GiB reserved CUDA memory, 128 GiB summed probe/loader RSS, and 3,600 seconds from child launch. Existing Amazon and other jobs are preserved. Co-resident elapsed time cannot establish uncontended efficiency.

`supervisor.py` starts a dedicated child session and preserves its PID/start-time identities. Its timeout and RSS watchdog may signal only members of that directly owned session. The unchanged probe enforces its own CUDA allocation and reservation ceilings at native forward and gradient boundaries. Failure, timeout, partial coverage and every owned signal remain in the receipts. There is no automatic probe retry.

## Staging failure

The first transport created only an empty execution directory. The bounded-plan directory had not been staged remotely, so dependency-binding writing could not complete. Read-only queries confirmed the empty directory and absence of binding, release, supervisor and launch receipt. No probe or constructor child started. `PRELAUNCH_STAGING_FAILURE.json` preserves that failure. A single staging-only continuation creates the missing repository-owned plan directory and then launches the still-unstarted, single authorized probe through the already qualified interpreter.

## Evidence placement

The full dependency file binding (7,620 admitted files) remains on the server. Its exact digest is in `LAUNCH_RECEIPT.json` and the approved release. The small local packet contains supervision source, transport records and copied resource receipts. The package inventory is not duplicated on the Mac.

The authoritative runtime output is the server packet's `run01` child directory. `monitor.py` reads only this execution's receipts and log tails; it does not read scientific experiment outcomes.

## Limit of a successful result

A zero-update resource PASS verifies complete coverage and finite native outputs/gradients with unchanged parameters. It does not establish predictive quality or full-fit readiness. Adam moments, optimizer-step temporaries and later-epoch costs are absent. The separately authorized scientific fit must check actual Adam allocation and finite state at its first ordinary update, retain resource bounds thereafter, and preserve a later failure if one occurs. This packet authorizes no scientific fit.
