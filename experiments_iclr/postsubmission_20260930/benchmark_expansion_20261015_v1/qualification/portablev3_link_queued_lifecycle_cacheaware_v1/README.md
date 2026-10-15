# Pending link lifecycle checks with a cache-aware host estimate

This successor changes only the prelaunch host-memory estimate in `supervise.py`.
The previous estimate counted inactive file pages alone. On this allocation,
roughly 225 GB of active file cache is also available for ordinary kernel
reclaim, while anonymous memory is about 1.4 GB. The new estimate includes 75%
of estimated clean active/inactive file cache after conservative exclusions.
No cache is cleared; host settings are unchanged. Peer growth, per-case RAM,
GPU and disk reservations, timeouts and the two-worker maximum are unchanged.
`CACHE_POLICY.json` gives the formula and its limits.

Both JOBs, native code, training loop, validation/restore work and source runtime
are byte-identical to `portablev3_link_queued_lifecycle_v1`. This is the same
unexecuted NCN34-update and BUDDY3170-update queue, not additional experiment
arms. The old package/checks/failed attempts are preserved. No execution or
publication follows from this source preparation. Root must supply current
resource evidence and the new manifest before launching each case.
