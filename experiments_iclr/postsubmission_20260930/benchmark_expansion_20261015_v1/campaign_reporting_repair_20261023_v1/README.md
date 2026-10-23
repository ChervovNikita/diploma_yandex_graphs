# Reporting recovery, 23 October 2026

The allocation queue stopped on 21 October at 18:19 UTC after two BUDDY
aggregation wrapper runs completed 149 training epochs and then failed in the
scalar reporting observer. The observer rejected the legitimate, SHA-pinned
`link_expansion_ports_BUDDY_production_v4/README.md` dependency because its file
extension was neither `.py` nor `.json`. The native writer had already saved
`selected_VALID.pt`. Four native fits and their I4 aggregation had completed.

This additive package leaves all frozen producer files, descriptors, model
operators, settings, source manifests, and the global smoke release unchanged.
`reporting_patch.install()` permits only the exact README path and frozen SHA,
retains every ordinary dependency hash check, and adds an explicit reporting
repair producer receipt. Install the same patch before using the frozen
collector to interpret repaired reporting sidecars.

`recovery_worker.py` inherits the original named queue with its admission,
ownership, two-job limit and process titles. It explicitly resumes only the two
audited failed attempts listed in the immutable recovery config. The native
runner restores their stopped recovery checkpoints and repeats finalization.
Acceptance requires zero new training epochs, zero optimizer transitions,
identical cumulative optimizer counts and unchanged selected epochs. Any later
failure still stops the queue for investigation; automatic retries are disabled.

Before finalization, the launcher preserves the previous export, result and
training records in the new attempt directory. The native runner separately
preserves the prior result in its history. Previous failed attempt receipts
remain intact. Successful earlier fits and all passing smoke tests are reused.

`activate_allocation.py` verifies the original plan and release, observes a free
supervisor lease, preserves the prior owner/status, and resumes only the existing
dead tmux pane. It never signals an existing process. This activation applies
only to the allocation; it does not change server51.

Validation: 15 focused CPU tests cover exact README allowance and rejection of
changed/unrelated/escaping dependencies, installer identity, frozen native argv,
exact failed-attempt authorization and zero-update recovery acceptance. A live
CPU audit inspected both existing checkpoints and complete finite VALID exports.
No TEST data were opened and no new scientific result is claimed by this repair.
