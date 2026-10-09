# Wiki15 V2 retention delta assessment

10 October 2026. Source-only review of the concrete WG1 repair and existing
supervisor command in `Wiki15_selected_member_geometry_descriptive_source_20261010_v2`.
No re-audit of the unchanged scientific formulas and no numerical execution.

## Conclusion

WG1 is addressed at source level for the fixed fresh execution directory and
bound existing supervisor. No new blocking source defect was found in this
delta. Actual publication, runtime costs, process cleanup and numerical success
remain unverified until execution. The disabled template grants no execution.

## Durable receipts

- `aggregate.py:21–38,126–145` creates an initial source/config-bound progress
  snapshot before NumPy/Torch imports. It records zero completed cells and worker
  wall/user CPU/system CPU/peak RSS through publication. File flush/fsync precede
  atomic rename; the parent directory is then fsynced.
- `aggregate.py:208–210` publishes every cell only after its original final
  checkpoint binding check. The snapshot contains all checked completed rows.
  A kill during a later write leaves the previous successfully published JSON
  intact; the temporary file remains evidence. No resume or automatic retry is
  introduced. An initial publication failure prevents numerical imports.
- `aggregate.py:233–245` retains caught failure details and publishes the final
  aggregate atomically. Hard termination can leave the last progress status
  `before_numerical_imports` or `cell_complete`; neither is a completed contrast.
  Completion progress is published after the final table, and the adapter requires
  both files, all fifteen cells and matching actual source/config bindings.

## Bound existing execution

`run_existing_owner.py:15–29,55–86` checks exact adoption/source/config/adapter/
owner/helper bytes and invokes the existing `run_fit` once. The concrete template
checks the exact peptide host and two-GPU inventory, enters the authorized
repository, and execs the adapter with the actual root adoption SHA substituted.

The command fixes a 100-second child wall limit, 2 GiB owned-process-tree RSS
threshold, zero owned GPU memory, empty CUDA visibility/PYTHONPATH and BLAS/OMP/
NumExpr thread values two. The unchanged worker sets Torch threads two and
inter-op threads one. The existing owner uses one-second sampling with bounded
telemetry, identity-checked group termination and direct Popen wait/poll. Its
unchanged EXIT receipt retains elapsed time, maximum sampled RSS/GPU/log/output
sizes, exit authority, reason, signal/refusal observations and log hashes. The
adoption and CURRENT_PROCESS bind the source/command. These cover worker failures
before initial progress as well as later external termination.

These are the existing sampled supervision thresholds, not a kernel-enforced
RSS ceiling or a zero-overshoot real-time timer. Last published worker CPU/peak
RSS and terminal sampled tree RSS have different scopes. Terminal worker CPU
after SIGKILL remains unknown, as PLAN explicitly states. No new supervisor,
termination primitive, retry, allocation or ownership framework is added.

## Narrow verification

Independent stdlib AST/hash checks confirmed all five source-manifest payloads
and seven disabled-adoption source bindings. Geometry and dispersion function
ASTs are identical to V1; the complete cell AST is identical after removing only
the completed-count assignment and progress call. Paired contrasts and complete
result construction are identical. Config bytes and V1 source hashes are
unchanged. Aggregate, adapter and template Python were parsed/compiled without
execution. Bindings are retained in `SOURCE_BINDINGS.json`.

## Receipt precision notes

These do not defeat WG1's durable custody or the adapter's completion checks:

- The final `persist_progress(result['status'])` call resets its `failure` field
  to null after a caught exception. The already atomically published failure
  aggregate retains the actual failure details. Read both retained files.
- `PHYSICAL_TERMINAL.json` sets `partial_progress_preserved=True` declaratively;
  that flag is not a check of file presence or bytes. Admission failure can occur
  before the initial progress snapshot. The actual progress file and the existing
  EXIT/log/adoption receipts supply evidence; do not use the flag as its substitute.

Only source and compact configuration metadata were read. No numerical providers,
fixtures, checkpoints, scientific arrays, server state or Q/K outcomes were
opened. V1, V2 and the existing owner/helper sources remain unchanged. This
assessment does not authorize a numerical readout or scientific interpretation.
