# Wiki15 selected geometry V2: retain compact progress

Source-only retention repair. V1 remains unchanged. No numerical work has run.
The geometry, folded head, class-margin, null-energy, roster, paired contrasts,
normalization and tolerances are identical to V1. Its scientific interpretation
and selected-epoch/local-global/VALID-only limits remain in force; see the V1
`PLAN.md` and `RESIDENT.json`.

## Minimal delta

V2 writes `AGGREGATE.json.progress.json` before importing NumPy/Torch and after
each completed cell passes the original final checkpoint binding check. Every
snapshot contains all completed compact cell rows, source/config bindings, UTC,
measured worker elapsed/user CPU/system CPU/peak RSS, and completion count. The
file and parent directory are fsynced after atomic replacement. An interruption
during the next publication leaves the previous complete JSON intact. A leftover
temporary file is retained; it does not trigger a retry or resume.

An exception updates failure progress, and final JSON publication is atomic too.
External termination may leave the last snapshot marked `cell_complete` or
`before_numerical_imports`; that is partial evidence, not a completed fifteen-cell
contrast. The last snapshot's costs cover work through that publication. The
existing owner's terminal record supplies actual total elapsed/RSS/exit/signal
costs after termination. Neither snapshot nor helper invents a terminal worker
CPU value after SIGKILL.

## Concrete existing owner binding

`run_existing_owner.py` is a small adapter to the already reviewed
`citeseer_known_ranking_control_matched_reference_owned_preparation_20261006_v2/owner.py`
`run_fit`, using the existing v3 `ownership_helpers.py`. It contains no new
monitoring loop, signal primitive or retry machinery. The adapter fixes child
wall time at100 seconds, owned process-tree RSS at2 GiB, owned GPU memory at0,
CUDA visibility empty and all thread controls2. The existing GPU telemetry is
retained, with a zero minimum free-memory requirement and no resource wait for
this CPU child. Existing log/output caps are256 KiB/1 MiB; poll/telemetry bounds
remain1/5 seconds. The original worker's90-second cell-start deadline is unchanged.

`ROOT_ADOPTION_DISABLED.json` binds the exact V2 source manifest, aggregate,
config, adapter, existing owner, ownership helper, original interpreter and fixed
limits. Source bindings are immutable; the template grants no execution.

After root reviews/adopts V2, use the existing77 MacLink transport to stage these
small source files and the unchanged V1 `INPUTS.json` at its bound path. Create the fresh
`Wiki15_selected_member_geometry_descriptive_execution_root_20261010_v2` directory,
copy the disabled template to its `ROOT_ADOPTION.json`, and set only
`execution_enabled` and `source_review_approved` to true. Compute that actual
adoption hash. `ROOT_COMMAND_TEMPLATE.txt` gives the exact route-checked command;
replace only `ACTUAL_ROOT_ADOPTION_SHA256`. No V1 output, source or receipt is
rewritten. The adapter creates fresh `owner/` and `result/` once; an existing
directory prevents a silent retry.

The original selected caches and head maps stay on77. Only the compact aggregate,
progress and existing owner terminal receipts may return to the Mac. No Q/K
candidate outputs, fits, new forwards, accuracy/NLL scores or raw arrays enter
this repair.
