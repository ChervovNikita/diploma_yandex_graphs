# Reuse the existing finite owner

Root owns release, execution, publication, terminal custody, and the interpretation of failures. Reuse only the pinned generic `run_fit(s, root, entry, queue, environment, output, c)` function in `citeseer_known_ranking_control_gpu77_b0_owned_preparation_20261006_v1/owner.py`, plus the pinned v3 ownership helper. Do not invoke that old owner module's specialized `main`, `run_child`, fit gates, or the initializer owner's fixed-four-update wrapper.

The existing function is numerical-import free, starts exactly one child with `subprocess.Popen(..., start_new_session=True)`, binds PID/start ticks/argv/cwd/pgid/sid, polls only its owned tree, caps wall time/RSS/GPU/logs/own output, signals only the verified owned group on its bound failure, and obtains terminal status from the direct Popen wait/poll. It has no automatic retry. Root must preserve EXIT, stdout/stderr hashes, resource maxima, child identity, signals/refusals, physical absence, and no-owned-CUDA postcondition. A returned receipt alone is not successful admission if any exit/reason/signal/terminal/absence check fails.

## Existing call interface to bind

- `s`: freshly loaded pinned ownership helper with `GPU` set once to root's selected physical UUID; never share a module instance across lanes that can rebind its GPU.
- `root`: a fresh owner output directory with existing `logs` child; create it through root's existing activation path.
- `entry`: exact `cell_id`, `job_relative` (the fresh activated release), `job_sha256`, `hard_seconds`, and `argv`. The argv uses the context-bound77 interpreter, `-B`, the exact new `qualify.py`, `--release`, its fresh absolute path, and `--release-sha256`, its hash. The existing function resolves `job_relative`; root verifies its hash immediately before the call.
- `queue['resource_limits']`: finite `resource_wait_seconds`, `poll_interval_seconds`, `telemetry_timeout_seconds`, `minimum_fresh_GPU_free_bytes`, `owned_tree_RSS_cap_bytes`, `owned_tree_GPU_memory_cap_bytes`, `combined_child_log_cap_bytes`, and `own_fit_output_cap_bytes`. `entry['hard_seconds']` supplies `external_hard_seconds_per_cell`. Suggested limits are in RESOURCE_PROPOSAL.json and are not authority.
- `environment`: exact existing provider context, selected singleton `CUDA_VISIBLE_DEVICES`, empty `PYTHONPATH`, absent `PYTHONHOME`, and root's usual bounded CPU-thread environment.
- `output`: the exact new worker output directory, not owner logs or an existing fit directory.
- `c`: existing context facade supplying `REPO`, `SOURCE_SHA`, `GPU_UUID`, `GPU_UUIDS`, `phase_file`, `physical_host`, `sha`, and `write`. `phase_file` must enforce phase-relative files; `physical_host` checks exact77 host/repository and full two-GPU inventory. `SOURCE_SHA` is the new qualification manifest hash, never an old fit source hash. The generic function does not need its old Citeseer science or any changed owner source.

## New external evidence without a hash cycle

Before final release creation, root writes a separate immutable owner evidence JSON that the release binds by path/hash. It states `root_execution_authorized=true`, exact qualifier hash, the two existing helper bindings, selected physical GPU, exact new output, active/external hard seconds, owned-tree GPU/RSS/output caps, and the source-only setup scope. The worker verifies these fields against the activated release before numerical imports.

That evidence does not contain the final release hash. The activated release then binds the evidence; root's actual `entry` and CHILD_STARTED/EXIT custody bind the final release hash and complete argv. This order avoids a release/evidence hash cycle. An evidence declaration never substitutes for actual owner/terminal receipts.

No new launcher framework, scheduler, remote command, process, retry/resume path, or specialized owner is implemented in this packet. The old fit/credit guard contracts remain unchanged.
