# Campaign process names

This additive launcher sets the process title to exactly
`AI Auto Research Pipeline` for the campaign queue owners, capacity waiters and
their native Python training processes. The frozen launch safeguards, memory
admission, original plans, source descriptors, barrier and training policies are
inherited unchanged. Only newly invoked campaign processes are affected.

`named_worker.py` accepts the launch safeguards CLI and adds the optional
`--process-title-dependency-dir`. The root operator prepares `setproctitle==1.3.6`
without dependencies in the separate project directory
`campaign_process_title_dependency_20261021_v1`, then passes its absolute path.
There is no global environment hook, `sitecustomize`, `PYTHONPATH` mutation or
change to the training virtual environment. A missing dependency fails before
the worker takes ownership.

Every invocation retains the complete original command as `native_argv` in
provenance. Its executed argument vector adds the named plaintext
`launch_named.py`, which calls `setproctitle`, restores the native script's
arguments and script directory, then runs that exact file with
`runpy.run_path(..., run_name='__main__')`. Native `SystemExit` is propagated.
The unchanged reporting entrypoint executes the qualified native runtime in the
same process, so GPU training retains the name. The label is a process command
title; Linux's shorter thread-name field may truncate its display in tools that
choose that field. NVTOP's full command display can show the complete title.

The outer producer schema is `campaign-named-processes-producer-v1`, with this
package at `actual_outer_source_manifest` and the pinned safeguards package at
`unchanged_launch_safeguards_source_manifest`. The subclass supplies that
producer before the parent first writes OWNER, STATUS or INVOCATION. Guardian
custody and native child leases are unchanged.

Use the original worker/plan/plan hash with this source manifest hash:

```sh
python3 -B -u /absolute/project/expansion/campaign_named_processes_20261021_v1/named_worker.py \
  --worker allocation/gpu0 \
  --plan /absolute/original/execution7/PLAN-allocation_gpu0.json \
  --plan-sha256 "$ORIGINAL_PLAN_SHA" \
  --source-sha256 "$NAMED_SOURCE_SHA" \
  --process-title-dependency-dir /absolute/project/expansion/campaign_process_title_dependency_20261021_v1
```

The same `launch_named.py --dependency-dir PATH -- watcher.py ARGUMENTS` form
can name a separately authorized Python watcher. The queue capacity waiter is
already named by `named_worker.py`; unrelated processes are untouched.

One small local fixture checks the requested label, original argv, resolved
script directory, `__main__`, and exit code 7. It uses a stub title library and
temporary plaintext script, with no native fits or remote calls. The first
fixture expectation used macOS's `/var` alias while Python correctly resolved
it to `/private/var`; the expectation was corrected. Runtime source was unchanged
by that fixture correction. Actual Linux title visibility is checked by the
root operator with the installed dependency.

`launch_named.verify_dependency(absolute_dependency_dir)` is available for a
read-only activation preflight. It pins the Python module and CPython 3.11 Linux
extension to the hashes verified live by the root operator, checks import origin
and version 1.3.6, and returns dependency receipts without changing the caller
process title. Those receipts also appear in the named producer provenance.
