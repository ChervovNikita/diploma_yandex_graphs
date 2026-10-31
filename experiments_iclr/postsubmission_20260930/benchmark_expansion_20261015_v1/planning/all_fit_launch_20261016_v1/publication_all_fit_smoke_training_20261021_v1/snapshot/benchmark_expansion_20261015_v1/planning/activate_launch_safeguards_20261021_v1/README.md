# Plaintext queue activation handoff

`activate.py` prepares the three existing queues for the user's 21 October
training authorization. Root stages and invokes this helper through the reviewed
route. The local preparation performed no remote action, STOP request, tmux
change, marker publication or training.

The helper checks the exact hostname and complete GPU UUID table before any
project metadata read. It then verifies its own supplied SHA256, the frozen
execution7, memory, safeguard and named-process packages, and the original plan.
The named package also verifies the pinned setproctitle 1.3.6 files, version and
module origin in the explicit project dependency directory before any STOP.
It verifies the current assigned smoke summary against its actual native result
receipts, latest attempt receipts and current PASS states. The global marker
must be absent. No assigned attempt lease may remain live.

The `inspect` action prints the owner, status, tmux pane, source receipts and
exact future command. It makes no handoff changes. `activate` requires the old
owner to be live and waiting at the global all-fit PASS gate, with every assigned
fit PASS and no active fits. It retains exact OWNER, STATUS, TMUX_LAUNCH and any
existing STOP bytes in `launch_safeguards_history_20261021` before writing the
original graceful STOP request. It never signals a process or stops an active
fit. Each call waits for at most the requested 1–55 seconds after preflight.

After the old process and supervisor lease have exited, the helper archives STOP
and respawns only a dead original tmux pane. If the old session disappeared, it
recreates that same session name. It never uses `kill-session` or `respawn-pane
-k`. The command names the canonical project venv interpreter and the frozen
`campaign_named_processes_20261021_v1/named_worker.py` with the explicit project
process-title dependency directory. Native jobs retain the inherited two-job
compatibility and capacity checks, the 20 GiB disk launch floor and I4 integrity
checks. The worker waits if its GPU cannot admit work.

`LAUNCH_SAFEGUARDS_HANDOFF.json` records intent and custody receipts. It grants no
training release. Intent is durable before tmux dispatch. If the result is
uncertain, repeat `inspect` or `activate` to reconcile; a recorded dispatch is
never automatically resubmitted. A new owner is accepted only after its OWNER
and STATUS both identify the frozen named-process source, the process is live,
and STATUS is waiting at the global gate with no active fit. An uncertain dead
or missing launch remains for root inspection instead of creating another owner.

The helper publishes no global PASS marker. Root releases training separately
after confirming all three new owners. The exact existing all-6,460-fit barrier
is inherited unchanged.

Existing interpreter paths were read from the retained 19 October
`REPAIR_LAUNCH_allocation_gpu0_20261019.json` and
`REPAIR_LAUNCH_server51_gpu0_20261019.json` receipts:

| Host | Python interpreter |
| --- | --- |
| allocation | `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python` |
| server51, both GPUs | `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.venv/bin/python` |

The existing sessions are `gnnm-allfit-allocation-gpu0-20261016`,
`gnnm-allfit-server51-gpu0-20261016` and `gnnm-allfit-server51-gpu1-20261016`.

Invoke the staged named script using the actual helper source hash supplied with
this package. Example inside the verified allocation route:

```sh
TASK_PROJECT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
TASK_E="$TASK_PROJECT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
"$TASK_PROJECT/.venv/bin/python" -B \
  "$TASK_E/planning/activate_launch_safeguards_20261021_v1/activate.py" \
  inspect --worker allocation/gpu0 --helper-sha256 "$TASK_HELPER_SHA"
```

For the authorized handoff, replace `inspect` with `activate` and optionally
add `--wait-seconds 45`. Server51 uses its project root from the table and the
corresponding worker name. No encoded command, generated executable payload,
global environment change, signal, job deletion or checkpoint alteration is
part of this helper.

Five local guard/identity/argv checks passed. They use mocked identity commands
and scalar records without network, GPU work or host changes. Source parsing and
the readable-command checker are recorded in `SOURCE_CHECKS.json`.
