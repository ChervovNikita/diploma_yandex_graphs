# Launch safeguards for the frozen execution7 campaign

This additive package is prepared locally for the readiness discussion on
21 October 2026. It has not been activated. Preparation created no global PASS
marker, launched no native work, contacted no remote host and changed no existing
queue owner, job, plan, scientific source, dataset, fit setting or attempt.
The current readiness review leaves the launch decision to the user's next
discussion. These source files do not authorize activation.

`queue_worker.py` extends the original execution7 Worker and calls the unchanged
clean file-cache memory probe from `campaign_memory_admission_20261019_v1`.
Both packages are verified against their frozen source manifests:

| Source | Manifest SHA256 |
| --- | --- |
| execution7 | `52a996e6effb77748339c57a716159a80a06272d88914d37beaa2dd680e7e890` |
| memory admission | `6905ecec8b840925d2c3e03fb9dd43f06083372a05077579082f468a93aa4c31` |

## I4 output integrity

The original result validation still checks schema, status, TEST exclusion and
the four member identities/job hashes. A completed I4 task additionally needs
`bank_receipt.path`, `bank_receipt.bytes` and `bank_receipt.sha256`. The receipt
must resolve to that attempt's `output/I4_selected_VALID.pt`, remain within the
project root and match the file's actual bytes and SHA256. A symlink to another
file is rejected. This check runs through the same validation method during
collection and when reloading a previously successful attempt.

A missing or changed bank prevents completion acceptance. Collection retains
the native result and records a failed attempt. Restart validation raises an
error while preserving the earlier attempt result and native receipt; it does
not rewrite historical success as a new measurement or automatically rerun it.
The I4 aggregator and every native invocation remain the frozen originals.

## Launch disk admission

The inherited `start` method still holds the host ledger lock while observing
live capacity, deciding admission and recording a reservation. Its disk check
now requires:

```text
free_disk >= 20 GiB + candidate.disk_bytes + sum(active[*].disk_bytes)
```

All active reservations on the host count, including work on the other GPU.
Equality is admitted if the other original checks also pass. A lower value
blocks new tasks and records the reason in `CAPACITY_WAIT.json`. Actual free
space already reflects retained results. Existing GPU/RAM/CPU checks, pairing,
remaining growth, the two-child limit and frozen per-task disk reservations
remain inherited. The memory repair still credits only conservatively bounded
clean active/inactive file cache and respects cgroup/host bounds.

This is a check before a new task starts. It does not reserve lifetime campaign
storage or enforce a retained-data cap, transfer allowance or checkpoint-boundary
check. It does not prevent another process from consuming space after admission.
It never deletes outputs or signals running jobs in response to low disk.

## Custody, provenance and full phase

The package uses the existing plans, state roots, locks, guardians, attempts and
smoke summaries. Existing smoke PASS records are validated and reused. The
original exact-ID all-6,460-smoke-PASS global barrier remains mandatory before
full training. This package does not create or relax that marker. The native
argv and resume policy are inherited unchanged, including the separation of
smoke and full outputs.

On a future authorized activation, the original supervisor lock must be acquired
before OWNER can be replaced. A narrow metadata hook first fsyncs the exact old
OWNER bytes into `OWNER-before-launch-safeguards-<sha256>.json`. New OWNER,
STATUS, INVOCATION and `LAUNCH_SAFEGUARDS_OWNER-<time_ns>.json` records identify
this outer source manifest, the inherited execution7 and memory manifests and
the original plan. Historical attempts keep their original producer identity.
An already-live supervisor prevents construction; this package cannot take over
its lease. It does not remove a STOP marker or perform a job handoff.

The ordinary full phase and I4 exports are supported. There is no failed-fit
retry switch. An ordinary full-task failure stops new starts and drains existing
children using execution7 behavior; recovery/retry needs separate handling.

## Plaintext command for a later authorized activation

Root owns route verification, plaintext staging, received-source checks, the
future launch decision and any necessary existing-owner handoff. Do not use the
legacy encoded repair or coordinator transport. The original execution7
`launch.py` invokes the original worker, so a future reviewed launcher must name
this wrapper explicitly. The command below belongs inside the authorized tmux
session and must use the independently verified hash of this package's
`MANIFEST.json` as `TASK_SAFEGUARDS_MANIFEST_SHA`:

```sh
TASK_PROJECT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
TASK_E="$TASK_PROJECT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
"$TASK_PROJECT/.venv/bin/python" -B -u \
  "$TASK_E/campaign_launch_safeguards_20261021_v1/queue_worker.py" \
  --worker allocation/gpu0 \
  --plan "$TASK_E/campaign_execution_20261016_v7/PLAN-allocation_gpu0.json" \
  --plan-sha256 19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad \
  --source-sha256 "$TASK_SAFEGUARDS_MANIFEST_SHA"
```

For server51, the project root is
`/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`.
Use the unchanged plan for the selected worker:

| Worker | Plan filename | Plan SHA256 |
| --- | --- | --- |
| server51/gpu0 | `PLAN-server51_gpu0.json` | `b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345` |
| server51/gpu1 | `PLAN-server51_gpu1.json` | `bc025b3209c0a7f5c2f10fd011fd54ca072c78269693b035ca3e53bbf243ad3c` |

These plan identities are retained in
`planning/all_fit_launch_20261016_v1/GLOBAL_SMOKE_BINDINGS_PORT6.json`; preparation
does not establish their current remote bytes. Optional `--poll-seconds` must
be at least 1 (default 10); `--smoke-wall-seconds` stays fixed at 1800.

## Focused local verification

Eight tests passed under Python 3.9.6:

```sh
python3 -B test_safeguards.py
```

They exercise bank receipt fields/confinement, actual missing/tampered bank files,
collection/reload rejection, exact disk thresholds and all-host reservations,
low-disk prevention of a new child, existing PASS reuse, inherited control/native
argv and owner preservation/current provenance. They use temporary local scalar
and file fixtures without model construction, GPUs, datasets or network.

The new source and its inherited admission/worker/guardian/common/capacity files
passed the readable-command static checker. Manual review traced ordinary
plaintext imports, argument-list child invocations and the reporting entrypoint's
receipt-bound import of the existing native source. These are normal imports of
inspectable files; no decoded executable payload is introduced. The local review
does not certify any remote transport or live worker. Source hashes, verification
scope and limitations are retained in `SOURCE_CHECKS.json` and `MANIFEST.json`.
