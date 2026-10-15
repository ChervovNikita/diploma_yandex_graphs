# Complete structural collection queue

Source only; not launched, uploaded or admitted. Root reviews/publishes this
package and supplies an approved copy of `ROOT_ADMISSION_TEMPLATE.json` with the
exact manifest/INPUTS hashes and real publication, interpreter and peer/resource
review receipts. The explicit delegation flag authorizes binding actual terminal
hashes after both prerequisites succeed. No terminal hash is guessed in advance.

## Fixed sequence

1. Wait at most 24 hours for all six structural fits in
   `results/structural_private_risk_v1` and all seven invocations in
   `qualification/pubmed_cuda_queue_24` to succeed. Use the frozen structural
   terminal gate, exact PubMed names/order, zero exit codes, terminal result
   equality, and the pinned launch owner identities. A failed prerequisite or a
   vanished owner without successful terminal metadata stops the queue.
2. Preserve actual END/OUTCOMES/QUEUE and seven RESULT receipts. Verify the paid
   33-bank analysis. Acquire `analysis_outputs/selected_VALID_collection_gpu_slot.lock`
   within ten minutes and keep it through collection and analysis. Recheck the
   terminal receipts after acquisition and before each collector.
3. Bind a real `COLLECTION_ADMISSION.json` under the root's conditional admission.
   Run unchanged `selected_VALID.py --mode commands` once. Require its six argv
   and terminal receipts to equal the frozen collection plan exactly.
4. Supervise those six original `analysis/selected_molecular.py` commands directly,
   serially. Reuse the existing collection gate/outcome helpers and emit actual
   START/OUTCOMES/END records in the format the original analyzer requires.
   The nested `--mode collect` launcher is not invoked: it would acquire the same
   lock and create collector grandchildren outside this owner's direct supervision.
5. Run unchanged `selected_VALID.py --mode analyze` once with CUDA hidden, only
   after all six collectors finish. Require its complete 39 banks / 45 unique
   paid fits result. The 33 existing banks are reused; no control is recollected.

The owner imports only stdlib bookkeeping and constructs no model. It uses the
existing `preflight`, `process`, `gpu_state`, `terminate` and collection helpers.
No fit, TRAIN update, TEST call, reselection, tolerance change, automatic retry,
partial summary, alternate collector or new supervision service is added.

## Bounds and ownership

- Proposed physical reservation: 8 GiB host RSS (owner plus current child), 4 GiB
  GPU and 2 GiB total output disk; two CPU threads. Root must admit these bounds.
  The prior residual collector recorded about 0.915 GB RSS and 0.082 GB CUDA
  reserved; those are supporting observations, not measurements of this queue.
- Command generation: 180 seconds. Each collector: the plan's 600 seconds.
  Saved-prediction analysis: 600 seconds. TERM cleanup: ten seconds before KILL.
  Monitoring adds ordinary polling/probe latency; no virtual-address cap is used.
- Before every invocation, recheck host and GPU headroom after the older peer's
  16 GiB host / 32 GiB GPU growth envelope. Authenticate PID 640428/start 6055039165
  if it remains present. Refuse any other GPU peer, PID reuse or a third model
  worker. Monitor own RSS/GPU/disk and the older peer envelope during each child.
- Only this queue's direct `Popen` process group is signalled. A fixed, exclusive
  queue-output directory is the durable one-attempt claim, including while waiting.
  Existing queue, collection or analysis outputs forbid duplicate work. A failure
  retains logs, known child costs and partial artifacts and stops later work.
  Missing terminal child receipts mean unrecorded work is unknown, never zero.

Fixed outputs are `qualification/structural_complete_collection_queue_run_v1`,
`analysis_outputs/structural_private_risk_routes_v1` and
`analysis_outputs/structural_private_risk_complete_v1`. Root must not run another
structural collector alongside this admitted queue. A hard owner kill is an
incomplete attempt requiring root cleanup; it is not permission to rerun.

## Allocation-local invocation for root

Set `QUEUE_ADMISSION` and `QUEUE_ADMISSION_SHA256` to the actual reviewed admission
file and SHA256. Use the existing root launcher/supervisor; this package schedules
nothing. A 26-hour outer watchdog covers the finite waits, seven work invocations
(six collectors and analysis), command generation and cleanup allowance.

```sh
QUEUE_ROOT=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
QUEUE_SOURCE="$QUEUE_ROOT/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/qualification/structural_complete_collection_queue_v1"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$QUEUE_ROOT/.venv/bin/python" -B -u "$QUEUE_SOURCE/queue.py" --admission "$QUEUE_ADMISSION" --admission-sha256 "$QUEUE_ADMISSION_SHA256"
```

The terminal `QUEUE.json` pins the complete analysis and collection END. Preserve
the scientific/PubMed costs separately from six collection costs and the single
saved-prediction analysis cost; do not sum concurrent wall times or memory peaks.
