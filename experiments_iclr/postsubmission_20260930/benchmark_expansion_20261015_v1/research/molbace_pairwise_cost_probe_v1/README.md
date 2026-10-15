# Pairwise MolBACE: bounded CUDA production-cost probe

Source only. Prepares one actual producer per root-admitted invocation:
`structured` shared or `native_common` body 0, seed 101. It launches no
owner, queue, child process, remote command or Git operation. No probe or
scientific fit has been run during preparation. The admission template is
closed. Root supplies publication/source review, the current one-worker
resource reserve and an external finite watchdog.

## Exact measured work

Each invocation uses the unchanged frozen producer descriptor, original
repository `.venv`, native runtime adapter and `adapter.train_epoch()`.
It makes **one construction**, traverses all **1210 TRAIN graphs once** in
the original shuffled native sampler (37 batches of 32 and a tail of 26),
and performs exactly **38 ordinary optimizer updates**. It then performs
**two complete VALID151 calls**. There is no selector, fitted checkpoint,
scientific state export, TEST indexing or retry. The engineering model is
discarded after this bounded work.

This whole epoch covers actual varying molecule/batch sizes. It is not a
two-batch throughput proxy. The already completed CPU two-update/fresh-
restore receipts are reused for correctness and exact descriptor identity;
their times do not authorize CUDA cost or a scientific budget.

| Producer | TRAIN native forwards | Each VALID | Total |
| --- | ---: | ---: | ---: |
| structured shared | 152 | 20 | 192 |
| native_common body 0 | 38 | 5 | 48 |

Actual optimizer attempts/completions and per-phase native forward
attempts/completions are counted and bounded. The original runtime
optimizer observer checks finite gradients. The full epoch must cover all
source records exactly once, with 38 maps and no skipped source batches.
Batch graph/atom/directed-bond sizes, pair-support/empty counts and
histogram, class visits and route weight masses are retained without
changing sampler/RNG behavior or adding a model forward.

## Timing, resources and output

Construction, the full TRAIN epoch and each VALID call use host elapsed
time bracketed by CUDA synchronization. TRAIN timing includes the native
loader, scalar objective, original backward/Adam and existing diagnostics.
The first epoch is cold. Only cheap passive batch/result bookkeeping is
added inside it; file writes occur outside timed phases. Full process wall
time includes source verification and reporting separately.

`RESULT.json` records actual native/private/total parameters, parameter
bytes, private BN bytes, actual Adam-state tensor bytes, total CUDA peak
allocated/reserved bytes and process peak RSS. It retains partial attempts,
failures, interruption and closure status. A hard kill may leave unknown
work for root's supervisor to retain. Native VALID computes its ordinary
metrics internally, but no validation scores or predictions are released
by this cost report. No selected state is saved.

One cold epoch and two VALID calls for two producers/seed101 do not measure
convergence or establish all-seed production throughput. Structured and
native-common costs, table construction and reader/writer overhead remain
separate; concurrent wall times or peaks must not be summed. Root reviews
actual variability, the frozen 21-fit controller limits and current worker
growth envelopes before choosing any scientific wall/memory budget.

## Interface after root source/resource review

```text
CUDA_VISIBLE_DEVICES=0 <repository>/.venv/bin/python -B probe.py
  --project-root <repository> --arm structured
  --admission <approved-root-admission> --admission-sha256 <actual-hash>
  --output <fresh-repository-contained-output>
```

The second producer is a separate root-controlled invocation with
`--arm native_common` and its exact admission. The probe requires the sole
authorized allocation GPU UUID before project metadata reads, source and
producer manifest closure, the actual CPU receipt join, one-construction/
one-epoch/38-map/two-VALID admission, and finite external watchdog/GPU/host
reservations. It does not run its own scheduler or watchdog.

`python3 -B check_source.py` checks source/CPU descriptor joins, stdlib-only
imports, exact work bounds, absence of selector/checkpoint/owner/Git calls,
closed admissions and the manifest. These checks execute no scientific
data, tensors, models or GPU operations.
