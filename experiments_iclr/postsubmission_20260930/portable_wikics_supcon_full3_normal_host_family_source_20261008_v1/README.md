# Full three-seed SupCon family owner

This is a complete finite launcher for the prospectively frozen three fresh
`supcon_eq2` fits. It runs the sealed public CLI unchanged for all 1,100 epochs
and reuses the original full predictor, data, own-CE views, selector and
mean-probability serving. Root creates the separate exact enabled admission
after reviewing this source. Nothing was launched during preparation.

| Lane | Original physical GPU | Fixed fit order |
|---|---|---|
| 0 | `GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998` | 6101, then 6307 |
| 1 | `GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced` | 6203 |

The controller is stdlib-only and creates at most one owned fit session per
lane. Each worker receives the original normal77 interpreter, empty
`PYTHONPATH`, its original physical UUID as `CUDA_VISIBLE_DEVICES`, and the
fixed public CLI argv. No dataset acquisition, installation, mount, namespace
or host setting is performed. Existing other jobs remain running.

## Frozen bounds

| Resource or envelope | Bound |
|---|---:|
| Per-worker PyTorch allocator | 24 GiB |
| Fresh free GPU memory before Popen and again inside entry | 28 GiB |
| Memory wait per cell | 3,600 seconds |
| Per-fit active time | 32,390 seconds |
| Reserved per-fit cleanup | 10 seconds |
| Per-fit hard time, including cleanup | 32,400 seconds |
| Entire family hard time, including waits and cleanup | 108,000 seconds |

These are the exact root-frozen protocol limits. They are safety bounds, not
an ETA. The family deadline reserves cleanup, and no fit is launched unless
its entire hard envelope fits the remaining family budget. A memory wait that
expires is retained even if a later poll finds free memory.

`entry.py` sets only its own allocator fraction, resets its own peak counters
and invokes `runpy.run_path` on the unchanged public `train.py`. It changes no
seed, Session, recipe, objective or selection function. The original Session
performs its normal initialization and RNG reset. An active signal deadline
and a self-only hard timer also bound an entry if its controller disappears.

## Root admission and execution

Copy `ADMISSION_TEMPLATE_DISABLED.json` into a separate root-owned file. Set
the four admission flags only after approval, supply this exact controller
manifest digest and select a fresh phase-relative family output directory.
The frozen protocol SHA is already included. The template is disabled; it is
not accepted for execution as supplied.

From the registered normal77 repository, with the exact interpreter in
`SOURCE_BINDINGS.json` and empty `PYTHONPATH`:

```sh
python portable_wikics_supcon_full3_normal_host_family_source_20261008_v1/controller.py \
  --admission ROOT_ENABLED_ADMISSION.json --admission-sha256 EXACT_ADMISSION_SHA
```

Root supplies the normal detached outer launcher. The controller checks its
own manifest, both source manifests/payloads, frozen protocol, provider
versions, GPU inventory and exact existing TRAIN/development/native files
before launch and before each subsequent Popen. The public CLI additionally
verifies all six ordered data arrays against its official projected pins.

The controller writes exact worker jobs, fresh logs, owner/cell/closure JSON,
and actual Popen PID/start-ticks/group/session custody. Timeouts stop only a
matching live owned session: TERM, CONT for a stopped child, bounded wait,
then KILL and bounded wait/reap inside the original hard deadline. It never
scans or signals unrelated processes. Per-cell failures and unlaunched cells
remain accounted; there is no retry or resume. Independent planned cells may
continue once after a reaped fit failure, under the same fixed roster.

## Completion and closed quality

After direct exit/reap, only `COMPLETE.json` and the outer entry's terminal
metadata are read to verify 1,100 updates, exact source identity, SupCon-only
calls, full forward/VJP/Adam/RNG accounting and actual allocator peaks.
`RUN`, validation traces, predictions and checkpoints are never read by the
controller. Selected-state hashes are retained from completion metadata
without opening the states. The pinned CLI prints only an engineering
completion summary; the controller and entry print only lifecycle summaries.
Per-fit CPU/RSS and CUDA costs are recorded by each worker, avoiding ambiguous
per-cell deltas from concurrent `RUSAGE_CHILDREN` totals.

`CLOSURE.json` accounts for all three cells and preserves every failure. It
does not authorize comparative opening. The entire original Wiki12 family,
all three fresh fits, exact setup/terminal/selected-state custody and all
three pre-reserved alignment-only pairs remain required. Original scores
are neither rerun nor replaced. This conventional-loss comparator is not
standalone novelty or shared-ensemble superiority evidence.

The native qualifier passed exactly two discarded full-TRAIN updates in the
original environment; its exact receipt and original alignment setup metadata
are bound as provenance. No further smoke comparisons or numerical fixtures
were run. Preparation used static AST, binding and arithmetic checks only.
