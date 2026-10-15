# Representative server51 throughput pair

This engineering qualifier measures whether two normal experiments on one server51 A100 finish a fixed native-plus-structured PubMed workload faster than serial execution. It reuses the existing full native PubMed adapter and frozen native/body0 and structured pilot definitions from the same server51 cohort. It creates no scientific fit result, checkpoint, selection or TEST score. It does not repeat restore checks or change the campaign port.

## Fixed workload and timing

The phase order is native alone, structured alone, then both concurrently on the same physical GPU. Each fresh child executes **10 warmup plus 100 timed complete TRAIN+VALID epochs**. Every TRAIN epoch traverses all11,829source TRAIN records and performs the original full-batch optimizer transition; every VALID call covers all3,943source VALID records. A complete invocation therefore costs **440optimizer updates,440full TRAIN epochs and440VALID calls**, of which400epochs are timed. These costs remain recorded even though no accuracy result is claimed.

Ten full epochs warm the native operators, allocator and provider. One hundred measured epochs provide a sustained window with per-epoch timings, avoiding a one/two-update throughput claim. This fixed bound is not a convergence study. The native and structured serial/concurrent replicas use the same frozen arm/seed/settings. Each process has two native CPU threads. No initialization, loss, routing, optimizer or forward code is modified.

Both concurrent children finish warmup and wait at an owner-released barrier. Their100-epoch jobs then run normally; the faster job can finish first. The mixed-workload makespan includes the slower tail, rather than pretending both jobs overlap for the whole phase. The scalar summary also reports actual overlap, fully-overlapping epoch counts/times, per-arm slowdown, start skew and startup/warmup separately. GPU synchronization brackets each measured full epoch. Progress JSON is written every25timed epochs; individual epoch durations exclude this I/O while whole-window makespan includes it.

There is one fixed phase order and one workload pair, so the result supports practical admission of this exact PubMed/native/structured profile on the observed GPU. It is not a throughput guarantee for every method/dataset, a paired statistical replication, or evidence about prediction quality. No automatic admission threshold is applied; root reviews the measured speedup and memory envelope.

## Admission and ownership

Copy and complete `ADMISSION_TEMPLATE.json` after root review. Keep `approved=false` until exact source, resource and prior-terminal receipts are present. The one-command form is:

```text
REPO/.venv/bin/python -B -u QUALIFIER/run.py --admission APPROVED_JSON --admission-sha256 EXACT_SHA
```

The starter detaches one supervised owner and prints its receipt. The owner:

1. Verifies hostname`olig`, the exact two-GPU table, assigned UUID, lexical repository venv, frozen source/map and admission bytes.
2. Requires the matching prior PubMed owner to be terminal and no longer live, then takes **that map's exact `slot_lock_relative` with nonblocking flock**. A busy slot is never stolen or waited out under a new claim.
3. Verifies original source/data bytes once and records a live source-verifying owner claim before constructing any model.
4. Checks measured free VRAM, cache-aware effective cgroup/host RAM, disk and reviewed CPU/thread capacity. Resource pressure waits before a phase; unrelated jobs are capacity consumers, never a PID whitelist. No cache clearing or host changes occur.
5. Starts at most two owned children on the selected GPU, checks their exact process births and resource envelopes, and signals only those owned identities during cleanup.

The prior owner, admission, qualifier source and actual target map are exact byte/hash pins. Children verify their live parent and the held slot lock. Failures retain JSON/logs, partial epoch/update counts and costs, with no retries. Watchdogs bound each child to1,800seconds and the owner to7,200seconds. No source process has been launched by this agent.

## Output for the resource-admission agent

`END.json` schema is `server51-native-structured-concurrency-qualification-v1`:

- `status`, `worker`, `hostname`, `physical_GPU_UUID`, `admission`, `source_manifest`, prior-owner receipts and exact slot lock.
- `phases.<phase>.arms.<arm>.result`: actual child JSON receipts, process identities and exit status.
- `timing`: serial sum, concurrent makespan, mixed-workload speedup, per-arm slowdown, overlap and epoch summaries, startup times and exact440-update cost.
- `all_actual_costs`: startup/warmup/barrier/invocation wall time, TRAIN/VALID/update counts, peak RSS and CUDA allocated/reserved memory per child.
- Capacity observations, total owner wall time, every failure/cleanup uncertainty, no-retry and closed-TEST declarations.

Each child `RESULT.json` includes all100measured epoch intervals,10warmup intervals and actual optimizer attempts/completions. Scores and tensors returned by the native validation callback are discarded; no prediction-quality statistic is persisted.

Three local stdlib source checks pass: syntax/no numerical imports, fixed protocol/cache-aware memory, and scalar makespan/overlap/cost calculations with incomplete-work rejection. Root still needs source review, actual admission, and an available prior pilot slot before execution.

## V2 terminal preservation

The V1 source remains immutable. V2 additionally records every observed failed or successful child terminal before handling its exit status, and records cleanup outcomes before clearing active children. `retained_child_terminals` includes the exact child RESULT receipt, actual reported update/epoch/VALID/wall/resource fields, exit code and `forced_KILL`. Missing or force-killed reports retain explicit unknown partial-cost flags. No timing or workload code is changed.

## V3 incomplete-cost correction

V1/V2 remain immutable. V3 treats every non-PASS/nonzero-exit, stale running, externally killed/OOM, missing/malformed/mismatched-authority, cleanup-error or forced-KILL terminal as potentially incomplete. Only exit0 plus final PASS, matching child/owner/phase/arm/admission/source, valid reported counters and finite wall time can clear that flag. Exact reported partial costs remain retained independently. A fourth scalar source test reproduces stale RESULT plus exit-9 and rejects it; native API/work/timing are unchanged.
