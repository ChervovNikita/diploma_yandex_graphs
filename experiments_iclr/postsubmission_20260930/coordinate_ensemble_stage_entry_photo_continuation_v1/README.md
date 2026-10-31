# Photo continuation bridge — source only

This is a separately versioned derivative of immutable
`coordinate_ensemble_stage_entry_v1/stage_entry.py`. It adds bounded metadata
settling after the original guard failure following the first successful Photo
qualification child. The precise failed GPU row was not retained; utilization
averaging after child exit is a hypothesis, not an established cause. This source
has not been imported, compiled, tested, uploaded, or executed during preparation.

## Interface and admission

Use the existing interface:
`stage_entry.py --request ABS_REQUEST_JSON --supervisor ABS_INNER_SUPERVISOR`.
The request schema remains `coordinate-ensemble-stage-entry-request-v1`.
Root must bind this new entry's exact path and SHA256 in a fresh request and the
existing supervisor chain. Existing immutable requests do not adopt this source.

The dataset mapping remains `LastFMAsia` / `AmazonPhoto`; acquire still requires
`coordinate_ensemble_runner_v1/runner.py`, and qualify still requires
`coordinate_ensemble_runner_v2/runner.py`. Fit, test, and preflight remain rejected.
Qualification remains exactly 100 complete updates with `eval_every=10` for one
explicit frozen cell and canonical public data manifest per operation.

Root supplies only the explicitly admitted remaining operations. There is no
automatic skip, model/data retry, source acquisition, resume, overwrite, or cell
selection. Runner output and bridge receipt directories must be new. All original
path, source hash, argv, route, GPU exposure, and inherited process group checks
remain in place. The Coauthor runner and Coauthor bridge are unchanged.

## Bounded GPU metadata settling

Before each child, the guard reserves at most 15 seconds for metadata checks.
It queries only `index,uuid,name,memory.used,memory.total,utilization.gpu` through
`nvidia-smi`. Query starts are spaced by 0.5 seconds; a query taking longer than
that interval is followed by the next query without an additional delay. Each
query timeout is capped by the remaining settling time, and sleep is truncated
to that remaining time. Query failures are not retried. No model, data, or child
operation runs during settling.

Every sample must immediately match exactly one physical GPU, index `0`, UUID
`GPU-44039938-fd82-41d2-fefd-de71514e2fac`, name
`NVIDIA A100-SXM4-80GB`, and total memory `81920` MiB. Used memory outside
`0..100` MiB fails immediately. Only utilization in `1..100` percent can trigger
another metadata sample. A child can start only after a sample reports used
memory at most 100 MiB and utilization exactly 0, within the settling deadline.
The guard does not query process names or commands.

The aggregate admission check now reserves
`operation_count * (per_operation_timeout_seconds + 15) + 3` seconds inside the
existing normal child budget. Before each guard, the remaining whole-bound
runway must exceed `per_operation_timeout_seconds + 15 + 2`; after the guard it
must still exceed `per_operation_timeout_seconds + 2`. The existing outer
supervisor remains the final wall-clock and descendant process group owner.

`START.json` declares the guard policy. Each attempted guard writes
`operation_NNN_GPU_GUARD.json`, including sampled metadata, timestamps relative
to guard start, elapsed time, and success/failure status. A failed guard is also
retained in the stage terminal's `gpu_guards` list even though no child START
receipt was written. Successful child START receipts bind their guard receipt
path and SHA256. Child timeout/cleanup and all scientific execution behavior are
unchanged.

Samples contain rows only from completed successful metadata queries. A query
timeout or nonzero exit is recorded by exception type/message; partial command
output is not retained. This limits diagnosis of query infrastructure failures.

## Preparation verification

Verification was limited to textual source/diff inspection and ordinary file
checksums. `SOURCE_PREPARATION.json` records the immutable base and protected
source hashes. `SHA256SUMS` binds this packet's three payload files. Independent
review is requested against the stable new source hash; root owns adoption,
request creation, runtime checks, and actual execution.
