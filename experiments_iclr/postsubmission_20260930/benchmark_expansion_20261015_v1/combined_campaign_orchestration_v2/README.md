# Combined campaign orchestration source v2

V2 repairs the interruption between shard and LOCAL_READY creation. The
unchanged v1 source/manifest remain available. See `REPAIR_NOTE.md` and the
focused filesystem fault-point result in `LOCAL_CHECKS.json`.

This is a small sequencer around the **existing training and final-scoring
supervisor commands**. It supplies the source design for one initial command
per physical GPU. It adds no trainer, native scheduler, resource allocator,
scientific arm, retry policy or new user approval.

**Current status: source preparation only.** No production run specs, GO,
supervisors, fits, native probes or TEST access were generated/executed.
Scorer/final-runtime bindings are deliberately null in
`PREPARATION_STATUS.json`. Source review and genuine final readiness remain
required. The candidate scorer v3 repairs private-BN BUDDY VALID context but
still refuses private-BN TEST before query/cache work; that is a real blocker.
The reviewed final integration v1 still names scorer v2 and needs a compatible
reviewed successor before it can be used with scorer v3. This bundle does not
edit either package or silently choose an unreviewed scorer.

## What one command will do

After qualification and the user's already-required campaign verification
and GO, each actual worker gets one `combined-portable-campaign-run-v1` spec:

```text
TRAIN existing supervisor
  → own durable terminal acknowledgment
  → local-host all-workers terminal barrier
  → host-primary creates one immutable terminal JSON shard
  → wait for later offline global-seal delivery
  → derive final run spec from the frozen template and same GO
  → FINAL existing supervisor
  → durable worker completion receipt
```

The original Engine still runs up to two compatible native children per GPU.
The allocation uses one worker;77uses two workers after future authorization
and route/resource qualification. Every fit's final serving remains on its
training worker. Four I4 native bodies and aggregation remain grouped within
that same worker/host under the current fixed partition. No raw checkpoints
or logits have to move between machines.

The initial form is one command per GPU, using the actual qualified lexical
Python path and pinned run spec. A launcher generated after readiness should
use this form, with every log path inside the repository:

```sh
nohup <qualified-python> -B -u <repo>/.../combined_campaign_orchestration_v2/run_gpu.py \
  --worker <allocation/gpu0|server77/gpu0|server77/gpu1> \
  --run-spec <actual-worker-spec.json> --run-spec-sha256 <actual-sha256> \
  > <repo>/.../<worker>-launch.log 2>&1 < /dev/null &
```

This is a template, not an authorized executable campaign command. The
sequencer itself ignores terminal hangup. Both existing supervisors start
new sessions, read `/dev/null`, and write file logs. A Mac disconnect does
not supply an ongoing dependency. The final stage automatically starts when
its valid offline delivery appears; a second worker launch command is not
required in the uninterrupted path.

## Offline synchronization is a real later step

Each host first finishes all of its fixed training tasks. The primary local
worker writes `HOST_TERMINAL_SHARD.json` and `LOCAL_READY.json` only after all
local terminal acknowledgments are present. Its existing ledger remains the
training authority until the final integration's reviewed phase handoff.
On restart the primary validates and reuses an existing exact shard, then
ensures LOCAL_READY independently. Both local workers wait for both files.
This validation happens once at the stage boundary, not on every poll.

Someone must later exchange those immutable **JSON shards** between the two
project repositories. `offline_exchange.publish_delivery` invokes the
existing authenticated global merge once on the receiving host. It writes a
host-local compatible seal, merge evidence, then `DELIVERY.json` last. Every
host needs its own locally valid receipt view. Foreign origin paths stay
metadata; they are not opened. The complete global partition and role
semantics are checked by the existing merge/final authority.

This file exchange is not hidden behind a claim of autonomous cross-network
coordination. No live Mac/network coordinator is required while workers
train or wait, but final scoring cannot start until the real offline
exchange and deterministic merge occur. It needs no second user GO, no
checkpoint-by-checkpoint approval and no new scientific decision.

## Durable state and failures

The sequencer has one permanent run-spec claim and one non-expiring local
worker flock. Actual portable GPU/worker leases and host resource ledger
remain with the existing supervisors. The extra flock prevents duplicate
stage sequencers; it does not reserve a GPU or replace native custody.

Each supervisor stage preserves START, PID, file stdout and RETURN using the
actual portable atomic/fsync metadata primitives. A successful completed
stage is skipped on restarting the same command. A failed/unknown START is
not repeated automatically, even if its supervisor might still be running.
Its original native outputs, reservations and existing review/recovery
mechanisms remain authoritative. No new approval layer is introduced here;
unknown custody simply cannot be guessed to mean no work occurred.

Ordinary failed/no-selected training outcomes are terminal and remain in the
global seal. Existing supervisor exit codes and durable terminal state must
agree before advancing. A nonzero supervisor exit blocks the stage sequence
and retains logs/results/costs. A successful stage receipt means its queue
finished; it is not a claim that every scientific comparison succeeded.

All original host STOP files are retained. The initial baseline is pinned;
a new STOP during an offline wait halts the sequencer and does not start
final scoring. The existing reviewed continuation policy remains in force.
The sequencer never signals other users' jobs. On an explicit interrupt it
can signal only its actual child supervisor, whose unchanged Engine owns
native STOP/drain behavior.

## Work is bounded

Own-worker TRAIN terminal metadata is checked once after its supervisor
returns. Host-primary shard creation happens once per host. Offline waiting
checks only expected file existence and host STOP receipts every30seconds;
it does not repeatedly scan every descriptor, fit, checkpoint or outcome.
The actual final supervisor performs its existing complete source/role/global
seal authentication once at startup and retains the bounded per-child
admission path. Native experiments are never launched directly here.

## Binding contract

The prepared execution template has schema
`prepared-combined-worker-execution-template-v1`, with:

- worker, fixed existing `training_entry` and reviewed `final_entry` receipts;
- `supervisor_python_environment`, scorer and final-integration manifests;
- frozen `final_static_fields`: release/readiness and optional existing STOP
  continuation only;
- scoring roster, orchestration output root, local host sync root and fixed
  offline delivery path;
- `maximum_native_children_per_GPU=2`, `original_paper_reruns=false`, and
  `new_human_action_at_barrier=false`.

The actual combined run spec supplies exact host config, template/source,
training run spec/release/GO and global decision/contract receipts. The same
host GO binds `combined_execution={source_manifest,execution_templates}` in
addition to its existing training/final authorities. These fields can be
part of the deterministic same-decision host templates; they do not reference
the eventual derived seal or create cyclic GO-file hashes.
`execution_templates` maps both77workers to their exact worker-template
receipts, so they share one unchanged host GO view rather than requiring
different GO bytes for each GPU.

Only the derived seal and already-bound parent authorities are inserted into
the final run spec. Arms, hyperparameters, data, roles, selection and native
operators cannot be changed at the offline barrier. Original GNNM paper
experiments remain excluded; original GNNM is retained on new datasets.

## Remaining release gates

1. Review this source bundle and bind a reviewed compatible scorer/final
   integration successor; private-BN BUDDY TEST remains unsupported.
2. Retain actual selected-state/VALID numerical qualification and inference
   resource classes/pair envelopes; the plan alone is not numerical evidence.
3. Qualify actual host projections, including future77authorization. No77
   access or launch was attempted here.
4. Prepare the concrete finite campaign/run specs for the user's verification
   and actual GO. No actual GO artifact has been created by this package.
5. Perform a narrow engineering qualification of stage start/restart and
   offline-delivery custody before advertising the launcher as ready.

`LOCAL_CHECKS.json` contains source/pure-metadata results only. They do not
qualify real orchestration execution, training, inference or paper claims.
