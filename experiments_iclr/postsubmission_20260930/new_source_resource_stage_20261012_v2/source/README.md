# Remaining resource admission proposal

12 October 2026. This is a local source and metadata review. No remote access,
CUDA execution, optimizer update, scientific fit or GO occurred in this work.
The master remains disabled. All proposed execution requires root source review.

## Scope and existing evidence

The v8 master contains 36 studies and 38,309 model-fit groups. Its new endpoint
VALUE and private bootstrap fit records still carry 32,768 MiB and 5,000,000
second scheduling placeholders. Those values are not measurements or an ETA.
Its original OGB author track has eight recipes and 80 independent processes.

Existing proofs should be retained and inherited within their actual scope:

- Endpoint VALUE: four numerical CPU checks; exact source/operator ownership,
  small-input gradient/state and genuine control ownership. No real-input CUDA
  resource, serialized engine restore or whole-graph export check.
- Private bootstrap: four numerical CPU checks; exact fixed-weight/block VJP,
  GAT/SAGE correction/reference and small-input selected-state replay checks.
  The metadata-only repair does not invalidate these numerical checks. No
  actual-input CUDA memory/time or serialized engine restore check.
- OGB author: eight standard-library metadata/parser/source checks and an
  allocation metadata preflight. No GNN/provider constructor or model forward.
  LayerNorm GINE measurements cannot qualify these BatchNorm GIN/GCN bodies.
- Existing SAGE private-CMCL, masked-core and native largest-width probes
  establish useful code-path and cost evidence for their own source packets.
  They do not measure the endpoint-generated edge tensors, corrected maps or
  two-VJP bootstrap tapes in the newly ingested packets.
- The optional route recomputation utility now has six CPU checks and two
  actual-input Physics memory checks. It is separate from v8 scientific source;
  integration and extra-forward accounting are root-owned gates.

No passed synthetic test needs to be repeated for this proposal. Source bytes
or callback changes require a focused delta check, not another whole test suite.

## Finite proposed stage

`PROPOSAL.json` binds exactly 32 fresh, sequential random-state constructor
processes. The requests are source-bound pointers, not executable fit commands.

| Family | Constructor cases | Input scope | Per-child deadline |
| --- | ---: | --- | ---: |
| Endpoint VALUE | 12 | six actual arms × WikiCS/Photo | 90 seconds |
| Private bootstrap | 12 | three weighted arms × GAT/SAGE × WikiCS/Photo | 90 seconds |
| Native OGB author | 8 | four GIN/GCN/virtual recipes × MolHIV/MolBACE | 120 seconds |

The 24 node cases each use one cold and three warm exact source callbacks.
The eight OGB constructors each profile two legal, label-free size-selected
TRAIN packs and the actual include-tail pack. Each pack uses one cold and three
warm exact native callbacks. This totals 192 single-process source callbacks.
Callbacks attempt the native optimizer action, but every action is intercepted
before any optimizer kernel or parameter/moment mutation. Report attempted
actions separately from actual updates, which must remain zero.

The per-child deadline sum is 3,120 seconds (52 minutes), not an ETA. Earlier
small source constructors/probes closed in roughly 9–17 seconds each; these
different operators may cost more. Prepare TRAIN packs once, outside the
deadline, then freeze them. No full OGB epoch or HPO fit is needed for admission.
An additional two reviewed same-GPU pairs, each capped at 120 seconds, would
bring the declared child/pair deadline sum to 56 minutes. The pairs are chosen
only after single-process peaks exist; they must not be automatically launched.
Source/data preparation and process cleanup are extra overhead. A timeout/OOM
is a retained failure; no automatic larger-cap retry or smaller model is planned.

The check is finite and targets three newly connected families. It cannot grant
whole-master admission to other still-unqualified families or to 18.77 hardware.

## Node constructors and inputs

Use width 128, depth 2, FFN multiplier 1, eight heads, LayerNorm and dropout .4
with the declared .002 LR metadata. The nine-point search changes LR and
dropout, not tensor dimensions. No optimizer step is used to select a setting.
Both actual graphs are needed: WikiCS has more edges/nodes; Photo has more input
features. Their source-bound shapes are:

| Dataset | Nodes | Directed records | Features | Classes | TRAIN rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiCS | 11,701 | 442,907 | 300 | 10 | 580 |
| Amazon Photo | 7,650 | 245,812 | 745 | 8 | 160 |

The fixed-weight manifest establishes identical TRAIN row counts across all
20 WikiCS and ten Photo masks. Freeze mask 0 for resource checks; read only its
GRAPH/TRAIN and, for bootstrap, its already fixed untrained weight table. Do not
decode SELECT/CAL/evaluate files, endpoint ID files, or any outcome artifacts.

The implemented helper uses each full frozen factory. The already reviewed
engineering clone replaces only the unused SELECT authority with a one-row
synthetic interface, then the RoleIO boundary denies every actual SELECT/CAL/
evaluate byte read, including hash reads. Validate/selection/evaluation/export
methods are disabled. GRAPH, TRAIN and fixed untrained bootstrap weights remain
exact. The source study, source code and its scientific selection rule are not
edited. The stub is an engineering input; it cannot establish SELECT gathering
or metric cost. Full-graph serving and metadata-shaped CPU role tensors qualify
the corresponding tensor/storage risks separately.

Endpoint source:

- The exact full factory calls its pinned `construct_source` with actual graph
  dimensions and edge/self-edge metadata for `BASE`, `E4`, `N4`, `equipped_M1`,
  `ordinary_M1` and `original_gnnm`. It receives only the unused synthetic SELECT
  interface instead of a real heldout role.
- Call unchanged factory closures `custom_train`, `logits_fn`, `capture` and
  `restore`. The factual batch, TRAIN IDs/labels and source private streams are
  constructed by their own source code. No closure is manually substituted.
- E4/N4 must both be measured. Edge incidence generators and gathered node
  maps have different live tapes. A zero-initialized identity transport must
  still retain differentiable builder operations; do not bypass them.
- Every node case also runs an eager no-grad, label-free full-graph serving
  forward and CPU tensor serialization. Record only shape, bytes and duration;
  do not export logits, scores, predicted labels or objective values.

Private bootstrap source:

- Use the exact pinned `source_build`, `bootstrap_module`, native Model,
  routes/factors/corrections, source `seeds/make/scope/logits` and exact
  `custom_train` closure. Preserve actual gradient ownership and the same
  four-route forward with its two separate VJPs; no scalar substitute.
- Measure `shared4_coherent_bootstrap`, `shared4_paired_graph_bootstrap` and
  `shared4_rank1_lora_graph_bootstrap` for each backbone/dataset. Keep the actual
  saved float64 [4,TRAIN] table and verify its immutable hash before/after.
- Construction metadata must enumerate every control, including original R/S/B
  GNNM, unweighted matched arms and separately fitted corrected M1/I4 controls.
  Establish exact kernel/parameter/topology containment before using a weighted
  M4 peak as conservative reservation evidence for an unweighted/M1 control.
  Do not call inherited controls individually measured, copy a speed claim, or
  silently assume that a different graph/backbone is covered. If source/layout
  containment fails, record the specific gap and propose only the necessary
  extra case. The 12 weighted checks are an initial finite stage, not proof of
  all possible control workspaces.

One native AdamW object may be constructed so parameter ownership and state
serialization remain source-compatible. Replace its step before any callback
with an interceptor that counts attempts and returns without running optimizer
code. Prepopulate zero first/second moment tensors and zero scalar steps by
reviewed explicit tensor creation, not an optimizer kernel. Their bytes and
hashes remain live through backward. Reserve an additional two-parameter-size
workspace allowance. Actual parameter/moment commits must be denied and hashes
identical. A source callback's declared `optimizer_updates:1` is intercepted
metadata, never the engineering result's actual update count.

## Native OGB source

Exact upstream source at `snap-stanford/ogb@61e9784ca76edeaa6e259ba0f836099608ff0586`
is mirrored read-only and checked against its four original pins. The constructor
for each case is:

```python
GNN(gnn_type="gin" or "gcn", num_tasks=1, num_layer=5,
    emb_dim=300, drop_ratio=0.5, virtual_node=False or True).to(device)
```

Keep omitted source defaults: residual=False, JK="last", graph_pooling="mean".
The source uses AtomEncoder/BondEncoder, BatchNorm and full categorical features.
The zero-step proxy preserves native Adam .001/weight decay 0 and its TRAIN
callback; it does not call `main`, native `eval`, Evaluator or epoch selection.

`prepare_native_TRAIN_packs.py` reads the already prepared official molecular
TRAIN.pt once per dataset, with categorical atoms/bonds and actual TRAIN y. It
compares its provider raw/scaffold receipt metadata with native author custody,
checks actual packed TRAIN IDs against the native official TRAIN index CSV,
then freezes the selected original graph fields. No model/provider/optimizer is
created. Do not invoke full PygGraphPropPredDataset during this probe: its full
processing materializes graph labels across all roles. Processed-cache resource
cost and official whole-provider protocol remain separate source-preparation
gates; the pack check makes no published-score equality claim.

For each dataset, prepare two legal batch-32 packs using structural metadata
only: 32 largest TRAIN node counts and 32 largest TRAIN edge counts, with graph
ID as the tie rule. Also prepare the actual deterministic TRAIN include-tail
pack: MolHIV five graphs; MolBACE 26. Bind the pack-selection code/geometry and
finite batch list before CUDA. Extract the exact upstream `train` function AST;
pass a one-batch iterable and a step interceptor. Native NaN masks, batch skip
logic, BCEWithLogitsLoss and reductions remain unchanged. Emit no loss or labels.

These two size-selected packs are concrete maximum-coordinate TRAIN probes, not
a theorem that every mixed node/edge batch has lower peak memory. Compute the
separate top-32 node and edge sum envelopes, retain headroom and live resource
monitoring. TRAIN-only probes do not qualify a larger label-free VALID/TEST
graph batch; source geometry metadata must close that separate serving bound,
without opening target-bearing outcomes. If it does not, that is a specific
remaining resource gate, not a reason to alter the author recipe.

BatchNorm changes running statistics and num_batches_tracked even when an
optimizer is disabled. Preserve the source train mode. Capture initial random
parameters/buffers/RNG before each probe, record the expected BN buffer changes,
then restore exactly before the next probe. Require unchanged parameters/manual
moments throughout and exact buffers after restoration. Disabling BN, using
eval-mode training or insisting that its live forward leaves buffers unchanged
would change the native code path. Label-free serving uses source eval mode.

The author source saves no selected model checkpoints. Do not invent selected
checkpoints or apply standardized engine resume criteria to its 80 processes.
Per-process native processed caches, source/raw copies, quarantined stdout/stderr
and 100-epoch TRAIN/VALID progress need disk capacity. Ten seeded recipes retain
the declared external seed wrapper; a random-state resource check is not a
published-score replication or fresh trained state.

## Serialization, restore and endpoint gates

For one widest exact node case per constructor topology, use the frozen engine
`capture_checkpoint`/`restore_checkpoint` directly on an engineering-only ID,
step 0, validation={}, decision={}. Do not run the engine fit loop, register a
fit, select a checkpoint or produce a phase gate. Supply the real adapter's
source external capture/restore and synthetic placeholder SELECT shape only;
never call validate or open actual SELECT. The exact factory/data-authority
replacement and source/import pins must be explicitly sealed and reviewed.

Serialize `DISPOSABLE_RANDOM_ENGINEERING_STATE.pt` inside the proposal case
directory. Include zero full-size Adam state; record serialized bytes, clone,
atomic save, load and verified restore time. Restore into the same exact
constructor after deliberately changing only disposable state, then compare
model, optimizer, private/global RNG, modes, factor member selectors, endpoint
counters and weight-table ancestry. Parameter/manual-moment hashes must remain
identical across every zero-step callback. This helper verifies the engine's
exact serialized state restoration; it does not require bitwise repeated CUDA
gradient identity. Existing numerical source replay tests retain their original
scope. Remove only this disposable snapshot after receipt/hash closure. This
qualifies random state restore and I/O; it does not qualify trained-state replay
or convergence.

The existing node exporter constructs, restores, forwards and frees one I4 body
at a time. It retains CPU role logits, so do not budget four simultaneously
resident CUDA model copies for I4 export. Qualify its peak CPU role tensor
shape/serialization with dummy shape-matched values and ordered synthetic IDs,
without using actual CAL/evaluate targets or claiming a real bank export.
Account all four separate constructor/restore/serving costs. Source inspection
shows mean-logit original GNNM and mean-probability adapted methods remain distinct.

Use measured checkpoint bytes to replace rough padding in retained-disk and
gross-write projections. Include selected/final/recovery, embedded selected
copies, atomic write headroom, prefix files, traces, parameter/body differences,
CPU bank tensors and graph/provider caches. Whole-master free disk and inode
capacity need a read-only project filesystem check before GO. A successful
disposable save cannot prove that tens of thousands of fits fit on disk.

## Two-slot gate and cost interpretation

Record actual other CUDA PIDs and usage; preserve Physics ownership. Use normal
host processes and only current-child allocator caps. Do not touch other jobs.
Take the larger sampled NVIDIA envelope and Torch peak reservation, add measured
nonallocator overhead, then explicit 15% plus 512 MiB headroom. Two starts are
admitted only when active reservations plus the new reservation fit both the
current free VRAM and declared worker capacity. Count CPU threads, RAM and disk
write overlap as well as VRAM.

After the 32 single-process cases, propose two finite pairs: the largest admitted
node reservation pair and one BatchNorm/native molecular pair. Each uses two
fresh processes, a synchronized start and four zero-step callbacks per process.
Measure per-child and aggregate elapsed time, overlap, VRAM, RSS and disposable
I/O. The allocation's pair results cannot establish server77 throughput or
capacity. When the user authorizes 18.77 again, verify that hardware and replay
only the necessary resource pair/maximum probes there before final assignment.
The current source supports two worker slots; two fits per GPU require admission
on the actual GPU. A failing pair remains explicit and needs root scheduling
revision before the user's final approval.

Zero-step forward/backward times omit native optimizer kernels, validation
metrics, checkpoint growth, full-epoch data loading, CAL/evaluation and failed
attempt costs. Use them as scheduling estimates with explicit allowance. No
exact ETA is defensible from the current placeholder weights or these probes.
Native OGB's registered 4,268,000 optimizer calls are an upper bound; source
singleton skips make exact executed calls unresolved before approved execution.

## Prelaunch versus scientific execution

Required prelaunch gates are: exact complete method/data/source coverage,
original-five exclusion and original GNNM new-data coverage, no-update source
interception, model/provider compatibility, representative memory admission,
random-state external restore, measured retention/write capacity, concurrency
admission, frozen source/helper/environment/data hashes, terminal engineering
cleanup, final measured-cost assignments and explicit user GO.

During the approved study, require actual optimizer accounting, full budget,
TRAIN/SELECT convergence traces, boundary-selected-checkpoint diagnosis,
trained-state resume/restore and selected ancestry, finite model/gradient states,
whole-cohort barriers and uncertainty. Native author convergence is judged from
its complete 100-epoch source curve, without converting its source protocol to
the standardized optimizer-update controller. These are scientific diagnostics;
they cannot be replaced by extra unapproved pilot fits during preparation.

All failures/costs are retained. No outcomes were inspected to choose or prune
these engineering cases. Roman Empire, Amazon Ratings, Minesweeper, Questions
and Tolokers are excluded from all methods. Original GNNM remains in both node
packets and all compatible new-data standardized studies. Original paper scores
are unchanged.
