# MolHIV GIN-VN whole-epoch GPU resource gate — source only

**Not launched. Root admission and an owned GPU slot are required.** This
package fills one missing resource workload from the accepted budget. It does
not repeat the already passed molecular v4 resume/selection lifecycle.

## Frozen representative

Use the exact proposed MolHIV/GIN-VN/dropout0.5 source recipe and existing
data root from the expanded v3 descriptor index. The descriptor itself is
inherited unchanged from generated_v1, with SHA verification recorded in
`SOURCE_DESCRIPTOR_METADATA.json`. This is the source's GIN with virtual node,
five layers, width300, native full atom/bond features, mean pooling,
AdamLR.001/WD0 and batch32. Official scaffold TRAIN32901 and VALID4113 remain
whole. Native operators, dimensions, skip law, optimizer and preprocessing stay
unchanged.

The shared method is the existing source-supported
`interaction_input_conditionalJ`: four boundary routes, private input
correction and conditional prediction aggregation, trained with its declared
joint objective. One source-sized update uses four native GNN forwards, one
backward collection and one Adam transition. This gives useful observation of
virtual-node and wrapper/aggregation work without adding a new method or
pretending it dominates every other recipe.

Engineering seed101/split0 and a one-epoch horizon are fixed. This is neither
a new scientific pilot nor a paper-source reproduction. Its AUC is an
engineering diagnostic and cannot be promoted as an accuracy finding.

## Exact work and unchanged execution

`qualify.py` invokes the real, unchanged `runtime_v4/run_fit.py:run` once,
without a resume argument. Instrumentation observes actual native calls;
there is no parallel training implementation or substituted finalizer.

| Work | Frozen count |
| --- | ---: |
| Native model construction | 1 |
| Full TRAIN epochs | 1 |
| TRAIN graphs / optimizer maps | 32,901 / 1,029 |
| TRAIN final batch | 5 graphs |
| TRAIN native GNN forwards / backwards | 4,116 / 1,029 |
| Whole VALID calls | 2 |
| VALID graphs / batches per call | 4,113 / 129 |
| VALID native GNN forwards, both calls | 1,032 |
| Total native GNN forwards | 5,148 |
| Ordinary selected finalization state/RNG loads | 1 each |
| Authenticated selected mean exports | 1 |
| TRAIN diagnostics, interrupted resumes, floating replay, TEST scoring | 0 |
| Automatic retries | 0 |

The two VALID calls are required production behavior: epoch selection and
selected-checkpoint finalization. The sole state/RNG load is that finalization,
not a separately constructed model or repeated training trajectory. Selection
saves its own epoch1 checkpoint, recovery checkpoint and actual/same-bank-mean
VALID export. No extra forward is introduced for mean aggregation.

The script stops on failure and preserves the output. There is one fresh
literal output for this gate. No automatic repair, second attempt, other
architecture, dataset, method or seed is authorized by the plan.

## Resource observations and limits

The watchdog is **1,800 seconds**, with an outer root supervisor recommended
at1,830seconds and escalation limited to its own gate process. This is a work
bound, not an ETA. CPU threads2/interop1 match the current molecular gate.

The proposed, unmeasured admission envelope is GPU8GiB with at least10GiB
currently free, own RSS4GiB with at least6GiB conservative CPU headroom, and
1GiB free repository disk. These are review assumptions, not measured bounds.
Root must account for peer growth and may decline admission. The existing
resource preflight refuses becoming a third CUDA process. No host, mount,
namespace, cache or peer-process setting is changed.

Save per-update time/work/component-gradient receipts, exact attempted and
completed forward/backward/optimizer counters, full-epoch and both VALID
timings, selected artifact sizes, allocator peaks, sampled own GPU-process
usage/RSS, RUsage peak RSS, and allocation-wide cgroup/cache/headroom. Native
forward hooks also record maximum TRAIN/VALID batch node and edge counts
without adding forwards. The existing accepted sampler is reused unchanged.

These measurements cannot establish dominance over other seeds, graph batches,
future epochs, GCN operators, dropout settings or methods. One batch order is
not an exhaustive native-shape maximum. Sampled process observations can miss
spikes; allocation-wide cache includes peers. Final file sizes are not a
continuous atomic-write disk peak. No concurrency or77portability claim follows.
An uncovered branch remains explicit rather than triggering a preset matrix.

## Preparation provenance and later admission

Source preparation fetched only the exact v3 descriptor-index entry and its
byte-pinned descriptor. It streamed SHA256 over the three referenced existing
MolHIV processed/TRAIN/VALID files inside the allocation repository, without
deserializing tensors or starting models. Exact TRAIN/VALID `.pt` role
alternatives are confirmed absent and must remain absent, so the native loader
uses the pinned CSV roles. The official processed graphstore
contains the public dataset normally; the native port indexes only authenticated
TRAIN and VALID roles and never creates a TEST view or computes TEST metrics.

`INPUTS.json` pins 102 source/metadata files and the three data files, including
unchanged runtime, source/native dependencies and interaction package
manifests. `SOURCE_CHECKS.json` records AST/schema/count checks only. No native
test, forward, optimizer update or GPU reservation occurred during preparation.

Root must publish/review this source, then create `ROOT_ADMISSION.json` with
`status: admitted`, the exact `INPUTS_sha256` and `qualifier_sha256`, the
published source commit, `one_owned_GPU_slot: true`, `automatic_retries: 0`,
and the exact output from INPUTS. The script refuses native imports otherwise.
The admission template is intentionally not an admission.

After admission, root may invoke the existing repository Python interpreter
with `CUDA_VISIBLE_DEVICES=0`, this qualifier, the literal repository root and
the exact INPUTS output. Guarded allocation execution is mandatory. Root owns
transfer, publication, launching and collecting results.
