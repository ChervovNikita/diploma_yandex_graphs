# PubMed/Cheb: three disjoint GPU queues

The frozen24-fit study is divided into three complete eight-fit optimizer
cohorts. Each retains native bodies0–3, original GNNM/G0, uniform, structured
and permuted residuals on one physical GPU. M1 reuses body0; I4 averages the
four independently selected native bodies. There are18reported banks and
12selected shared-bank collectors. No original-paper experiment, HPO, TEST,
large-campaign fit, or new scientific setting is added.

| Worker | Cohort | Host | Physical GPU |
| --- | ---: | --- | --- |
|allocation/gpu0|101|anogena-2-0|GPU-44039938-fd82-41d2-fefd-de71514e2fac|
|server51/gpu0|202|olig|GPU-9c6c20fa-a620-479d-f72f-163ac74deef2|
|server51/gpu1|303|olig|GPU-30283973-140d-9e7b-e780-0fcc487dec54|

`CANONICAL.json` contains the original24descriptors and hashes; `SHARDS.json`
contains disjoint ordered IDs. Existing typed path relocation changes only
declared runtime paths. Execution jobs additionally include explicit source,
map, canonical-job and metadata-compatibility hashes. Original descriptors
and source files remain unchanged. Each checkpoint therefore binds its exact
execution view and remains traceable to the canonical job.

`binding.py` gives the frozen residual extension's exact descriptor validator
a host-specific metadata factory in each isolated fit/collection process.
Its transform/update/forward/state code remains the original code object.
The original native fit API, optimizer, whole-VALID selector, minimum2000,
patience100, cap10000, initialization offsets, split and recipe are retained.
Original `study.fit` checks own selected state and runtime/source dependencies.
Host/seed are coupled; retain within-cohort contrasts and disclose hardware.

## Review, sync and one-command starts

`TARGET_MAP_TEMPLATE.json` is closed (`approved=false`). Root creates a
separate actual map with the reviewed source-manifest SHA, publication commit,
host runtime/resource evidence, peer identities/growth reserves and approval.
All workers use identical map bytes. `SOURCE_BINDINGS.json` names58exact
phase-relative source/support files. `PARENT_INPUTS.json` names the three exact
data/basis/role payloads. No51readiness is inferred by the source template;
the separately retained actual qualification receipts belong to root.

After root review, publication, sync and actual-map approval, run one command
per GPU. Replace `MAP_SHA` with that actual map's SHA and use the actual map
filename in the commands. Starts detach immediately with stdin closed and
logs under each shard; training survives the MacLink/SSH connection closing.

```sh
# Allocation: repo root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_three_gpu_queue_v1/run_queue.py --worker allocation/gpu0 --target-map experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/PUBMED_THREE_GPU_ACTUAL_MAP.json --target-map-sha256 MAP_SHA

# Server51: repo root /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_three_gpu_queue_v1/run_queue.py --worker server51/gpu0 --target-map experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/PUBMED_THREE_GPU_ACTUAL_MAP.json --target-map-sha256 MAP_SHA
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_three_gpu_queue_v1/run_queue.py --worker server51/gpu1 --target-map experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/PUBMED_THREE_GPU_ACTUAL_MAP.json --target-map-sha256 MAP_SHA
```

The owner authenticates host/inventory, original repository virtualenv,
published source, exact files/data and root approvals. Workers must map to
three distinct host/GPU pairs. Each owner holds its GPU slot; a child must
have the exact CUDA UUID mask, live authenticated parent claim and held lock.
One scientific child runs per GPU. Larger two-child concurrency remains a
separate measured admission; this package does not claim six running children.

Outputs are fresh, immutable attempts under
`benchmark_expansion_20261015_v1/results/pubmed_three_gpu_v1/WORKER_NAME`.
Existing output/owner claims refuse duplicates. No retries, skip, resume,
seed replacement, or partial-family quality summary is implemented. Each fit
has a12-hour bound; the owner has96hours. Root's map preserves reviewed8GiB
GPU/host/disk envelopes and peer growth. Failures and elapsed child costs are
kept. TERM reaches only the owned child's process group, then escalates after
20seconds; no peer or host setting is changed.

## Selected collection and portable joint analysis

After its eight fits succeed, each owner runs four zero-training selected
collectors, then exports a compact `bundle/`. `pipeline.selected_core` retains
the original collector's three runtime imports and unchanged scientific body
from `started` through its return. It makes four native evaluation forwards,
blocks TRAIN and optimizer calls, restores each own selected checkpoint and
preserves the saved selected mean as scoring authority. Reconstruction
differences are diagnostic warnings, never replacement scores or retries.

Each bundle contains original selected logit/member exports, copied metadata
with origin hashes, genuine saved I4 aggregation, and six portable prediction
banks. Checkpoints remain on their training server. Bundle paths can move;
their original evidence headers and receipts are not rewritten.

Copy all three complete `bundle/` directories into an aggregation directory
inside one project repository. Then run:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pubmed_three_gpu_queue_v1/analyze.py --root "$PWD" --target-map experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/PUBMED_THREE_GPU_ACTUAL_MAP.json --target-map-sha256 MAP_SHA --bundle PATH_TO_COHORT101_BUNDLE --bundle PATH_TO_COHORT202_BUNDLE --bundle PATH_TO_COHORT303_BUNDLE --output PATH_TO_FRESH_JOINT_ANALYSIS
```

Intake requires all24unique fits and12collectors before tensor/quality access.
It verifies copied bytes against original receipts, exact execution/canonical
jobs, selected epochs/checkpoints/exports, collector-to-fit links and zero
TRAIN work. It compares the portable banks against copied original selected
and route exports, including M1/body0 and exact I4 membership/mean. Original
provider IDs/label digests and stable scalar diagnostic functions are reused.
All42signed comparisons, per-class harms, means/sample SDs, outcomes, costs
and sanity warnings remain. This is exposed development on one graph/split,
not protected confirmation, independent graph replication or a novelty result.

`check_source.py` runs only scalar/AST/source checks:24canonical relocation
roundtrips, disjointness, unchanged scientific fields, duplicate GPU/wrong-mask
refusals, collector globals/imports, standard-library queue resolution and
incomplete intake refusal. It performs no Torch, model, data, host or remote
execution. Queue32 remains preserved; unfinished queue33 is excluded.
