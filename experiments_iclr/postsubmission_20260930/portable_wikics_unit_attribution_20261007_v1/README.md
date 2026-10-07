# Public WikiCS unit-factor attribution CLI

Run one explicit condition in your own environment using caller-supplied safe
TRAIN/development NPZ files, native Polynormer source and a fresh output path.
This wrapper uses the unchanged portable public v2 full driver and a byte-identical
copy of the v2 recomputation kernel. The original study sources remain unchanged.
External runs are labeled separately from the registered author Wiki12 study.

## Setup

Use Python3.11. Keep this folder beside
`portable_internal_be_public_interface_20261007_v2`, or pass that public directory
at any local path with `--public-interface`. Its original manifest/code seal is
checked. The bundled `recompute.py` is checked against its original v2 digest.
SOURCE_ORIGIN.json records provenance; its original study paths are not read at
runtime. All runtime inputs and outputs are caller-owned paths. The CLI uses
ordinary Python execution, with no host, GPU UUID, author release or private
evidence requirement.

Recorded v2 providers: NumPy1.26.4, Torch2.1.2+cu118, PyG2.7.0,
torch-scatter2.1.2+pt21cu118, torch-sparse0.6.18+pt21cu118 and OGB1.3.6.
Use Torch/scatter/sparse wheels matched to your operating system and CPU/CUDA
ABI; `requirements.txt` records package versions. Actual versions are recorded
in outputs rather than enforced as an author-host condition. CPU and CUDA
devices are supported by the original public Session; alternate environments
and this new public wrapper have not been numerically qualified.

For a Linux CUDA11.8 environment matching the recorded provider family, these
are setup commands for your own environment (not executed during preparation):

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install numpy==1.26.4 torch-geometric==2.7.0 ogb==1.3.6
```

For CPU, choose the corresponding CPU wheels. See the
[PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
The inherited model module imports OGB encoders, so OGB is required for WikiCS
even though the WikiCS metric is accuracy. Only the Polynormer native source is
loaded for this task; no NCN source is needed.

Obtain Polynormer from [the upstream repository](https://github.com/cornell-zhang/Polynormer)
at commit `fc8c276c9c5dfbd616d83f65338a3392188a5e08`. Supply its matching `model.py`
(SHA256 `9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8`).
No third-party native code is bundled here.

Supply complete safe NPZ roles, or use the public v2 converter on the official
WikiCS raw JSON you obtained independently:

```sh
python portable_internal_be_public_interface_20261007_v2/export.py wikics --raw-json data.json --output roles/wikics
```

The input loader checks complete WikiCS populations/shapes/domains. TRAIN is
`{x, edge_index, ids, y}` with11701 nodes,300 features,442907 prepared edges and580
TRAIN labels; development is `{ids, y}` with5274 objects. Development is the union
of official split0 validation and stopping masks. The public converter checks
public source/ordered-array pins. Caller NPZ serialization bytes need not match
author files; supplied data hashes and source-verification limits are recorded.
The trainer reads no TEST role or TEST metric. See public v2 DATA_FORMAT.txt for
complete data and conversion semantics.

## Run

```sh
python portable_wikics_unit_attribution_20261007_v1/train.py --condition combined --seed 6101 --device cuda:0 \
  --public-interface portable_internal_be_public_interface_20261007_v2 \
  --polynormer external/Polynormer/model.py --train roles/wikics/train.npz \
  --valid roles/wikics/valid.npz --output runs/wiki_combined_6101
```

Use `--device cpu` for CPU or another CUDA index in your environment. The default
seed is6101; other integer seeds may be supplied. The author's exploratory design
used6101/6203/6307. For a matched four-condition reproduction, predeclare your
seeds and keep device/runtime/data fixed across conditions within each seed.

| Condition | Supervised alignment | Residual member contrast |
|---|---:|---:|
| plain | 0 | 0 |
| alignment_only | .05 | 0 |
| residual_only | 0 | .05 |
| combined | .05 | .05 |

Every condition uses shared four-member unit factors and two full own-loss views,
all580 TRAIN labels, original Adam/settings and unchanged1100-epoch schedule:
100 local +1000 global, original joint local transition, strict-first best complete
development checkpoint selection, original mean class-probability serving. The
fixed auxiliary sample has at most512 original evenly spaced TRAIN objects and
temperature.2. Active original loss functions run unchanged; inactive terms are
zeroed by a per-session facade. There are no coefficient, model, horizon,
early-stop or resume options. TF32 is disabled and cuDNN benchmark is false;
Torch threads are2 and interop threads1, matching v2.

Recomputation evaluates two original shadow views under no_grad, computes joint
output cotangents once, restores the same member streams and replays each
member/view separately. Eight parameter VJPs accumulate before one Adam bank
update. Exact replay/shadow RNG endpoints and unchanged parameters before Adam
are checked. This costs16 training member forwards/update, with one backbone
graph live at a time. Replays reuse the original two stochastic views.
Process/time/memory management remains your responsibility.

## Outputs and interpretation

The original driver writes RUN/PROGRESS/VALID_TRACE, selected/local joint
snapshots, COMPLETE or FAILURE. Public condition, method/source identity, actual
providers, loss-call counts and execution costs enter run/progress/checkpoint
metadata. A complete cell charges17,600 training and4,400 development member
calls,1,100 joint output-cotangent collections,8,800 member VJPs and1,100 Adam
bank updates. PUBLIC_COST.json records actual attempt/return counts, update and
inclusive wall seconds, available CPU/RSS and CUDA peaks. Partial failure counts
remain visible. No automatic retry, next condition, orchestration or exact-fit
resume is provided.

Selected snapshots can remain local even after1100 training epochs. Their saved
Python `global` flag must be restored explicitly when serving; it is absent from
state_dict. Preserve original selected snapshots and selection rules in readout.
The same development population selected these checkpoints, so results carry
selection optimism and supply no independent TEST evidence. Treat three reused
optimizer seeds on one graph as exploratory; nodes and members are not additional
independent models. This source offers no novelty/confirmation or bitwise
author-execution-equivalence claim. Sparse kernels/hardware/provider builds and
the changed floating-point accumulation order can affect trajectories.

Preparation verified code seals, AST and stdlib-only CLI help. No model, data
conversion, numerical test, full run, remote access or scoring was executed for
this wrapper. Actual runtime qualification and root publication review remain
pending.
