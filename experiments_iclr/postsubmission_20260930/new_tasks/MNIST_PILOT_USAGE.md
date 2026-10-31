# MNIST-superpixel architecture transfer pilot

This development study tests transfer of the common/narrow architecture from PPI to a distinct image-graph corpus. It is not a claim of state-of-the-art MNIST performance. `mnist_adapters.py` composes the frozen SAGE baseline adapters with the repaired common/narrow source. `mnist_pilot.py` supplies prospective freeze, CPU preflight, guarded full training, and CPU checkpoint/evidence replay. No MolHIV or frozen source file is changed.

## Data and readout

Use the already frozen released MNIST-superpixel splits: 55,000 training, 5,000 validation, and 10,000 test graphs. The runner loads train/validation only. It concatenates released node intensity and the two released coordinates into float `x[N,3]`, with no rescaling. It preserves released directed edge entries and edge attributes. All SAGE variants ignore edge attributes and add no implicit self-loops; their separate root transformation remains part of SAGE.

Every arm uses **mean node-logit pooling**: apply the architecture's node normalization/readout to get member node logits `[M,N,10]`, then average those logits within each graph to obtain `[M,G,10]`. The source architecture's LayerNorm is at individual nodes. Ensemble predictions then average the members' raw graph logits. The readout choice is common to the candidate, source, multihead, and packed controls; it is part of this prospective recipe and need not match recipes used by published MNIST leaderboard methods.

Loss is mean cross-entropy over graphs and members. Graphs have equal loss weight regardless of node count. There is no node subsampling.

## Fixed comparison

The four common/narrow cells use width 128, rank 16, four members, and three layers. For a given seed they consume the same shared/narrow initialization draws. Feedback-disabled expansion maps are absent from registered parameters and optimizer state.

| Arm | Centered coordinates | Mean feedback | Parameters | Adjacency channels/layer |
|---|---:|---:|---:|---:|
| `centered_feedback` | Yes | Yes | 281,006 | 176 |
| `centered_no_feedback` | Yes | No | 278,958 | 176 |
| `uncentered_feedback` | No | Yes | 281,006 | 192 |
| `uncentered_no_feedback` | No | No | 278,958 | 192 |

The centered model folds the discarded mean of the raw narrow state into the common state. The uncentered feedback control feeds the mean current narrow update into the common state while retaining uncentered narrow states. These are the repaired source's four controls. This adapter does not introduce a new feedback mechanism.

The controls are frozen-source `gnnm`, copied-private `untied`/`first_private`/`middle_private`/`last_private`, `base`, `multihead`, source `ens`, source-parameter-copied `packed_gnnm`/`packed_full`, `packed_channels`, `packed_parameters`, and `base_parameters`. Tied/private arms share boundary projectors and start from the same source initialization; private blocks are copied before optimizer construction. The independent ensemble uses independently initialized complete models. Packed copies are checked for evaluation equivalence; stochastic dropout trajectories need not be identical across execution layouts.

Matching targets are fixed to `centered_feedback`, not selected from validation outcomes. Channel matching gives four independent width-44 models at 176 total adjacency channels. Parameter matching chooses the closest exact source-style count over integer widths 8–512, breaking ties toward the smaller width, with no label/score access:

| Control | Width | Parameters | Difference from 281,006 |
|---|---:|---:|---:|
| `packed_channels` | 44/member | 121,656 | Channel match only |
| `packed_parameters` | 67/member | 277,688 | −1.1808% |
| `base_parameters` | 136 | 281,666 | +0.2349% |

The displayed counts are derived from source structure and asserted by the synthetic verifier. The prospective freeze also records every registered arm's actual count, execution path, and matching metadata. Adjacency channels are structural counts, not measured latency or memory.

## Recipe and selection

Use AdamW, learning rate 0.001, weight decay 0, epsilon 1e-8, dropout 0.2, source SAGE MLP multiplier 1, float32, no AMP, batch size 128, zero loader workers, and 100 epochs. Optimizer seeds are 0/1/2. Epoch graph order uses `19709 + optimizer_seed*1000 + epoch`.

Training conditions are nested 10%/100% graph-label budgets: 5,500/55,000 graphs. Stratification uses training labels only. Allocate floor(total × 0.1), distribute class quotas by largest fractional remainder with digit-order tie breaks, permute the ten classes using seeds 101–110, and retain sorted split-local official training positions. These positions are common to every arm and optimizer seed. Budget levels are conditions on one corpus; they are not independent tasks or a best-level selection axis.

Use all 5,000 validation graphs after each epoch. Select maximum pooled validation accuracy, then minimum pooled cross-entropy, then earliest epoch. Every registered arm gets the same fixed recipe and 100 validation opportunities. No hyperparameter search is exposed. A prospective freeze may register a smaller explicit arm list before fitting; there is no automatic arm, rank, width, or budget search.

## Commands

Run from the authorized repository with its existing interpreter. All inputs and outputs must remain within this phase's `new_tasks` directory. Do not install dependencies, recreate the data freeze, or use an occupied GPU. Existing output paths are rejected.

```sh
MNIST_PHASE="$PWD/postsubmission_research_20260930"
MNIST_RUNNER="$MNIST_PHASE/new_tasks/mnist_pilot.py"
MNIST_DATA="$MNIST_PHASE/new_tasks/data/mnist_superpixels/frozen_shareable_manifest_v2.json"
MNIST_PROTOCOL="$MNIST_PHASE/new_tasks/protocols/mnist_pilot_v1.json"
```

The default is the coordinator's shareable derivative of the existing data freeze, with individual test-label digests removed and the original private manifest hash linked. Explicit original `frozen_manifest.json` inputs remain valid for engineering history. No new acquisition or processing is required.

Synthetic CPU verification accesses no dataset and performs no optimizer step or real scoring:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python "$MNIST_PHASE/new_tasks/verify_mnist_pilot.py"
```

The default CPU preflight performs one forward/backward/AdamW step on **128 complete training graphs**, checks that every registered parameter receives a finite gradient, and creates no validation predictions. Its training loss and timing are engineering checks.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python "$MNIST_RUNNER" \
  --data-manifest "$MNIST_DATA" --arm centered_feedback --seed 0 --label-fraction 0.1 \
  --device cpu --output "$MNIST_PHASE/new_tasks/preflights/mnist_centered_seed0_frac01_cpu_v1"
```

Freeze code, data, environment, recipe, arm/matching plan, training budgets, and official development indices before any full fit:

```sh
PYTHONDONTWRITEBYTECODE=1 python "$MNIST_RUNNER" freeze \
  --data-manifest "$MNIST_DATA" --output "$MNIST_PROTOCOL"
```

A full fit requires that exact prospective protocol and one explicitly authorized idle GPU. Replace the placeholder only when the coordinator has selected a free authorized device. The runner checks single-device visibility, memory already used ≤100 MiB, zero GPU utilization, and no active compute applications before moving the model to CUDA. The coordinator must prevent concurrent launches; this guard does not reserve a device.

```sh
MNIST_GPU="REPLACE_WITH_AUTHORIZED_IDLE_GPU_INDEX"
CUDA_VISIBLE_DEVICES="$MNIST_GPU" PYTHONDONTWRITEBYTECODE=1 \
  python "$MNIST_RUNNER" train --data-manifest "$MNIST_DATA" --protocol "$MNIST_PROTOCOL" \
  --arm centered_feedback --seed 0 --label-fraction 0.1 --device cuda:0 --physical-gpu "$MNIST_GPU" \
  --output "$MNIST_PHASE/new_tasks/runs/mnist_centered_seed0_frac01_v1"
```

Replay a complete fit on CPU, writing outside the immutable run directory:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python "$MNIST_RUNNER" replay \
  --data-manifest "$MNIST_DATA" --run "$MNIST_PHASE/new_tasks/runs/mnist_centered_seed0_frac01_v1" \
  --output "$MNIST_PHASE/new_tasks/replays/mnist_centered_seed0_frac01_v1"
```

Use the archived runner under `run/source/new_tasks` if the current runner/recipe changed. The archived adapter resolves its own archived baseline/common-narrow/model dependencies. Frozen absolute data paths belong to the environment that made the data freeze; they must not be silently rewritten.

## Evidence and costs

Preflight/fit artifacts include snapshots of the runner, graph adapter, frozen baseline adapters, repaired common/narrow paths, original model source, runtime provider/SAGEConv/LayerNorm class sources, the data manifest, environment, exact train positions/indices, initial model/RNG, and hashed contract. Completed fits additionally retain the prospective protocol, all 100 epochs of complete validation member logits, validation targets/indices, selection trace, selected model/optimizer/RNG, selected logits/metrics, and final model/optimizer/RNG. At four members, the raw 100-epoch validation logits occupy about 80 MB before file-format overhead.

`artifact_manifest.json` covers every retained file except itself. Failure records and manifests preserve unsuccessful fit/preflight attempts after artifact preparation. Setup/guard failures may exit before creating a run directory; keep command, stdout/stderr, and exit status in the coordinator's external attempt log as well. Do not add logs or other unregistered files inside a completed immutable run.

The preflight records whole-batch shape, one-step time, sparse aggregation widths, and host operator-construction time. Full fits record per-epoch training and complete validation seconds, total/end-to-end seconds, and peak allocated/reserved CUDA memory. Timing includes data transfers/operator preparation; total time also includes evidence writing. These fields do not establish deployment inference latency or speedup. The composable adapter is available to a separate fixed-batch cost profiler.

Replay verifies exact file/hash coverage, all retained epoch metrics, checkpoint selection, selected result/report/checkpoint/evidence agreement, and CPU checkpoint predictions. CPU/CUDA member logits use absolute/relative tolerances 2e-4 and report maximum error. Replay is a numerical checkpoint audit, not a bitwise rerun of stochastic training; deterministic CUDA algorithms are not enabled.

## Runner test and confirmation scope

The released provider cache contains public test labels. The original data freezer loaded and fingerprinted test targets without predictive scoring. The original manifest stays private; the shareable derivative omits per-test-graph target digests. Existing archive/file hashes are retained. This runner loads train/validation files only, has no test subset loader, and supplies no test-evaluation command. Confirmation datasets are untouched. GOOD-CMNIST shares source images with MNIST and cannot serve as an independent image-corpus confirmation of this study.

## Verification status

Local checks are Python AST syntax and source/manifest inspection only; the local interpreter lacks torch/PyG. The coordinator runs synthetic helper checks and the real CPU minibatch preflight in the existing runtime. Their outcomes must be recorded separately from a complete development fit or replay. No MNIST model fit or scoring job has been launched by this agent.
