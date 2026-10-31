# MolHIV development pilot

`molhiv_pilot.py` provides `freeze`, `preflight`, `train`, and `replay`. Its default command is a CPU engineering preflight with one complete training minibatch. Full fits are reserved for the coordinator after a prospective protocol is frozen and an authorized GPU is idle.

## Fixed study contract

The recipe is width 128, three residual bond-aware GINE blocks with internal MLP multiplier 2, four members, node LayerNorm, mean graph pooling, dropout 0.1, and mean raw-logit pooling. `base` has one member; `multihead` has one hidden propagation path. BCE gives each labeled graph equal weight, then averages over members. Optimization is AdamW with learning rate 0.001, weight decay 0, epsilon 1e-8, batch size 128, 100 epochs, float32, no AMP, and no loader workers.

Supported arms are `base`, `gnnm`, `untied`, `first_private`, `middle_private`, `last_private`, `multihead`, and `ens`. The tied and copied-private arms start from the same initialization for the same seed. Private blocks include their BondEncoder; AtomEncoder and boundary factors remain shared. ENS initializes separate complete models. See `GINE_MODEL_SPEC.md` for the model interface and intervention details.

The official training split contains 32,901 graphs; the complete official validation split contains 4,113 graphs. Training conditions use a nested 10% subset (3,290 graphs) and the complete training split. Subset construction uses training labels only: floor the total budget, assign binary-class quotas by largest fractional remainder, permute classes with seeds 101/102, and retain sorted official training positions. This subset is common across optimizer seeds 0/1/2 and all arms. The two budgets are conditions on one dataset, not independent tasks.

Checkpoint selection is maximum pooled validation ROC-AUC, then minimum pooled BCE, then the earliest epoch. ROC-AUC uses the official OGB evaluator. All arms have one fixed recipe and the same 100 validation opportunities; there is no hyperparameter search interface.

## Commands

Run from the permitted repository with its existing torch/PyG/OGB interpreter. Every input/output path must resolve inside this phase's `new_tasks` directory. Output files and run directories must be new; the runner refuses to overwrite previous attempts.

```sh
MOLHIV_PHASE="$PWD/postsubmission_research_20260930"
MOLHIV_RUNNER="$MOLHIV_PHASE/new_tasks/molhiv_pilot.py"
MOLHIV_DATA="$MOLHIV_PHASE/new_tasks/data/molhiv/frozen_manifest.json"
MOLHIV_PROTOCOL="$MOLHIV_PHASE/new_tasks/protocols/molhiv_pilot_v1.json"
```

First prepare the data freeze described in `FREEZER_USAGE.md`. The runner verifies its archive, raw/processed files, provider snapshots, and the installed OGB dataset-provider source before proceeding. Frozen paths belong to the environment that made the data freeze; relocating the cache requires a new valid freeze.

Synthetic helper verification uses no dataset, model fit, validation scoring, or CUDA context:

```sh
PYTHONDONTWRITEBYTECODE=1 python "$MOLHIV_PHASE/new_tasks/verify_molhiv_pilot.py"
```

Create the immutable prospective protocol before any full fit. `--arms` may restrict the registered comparison; without it, all eight arms are registered. All seeds and both label budgets are fixed in the recipe.

```sh
PYTHONDONTWRITEBYTECODE=1 python "$MOLHIV_RUNNER" freeze \
  --data-manifest "$MOLHIV_DATA" --output "$MOLHIV_PROTOCOL"
```

The protocol binds the source files, data manifest, package/environment report, train/validation index hashes, train-only budget hashes, recipe, arms, seeds, and UTC creation time. A changed bound value requires a new prospective protocol before fitting.

The default CPU preflight uses one full 128-graph minibatch from the selected training condition. It checks finite loss and gradients, performs one AdamW step, and records batch shape/counts. It produces no validation predictions or research performance evidence.

```sh
PYTHONDONTWRITEBYTECODE=1 python "$MOLHIV_RUNNER" \
  --data-manifest "$MOLHIV_DATA" --arm gnnm --seed 0 --label-fraction 0.1 \
  --output "$MOLHIV_PHASE/new_tasks/preflights/molhiv_gnnm_seed0_frac01_v1"
```

For a full fit, the coordinator supplies an explicitly authorized idle physical GPU index. Replace the placeholder only after authorization and the idle check. The runner requires exactly that one GPU in `CUDA_VISIBLE_DEVICES`, logical `cuda:0`, at most 100 MiB already used, zero GPU utilization, and no active compute applications. It checks these conditions before moving the model to CUDA. An idle check is a launch guard, not a GPU reservation; the coordinator must prevent simultaneous launches on that device.

```sh
MOLHIV_GPU="REPLACE_WITH_AUTHORIZED_IDLE_GPU_INDEX"
CUDA_VISIBLE_DEVICES="$MOLHIV_GPU" PYTHONDONTWRITEBYTECODE=1 \
  python "$MOLHIV_RUNNER" train \
  --data-manifest "$MOLHIV_DATA" --protocol "$MOLHIV_PROTOCOL" \
  --arm middle_private --seed 0 --label-fraction 0.1 \
  --device cuda:0 --physical-gpu "$MOLHIV_GPU" \
  --output "$MOLHIV_PHASE/new_tasks/runs/molhiv_middle_private_seed0_frac01_v1"
```

CPU replay verifies a completed fit using its archived adapter and selected checkpoint, recomputes the complete validation predictions, and audits all 100 retained epoch metrics and the selection rule. Its output must be outside the immutable fitted run. If the current runner/recipe differs, invoke the runner archived under the run's `source/new_tasks` directory with the same arguments.

```sh
PYTHONDONTWRITEBYTECODE=1 python "$MOLHIV_RUNNER" replay \
  --data-manifest "$MOLHIV_DATA" \
  --run "$MOLHIV_PHASE/new_tasks/runs/molhiv_middle_private_seed0_frac01_v1" \
  --output "$MOLHIV_PHASE/new_tasks/replays/molhiv_middle_private_seed0_frac01_v1"
```

## Preserved evidence

Each preflight/fit retains source snapshots for the runner, molecular adapter, and original boundary implementation; runtime class source snapshots for AtomEncoder, BondEncoder, GINEConv, and the OGB provider; the data manifest; the environment; exact training positions/global indices; the initial state/RNG; and a hashed contract.

A completed fit additionally retains the prospective protocol, every epoch's member logits on all validation graphs, validation targets/global indices, `validation_trace.csv`, the selected model/optimizer/RNG checkpoint, selected member logits and metrics, the final checkpoint, wall-clock times, and peak allocated/reserved CUDA memory. `artifact_manifest.json` covers every retained file except itself. Fit and preflight failures after artifact preparation retain `failure.json` and a manifest; guard/setup failures exit before the fit and may have no run directory. Keep command/stdout/stderr and process exit status in the coordinator's attempt log as well.

Replay verifies exact manifest coverage and hashes; all retained epoch scores and selection; agreement between the result, selected report, checkpoint metrics, and selected epoch logits; and selected checkpoint predictions on CPU. CPU versus CUDA member logits use absolute/relative tolerances of 2e-4 and report maximum error. This is a numerical checkpoint replay, not a bitwise replay of the complete stochastic training trajectory. The runner records its environment/RNG but does not enable deterministic CUDA algorithms.

Timing covers the development fit, including validation and artifact writing. A separate end-to-end time includes preparation. CUDA memory is measured after initial artifact preparation. These fields do not supply inference latency or establish a deployment speedup; use the separate efficiency protocol for those claims.

## Test and confirmation scope

OGB's processed cache contains the complete graph archive. The runner instantiates and iterates official train/validation subsets only. No test subset loader, test-label summary, or test-scoring command is provided. Confirmation datasets are not accessed. Development choices must be frozen before independent confirmation is opened.

## Verification status

Local verification is limited to Python AST syntax checks because the local interpreter lacks torch/PyG/OGB. The coordinator has reported successful CPU synthetic GINE interface/initialization/gradient checks; `GINE_CHECKS_v1.json` records that result. MolHIV runner helper checks and real one-minibatch preflight require the coordinator's existing runtime; they must be reported separately from a successful full fit or replay. No full MolHIV fit has been launched by this agent.
