# Training-only block selection for molecular GINE

This packet transfers the existing sharing selector to graph-level molecular prediction. A tied four-member GINE is fitted only on training fold P. The three gradient checkpoints use training fold Q to decide whether one complete GINE block should become private. The selector can choose no block. There are no validation metrics or predictive results in this packet.

## What changes

The numerical selector is unchanged: nine train-only recurrence/noise measurements, the same eligibility rule, and the same added-parameter ranking. A probed block includes its node normalization, learned bond embeddings, GINE epsilon, and MLP. Copying a block for four members adds exactly three block copies. All three positions have equal actual parameter cost; first, middle and last must remain fixed predictive controls.

Scaffold and duplicate-molecule groups remain together in P/Q and every A/B split. All empty scaffolds remain one group. Large groups may produce uneven folds; the export and freeze report this rather than breaking a group to get a better split. Each observed graph has equal loss weight; node count, label count and group size do not become hidden loss weights. The batched collector preserves that reduction, including a short final minibatch.

## Files

- `source/`: unchanged, hashed copies of GINE, BatchEnsemble, loss guard and recurrence policy sources.
- `make_train_bundle.py`: separate existing-cache exporter; no provider constructor or download.
- `group_folds.py`: standard-library deterministic grouped allocation.
- `gine_collector.py`: equal-graph batched gradients and no-mutation records.
- `probe.py`: prospective freeze, GPU-guarded P-only fitting, hard selector lock, and CPU autograd/decision replay.
- `verify_cpu.py`: unequal-node/label-count synthetic GINE autograd gate. It takes no optimizer step and does not use a dataset.
- `PLAN.json`: scope, controls, locking and remaining evidence requirements.

## Runtime commands

Run from the authorized remote research phase, under the repaired repository environment and supervisor. The first command is:

```sh
.venv/bin/python experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_v1/verify_cpu.py \
  --report experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_v1/engineering/CPU_GATE_run01.json
```

Here `.venv/bin/python` is relative to the repository root. If the gate fails, preserve its supervisor logs and amend this packet before using it. Local source checks do not establish a passing autograd gate.

After the root supplies a separate, repository-local RDKit runtime, create the training-only bundle. Every input must be inside the research phase and exactly covered by the existing frozen shareable manifest:

```sh
python sharing_policy/gine_transfer_v1/make_train_bundle.py \
  --data-manifest new_tasks/data/molhiv/frozen_shareable_manifest_v2.json \
  --processed new_tasks/data/molhiv/provider/ogbg_molhiv/processed/geometric_data_processed.pt \
  --split-train new_tasks/data/molhiv/provider/ogbg_molhiv/split/scaffold/train.csv.gz \
  --molecule-map new_tasks/data/molhiv/provider/ogbg_molhiv/mapping/mol.csv.gz \
  --output sharing_policy/gine_transfer_v1/data/official_train_v1.pt
python sharing_policy/gine_transfer_v1/probe.py freeze \
  --train-bundle sharing_policy/gine_transfer_v1/data/official_train_v1.pt \
  --output sharing_policy/gine_transfer_v1/runtime/protocol_v1.json
```

The exporter explicitly records that `torch.load` materializes the all-target tensor in the existing cache. It does not index, summarize, hash or export held-out targets. The probe reads only its training-only output. Thus the probe is isolated from test targets; the export is **not** certified to avoid all-target storage materialization.

Run the gate again with `--train-bundle sharing_policy/gine_transfer_v1/data/official_train_v1.pt` and a fresh `--report` filename after export. This adds a real128-graph training-batch autograd check at production width128. It compares grouped graph-mean gradients with independently reduced direct gradients and checks replayable Gram statistics, without an optimizer step. MolHIV has one target per graph; unequal finite-target counts are checked separately by the synthetic GINE gate. The real batch is an engineering check, not an accuracy pilot.

Once an explicitly authorized idle GPU is available, invoke one seed/fraction cell:

```sh
CUDA_VISIBLE_DEVICES=0 python sharing_policy/gine_transfer_v1/probe.py probe \
  --train-bundle sharing_policy/gine_transfer_v1/data/official_train_v1.pt \
  --protocol sharing_policy/gine_transfer_v1/runtime/protocol_v1.json \
  --seed 0 --label-fraction 0.1 --device cuda:0 --physical-gpu 0 \
  --output sharing_policy/gine_transfer_v1/runs/seed0_label10_run01
python sharing_policy/gine_transfer_v1/probe.py replay \
  --train-bundle sharing_policy/gine_transfer_v1/data/official_train_v1.pt \
  --run sharing_policy/gine_transfer_v1/runs/seed0_label10_run01 \
  --output sharing_policy/gine_transfer_v1/replays/seed0_label10_run01.json
```

GPU0 in this example is a placeholder, not an allocation. Finish all three seeds at both fractions, and audit all six locks, before any predictive validation comparison. All outputs use exclusive creation. Every checkpoint retains optimizer and RNG state; source snapshots, masks/folds, losses, costs, failures and decisions remain immutable.

## Limits

No CPU autograd gate or real-data export has run on this Mac: its available Python lacks torch, PyG, OGB and RDKit. The source-only checks are recorded separately. The production code needs a supervised real train-batch gate after export, then a runtime freeze and P-only fits. The predictor amendment is still required to compare selected against tied, first/middle/last, full-private and relevant size-matched controls. Training-only selection is not evidence of accuracy gain. MolHIV is development; its nested label conditions and seeds are not independent task confirmation. Existing reported scores and frozen pilot protocols are unchanged.
