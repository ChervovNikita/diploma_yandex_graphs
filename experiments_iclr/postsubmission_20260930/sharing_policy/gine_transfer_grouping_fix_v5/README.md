# Training-only molecular GINE: uniform grouping preparation

This sibling preserves both the failed strict-sanitization v2 exporter and the failed v3 aromatic-hydrogen fixture gate. Every official training molecule and its scaffold uses the pinned RDKit Cookbook partial-valence procedure plus the standard ADJUSTHS stage. This addition requires a new supervised runtime gate before export. V4 repaired aromatic nitrogen H bookkeeping but retained explicit input H atoms because the Boolean sanitize=False parser also disables H removal. An exhaustive16-fixture diagnostic isolated that one difference from the standard route. V5 uses explicit parser params sanitize=False/removeHs=True uniformly; standard isotope and isolated-H exceptions remain intact. Graph allocation, loss, probe and selector remain unchanged. CSV row position remains the official graph identity; mol_id is ignored. No graphs are dropped and exceptions never become singleton groups.

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
.venv/bin/python experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_grouping_fix_v5/verify_cpu.py \
  --report experiments_iclr/postsubmission_20260930/sharing_policy/gine_transfer_grouping_fix_v5/engineering/CPU_GATE_run01.json
```

Here `.venv/bin/python` is relative to the repository root. If the gate fails, preserve its supervisor logs and amend this packet before using it. Local source checks do not establish a passing autograd gate.

After the root supplies a separate, repository-local RDKit runtime, create the training-only bundle. Every input must be inside the research phase and exactly covered by the existing frozen shareable manifest:

```sh
python sharing_policy/gine_transfer_grouping_fix_v5/make_train_bundle.py \
  --data-manifest new_tasks/data/molhiv/frozen_shareable_manifest_v2.json \
  --processed new_tasks/data/molhiv/provider/ogbg_molhiv/processed/geometric_data_processed.pt \
  --split-train new_tasks/data/molhiv/provider/ogbg_molhiv/split/scaffold/train.csv.gz \
  --molecule-map new_tasks/data/molhiv/provider/ogbg_molhiv/mapping/mol.csv.gz \
  --output sharing_policy/gine_transfer_grouping_fix_v5/data/official_train_v1.pt
python sharing_policy/gine_transfer_grouping_fix_v5/probe.py freeze \
  --train-bundle sharing_policy/gine_transfer_grouping_fix_v5/data/official_train_v1.pt \
  --output sharing_policy/gine_transfer_grouping_fix_v5/runtime/protocol_v1.json
```

The exporter explicitly records that `torch.load` materializes the all-target tensor in the existing cache. CSV parsing can also internally materialize public heldout label columns and SMILES strings. Only official training tensor targets and train-row SMILES are selected, used for grouping or exported. No heldout target or SMILES value is inspected or printed for diagnostics. The probe reads only the training-only output. The exporter is **not** certified to avoid heldout field materialization.

Run the gate again with `--train-bundle sharing_policy/gine_transfer_grouping_fix_v5/data/official_train_v1.pt` and a fresh `--report` filename after export. This adds a real128-graph training-batch autograd check at production width128. It compares grouped graph-mean gradients with independently reduced direct gradients and checks replayable Gram statistics, without an optimizer step. MolHIV has one target per graph; unequal finite-target counts are checked separately by the synthetic GINE gate. The real batch is an engineering check, not an accuracy pilot.

Once an explicitly authorized idle GPU is available, invoke one seed/fraction cell:

```sh
CUDA_VISIBLE_DEVICES=0 python sharing_policy/gine_transfer_grouping_fix_v5/probe.py probe \
  --train-bundle sharing_policy/gine_transfer_grouping_fix_v5/data/official_train_v1.pt \
  --protocol sharing_policy/gine_transfer_grouping_fix_v5/runtime/protocol_v1.json \
  --seed 0 --label-fraction 0.1 --device cuda:0 --physical-gpu 0 \
  --output sharing_policy/gine_transfer_grouping_fix_v5/runs/seed0_label10_run01
python sharing_policy/gine_transfer_grouping_fix_v5/probe.py replay \
  --train-bundle sharing_policy/gine_transfer_grouping_fix_v5/data/official_train_v1.pt \
  --run sharing_policy/gine_transfer_grouping_fix_v5/runs/seed0_label10_run01 \
  --output sharing_policy/gine_transfer_grouping_fix_v5/replays/seed0_label10_run01.json
```

GPU0 in this example is a placeholder, not an allocation. Finish all three seeds at both fractions, and audit all six locks, before any predictive validation comparison. All outputs use exclusive creation. Every checkpoint retains optimizer and RNG state; source snapshots, masks/folds, losses, costs, failures and decisions remain immutable.

## Limits

No CPU autograd gate or real-data export has run on this Mac: its available Python lacks torch, PyG, OGB and RDKit. The source-only checks are recorded separately. The production code needs a supervised real train-batch gate after export, then a runtime freeze and P-only fits. The predictor amendment is still required to compare selected against tied, first/middle/last, full-private and relevant size-matched controls. Training-only selection is not evidence of accuracy gain. MolHIV is development; its nested label conditions and seeds are not independent task confirmation. Existing reported scores and frozen pilot protocols are unchanged.

## Mapping repair check

Run `python sharing_policy/gine_transfer_grouping_fix_v5/verify_mapping_identity.py --report sharing_policy/gine_transfer_grouping_fix_v5/engineering/MAPPING_ID_run01.json`. This fixture exercises blank and non-numeric mol_id values and rejects any explicit target-column or nontraining-SMILES selection. The real mapping README is322bytes and has SHA256 `e6b64bdc76f68f18bb3c4c9c9826eefa64be48748848670f6c5a11835d618c3f`. No new predictive data is read by this repair. V1 runtime gate and failed exporter logs remain under v1; they are not overwritten or recast as a passing export.

## Uniform chemical grouping gate

Run `python sharing_policy/gine_transfer_grouping_fix_v5/verify_chemical_groups.py --report sharing_policy/gine_transfer_grouping_fix_v5/engineering/CHEMICAL_GROUPS_run01.json` with the separately pinned RDKit runtime. It checks aromatic/Kekule equivalence, organic scaffold relatives, chiral identity, disconnected salts, explicit hydrogen, seven-ligand aluminium including a ring-bearing scaffold, and the documented hypervalent phosphorus example. Pyrrole and N-methyl pyrrole must share the standard H1 scaffold; pyridine must retain H0 and a different scaffold. The ordinary explicit-H fixture must match standard atom count/order/connectivity, and added isotope, H2 and isolated-H fixtures preserve the standard parser exceptions. All original fixtures and equality checks remain. These are data-preparation fixtures; they do not train a model.

The uniform procedure is the [RDKit Cookbook partial-sanitization recipe](https://github.com/rdkit/rdkit/blob/Release_2025_03_6/Docs/Book/Cookbook.rst): parse with explicit SmilesParserParams(sanitize=False,removeHs=True), update properties with strict=False, then use FINDRADICALS, KEKULIZE, SETAROMATICITY, SETCONJUGATION, SETHYBRIDIZATION and SYMMRINGS, plus SANITIZE_ADJUSTHS. The pinned [MolOps.cpp source](https://github.com/rdkit/rdkit/blob/Release_2025_03_6/Code/GraphMol/MolOps.cpp) documents the disappearing aromatic hydrogen and corrects it without strict valence checks. This is the standard general bookkeeping stage, with no aromatic-N blanket override or case-specific fallback. Molecules and MurckoDecompose scaffolds follow the same procedure. The default GetScaffoldForMol helper has a strict property-cache call, as the exact installed source evidence shows.

This is a version-pinned structural identity for grouping. It does not certify chemical validity, reproduce older RDKit chemistry exactly, or normalize tautomers/protonation. Legacy stereo assignment has documented limits. The exporter compares atom order/atomic numbers and every inverse-edge pair with each original frozen OGB graph, and verifies original tensor hashes. It leaves all model features untouched. If any input cannot complete the uniform procedure or differs in topology, the entire export fails; no molecule is silently removed, assigned an exception identity or selected using a label/score. Both processed-tensor and CSV field materialization limitations remain.

## Preserved failed-gate evidence

The exact supervised stage diagnostic and pinned RDKit primary sources are retained and hash-bound in `SOURCE_PROVENANCE.json`. In v3, initial pyrrole normalization lost its explicit aromatic nitrogen hydrogen and produced an invalid canonical form; scaffold re-normalization then failed. The N-methyl pyrrole scaffold showed the same loss. `ancestor_v3/` retains the failed packet source and delivery evidence. Neither that failure nor this new source-only amendment is a successful data export or predictive result.

## Explicit parser H suppression

The pinned [SmilesParse.h](https://github.com/rdkit/rdkit/blob/Release_2025_03_6/Code/GraphMol/SmilesParse/SmilesParse.h) states that its Boolean sanitize argument controls both sanitization and H removal. Its false branch disables both. V5 uses the separately exposed parser parameters to retain H removal while disabling full sanitization. [SmilesParse.cpp](https://github.com/rdkit/rdkit/blob/Release_2025_03_6/Code/GraphMol/SmilesParse/SmilesParse.cpp) applies the standard RemoveHs procedure with explicit-count updates and sanitize=False. This changes grouping parser representation only; every resulting atom order/connectivity must still equal the original frozen OGB graph. Original model features remain untouched.

`engineering/ALL_V4_FIXTURES_run01.json` retains the exact runtime diagnostic, and `ancestor_v4/` retains the failed source, manifest and gate evidence. Canonical equality never overrides the frozen graph topology check. No empirical dataset has been examined by this amendment preparer.
