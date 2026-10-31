# Author-side fixed-mask optimization-seed pair study (predeclared)

**Frozen before any run:** 2026-09-25 UTC. This diagnostic tests whether the Roman Empire SAGE result for GNNM versus initially matched untied propagation persists when only the optimization seed varies on one official mask. The seed count is fixed at **three** because of the available run window: seeds **0, 1, 2**, each with one fresh GNNM run and one fresh `untied_backbone` run. No result from the earlier five-mask matrix is reused. The older split-0 jobs have no preserved exact launch-time source hash and the old five-row series changed both mask and seed.

## Fixed design

- Dataset: the pinned `data/roman_empire.npz`, official train/validation/test **mask 0** throughout; bidirected graph with self loops as in the original control runner. Test labels enter only after the validation-selected checkpoint is loaded.
- Models: `gnnm` and `untied_backbone` from `projector_controls.py`. Both use SAGE, five layers, width 512, four members, LayerNorm, 0.2 dropout, and the default BatchEnsemble initialization. For the untied variant, four propagation stacks are deep copies of one initialized stack; its input and output BatchEnsemble blocks and output normalization remain shared. No parameters or training randomness are copied from a trained model.
- Per pair, call `set_seed(seed)` immediately before each model construction. The runner requires identical canonical initial parameters and CPU/CUDA RNG states; it saves all initial member logits and requires their maximum absolute difference to be at most `1e-6`. A failed pair stops before training the second variant.
- Train with AdamW, learning rate `3e-5`, zero weight decay, and the mean of four member cross-entropies. Maximum 5,000 steps. Validate pooled logits at step 1 and every ten steps. Save the first checkpoint with a strictly improved pooled validation accuracy. Stop after 300 validation-counted steps with no strict improvement, exactly as in the original control runner. Never use test accuracy to select a checkpoint, seed, or variant.
- Run order is seed 0 GNNM then untied; seed 1 GNNM then untied; seed 2 GNNM then untied. All six runs and their artifacts are required for completion. A partially written run stays under `.inprogress` and is not silently reused.
- New result root: `experiments_iclr/fixed_mask_seed_pair_results/`. The source manifest pins the runner, verifier, protocol, imported model/data modules, and Roman Empire data bytes. A changed input causes the runner and gate to fail. The active queue, original result roots, existing launchers, and manuscript are outside this study.

## Locked evaluation and reporting

For each seed, compute the percentage-point difference in **selected test accuracy** (`gnnm - untied_backbone`) on mask 0. Report all three paired differences, their mean, sample standard deviation, and the three seedwise validation-selected steps. Call this a fixed-mask optimization-seed diagnostic: all runs use one graph and one official mask, so it does not estimate across-graph variation. Report the complete result, including mixed signs or failure to support the original direction. No sequential stopping, seed substitution, hyperparameter revision, or selection by test result. A failed or incomplete run is disclosed and investigated; it is not replaced under the same protocol.

The independent completion gate requires exactly the six result JSON files and checks the pinned source/data hashes, required artifact SHA-256 values, validation trace and earliest-tie checkpoint rule, seed/mask/configuration, canonical initial-state and RNG hashes reconstructed from each seed, pairwise initial-logit equivalence, checkpoint-loaded pooled validation and test metrics, and saved test predictions against official mask 0. It writes `completion_audit.json` only after all checks pass. The report remains author-side until that gate passes.

## Launch after the current GPU queue is idle

From the repository root, using the local environment:

```bash
.venv/bin/python experiments_iclr/fixed_mask_seed_pair.py --preflight-only
.venv/bin/python experiments_iclr/fixed_mask_seed_pair.py --cpu-smoke
.venv/bin/python -u experiments_iclr/fixed_mask_seed_pair.py --device cuda:0 --seeds 0 1 2 \
  > experiments_iclr/fixed_mask_seed_pair_results/run.log 2>&1
.venv/bin/python -u experiments_iclr/verify_fixed_mask_seed_pair.py --device cuda:0 \
  > experiments_iclr/fixed_mask_seed_pair_results/verification.log 2>&1
```

Do not launch concurrently with the active SAGE/OGB/optional control jobs. Historical five-mask SAGE runs averaged 984 seconds for GNNM and 1,104 seconds for untied propagation per run (early-stopping dependent). Six runs therefore suggest about **1 hour 44 minutes** of training; allow **2–3 GPU hours** including initial-logit checks, checkpoint I/O, and the independent gate. The 5,000-step cap and new seeds can extend this estimate.
