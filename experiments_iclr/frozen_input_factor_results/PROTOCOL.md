# Frozen input BatchEnsemble factors: prespecified Roman Empire SAGE control

Saved 2026-09-25 UTC before the control runner or verifier was written. This is a post hoc mechanistic diagnostic, not an independent replication.

## Fixed experiment

Use official Roman Empire masks 0, 1, 2, 3, 4. For mask `s`, seed is `s`. Use the same `TABMAblationModel` constructor and training/data/selection path as the existing SAGE `gnnm` case in `experiments_iclr/projector_controls.py`: depth 5, width 512, four members, LayerNorm, dropout 0.2, AdamW with learning rate 3e-5 and weight decay zero, at most 5000 steps, validation at step 1 and every 10 steps, stop after 300 steps without pooled-validation improvement. Select the checkpoint by accuracy of the mean of four validation logits, with first observed tie retained. Test labels enter only the final evaluation. The new variant name is `freeze_input_factors`, with only `input_be_block.R`, `.S`, and `.B` frozen immediately after initial construction; all other parameters, including input W, output factors, and residual stack, train. The result root is `experiments_iclr/frozen_input_factor_results/` and must contain exactly one row, checkpoint, and prediction archive per mask. No current runner or running launcher may be modified.

For each mask, construct an ordinary GNNM and a frozen-input copy from the same seed using the same constructor on the same device. Before training, assert bitwise equality of every initial state tensor, all four member logits in eval mode on the loaded graph, and Python/NumPy/PyTorch RNG state after construction; on CUDA also compare CUDA RNG state. Assert that only the three input factors differ in `requires_grad`. Restore the paired post-construction RNG state before training. Keep a SHA256 digest of each frozen initial input factor tensor in a per-mask initialization record. Do not use a checkpoint or learned factor values to initialize the control.

Compare each selected control test accuracy with the corresponding **existing** SAGE `gnnm` row for the same mask in `experiments_iclr/results/projector_controls.csv`, then report five signed control-minus-GNNM differences and their descriptive mean. The five official masks overlap on one graph; they are not independent replications or a basis for a significance test. This contrast estimates the effect of *learning* the input factors in this architecture, conditional on these seeds and optimization settings. Fixed random sign-coded input factors remain, so this does not supply a clean ordinary-projector ablation or directly resolve that reviewer concern.

## Locked inputs and baseline

Repository HEAD at specification: `35723148a7fbcd199cbfc0e3da1e830a9cdae5e1`. Before launch and at verification, require these SHA256 values:

- `experiments_iclr/projector_controls.py` 3b24562b132c439733f6e0599a84e62aa5d1d5b0ce04d0239574d3b07eea1fcf
- `ablation/models_ablation.py` 2c03998028ff80dc3e0ea49149d8281da073797b52c663f8229bc8b61ace51a8
- `models.py` 07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303
- `datasets.py` 33e4d1c7fb8033e45065db8f4dde67f4f076bf524488e301ceddf5586c9c7893
- `run_common.py` e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc
- `data/roman_empire.npz` a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428
- `experiments_iclr/results/projector_controls.csv` fa5682e1e2bd2c1d2f4d5ccdb6efb8e67850992df1acaf6c8a8e859b4573d47a

Existing GNNM checkpoint hashes for masks 0–4, in order: `072b0e53196c519dcc32efaa019020e9fceb27ad3dd7c2025d30705fa5371c12`, `e55a995978848538fca4384797955d4eead7d18bd0fe2aa6b3dfa41c6a4611f7`, `fe3c00526816aa0686bb890946b8defd492fbca9bd66880dc72535ecbd99f7c9`, `7f85dbd05a969a7939ac2043f43a0d28cb780b3b376466c066f17c403a8188b9`, `51890f54550bde90c06e38c1fcdaa3dedd890aabf118afc0560f51f2a8764775`.

Existing GNNM prediction archive hashes for masks 0–4, in order: `bbbd675b8ea1d4c30968e7318c307311e33a458a0339313b0a6f66261e035754`, `b537043f1edbd5e0e9c63a1c4a03ec10b27174f60e9c0f32c1779da88d5c02d5`, `3c56df27e30f4fb9999ca9fb712a20853b514ec7a3905ac1595c8c470e1c5d87`, `7e8257b6f7641a47cc617ecfa7081cc2e4975e64b6da151b7c9331a169911298`, `0f4254ce05f717813fa26bd82b0cffb9861b3129c1ff87fbe85b9ca4102ff1cb`.

The verifier must reject missing, duplicate, partial, reordered, or mixed-protocol rows; changed locked files or baseline artifacts; malformed or inconsistent checkpoint and prediction archives; nonfinite values; incorrect split masks; and changed frozen factor values. A full-run success marker is valid only after an independent complete verification. The one-shot launcher is prepared only; this preflight does not authorize or schedule GPU work.
