# Literal SeHGNN driver changes for IMDB development

Use the pinned `hgb/main.py`, `hgb/utils.py`, `hgb/model.py` and actual `data/data_loader.py` at SeHGNN commit `e92bd37d0b803457339555684f139b4c8f3e160d`. The following seams prepare a literal 200-epoch TRAIN/VALID reference. A runnable scientific driver and its release remain root's next reviewed implementation.

## Preserve the model and native computation

Keep `hgb/model.py` byte exact. Preserve per-metapath embeddings, both feature projection layers and their LayerNorm/PReLU/dropout, semantic Q/K/V fusion with learned zero-initialized gamma, concatenation projection, all three residual hidden task blocks, and final nonaffine five-logit LayerNorm. The false optional raw-feature residual flag does not remove task residuals. Preserve the original native `train`, feature propagation and sparse label propagation helpers, Adam and AMP/GradScaler behavior.

Keep raw node/type blocks, attributes and keyword identity. Keep all six directed relation supports. Construct each SparseTensor exactly in the native destination-row/source-column convention, with raw head as row and raw tail as column; use native binary row normalization. Preserve movie/actor/director/keyword DGL edges and original feature aggregation. Verify full reverse supports with the input schema; add no synthetic reverse relation or graph self loops.

## Replace the input seam

1. `data/data_loader.py` constructor lines 24–27 opens both `label.dat` and `label.dat.test`. Do not instantiate it. Build the native static graph/feature context from the reviewed `RoleData` buffers and frozen role descriptors.
2. `hgb/utils.py` lines 238–267 loads TEST membership and copies its values into `init_labels`. Replace this IMDB input seam with explicit TRAIN and VALID target buffers and a count-only graph metadata object. The replacement must expose no `labels_test` or TEST scorer. A native helper requiring global movie indexing may use an explicitly masked target buffer with unavailable entries set to NaN; it must never turn unassigned rows into negative target truth.
3. `hgb/main.py` line 22 calls the stock seam. Dispatch only to the reviewed IMDB role-isolated seam. Restrict this preparation to `dataset=IMDB`; it establishes no general dataset adapter.
4. Retain original full transductive feature propagation before the per-seed fit. Keep raw cached features, source dtype and all native channel keys. Bind and report the actual feature/label channel counts and parameter count before opening quality results.

## Preserve role and RNG practice

The native seed block at main lines 160–179 seeds Python/NumPy/Torch and then shuffles sorted development IDs. Load the fixed descriptor for that seed and verify it against the **same seeded global NumPy shuffle**, consuming that original shuffle before model construction. Use the matching frozen TRAIN/VALID IDs thereafter. This preserves the native stream position and prevents a new private splitter from silently changing the loop.

Replace `labeled_nid = concatenate(TRAIN,VALID,TEST)` at main line 188 with `concatenate(TRAIN,VALID)`. Remove TEST counts and `flag[test_nid]`; all remaining movie IDs are unassigned. Complete TRAIN and complete VALID remain the only scoring populations. Unknown membership supplies no barrier to these fits.

## Keep TRAIN only label propagation

At main lines 204–215, retain the native zero source tensor and copy **only `init_labels[train_nid]`** into its TRAIN rows. No VALID values enter the propagation source. Preserve the original native row-normalized metapath products and `remove_diag(v) @ label_onehot` at lines 300–307, including its absence of post-diagonal-removal renormalization.

Remove **every** `check_acc` call at lines 281, 313 and 316. The helper at utils lines 141–178 computes TEST equalities and TEST NLL/BCE even when `show_test=False`; changing its display flag is insufficient. It has no training-update effect and is an inadmissible diagnostic for this development stage.

Preserve all 12 movie-returning label channels if actual source topology produces them. Record discrepancies as schema/native qualification failures; do not reduce hops or discard channels to create a weaker reference.

## Remove TEST scoring and export

In main lines 399–405, remove TEST best-loss/accuracy variables and TEST histories. At lines 438–468, compute BCE and F1 only for complete TRAIN and VALID. Remove `loss_test`, `test_acc`, their saved best values and TEST log strings. Keep the strict `loss_val < best_val_loss` selector and the literal `epoch-best_epoch > patience` stop.

Remove all TEST submission generation and TEST accuracy from main lines 504–526. Remove the unknown-row `full_loader` scoring/export path for this development reference. Neither complement IDs nor zero/NaN placeholders may be scored as TEST. Final official membership and scoring belong to a later actually frozen confirmation release.

## Save and freshly restore the selected reference

The stock source saves only `model.state_dict()` and serves saved `best_pred` for known rows; selected weights are reloaded only for its extra-node path. Save each selected model/buffers, Adam, AMP scaler, epoch/role bindings and Python/NumPy/Torch CPU/all visible CUDA streams. Preserve the exact strict selector and training updates. Saving additional ownership state changes custody, rather than model math.

Always restore the selected model for fresh TRAIN/VALID serving, including when no unknown rows are inferred. Retain the original final serving operation `sigmoid(raw_logits) > 0.5`, native full-population BCE and official five-column micro/macro F1. Record any material reconstruction differences without a microscopic parity campaign. The native loop's printed raw-logit `>0` F1 diagnostic can remain explicitly distinguished from fresh served metrics.

## Fixed competent command and next gate

Preserve the literal author settings:

```text
--epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4
--num-hops 4 --num-label-hops 4 --label-feats
--hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0
--amp --lr 0.001 --weight-decay 0 --batch-size 10000 --patience 50
```

Use the already frozen role seeds and a fresh owned fit per seed. Keep CPU feature propagation, all full-data preprocessing/cache costs, original Torch/SparseTensor/DGL providers and dynamic state. Root should first bind real schema/roles using `qualify_schema.py`, then qualify the literal native full TRAIN/backward/Adam and selected-state serving under the existing runtime. No narrower width, omitted keyword type, lost label channels, changed classifier normalization or cheaper inference substitution is part of this preparation.
