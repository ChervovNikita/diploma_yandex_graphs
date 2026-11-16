# Paper text fixes for the next revision

Submission 42988. Nothing here has been applied to the paper yet: the text stays
unchanged in this iteration. Add every required change to this log as it is found.

Status values: `open` = still to apply, `check` = the wording depends on a check that is not done.

## 1. Training steps on Tolokers (`open`)

- **Where:** Table 7, "Maximum training steps: 5000".
- **Problem:** Tolokers was trained with a limit of 20000 steps, not 5000
  (commit edef5c3c, `run_all_experiments_dual_gpu_alex_tolokers.sh`).
- **Change:** state 20000 steps for Tolokers and 5000 for the other datasets. Explain that
  Tolokers needs the longer limit to converge.
- **Evidence (`results_merged/tolokers`, best-checkpoint step):**

  | Method | Runs | Median | Max | Runs with best step > 5000 | Best step >= 19700 |
  |---|---|---|---|---|---|
  | BASE | 399 | 790 | 8470 | 5 (1.3%) | 0 |
  | ENS (latest member) | 400 | 1045 | 9530 | 12 (3.0%) | 0 |
  | GNNM | 390 | 1905 | 12580 | 35 (9.0%) | 0 |

  No Tolokers run has its best checkpoint near the 20000 limit, so 20000 is enough.
  5000 would cut off 35 GNNM runs, most visibly ResNet (42.5% of its GNNM runs).

## 2. "Trained to convergence" must not be claimed for every dataset (`check`)

- **Where:** any sentence saying that all models are trained to convergence.
- **Problem:** the archived logs show that some runs never reached early stopping and used the
  whole 5000-step limit.

  | Dataset | BASE | ENS (runs with a member at the limit) | GNNM |
  |---|---|---|---|
  | Roman-empire | 10/350 | 33/350 | 54/350 |
  | Amazon-ratings | 10/350 | 17/350 | 68/350 |
  | Minesweeper | 0/349 | 0/350 | 8/350 |
  | Questions | 0/350 | 0/350 | 0/350 |
  | Tolokers (20000 limit) | 0/349 | 0/350 | 0/340 |

  Counts cover the seven paper backbones and come from the "Early stopping" lines of the logs.
- **Change:** limit the claim to Tolokers and Questions, or describe the 5000 limit as a
  fixed budget elsewhere. To be finalised with item 9 of the rebuttal list (training steps).

## 3. Depth selection (`open`)

- **Where:** Appendix A.
- **Problem:** the text says depth is selected on split 0 and reused. Table 1 actually uses
  depth selected by validation for each split (102 of 105 cells reproduced).
- **Change:** describe per-split selection. Selecting on split 0 would make GAT-sep ENS the
  best result on Amazon-ratings.

## 4. Checkpoint rule of ENS and GNNM (`open`)

- **Problem:** the text does not say that ENS selects a best checkpoint for each member,
  while GNNM uses one checkpoint selected on pooled validation.
- **Change:** state both rules.

## 5. Missing citation (`open`)

- **Change:** cite Kim 2023.

## 6. Incomplete Tolokers ResNet runs (`check`)

- **Problem:** `results_merged/tolokers/tabm.csv` has no ResNet rows for splits 0 and 2 at any
  depth (390 of 400 rows). `tolokers/base.csv` and `minesweeper/base.csv` have no ResNet row
  for depth 1, split 0.
- **Change:** rerun the missing splits, or state the number of splits used for these cells.

## 8. Source of the PROTEINS result (`check`)

- **Where:** Appendix D, Table 9, and the code supplement.
- **Problem:** the code is in a separate repository (`graph_classification_gnnm`), and its
  per-fold results are git-ignored and exist only locally in `RESULTS_ENS/`. The local files
  match Table 9 for GNNM (73.62 +- 3.31). `experiments_iclr/README.md` says the result cannot
  be reconstructed from that clone.
- **Change:** include an anonymized copy of the code and the per-fold results in the supplement.

## 7. Definition of step time in Appendix C and Table 8 (`check`)

- **Where:** Appendix C ("mean wall-clock time of one training step"), Table 6, Table 8.
- **Problem:** the reported step time covers the update only, while the training loop also
  runs validation at every step. The script `ablation/run_cost.py` keeps all four ENS models
  in memory, while the archived ENS was trained one model at a time.
- **Change:** state what the step time covers and how ENS memory was measured, or replace the
  numbers with the `ablation/p07_cost_small_setup` measurements, which report the update,
  forward pass and metric separately.

## 9. Step limit of the final runs (`open`, after the fixed-protocol campaign)

- **Where:** Section 4 (training setup), Appendix B, Table 1 caption.
- **Problem:** under the paper limit (5000 steps) 7.5% of GNNM runs, 1.1% of BASE runs and
  0.8% of ensemble members stopped at the limit instead of by patience
  (`ablation/fixed_protocol/rerun_list.csv`). They are retrained with the same protocol and a
  limit of 40000 steps (`ablation/fixed_protocol/PROTOCOL.md`).
- **Change:** state the 40000-step limit, that all other settings are unchanged, and which
  Table 1 numbers changed (`rerun_changes.csv` of the collector). Also fixes item 6 (the 12
  missing Tolokers/Minesweeper ResNet runs are trained in the same campaign).

## 10. Private normalization statistics for BatchNorm backbones (`open`)

- **Where:** method section (Batch normalization paragraph), molecular appendix.
- **Problem:** members of a BatchNorm backbone need their own normalization statistics; a
  shared BatchNorm reduced MolBACE ROC-AUC from 0.827 to 0.670. Node backbones use LayerNorm,
  where the question does not arise.
- **Change:** say that GNNM keeps per-member BatchNorm statistics when the backbone uses
  BatchNorm, and that the node-classification backbones use LayerNorm.

## 11. Collab link prediction with GCN and GraphSAGE (`open`)

- **Where:** link-prediction appendix and the response to the reviewer.
- **Change:** report GCN and GraphSAGE backbones (official OGB recipe) next to BUDDY, with
  Hits@20, Hits@50 (primary) and Hits@100; the BUDDY Hits@20/100 are recomputed from saved
  predictions, without retraining (`ablation/p14_collab_ogb`).

## 12. Optional contrastive term (`open`, only if the final runs show no harm)

- **Change:** if the contrastive variant is reported, present it as an optional regularizer
  with its coefficient and temperature (0.05, 0.2), with the node-task result next to plain
  GNNM at the same cell (`node_contrast.csv`). The earlier runs showed no effect
  (mean -0.03 points, better in 13 of 35 cells), so it must not be claimed as an improvement.
