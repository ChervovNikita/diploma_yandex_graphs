# Retrospective depth selection sensitivity

The three policies below use the same 15 archived CSVs and fixed width 512 and learning rate 3e-5. They consider depths 1–5 for every dataset, backbone, and method. All depth decisions use validation metrics only. Values within 1e-12 of the best validation value count as ties, broken toward the smaller depth. The selected depth's test metric is read afterward. The outputs are `depth_policy_per_split.csv`, `depth_policy_cell_summary.csv`, and `depth_policy_dataset_effects.csv`.

- **Per split validation:** Choose depth separately on each of the ten official masks. This reproduces the main reconstruction.
- **Mean validation:** Choose one depth using the mean validation metric across all ten masks, then apply it to all ten masks.
- **Split 0 validation:** Choose one depth from mask 0 validation and reuse it on all ten masks.

GNNM minus four independently trained networks (ENS), averaged over eight backbones within each dataset, in percentage points:

| Selection policy | Roman Empire | Amazon Ratings | Minesweeper | Questions | Tolokers | Mean of five datasets | Dataset wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per split validation | +1.122 | +0.088 | −0.339 | +1.010 | +0.170 | +0.410 | 4/5 |
| Mean validation | +1.101 | +0.085 | −0.406 | +1.012 | +0.165 | +0.391 | 4/5 |
| Split 0 validation | +1.072 | −0.074 | −0.352 | +1.005 | +0.009 | +0.332 | 3/5 |

These are retrospective sensitivity calculations, not three prospectively planned benchmarks. The ten masks of one graph overlap. In particular, a validation label used by the mean-validation or split-0 policy may be a test label in another mask, so those two policies should not replace the per-split primary estimate. The pattern of SAGE gains on all five datasets and GAT losses on four of five is unchanged across these policies. The dataset-level aggregate is sensitive to the depth policy, so the paper should state its primary rule exactly and avoid a universal gain claim.
