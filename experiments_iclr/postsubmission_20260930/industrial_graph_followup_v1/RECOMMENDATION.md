# Industrial graph follow-up recommendation

Date: 2026-10-02. Status: bounded literature and static source review; two new primary reads. This folder proposes future preparation and supplies no authority to acquire datasets, load models, train, or change the sealed R17 study.

## Recommended sequence

**First: Tolokers2, fixed GraphLand RL, one residual GAT backbone.** Use the modern GraphPFN paper's released GAT tuning configuration as the preferred source anchor: three blocks, width 512, residual connections, LayerNorm, GELU, AdamW, and a two-logit cross-entropy classifier. Its published RL AP is 57.41±0.85, versus 56.27±0.31 for GCN; GraphLand v5 separately reports its local GT and GAT ahead of GCN on this task. A competent native recipe is needed before testing the shared/private extension. The saved tuning config is a search specification, not a verified winning GNN configuration.

Freeze one backbone and one split prospectively. Retain the five mechanism arms and an independently trained native ensemble control, using a declared common pooling rule and matched selection/budget conventions. Average precision should be the task performance endpoint; add Bernoulli NLL and calibration diagnostics if the research question includes predictive uncertainty. AP improvement alone cannot establish uncertainty quality. Any NLL-based checkpoint rule replacing native AP selection is an explicit protocol adaptation. Select a bounded configuration using permitted source validation roles before final evaluation labels are available; do not call package defaults the published optimum.

**Second: Artnetviews, fixed RL, residual GCN, after regression preparation.** GCN is competitive with LightGBM+NFA on the published RL task. Use a one-output MSE model and report R² and original-scale squared error. A new initializer residual, output-dimension contract, target-transform custody, and automatic-differentiation/optimizer qualification are necessary. The sealed R17 cross-entropy implementation cannot be used unchanged. Deterministic MSE fits do not define a Gaussian likelihood; a Gaussian NLL or epistemic-uncertainty claim would require a separate variance/predictive-distribution protocol.

**Later: choose TH or THI for temporal robustness.** Both tasks support temporal splits. Native GCN is a sensible common backbone for a separately frozen TH experiment. GraphPFN v4 reports RL results only, so its published advantage cannot be claimed for TH/THI. Avoid expanding this lead into a backbone × task × split grid.

## Exact task identity

| Release name | Target and graph | Nodes | Undirected edges | Stored bidirectional edges | Raw attributes |
|---|---|---:|---:|---:|---:|
| `tolokers-2` | Worker ban/fraud classification; workers linked when they worked on the same task | 11,758 | 519,000 | 1,038,000 | 16 |
| `artnet-views` | User views regression on the Shedevrum/Artnet friendship graph | 50,405 | 280,348 | 560,696 | 50 |

These are paper/provider metadata, not measurements from acquired datasets. Tolokers2 is distinct from the older Tolokers release. Artnetviews uses the Artnet graph with a different target from `artnet-exp`. Raw attributes include typed numerical/categorical/fraction features; one-hot encoding changes neural input width. Final encoded width and labeled role counts remain unknown. Native source intersects split masks with non-missing (not NaN) targets, so the nominal percentages cannot certify exact labeled counts.

RL is fixed random stratified 10/10/80 train/validation/test; RH is 50/25/25. TH is temporal 50/25/25; THI uses the same roles with three evolving graph snapshots, excluding later nodes at training. Repeated model seeds on a fixed split are not independent graph or split replication.

The Zenodo record `16895532` lists `tolokers-2.zip` (3,361,114 bytes; MD5 `f453141143f3a93a61502e461a03d222`) and `artnet-views.zip` (13,192,474 bytes; MD5 `48e3eb3f45bb3ff50b4c1ba323135f5b`). Only record metadata was retrieved. A later acquisition must bind and inspect actual archive contents and provider/preprocessing identity.

## Contemporary references

GraphPFN v4 actually evaluates both exact GraphLand tasks. Its Table 1 reports AP or R² ×100, mean±standard deviation over repeated runs:

| Method | Tolokers2 AP | Artnetviews R² |
|---|---:|---:|
| LightGBM+NFA | 56.34±0.06 | 56.10±0.02 |
| GCN | 56.27±0.31 | 56.03±0.25 |
| GAT | 57.41±0.85 | 53.60±0.24 |
| GT | 56.98±0.55 | 53.37±0.46 |
| G2T-LimiX ICL | 61.60±0.18 | 61.58±0.08 |
| GraphPFN ICL | 61.29±0.12 | 62.79±0.08 |
| G2T-LimiX FT | 59.75±1.14 | 63.24±0.07 |
| GraphPFN FT | 62.80±0.39 | 65.35±0.06 |

Use a qualified LightGBM+NFA baseline and GraphPFN FT as the main contemporary performance references; GraphPFN ICL and G2T-LimiX are informative references if the final question warrants their distinct cost/model regimes. GraphPFN uses synthetic pretraining plus pretrained LimiX and a ten-member **shared-weight preprocessing ensemble**, not ten independently trained models. Its members permute/sample features (eight random features per forward); fine-tuning samples one ensemble member per gradient update and evaluates the ensemble every ten updates. Native independent GNN ensemble utility and pretrained PFN utility answer different comparisons, so report both costs and semantics explicitly.

GraphLand v5 evaluates OpenGraph ICL, AnyGraph ICL, TS-GNN/TS-Mean ICL, and GCOPE FT; implemented regression support is absent in its tables. GraphPFN v4 additionally evaluates SAMGPT, MDGFM, TAG-TabPFNv2, TAG-LimiX and G2T-LimiX. Classification-only rows do not supply an Artnetviews baseline. Appendix E's NodePFN run exceeds its recommended training-context size, a stated limitation of that comparison. No third primary was read for these methods.

## Source and execution requirements

1. **Keep primary versions and recipes separate.** GraphLand v5 Table 2 gives Tolokers2 GAT 53.78±1.34, GT 54.50±1.20, GCN 51.32±0.96, and LightGBM-NFA 56.16±0.28; Artnetviews GCN 55.99±0.26 and LightGBM-NFA 56.55±0.04. Do not splice these into GraphPFN v4's table as one experiment. Static source confirms materially different loss heads, preprocessing, stopping and loop policies. Exact result parity was not tested.
2. **Full graph custody is required.** The GraphLand release implements full-batch training; minibatch mode raises `NotImplementedError`. Its residual GCN/SAGE/GAT/local GT bodies and multi-layer input/output heads are stronger than bare defaults. Four private routes must still compute their complete member-dependent trajectories. DGL graph operations and GAT attention need prospective JVP/VJP and private-boundary qualification; feasibility has not been measured.
3. **Bind native optimizer and precision semantics.** GraphLand's paper says Adam, while pinned release code uses AdamW, GradScaler and LambdaLR. Source defaults use weight decay zero and no warmup. The release scripts search five learning rates × three dropouts × two task-specific transforms (30 configurations for Tolokers2, 60 for Artnetviews including target scaling), one initial run and ten runs for the selected setting. The modern GNN configs instead use patience 100, no declared AMP, fixed quantile-normal numeric policy, tuned fraction policy, and task-specific self-loops; binary classification uses two-logit CE. Do not transplant R17 coupled-Adam decay transport into AdamW without qualification.
4. **Preserve preprocessing scope.** Native GraphLand transductive feature transforms/imputation/one-hot vocabulary use all public node features, with regression target scaling fitted on training targets only. THI uses training-snapshot feature fitting. GraphPFN uses typed features with ordinal categories and different fraction transforms. These providers are not interchangeable. Freeze loop removal/addition, symmetrization, missing-target masks, target scaling and snapshot maps.
5. **Guard final labels.** Both inspected native GNN training scripts compute test metrics during training. A closed-label experiment requires a reviewed wrapper that preserves training and validation selection while withholding final labels. No native script should be run unchanged as the prospective closed-label study.
6. **Review NFA before using it.** Saved `graphland-baselines/scripts/nfa.py` contains a fraction branch that aggregates `num_features` rather than `frac_features`. Its inductive categorical and fraction branches use the full graph where the numerical branch uses a snapshot subgraph. Artnetviews is commented out in that script's default dataset list. These static findings require resolution and source identity before a new baseline; they do not demonstrate that published scores were generated incorrectly.
7. **Use separate pinned environments where needed.** GraphLand DGL metadata pins Torch 2.5.1/Python 3.11; GraphPFN package requires Torch <2.5, and its paper environment pins Torch 2.4.0/Python 3.12.9. Neither environment is qualified against the current R17 runtime. Paper GNN experiments use A100 80GB; this is author hardware, not a measured minimum. GraphPFN Table 9's end-to-end optimal-configuration runtimes exclude tuning/pretraining costs and are not local forecasts.

This follow-up is outcome-aware after the existing mechanism lead and after inspection of published results. It can test transfer to an industrial domain; it cannot be described as an untouched task-selection cohort or as a guarantee across graphs. Continue to report accepted alpha/fallbacks and complete-operation utility; branch-specific backtracking does not support a same-alpha causal alignment claim.

## Evidence

Primary IDs: [GraphLand 2409.14500v5](https://arxiv.org/abs/2409.14500v5), revised 2026-05-02; [GraphPFN 2509.21489v4](https://arxiv.org/abs/2509.21489v4), revised 2026-08-20. Source pins: GraphLand `7246fe3f6b53cffaa76e3dc22726a6db3984bbe3`; GraphPFN `3b9b115490249cc777227c846babfb55f35bd8c4`; GraphLand baselines `4629cd7cfd7e69c9797f4c2fd389e1d419885899`. Exact reading scopes, SHA-256 descriptors, reused memory and retained page renders are recorded in `PAPER_CONCLUSIONS.json`, `SOURCE_RECEIPTS.json` and `MANIFEST.json`.
