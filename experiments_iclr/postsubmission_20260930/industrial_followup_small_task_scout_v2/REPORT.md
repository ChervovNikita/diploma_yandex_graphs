# Smaller follow-up scout: F1 driver-position with RelGT

**Recommendation: retain the task, decline native execution readiness.** RelBench `rel-f1` / `driver-position` is a substantially smaller temporal heterogeneous regression candidate, and RelGT supplies a recent complete trainer. The pinned trainer has material forecast-context defects, so this scout does not satisfy the full request for a ready reproducible graph comparator. No source repairs or execution were performed.

The completed H&M scout remains unchanged. This is a separate research lane; it does not alter either the H&M recommendation or the sealed industrial execution packet.

## Size and scientific setting

Published metadata gives **7,453 training forecasts, 499 validation forecasts and 760 test forecasts**, versus H&M’s 5,488,184 training forecasts: about 736 times fewer. The F1 database has 74,063 rows across nine tables and 67 columns. These are paper metadata, not inspected local data. A small forecast table does not establish cheap model training: RelGT still uses a width of 512, 300 local tokens and 4,096 global centroids, with roughly 10–20 million parameters across its reported configurations.

The task predicts a driver’s average finishing placement across races in the next **60 days**, using `mean(results.positionOrder)`. The paper’s phrase “two months” is not a calendar-month definition. Forecast identity is `(driverId, date)`, with target interval `(t, t + 60 days]`. Rows are selected by the native outcome-table SQL, rather than a Cartesian product of all drivers and cutoffs. The task’s driver eligibility subquery has a one-year lower bound without a seed-time upper bound; preserve and disclose that benchmark population definition. It does not certify a production list of eligible drivers at forecast time.

The native validation and test cutoffs are **2005-01-01** and **2010-01-01**. The task requests up to 40 evaluation timestamps, with 60-day spacing. The source-derived validation calendar has 30 candidate cutoffs, ending 2009-10-07, whose final target ends 2009-12-06. The training schedule descends by 60 days from 2004-11-02 to the database minimum; using the published minimum 1950-05-13 yields 332 candidate dates, earliest 1950-06-19. If the database covers the full test schedule, the 40 test cutoffs end 2016-05-29, with the last horizon ending 2016-07-28. Actual retained rows and database maximum were not inspected; a future data admission must verify them. [TASK_PROTOCOL.json](TASK_PROTOCOL.json) preserves the exact source formulas.

The paper reports 826 unique forecast entities and 44.6% train/test entity overlap. This setting includes chronological evaluation with less entity overlap than H&M, but does not by itself establish a clean cold-start experiment or general industrial transfer.

## Recent graph comparator and competent tabular controls

The new scoped primary is **Relational Graph Transformer**, arXiv **2505.10960v2**, revised **2026-02-05**. RelBench v2 cites it as ICLR 2026. The current author repository snapshot is commit **19e423ca3e7cac761130aba790857f2dc3a46ef7**, dated 2025-07-10; its relationship to the revised paper is not independently certified.

| Published F1 result | Validation MAE | Test MAE | Context |
|---|---:|---:|---|
| Raw entity LightGBM | 3.450 | 4.170 | Original RelBench; meaningful tabular control, unlike H&M’s zero-like raw baseline |
| Global median | 4.136 | 4.399 | Original RelBench simple control |
| HeteroGNN | 3.193 ± .024 | 4.022 ± .119 | RelGT’s historical graph reference |
| RelGT headline | 3.3257 ± .5618 | 3.9170 ± .3448 | Latest RelGT appendix; spread is reported, run count was not established in the scoped source |
| RelGNN | — | 3.798 | Saved primary context; its missing native trainer remains unresolved |

All values are published results, not new measurements or a common locally controlled experiment. RelGT’s gain over HeteroGNN is modest here and smaller than its reported spread. It is a modern comparator, not the best known graph model on this task.

The pinned legacy RelBench `examples/lightgbm_node.py` provides a complete raw-feature tuner: training-fitted materialization, MAE selection and ten trials. Its default 50,000-row cap does not subsample this 7,453-row task; explicitly declare full data. Keep the optional autoregressive-label branch disabled unless its separate chronology and label custody are qualified.

For a stronger non-graph comparison, the saved author user-study trainer and newly retrieved F1 SQL provide **50 engineered features**: driver/constructor standings, recent race performance and upcoming race attributes. The source stype mapping includes the target and two keys as well, yielding 52 non-target columns before explicit drops; materialization was not executed. The inherited identifier-drop quirk must be declared before outcomes. The SQL reads future race dates/circuits for upcoming events. A fair historical-input study must either establish that these schedule attributes were available at each cutoff and make the same permitted information available to graph arms, or prospectively declare a history-only feature adaptation. Do not silently call that adaptation an exact author reproduction. An exact engineered-GBDT score was not numerically inspected.

## Why the native trainer is not admitted

The author `main_node_ddp.py` contains full optimization, validation checkpoint selection and final inference. It uses scalar L1 loss, MAE, Adam, gradient clipping at 1, and native train-target 2/98 percentile inference clipping. The small-task launcher specifies nine layer/dropout configurations, 100 epochs, batch 256, learning rate .0001, and seed 0. It launches each configuration with one GPU; the eight-GPU server recommendation concerns its scheduling environment. The driver still requires NCCL, CUDA and NVML even for one process. Effective learning rate scales with world size. The `warmup_steps` argument is present but no active warmup scheduler was found in the trainer.

Three material concerns prevent an exact native reproduction claim:

1. **Forecast contexts are keyed only by driver ID.** In `utils.py`, sampling initially receives entity and seed time, but returns a dictionary `S[type][entity_id]` (lines 270–272). Precomputation retrieves that same entry for every repeated entity row (lines 440–446). Its chunk size is 10,000 (line 405), so the published 7,453 training rows fit in one chunk. Since they refer to at most 826 unique entities, repeated drivers necessarily collide. Some forecasts can therefore receive neighborhoods and relative times constructed for another cutoff. Later-cutoff context can cause temporal leakage, but its actual incidence was not measured.
2. **Fallback tokens lack the temporal predicate.** When no legal one/two-hop neighbors exist, the source samples from all graph nodes and calculates relative time without requiring `event_time <= seed_time` (lines 168–180). The branch’s frequency and realized leakage were not inspected.
3. **Configuration selection is unresolved.** Table 8’s headline configuration, one layer/dropout .5, has test MAE 3.917 and validation 3.3257. The grid’s lowest validation MAE is instead 3.1046 at four layers/dropout .3, whose test MAE is 4.6316. The headline matches the lowest test entry. The scoped paper says configurations were tuned but does not close the cross-configuration selection rule. A future study must select configurations solely by its declared validation rule, never by this test minimum.

These are findings about the inspected source and reporting protocol. They do not establish which artifacts produced the published numbers or invalidate the paper’s empirical results. Dependencies, cached sampling provenance and Python hash seeding are also unpinned. Native GPU/runtime reproducibility was not attempted.

## Fair comparison after the source issues are resolved separately

First establish full-data competence with global/driver history medians, the native raw LightGBM, a declared engineered GBDT view, and a qualified RelGT trainer. Use MAE primary and validation selection; fix R² and RMSE secondary. This is the legacy profile: the latest RelBench v2 paper instead uses R² primary. Retain a single validated configuration across all route arms and charge all nine native grid trials if that search is used. Three paired seeds support an exploratory pilot; do not claim the paper’s exact run count or choose seeds after outcomes.

Then compare the unchanged warm continuation, common-only initialization, matched random tangents, type/time-valid topology permutation and graph-filtered residual initialization. Add shared-trunk private heads and a reused independent native scalar ensemble to distinguish head freedom and ordinary ensembling. Match factor support, warm state, head, scalar loss, perturbation budget, RNG and selection budget. Freeze member count, clipping/pooling order and stopping before execution. Use arithmetic scalar pooling; memberwise native clipping then mean is a possible declared rule.

Hypothesis: **At a competent identical temporal heterogeneous forecaster and warm state, do history-valid graph-filtered training residual directions improve a parameter-shared committee’s future placement MAE beyond matched random/common/topology controls at measured total cost?**

The adaptation remains separate work. L1 has a sign-residual cotangent and a nonsmooth point at zero. The graph/filter axis must retain driver and cutoff. Copy/freeze or explicitly account for mutable EMA centroid buffers and BatchNorm state during initialization differentiation, and pair stochastic positional-encoding/dropout draws. The source creates random GNN-PE inputs even in evaluation; declare the prediction/RNG protocol prospectively. A richer two-layer RelGT head is native here and must be held fixed across controls. Replacing L1 with MSE or removing global attention changes the comparator and needs its own matched control.

Charge all feature/token preparation, tuning, warm training, AD/filter products, private states and complete graph trajectories. Stop before execution while the temporal context defects or selection/custody rules remain unresolved. A negative result or gains fully explained by heads/randomization/independent ensembling must be retained.

## Evidence and scope

[PAPER_CONCLUSIONS.json](PAPER_CONCLUSIONS.json) records the latest RelGT primary and the narrowly revisited F1 passages from the saved RelBench primary. [INSPECTED_PASSAGES.json](INSPECTED_PASSAGES.json) preserves exact HTML blocks and author-source line ranges; [REUSED_EVIDENCE.json](REUSED_EVIDENCE.json) records the consulted memory and prior-source hashes. Retrieval receipts and source pins are saved locally.

No dataset or model downloads, labels, notebook outputs, imports of scientific code, fits, installations, SSH, local source adaptations or runtime infrastructure occurred. Only source/paper text and metadata were inspected. There are no new execution gates or expanded static-check package. H&M’s existing manifest and payload remain unchanged.
