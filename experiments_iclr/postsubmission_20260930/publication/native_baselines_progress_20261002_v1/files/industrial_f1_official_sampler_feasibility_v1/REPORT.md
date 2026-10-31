# F1 feasibility: official temporal sampler and native GraphSAGE

**Recommendation: feasible for a small independent pilot.** Use the official RelBench v1.1.0 temporal `NeighborLoader` path, native HeteroGraphSAGE and full-data LightGBM controls. This avoids RelGT’s entity-only neighborhood cache and requires no RelGT configuration search or repair. Source feasibility is established; runtime, data custody and empirical competence remain unqualified.

This packet extends the [saved F1 scout](../industrial_followup_small_task_scout_v2/REPORT.md). Its exact task and calendar definitions remain authoritative: `rel-f1/driver-position`, 60-day average `positionOrder`, forecast key `(driverId,date)`, 7,453/499/760 published train/validation/test rows. No prior scout or live study was changed.

## Correct forecast identity and sampling

Pin RelBench **v1.1.0**, commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`. Its `get_node_train_table_input` preserves one node ID and timestamp for every forecast row. `AttachTargetTransform` explicitly handles repeated entities and attaches targets through `batch[entity].input_id`, the forecast-row index.

With proposed PyG **2.6.1**, temporal sampling automatically sets `disjoint=True`; request it explicitly. Use `time_attr="time"`, the row-aligned `input_time`, uniform sampling, directional subgraphs and fanouts **128/64**. The sampler forwards each seed time to `pyg.hetero_neighbor_sample` and retains `input_id`/time metadata. A driver at two cutoffs therefore has two disjoint forecast components. There is no entity-only Python dictionary or all-node fallback. **pyg-lib is required**: the torch-sparse fallback rejects this disjoint path.

Before any fit, a future disposable qualification must demonstrate repeated entity/different-time rows, correctly aligned input IDs, per-component event times no later than their seed, and complete keyed predictions. This source inspection did not run that qualification.

## Fixed native graph recipe

The original RelBench Appendix B.2 and released trainer agree on a fixed regression recipe: two graph layers, width **128**, sum aggregation, four-layer per-table ResNet encoders, one-layer scalar head, L1 loss, Adam **lr .005**, **10 epochs**, batch **512**, uniform temporal sampling. Select the strictly lowest validation MAE checkpoint. Keep the source’s train-target **2/98 percentile** prediction clamp. Report R² and RMSE as fixed secondary metrics under the explicitly named legacy MAE profile.

Use seeds **42, 43, 44**, all rows, and no graph hyperparameter search. The historical native result, validation/test MAE **3.193/4.022**, is a competence reference; this adapted prospective protocol cannot be called an exact reproduction of that score. Do not choose RelGT’s best test configuration. Native raw LightGBM’s **3.450/4.170** is also published context, not a new result.

## Fair tabular baselines and source declarations

Required controls are a training global median, a driver-history predictor with declared fallback, native full-data raw LightGBM with **10 validation trials**, and a stronger history-based engineered LightGBM. The user-study SQL has 50 engineered features. Exclude its six upcoming-race round/circuit features for the preferred historical-input comparison, yielding **44 engineered features**. Preserve the source-default driver ID/date predictors explicitly, so this view has **46 non-target columns**. This is a declared history-only adaptation, not the published user-study baseline. Its training materializer fits on training forecast rows only.

Do not pass key columns to the existing trainer’s drop option: it later needs those same columns to remap predictions. If a 44-predictor view without keys is desired instead, preserve a separate keyed row map before dropping them and declare that additional change. A future-schedule comparator can be added only under an explicit historical-availability and information-parity policy.

The graph’s native feature materializer fits statistics on the whole database snapshot through the test cutoff. For the preferred chronological profile, fit each timestamped table’s schema/statistics/vocabulary using rows through **2005-01-01**, then use its training-fitted converter for the stable-ID public snapshot through **2010-01-01**. Keep native untimestamped metadata tables and disclose that benchmark assumption. Do not infer production point-in-time availability of those tables. PyTorch Frame 0.2.3 exposes the required training-fitted converter. This preprocessing change and fixed validation sampling must be attributed as independent adaptations.

The native trainers directly evaluate test at the end and their ordinary download caches can contain hidden target columns. A separate development/final-evaluation custody wrapper is necessary. [SOURCE_DECLARATION.md](SOURCE_DECLARATION.md) specifies these changes precisely; no wrapper or source repair was implemented here.

## Minimum prospective cohort and cost

First run **three native graph fits** and one ten-trial search for each tabular view. Use validation only. Require mean graph validation MAE to beat the global median and be no worse than the matched raw LightGBM; retain driver-history and engineered results even if they win. If that competence condition fails, report it and stop route expansion without selecting a new model on test.

After competence, reuse each graph seed’s selected warm checkpoint. Compare five matched three-member arms: unchanged warm copy, common-only, random tangent, type/time-valid topology permutation and graph residual initialization. Add a shared-trunk private-head control; reuse the three native fits as the independent scalar ensemble. Use the same native head, L1 objective, reset Adam state, 10-epoch continuation budget, input contexts, factor scope and validation rule. Mean memberwise-clamped scalar predictions. [COHORT.json](COHORT.json) records the fixed proposal.

Each native fit needs **15 training batches per epoch, 150 updates total**, plus 11 validation batches. Three fits use **450 updates**. These are source/metadata counts, not measured timings. [COST_MODEL.json](COST_MODEL.json) gives one-GPU planning scenarios: at 1/5/20 seconds per full batch, a native fit is about **2.7/13.4/53.7 minutes**, excluding preparation. Three fits are about **8/40/161 minutes** plus one shared preparation pass and tabular searches. Target a single 24 GB CUDA GPU, subject to measured memory qualification; no fit or memory measurement occurred. Batch reductions would change the native protocol and must be declared prospectively.

Parameter sharing does not reduce a three-member committee to one graph evaluation. The proposed continuation cohort has **7,650 graph-trunk update equivalents** before AD/filter work and validation: 450 native, 6,750 across the five three-member arms, and 450 for shared-trunk heads. Charge preparation, failures, warm fits, derivatives, all member trajectories and reused ensemble training. No cheap wall-time claim follows from the parameter count.

## Scope of the method test

This can test supervised initialization utility on one competent small temporal heterogeneous forecaster. Initially bind member-private input/output factors only to the native relation-specific SAGE linear maps, retaining the backbone’s type-specific matrices. At identity factors, copied heads and copied normalization buffers must reproduce the warm scalar function. Do not tie different relation types together. Retain a separate driver/cutoff axis for every residual filter and use only training targets and history-valid structure.

L1 nonsmoothness, typed factor derivatives, BatchNorm state and finite-step descent need separate qualification. The current homogeneous CE route implementation is not an admitted regression implementation. This pilot does not establish foundation-model competitiveness or transfer the old certificate unchanged.

Only saved literature, official source text, library source and package metadata were used. No SSH, GPU calls, data/weight downloads, final labels, scientific imports, model execution or live-study edits occurred.
