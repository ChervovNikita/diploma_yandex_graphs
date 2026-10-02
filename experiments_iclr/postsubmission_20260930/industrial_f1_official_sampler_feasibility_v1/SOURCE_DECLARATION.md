# Independent protocol declaration

Proposed name: **F1 chronological GraphSAGE initialization pilot v1**. This is an independent protocol using the official RelBench architecture and sampler. It is not an exact author reproduction, a RelGT repair or an executed study.

## Source and dependency profile

- Task/database/model/trainer: RelBench v1.1.0 commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`.
- Sampler interface inspected: PyG 2.6.1. Proposed qualification stack: Python 3.11, Torch 2.5.1 with CUDA 12.4, PyG 2.6.1, compatible pyg-lib 0.4.0, PyTorch Frame 0.2.3 and LightGBM 4.5.0. These are candidate versions, not an installed or qualified lock.
- Freeze all remaining packages, compiled operator/build identifiers and the native GloVe text-embedding asset before actual preparation. Keep native text columns; do not silently drop them to reduce cost.
- Required operations are heterogeneous temporal disjoint sampling, table-feature conversion, scalar L1 forward/backward and validation. Reject a torch-sparse-only runtime.

## Exact changes to the released graph trainer

1. Separate development and final inference. Development creates only train/validation loaders and cannot load a full test target table or evaluate test. A final worker later receives masked test row keys and produces keyed predictions. A separate custodian owns final target evaluation.
2. Consume a custody-provided public database snapshot through 2010-01-01 with stable primary/foreign key IDs, train/validation task tables and masked test keys. Do not give the worker ordinary caches or raw files containing future target-window events or full test labels. Existing native download calls cannot implement this isolation by themselves.
3. In graph preparation, fit stypes and feature converters for timestamped tables only on rows through 2005-01-01. Retain the native untimestamped metadata policy explicitly. Use the fitted converter on the stable-ID public table through 2010-01-01; build edges/times from the full public table without reindexing the fitting subset. Keep converter statistics frozen. PyTorch Frame `Dataset(...).materialize().convert_to_tensor_frame` supplies this operation.
4. Keep the official `get_node_train_table_input` and `AttachTargetTransform`; request `disjoint=True`, `subgraph_type="directional"`, `time_attr="time"`, row-aligned `input_time` and native fanouts128/64. Add no synthetic fallback neighbors. Empty valid neighborhoods retain only their legal seed/context.
5. Preserve native architecture, Adam .005, L1, 10 epochs, batch512, sum aggregation and training-target 2/98 clipping. Use the earliest strictly best validation-MAE checkpoint, matching the source’s strict comparison. Export copied weights/buffers and keyed validation predictions.
6. Use num_workers0 for the initial qualification and pilot. Freeze validation sampling draws per seed and reuse them at each checkpoint comparison; restore training RNG state after validation. Record the exact RNG policy. This differs from the native successive-draw validation schedule and must be attributed.
7. Assemble predictions using `input_id`, with complete unique forecast-row coverage and the original `(driverId,date)` order. Never use driver ID alone as a prediction key.
8. Namespace prepared artifacts by task/source, actual row hashes, feature policy, dependency profile and sampler settings. Refuse an unrelated existing cache. No cache or data hashes were available in this scout.

No changes to the native temporal sampling implementation or legacy outcome-table cohort are proposed. Do not silently add a seed-time upper bound to the task’s eligibility SQL; changing that population would define another task.

## Tabular declaration

Raw LightGBM: retain the pinned `examples/lightgbm_node.py` view, disable AR labels, use sample_size0, ten MAE validation trials and training-fitted table conversion. Separate its final test evaluation through the same custody boundary.

History-engineered LightGBM: use the pinned user-study F1 SQL with its six upcoming race round/circuit columns removed from the feature view. Keep 44 engineered features plus the two source-default key predictors. The full source stype mapping and row keys remain available for alignment; materialize and fit on train only. Ten validation trials are fixed. Seed the native wrapper and record its actual Optuna/LightGBM sampler and booster seeds/settings; no search outcome or exact seeded runtime was verified here.

If both keys are excluded, preserve a separate row-key frame and remove their stypes only from predictors; the original trainer drops the validation key columns before its later mapping call, so a key-dropping command alone is insufficient. This extra adaptation is excluded from the preferred minimum cohort.

## Shared-backbone/private-factor declaration

Use three members and the same warm state per native seed. Start with factors on all relation-specific SAGE `lin_l` and `lin_r` maps; leave the table/temporal encoders’ parameterization native. Each member map is `diag(r) W diag(s) x + b`, preserving the native bias separately. Identity factors and copied scalar heads reproduce the warm model. The F1 schema gives 26 directed relations and two graph layers: 104 factorized linear maps, or 79,872 input/output factor scalars across three members, before optional head copies. Actual module inventory must be checked against admitted data/schema.

Reset Adam state equally for every continuation arm, including the unchanged warm copy. Copy all normalization buffers; hold them fixed during initialization differentiation, and declare member-specific running state for subsequent training. Average independent member L1 losses. Use the same sampled batch context for matched arms/members; charge every graph-trunk evaluation. Head-only control performs one shared trunk computation per batch. The independent ensemble reuses all three native fits while retaining their full original cost.

Filtering training residual cotangents requires a declared history-valid operator per cutoff and a `(driverId,date)` axis. Do not reuse the RelGT entity-only cache or a future-informed single graph. Qualify the scalar L1 cotangent, warm-function parity, mutable state and finite-step checks independently before route fits. MSE, larger heads, tied relation matrices, different sampling budgets or lower batch sizes are separate prospective changes.
