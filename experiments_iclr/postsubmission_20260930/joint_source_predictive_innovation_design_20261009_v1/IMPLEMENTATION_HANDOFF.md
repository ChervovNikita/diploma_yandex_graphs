# Source implementation handoff — inactive

The design is a document only. No new adapter/optimizer/runner is written or authorized.

Native SeHGNN HGB IMDB body: `sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/model.py`, pinned original commit `e92bd37d0b803457339555684f139b4c8f3e160d`. Preserve all native dimensions, biases, residuals, heads, normalization, activation/dropout and semantic gamma.

Qualified existing affine boundaries are defined by `sehgnn_grouped_member_factor_adapter_source_20261009_v1/grouped_member_factors.py`:

- Grouped: `feature_projection.0`, `feature_projection.4`. Keep exact namespace/channel/group order and per-group input/output dimensions.
- Semantic: `semantic_fusion.query`, `semantic_fusion.key`, `semantic_fusion.value`, `fc_after_concat`.

Use its ownership/channel-order/contract findings and `PRIVATE_SITE_JACOBIANS.md`. These qualify the original sites/BE placement; the new additive rank-one composition, zero-output initial derivatives, mixed/native precision, private optimizer ownership and reconstruction still require a separate bounded whole-native discarded-copy engineering qualification. Zero semantic gamma and zero raw-family inputs remain known nulls; do not bypass them.

Source producer: `sehgnn_literal_raw_family_source_views_source_20261009_v1/native_family_views.py`, specifically `raw_support_plan`, `build_one`, `build_family_views`, under the unchanged ViewConfig/protocol. It removes MA/AM, MD/DM or MK/KM raw typed supports, preserves node counts and rebuilds all native feature/TRAIN-label propagation under the literal normalization/diagonal rules. Do not replace it with a cache mask, source logit mask, private edge operator or arbitrary augmentation. The exact full context and view bindings/RNG/cache immutability need preservation.

Existing native bank boundary: `sehgnn_grouped_bank_training_callback_source_20261009_v2/bank_training.py` provides full native contexts, ownership, differentiable `native_forward`, factual evaluation, coherent checkpoint/restoration and original selector. Its current `train_epoch`/post-step `correct` implement a different sealed policy; reuse the contracts, not that update as an implicit implementation of this design. A new version must expose P/S/component reductions and source-gradient recipients explicitly, accumulate at one old state, and charge all16 view forwards/backwards/replays.

Readout authority: `sehgnn_paired_six_arm_source_supply_pilot_source_20261009_v2/selected_diagnostics.py`, with existing `analyze`, `compare_selected`, and genuine-reference comparison code. Reuse exact saved selected current FP32 full/ablated metrics/masks; retain all member/label/current4×3U/D, absent/restored risk bases and both repair/harm contexts. Native single fresh per-label prediction availability remains whatever its actual source supports; never reconstruct it or invent a gate.

Every listed source is hash-bound in REUSED_BINDINGS.json. No current actual qualification receipt, server, array/checkpoint/data or outcome was opened in this design task. Root owns separate source authorization, engineering/competence admission, prospective freeze, resources and complete actual custody. Positive useful quality evidence is worth testing; this handoff grants no execution.
