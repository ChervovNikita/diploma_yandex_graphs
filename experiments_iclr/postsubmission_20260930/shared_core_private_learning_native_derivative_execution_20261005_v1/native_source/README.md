# Citeseer-HeaRT NCN TRAIN/VALID runner

Source only; no numerical imports/runs by this preparer. Native donor: Juanhui28/HeaRT at c447cbff4c493b60d14b6544c3c39d3b9c5ddff0, Citeseer `cn1` recipe. Exact donor model/util bytes are vendored. The runner avoids Planetoid bootstrap, OGB, sklearn and SciPy imports; runtime needs torch, numpy, torch_geometric, torch_sparse and torch_scatter.

## Operations and arms

Native encoder: input dropout.4, learned3703→256 projection/dropout.3, one puregcn layer, native JK scalar, no residual, no edge dropout. Native head: CNLinkPredictor, width256, dropout.3, LN, xlin, tailact, twolayerlin, complete common-neighbour supports. This is NCN, **not Collab NCNC**: no recursive missing-link completion or private recursive contexts exist. The adaptation tests member-specific endpoint interaction frames before the coordinate product in additive nonlinear endpoint/common-neighbour fusion. Private-head context features remain; no graph-coherence claim follows.

| `arm` | Explicit `member_count` | Construction |
| --- | --- | --- |
| native_single |1| Direct author NCN head |
| independent_member |1| Ordinary constituent with explicit ensemble_member_index0–3 |
| independent_frame_member |1| Ordinary constituent with one H at explicit axis_index=ensemble_member_index |
| unframed_f4 |4| Shared encoder, rank-one r/s head maps and private LN/beta |
| shared_frame_f4 |4| Same bank, one shared learned H |
| private_frame_f4 |4| Same bank, four learned H initialized at e0,e1,e2,e3 |
| same_four_frames_single |1| One native trajectory, four H products concatenated into one wider xij map |

Frames reuse the strict nonzero Householder primitive and zero-RNG axis initializer in `native_endpoint_private_factor_mechanism_assessment_20261005_v1/adapter.py`. The capable single reuses `endpoint_frame_same_operation_controls_20261005_v2`: blocks `[W,C,C,−2C]`, C=W/4, algebraic initial function preservation with accessible nonzero-block derivatives. It has3d² extra weights, not matched whole-model capacity. Banks require an explicit `factor_seed`; sorted r/s signs use a separate CPU generator and do not consume native model RNG. Use the same factor seed within a paired block across its three bank arms. No default cohort/initializer sweep exists.

Complete native fit: Adam encoder/head lr.001, weight decay0, batch1024; mask the whole current positive batch; native PyG negatives and PermIterator; native shuffled incomplete tail is dropped. Authenticated TRAIN has3870 nonself positives:3 updates per epoch,798 shuffled tail records omitted. Maximum9999 epochs, complete VALID every5, stop after eleven strict-improvement misses. First maximum of rounded4-decimal VALID MRR selects weights. Serving averages raw logits, with balanced per-query/member native log-sigmoid loss. Native global RNG is preserved per fit; architectures consume different dropout draws, so pairing is by seed block, **not identical negative/mask replay**. The TF32-off modern runtime is explicitly recorded, not a claim of exact historical CUDA reproduction.

## Qualification and launch

Stage this packet inside the authorized one-GPU repository phase. First root reviews the source and qualifies native sparse/encoder/head operations and actual runtime versions. Root's feature qualificationv2 reports exact float32[3327,3703] equality to independently pinned raw Planetoid features, TRAIN/VALID IDs and VALID anchor association; bind its direct remote RESULT.json hash as evidence. The runner independently authenticates the four available files, exact split counts and pool shape/association and accepts no TEST file roles.

Invoke `python3 -B <packet>/run.py --job <reviewed-job.json> --output <fresh-phase-path>`. Job keys:

- `source_review_approved=true`, `runner_sha256`, `source_manifest_sha256`, independently observed `expected_hostname`;
- `arm`, explicit `member_count`, `seed`, `paired_seed_block`, immutable `cohort_plan_sha256`;
- `ensemble_member_index` for independent constituents, matching `axis_index` for framed constituents; `factor_seed` for banks;
- `available_manifest_relative` and its SHA256, `qualified_feature_shape=[3327,3703]`;
- `feature_authority` and `negative_pool_authority`, each `{verified:true,origin:<description>,evidence:[{path:<phase-relative-qualified-receipt>,sha256:<digest>}]}`;
- `runtime_qualified=true`, exact `runtime_versions` object with torch/numpy/torch_geometric/torch_sparse/torch_scatter/CUDA version strings.

Route checks pin Linux, exact singleton authorized GPU UUID and repository/phase before data or writes. Use ordinary execution. All outputs stay inside the phase. No namespaces, server changes, sudo, retries or silent CPU fallback.

Outputs: CONFIG, progress, complete VALID history, selected encoder/head/Adam/RNG checkpoint, selected VALID logits bound to that checkpoint, final FREEZE with exact hashes, inclusive time and CUDA peaks. Failure preserves artifacts and writes FAILURE. Both successful and failed work remain evidence. No TEST loader/evaluator is supplied.

## Independent ensembles and representative screen

Four ordinary constituents are fitted **independently**, each with its own seed and native validation selection/stopping. `aggregate.py` averages their saved selected VALID logits after the entire prospectively fixed cohort finishes; it does not refit or select the ensemble. Its reviewed job uses the same authorization fields plus `mode=aggregate`, `complete_frozen_cohort=true`, `ensemble_family` (`ordinary_independent4` or `independent_frame4`), and four ordered `component_freezes=[{path,sha256}]`. Use distinct constituent seeds. Explicit `reuse_native_single_as_member0=true` can reuse the identical declared single as ordinary member0; all other component indices/operations must match. Framed ensemble axes are0–3. No candidate result drives component selection.

Minimum proposed representative comparative screen: all seven arm families, three prospectively fixed paired seed blocks, complete native schedule and all227×500 VALID negatives. There are13 component fits/block, or12 with declared exact single/member0 reuse;36 total fits after reuse. This is a proposal, **not an adopted/started cohort**. A complete native-single qualification first determines measured cost and healthy coexistence. Do not shorten schedules or evaluation pools to claim a representative result. TEST remains closed until root freezes the complete cohort and chooses a separately reviewed confirmation/evaluator. No original node-classification score changes occur.
