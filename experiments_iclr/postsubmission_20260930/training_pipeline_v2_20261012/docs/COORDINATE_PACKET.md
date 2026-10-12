# Exact coordinate-family v2 preparation

12 October 2026. Owned preparation files: `adapters/coordinate.py`, `compile_coordinate_matrix.py`, `tests/test_coordinate.py`, and this document. No native runtime/profiler file is edited. **Disabled source preparation, not scientific READY**: the child runs standard-library syntax/accounting checks only; root owns CPU/model/provider/resource/endpoint QA and allocation execution. No remote or benchmark work is authorized by this packet.

## Exact sources and mechanism coverage

The adapter imports the unchanged `coordinate_ensemble_source_v1/models.py::build_model`, SHA256 `a74a87dc26b7675a2d3fa0aaf0e7734b4786fa6c49411ef52f9bb21c0516dbc1`, with native Model/TABMModel SHA256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`. It binds/relocates only the source module's explicitly hashed `FROZEN_SOURCE` dependency; it changes no source bytes or forward/parameter semantics. Source construction defaults are members4, layers2, dropout.5, FFN multiplier1, affine LayerNorm, heads4. The actual historical stage1 training recipes override these to layers3/dropout.2/heads8, with the separate GT-sep comparator layers5/width512. The compiler binds the exact stage1 specification SHA256 `6da6a44bbc2b55118296b8e0de007330fd5d83983b57fb3d17fdd093e0cef44d` and imports none of its models.

| Historical alias | Source architecture / ownership | Supported operator |
| --- | --- | --- |
| coordinate_family_original | Unmodified original boundary R/S/B and shared residual maps/norms | SAGE |
| coordinate_family_factor | Live boundary BE; shared hidden W; unit-initialized private R/S; private biases wherever original bias existed | SAGE |
| coordinate_family_coordinate | Factor architecture plus independent fixed input/output gather buffers at each hidden map | SAGE |
| coordinate_family_permutation | Live boundary BE; fixed hidden gathers/private biases; interior R/S omitted | SAGE |
| coordinate_family_bias_only | Live boundary BE; shared hidden W; private biases; interior R/S and gather buffers omitted | SAGE |
| coordinate_family_single | Complete ordinary single; same-width reference TABM common maps/norms copied; ordinary boundaries omit private factors | SAGE |
| coordinate_family_untied | Four complete ordinary bodies, four disjoint optimizer objects, sum own-CE backward, one synchronized cohort selector | SAGE |
| coordinate_family_heads | One complete shared ordinary trunk, computed once, plus four private h→h→C GELU/dropout heads; unused native output map removed | SAGE |
| coordinate_family_gt_sep_single | Separate complete native GT-sep ordinary single | GT-sep only |

The four hidden sites in each SAGE block are mean-neighbor `lin_l`, root `lin_r`, FF1 and FF2. The column-vector map is `D_s P_out W P_in D_r h + b_m`: scale input, gather input index, shared W, gather output index, scale output, then private bias. Root `lin_r` remains bias-free. SAGE adds neighbor/root results, concatenates with the normalized self state, then uses the original FF dropout/GELU/dropout ordering and residual addition. Common affine LayerNorm and identity skips remain shared and unpermuted. Input/class axes and boundary BEBlock.W are never given hidden permutations. Output normalization and the original boundary output BE remain at the tail; heads use their own capable two-map tail.

These independent input/output permutations are **not a coherent full-network gauge or a tied conjugation**. A source lacking such a named variant is not expanded into one. The source permutation arm keeps learned boundary factors, so it is not globally factor-free. Generic all-map instead modifies every 2D weight with different factor/bias ownership and initialization; it cannot stand in for these aliases. Shared factors, biases, norms, index buffers and every optimizer state are retained by the exact factory and checkpoint bridge.

## Source recipe and full-budget translation

All historical arms use arithmetic mean of per-member softmax probabilities as primary serving, mean member TRAIN CE except untied's sum for backward, and strict earliest SELECT pooled NLL. Mean-logit NLL/accuracy is a same-checkpoint sensitivity only. **The coordinate original GNNM alias uses probability mean**, while the native packet's original alias uses raw-logit mean. They are different source/serving recipes and must not be pooled or claimed to reproduce each other.

Historical stage1 uses constant AdamW, betas(.9,.999), eps1e-8, decay0, evaluation every10 full TRAIN cycles, minimum1000/maximum2000, and patience200. AmazonPhoto uses LR.00003; CoauthorCS uses .0003, except its GT-sep comparator .00003. Main context width256/depth3/dropout.2, FFN1/heads8/LayerNorm; GT-sep width512/depth5. Historical storage-matched source defaults are AmazonPhoto single267/untied122/heads239 and CoauthorCS single264/untied88/heads247.

V2 retains **2000 full cycles**, with historical minimum1000/patience200 recorded as stop diagnostics and never used to truncate the authorized full-budget controller. `budget.unit=epochs` means one complete full-graph TRAIN cycle per adapter call. Historical untied performs **four actual optimizer step calls per cycle**, hence8000 calls over2000 cycles. One atomic run group pays for four optimizer trajectories/physical bodies; it is not one paid trajectory or genuine separately selected I4. The epoch unit avoids the optimizer-update controller's end assertion that would incorrectly equate2000 cycles with8000 actual calls.

`OptimizerBundle` retains all four actual AdamW objects, disjoint parameter groups and separate state dictionaries under a checked schema. Ownership inspection sees every parameter once; checkpoint/restore preserves each optimizer object/state. Untied gradients use sum of member mean CEs, while logged CE is their mean. Calls completed before an optimizer failure and failed step attempts are recorded. An internally failed optimizer step can partially mutate tensors; partial-cycle recovery requires root inspection/QA and is not asserted source-faithful by this preparation. No controller replacement or silent single-AdamW collapse occurs.

## HPO, controls and defaults

Each of thirteen search arms uses a deduplicated union:80 broad points (LR .00025/.0005/.001/.002/.004 × nominal width128/256 × depth2/3 × dropout.2/.4 × decay0/.0001),20 source anchors (LR .00003/.0003 × nominal width256/512 × depth1–5 × dropout.2 × decay0), plus actual applicable historical fixed defaults. Frozen FFN1/heads8/LayerNorm, source optimizer and2000-cycle controller stay fixed.

For ordinary single, untied and heads, nominal widths are resolved by the exact source parameter-count function extracted with AST, plus int64 index bytes, against a coordinate candidate's complete deployed tensor bytes. Integer width is chosen deterministically by nearest bytes with smaller-width ties. Coordinate/permutation index bytes are `8×M×L×9h` for FFN1 and are always charged. Actual fixed defaults are included even when current provider shapes differ from historic width matching. A new dataset keeps both actual source LR/default anchors as explicitly transferred source points. Full resolved numeric values, not source labels, deduplicate the domain. Counts are reported from actual deduplication; adding defaults can produce more than100 points, and no default is silently removed to force100. Matching is model-byte arithmetic; it is neither peak-VRAM matching nor a claim of identical initial functions across widths. Root must check actual storage reports and tolerances before an equal-byte scientific claim.

The nine historical arms remain explicit. Four additional prospective one-member controls use the exact factor/coordinate/permutation/bias-only factories with M1. Their boundary/permutation construction consumes different RNG shapes from M4, so identical initial functions across route counts are not claimed. Each search uses seeds71101/71103 on split0 once, pooled-NLL-first ranking, then proper loss/stable config ID. Choices are frozen across every prepared split with comparative seeds71201/71203/71207.

Six genuine I4 banks inherit the matching single's HPO: ordinary SAGE, ordinary GT-sep, and factor/coordinate/permutation/bias-only M1. Each has four independently constructed, optimized and selected complete learners. Member0 reuses the exact comparative single artifact; only three additional fits are charged. Actual model seeds differ by1000003×body; permutation seeds are model seed+1000, preserving the source local-generator separation without putting differing per-member seeds into the shared static config. Heads and synchronized untied are not labeled I4.

For equal100-point resolved domains, one dataset with J prepared splits has2600 atomic search run groups and3200 paid search optimizer trajectories,93J comparative run groups and102J paid comparative trajectories,57J banks/exports/CAL/evaluation jobs each,18J I4 banks/member0 reuses, and174J served route views. Exact materialized counts supersede this conditional arithmetic when source defaults increase a domain. Optimizer-call ceiling is `2000×paid trajectories`; full-cycle ceiling is `2000×atomic groups`. These are prospective maxima, not measurements or executed work.

## Compiler, endpoints and gates

Allocation-only `compile_coordinate_matrix.py` requires actual prepared node CUSTODY, frozen runtime containing `adapters/coordinate.py`, frozen exact coordinate/native sources, frozen exact stage1 source specification, explicit resource JSON, frozen fixture receipts and a fresh output directory. It verifies allocation identity before project metadata, binds individual frozen sources/inputs, and creates no GO receipt. It accepts existing prepared multiclass small-node TRAIN/SELECT providers. Resource keys are `dataset_id/coordinate-SAGE-source`; native all-map resource profiles do not certify coordinate gather/index/optimizer storage. An explicit fallback remains only a declared scheduling assumption. Source defaults never override the prepared class/feature counts.

Outputs:disabled `PLAN.json`, compressed study contracts, `SUMMARY.json`, and `SOURCE_COVERAGE.csv` covering all nine historical aliases plus explicit provider blockers. The plan partitions complete dependency groups over supplied positive logical capacities; remote capacity metadata does not authorize remote work. Its gates enforce whole coordinate search → comparisons → label-free export/CAL → evaluation. Root must integrate these with the massive study's **all-packet global embargo**, not treat one packet closing as permission to open another packet's CAL/evaluate targets.

Existing label-free node bank export and endpoint factories are reused through each declared exact adapter. All heldout targets stay in sealed separate files. Scalar CAL is prospectively standardized after the native serving pool,500 float64 Adam steps/LR.01/logT[-4,4], and is not historical coordinate calibration replication. Factory paths bind the frozen runtime; future live migration edits do not invalidate this packet's jobs.

## Blockers and qualification

No non-SAGE hidden coordinate/factor/permutation/bias placements, binary BCE, heterogeneous providers, link/molecule tasks or sampler interfaces are claimed. GT-sep is an ordinary single comparator, not a coordinate transplant. LastFMAsia remains blocked because no prepared v2 provider exists. Historical `as_pyg` edges and custom development role protocol differ from the v2 undirected/self-loop graph and four-role custody: this is an exact mechanism on standardized new roles, not exact historical data/protocol repetition. Original five datasets remain excluded whenever the packet includes original GNNM, with historical scores unchanged. This packet does not migrate or cover all249 predictive candidates.

Root QA must qualify full-input semantics, source construction/storage/gradients, four-object optimizer checkpoint and restoration,2000-cycle/cadence/selector behavior, partial-cycle failure handling, device/RNG custody, endpoint I4 ancestry, actual coordinate VRAM/time and serializer/disk cost. Present these costs and the concrete disabled plan before human GO.

Local metadata-only command:

```sh
python -B -m unittest tests.test_coordinate.CoordinateMetadataTests
```

After source admission, root tiny CPU fixtures are deliberately gated:

```sh
GNNM_COORDINATE_CPU_FIXTURES=1 python -B -m unittest tests.test_coordinate
```

Fixtures exercise all nine source-arm shapes/ownership, live boundary/shared/context gradients, private bias flags, actual four-optimizer calls/state restoration under the epochs controller, disjoint M1 learners and fixed-index restoration. They use prescribed synthetic graph/TRAIN/SELECT tensors only. Their numerical execution has not been performed by this child. They complement, without overwriting, the archived source analytic qualifications.
