# Three missing residual-only MolBACE fits

Prepared 15 October 2026 for root review. **Unlaunched.** This packet uses the existing full-epoch runner and existing mechanisms. It creates exactly three candidate jobs; completed controls are reused with their original producer identities. It does not alter the broad 6,460-fit roster or the forthcoming normalization successor.

## Question and reason for this contrast

Under the same private BN law, does replacing BE with native-initialized additive private routes help? The completed private-BN base scored **73.6142% AUC**, BE plus correction **72.1856%**, geometry **73.6508%**, and both **73.0281%**. The old shared-BN correction gain therefore does not establish a gain under private BN. Residual-only is a distinct missing method, not a selected winner.

The three new jobs use seeds **101/202/303**, all 1,210 TRAIN graphs, the complete official 151-graph VALID scaffold split (130 positive/21 negative), and no TEST. Retain every seed, failure and cost. No selected-seed pruning, coefficient/rank/site search, rescoring rule change, or fresh-confirmation claim follows from this small, already-explored VALID population.

## Exact implementation and reuse

The new `session_extension.py` composes, in order:

1. **`portable_private_input_v2` residual-only:** remove the untrained input/output BE hooks and parameter ownership; retain the original native body, parameters and optimizer groups; attach the existing four-route residual wrapper.
2. **Existing molecular update:** `molecular/session_extension.py` with both auxiliary and input-adapter switches false. This retains the exact own-mean-BCE training implementation used by the completed correction control, without attaching a second residual.
3. **Exact `private_bn_state_v3`:** private running buffers, shared native affine parameters, TRAIN updates only its own route and VALID reads its own state. No recalibration or arithmetic/mixture buffer substitution.

The residual is `h + U_m GELU(V_m h)` at the complete native AtomEncoder output. At MolBACE width 300, the current v2 residual and old correction use the same rank 8, float32 Gaussian QR frame, seed `seed+901003`, four disjoint initial down subspaces, and zero up matrices. The native callback asserts float32 construction. Thus the dimension-general v2 fallback is inactive here. The primary contrast removes BE under the same residual law, placement, initialization, own-loss update and BN policy; it is not an isolated initialization-only test. Hook/module registration and checkpoint identities differ and are disclosed.

All native atom/bond inputs, graph operators, native residual=False, mean graph pooling and output head remain unchanged. Four fresh private BN banks clone the native initial state. With zero U and BE removed, all evaluation routes initially equal the native function. Different learning after initialization is a hypothesis, not guaranteed diversity.

The new jobs copy the completed common recipe, native callback, policy, data paths and split. The only recipe metadata addition is the repository root required by the current portable adapter. Native settings remain five GIN layers, width 300, dropout .5, batch 32 and Adam .001 with original defaults. The selector remains first strict maximum whole-VALID AUC, minimum 100 epochs, patience 50 counted under the original controller, cap 1,000; no checkpoint zero selection. A cap is retained as unresolved convergence, without an automatic extension.

## Controls and reporting

`SPEC.json` binds nine exact wrapper control rows: original G0, private-BN base and private-BN BE-plus-residual for each seed. It also binds all six native M1/I4 summaries and their selected epochs; M1 is I4 body0. Native I4 retains four independently trained, independently selected unchanged native bodies. The original G0 remains in the report even though it uses shared BN.

Primary: residual-only minus private-BN BE-plus-residual. Required comparisons: private-BN base, original G0, native M1 and genuine native I4. Report every signed AUC difference and mean/sample SD only when all three seeds complete. A G0-only recovery is insufficient for the research target. Include mandatory BCE, selected/trained epochs, traces, stop reasons and measured cost. Native RESULT BCE and old I4's saved-prediction float64 uniform-graph BCE have slightly different numerical reductions; retain their provenance and do not interpret rounding-sized differences. Any later common-precision saved-output analysis must be explicit and must preserve selection/aggregation.

Three seeds on one scaffold split do not establish generalization, statistical significance or novelty. No retrospective probability averaging, calibration, gate or geometry ingredient is selected.

## Parameter/state and cost accounting

The candidate has **19,200** private trainable parameters (`4×2×300×8`), no BE parameters and zero additional learned BN parameters. FP32 residual weights occupy **76,800 bytes**; their two Adam moments add **153,600 bytes**, plus actual scalar steps. Four private BN banks add **144,320 buffer bytes** for the declared native GIN, while canonical native buffers remain registered. Native affine parameters remain shared.

Construction records actual native/private/total parameters, parameter bytes, removed BE count, native/private buffer bytes and overhead relative to native M1. Each checkpoint records actual total/private optimizer tensor bytes including scalar steps. Dense-moment estimates, serialized checkpoint bytes and peak process/CUDA memory are separate quantities. The BE count is measured from the old wrapper; its source formula is `4×(sum(atom vocabulary sizes)+3×300+2)`.

The three exact BE-plus-residual/private-BN controls took **4,491.63 seconds** across 149/152-epoch trajectories, about 10 seconds per epoch. Rough planning: **75 minutes** if stopping is similar, **8.3 hours** at the three 1,000-epoch caps. Residual-only time is unmeasured. Bound: 3,000 epochs, at most 114,000 optimizer transitions, one new fit at a time. Reuse costs zero new control training but retains all historical costs. A failure halts later admission; no automatic retry/resume or seed replacement.

## Readiness and execution

`JOB_seed101.json`, `JOB_seed202.json` and `JOB_seed303.json` are executable descriptors for the unchanged `runtime/run_fit.py`; their exact future argv is in `SPEC.json`. No launcher is added. Root review and guarded authorized-host source/data/control checks precede any launch. The old provider cache lacked a historical control hash; role pins and the currently bound cache hash preserve that disclosed limit.

Static descriptor, hash, composition-order, ownership/accounting and checkpoint-binding mock checks pass. Existing actual CPU MolBACE v2 residual-only checks passed initial function parity, two updates and fresh restore. **The new private-BN composition has not had a numerical lifecycle check.** Root should use the existing engineering machinery on these exact bytes to check initial parity, route BN isolation, complete VALID and fresh checkpoint restore before the three scientific fits. This is engineering overhead, not a fourth scientific fit. Broad final-serving dispatch is not silently admitted for this new identity.

## Different transfer setting, separately prospective

Use **Coauthor-CS / PolyFormer-Cheb**, a full-graph node task with native polynomial/attention operators and no BN. Keep its declared source preset: hidden 64, K4, one layer, eight heads, dropout .8, dprate0, ordinary/attention Adam groups .0005/.01 with WD .001. No synthetic BN arm is needed.

Required arms are native M1/I4, original G0, portable BE-plus-residual and residual-only. With three fixed paired seeds and no proven exact reusable controls, this is **21 trajectories**: 12 native bodies plus nine shared banks, with M1 reused as I4 body0. Use a predeclared source split and full TRAIN/VALID; the existing long controller is minimum2,000, patience100, cap10,000. The cap is 210,000 optimizer transitions and 480,000 native-route TRAIN forward equivalents, excluding validation/recomputation.

Historical four-update smoke peaks were **5.43 GiB native / 12.43 GiB shared**, not full-fit timing. Exact Coauthor-CS residual lifecycle, full-epoch CUDA timing and checkpoint/resource qualification are needed before a defensible wall-time budget. No transfer jobs are created here, and MolBACE does not justify omitting any of these controls.
