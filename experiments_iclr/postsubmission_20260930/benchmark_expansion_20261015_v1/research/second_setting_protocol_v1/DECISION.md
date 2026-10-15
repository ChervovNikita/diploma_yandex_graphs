# Second-setting transfer protocol and choice

15 October 2026. Source and metadata preparation only. No runnable scientific JOB, launch, source recipe edit or new literature search is included.

## Decision

**Select native PubMed / PolyFormer-Cheb for next source preparation only.** Root's completed provider-only check confirms literal split0 TRAIN counts [3943,3943,3943] and VALID counts **[72,1919,1952]**. All three classes occur, but class0 is only **72/3943 = 1.83% of VALID**. Keep the original whole-VALID accuracy selector; require per-class support/accuracy/NLL and descriptive macro metrics alongside it. This is not balanced validation. One class0 error changes its recall by about1.39 points, while changing whole-VALID accuracy by about.0254 points.

Root accepted this setting for source preparation, with **no 24-fit launch or additional scientific budget**. Keep Coauthor-CS / PolyFormer-Cheb as the fully specified nine-supported-class alternative. This revises the earlier CS preparation priority because its native class exhaustion restricts the scientific question and its measured legacy private-credit memory is substantially larger. It does not adopt a new run matrix or infer cheap exact residual training from old smoke timings.

The original dataset inventory lists Roman Empire, Amazon Ratings, Minesweeper, Tolokers and Questions; PubMed is outside that five-dataset rerun restriction. The saved manuscript states the same five. This is an allowed new native setting, **not an unused dataset**: the PubMed exposure ledger and numerous retained studies document repeated development. Protected confirmation and pristine TEST status are not established.

The completed provider gate measured coverage, not predictive quality. It took6.412 seconds, with zero model construction, task forwards, optimizer updates or CUDA context; the native provider container was deserialized, but no TEST label was indexed or TEST-specific tensor/score loaded. `PUBMED_PROVIDER_CLASS_SUPPORT_22.json` preserves the actual result and source/role receipts. No repeat materialization or additional metadata probe is needed. Do not search split seeds or choose another setting from new scores. Scientific fit admission remains separate.

## What the native CS split can and cannot answer

The verified author splitter uses `round(.6*N/C)` TRAIN nodes per class, taking every node when a class has fewer than that cap. For CS, N=18,333, C=15 and the cap is 733. All 708/462/429/371/118/420 nodes in classes **0/1/3/6/9/12** enter TRAIN. Thus six classes, 40% of the class identities and 2,508 nodes, have no VALID support. Those classes also have no remaining heldout nodes under this source splitter. All ten literal source seeds have this exhaustion; a different seed cannot cure it.

The fixed source split 0 / seed 1941488137 has 9,105 TRAIN and 3,667 VALID nodes. Keep the source masks and 15-class head intact. Train all 15 classes and report the exact per-class TRAIN/VALID counts. Absent-class recall/NLL is **undefined**, not zero; report null with support zero. Macro recall over the nine observed VALID classes must be labeled as such. False predictions into an absent class can still be counted, but they do not measure that class's recall. Do not silently create a stratified replacement split or call the selection all-class validation.

CS remains a legitimate restricted transfer question: a different prediction unit, a polynomial/attention backbone, no BN and 24.28 times MolBACE's VALID record count. Its nine supported classes can test usable specialization. It cannot test performance on the six exhausted classes, and the dependent nodes of one transductive graph are not 3,667 independent graphs. The paper's 96.49% contextual CS score is neither our achieved score nor an expected effect size.

## Exact alternative and exposure

PubMed-Cheb has 19,717 nodes, 11,829 TRAIN and 3,943 VALID on literal source split0 / seed1941488137. Its verified TRAIN class counts are [3943,3943,3943]. The old documented floor-stratified TRAIN counts [2461,4643,4725] imply every total class population exceeds the new3943 cap, so the CS class-exhaustion mechanism is absent. The original saved provider audit omitted VALID class counts. Root's new direct support check now verifies **[72,1919,1952]**, resolving that uncertainty while exposing substantial minority underrepresentation. Preserve those exact masks and counts; do not amend the source split to improve balance.

| Native author preset | CS / Cheb | PubMed / Cheb |
| --- | ---: | ---: |
| Hidden width / K / layers | 64 / 4 / 1 | 128 / 6 / 1 |
| Attention heads / FFN / multi / q | 8 / 32 / .5 / 1 | 16 / 64 / 2 / 2 |
| Dropout / attention dropout | .8 / 0 | .30000000000000004 / 0 |
| Ordinary / attention LR | .0005 / .01 | .0005 / .0005 |
| Ordinary / attention WD | .001 / .001 | .001 / .0001 |
| Old native smoke peak allocated | 5.425 GiB | 1.645 GiB |
| Old G0 smoke peak allocated | 12.430 GiB | 2.449 GiB |
| Old private-responsibility smoke peak allocated | 21.731 GiB | 3.482 GiB |

These are exact author presets, not a mixture of their settings. Both have native LayerNorm and **no BN**. Do not add a BN arm or private-normalization state. The old private-responsibility smoke used BE and is a memory warning/comparison, not qualification of the proposed residual composition. Neither setting has an exact full-fit ETA from these four-update invocations. MolHIV / plain GIN is a retained alternative with a larger validation population, but keeps the molecular task and substantial minibatch work; its measured GIN-VN sibling does not establish a cheaper plain-GIN transfer.

PubMed history is counted as exposure. The retained CORE9, native-reference acquisition, factorized-single six, private-hop eighteen, private-Adam three and LM4 (36 groups / 54 trajectories) families remain documented, including their failures. These overlapping ledgers must not be summed as unique new fits. The old reference protocol used **Mono, K2, hidden256, two layers, LR .005, attention WD 1e-8, dropout .5 / attention dropout .8, split190111 with VALID3942, seeds9101/9203/9307, probability pooling and internal/all-23-site factors**. The proposed Cheb protocol uses its author preset, source split0 with VALID3943, boundary-only G0 or input-only residuals, seeds101/202/303, mean raw logits and the matched long controller. Those old checkpoints, predictions and selectors are not reusable controls. This new identity tests a distinct native recipe on an encountered graph; it does not rescue any closed old recipe or establish fresh confirmation.

## Minimal cold-fit comparison

Exactly **24 physical trajectories in one chosen setting**, eight per paired seed 101/202/303, produce **18 bank results** (six per seed). The two candidate presets are alternatives, not a 48-fit matrix:

| Bank | Physical trajectories per seed | Selection and reuse |
| --- | ---: | --- |
| Native M1 | 0 additional | Exact native I4 body0 reused, under the same controller |
| Genuine native I4 | 4 | Four independent native bodies, each own-selected; mean saved raw logits |
| Original boundary GNNM (G0) | 1 | Input/output BE, one common shared-bank selector |
| Uniform residual-only | 1 | No BE; identical residual law/partition to next two arms |
| Structured residual-only | 1 | Fixed TRAIN degree responsibility |
| Permuted residual-only | 1 | Exact within-class whole-vector permutation control |
| **Total** | **8** | **24 trajectories across three seeds** |

Use the retained independent-body offsets [0,100003,200003,300007], body0 offset zero. M1/I4 body0 is the same completed fit and charged once. No exact completed CS/Cheb controls were verified in the inspected authorities; for the proposed PubMed identity none are established as reusable here. Treat all 24 as cold until an exact recipe/split/seed/controller/selection/serving receipt proves otherwise. Engineering callbacks are not selected controls.

This is the narrower **structural-transfer** question requested now. The earlier residual-removal proposal also required BE-plus-residual: including it would add three shared fits, totaling **27**. This 24-fit comparison can compare whole G0 and residual procedures, but cannot attribute that difference purely to removing BE. It makes no such isolated removal claim. No discarded factorized-single, equipped, joint-untied or capacity-control arm is added.

## Source-compatible residual and objective

Reuse the portable input residual mechanics at native `lin1` **output**, before attention, preserving every polynomial token and native operation. CS output is [N,5,64]; PubMed output is [N,7,128]. One route's residual acts on every token with the same route matrices:

`h -> h + U_m GELU(V_m h)`.

Keep rank8, the existing isolated CPU FP32 Gaussian-QR initializer with seed+901003, disjoint down subspaces when 4×8 fits the native width, and zero up matrices. Width64 adds 4,096 learned coordinates (16,384 FP32 bytes; 32,768 dense two-moment bytes); width128 adds 8,192 (32,768 / 65,536 bytes). Zero-up evaluation initially equals a fresh native constructor. This is native initialization, not preservation of a competent trained checkpoint.

Residual-only removes the untrained input/output BE factors and their optimizer ownership. Keep every native parameter, LayerNorm affine, head, graph/basis preprocessing and Adam group option/default. Assign only residual down/up to the original `lin1.weight` group with that preset's ordinary LR/WD. The output `lin3` hook is observer-only, with no output residual/factor. Original G0 retains the exact native boundary BE insertion at `lin1` and `lin3`.

For all three residual arms, use four source route forwards at one old state and the same **two-collection block updater**: all native parameters receive F, the mean of four original full-TRAIN masked CE losses; only residual down/up receive W, the mean of fixed-weight per-node CE under the same native mean reduction. Uniform uses w=1 and W=F. This makes it the same-operation control for the two prospective responsibility arms, with the same mathematical ordinary objective. Native/G0 retain their original update. Coupled Adam decay remains in the optimizer; do not add it to W or change attention groups. No clipping, auxiliary loss, minibatch renormalization or new scalar-objective claim follows.

Existing portable residual code is reusable tensor/ownership machinery. The immutable MolBACE v2 extension is **not** a node port: it checks molecular widths, parent JOB identity and private BN. A small explicit node composition with exact same-old-state gradient/restore checks is still required; no such code is claimed implemented by this plan.

## Fixed structural task

Use **incoming native degree** at each TRAIN node, counting the exact edge records of the already-source-preprocessed public transductive graph. Retain duplicate multiplicity and any existing self records. Do not add self loops, coalesce, symmetrize again or use normalized propagation/basis edges for the count. This degree describes neighborhood size in the factual graph; it does not inject labels or change graph information.

Within every observed TRAIN class, reuse the retained exact fractional tie-overlap quartiles q and `w=.5+2q`. Each route retains the original class mass, each TRAIN node's mean weight is one and weights stay in [.5,2.5]. Equal-degree ties receive equal vectors; no ID tie breaking. Only TRAIN labels determine assignment; no VALID/TEST labels, losses or errors enter the table. Public topology can include all nodes under the native transductive protocol.

The permutation control uses the whole four-route vector, shuffled once within class by isolated seed+7030031; it preserves all weights and class/object mass. No redraw. Record degree, tie-group, nonuniform-vector and exact table/permutation hashes before training. If the entire table is uniform, there is no structural intervention to test; stop preparation and report that fact. Do not switch to homophily, centrality, altered quartiles or another strength after observing support or outcomes.

## Convergence and scientific release

Use one prospectively fixed literal source split 0 and the three paired optimizer seeds, keeping full native TRAIN/VALID and mean raw-logit serving. Use the retained matched controller: minimum2,000 full epochs, then patience100 checks under the original Selector bookkeeping, cap10,000, complete VALID accuracy each epoch, first strict maximum, no checkpoint0 and no loss tie breaker. This is an explicitly extended matched protocol; native source reference is at most2,000/patience250 with its own advancing-RNG repetition policy, and remains separate.

There is one full-graph Adam transition per epoch per physical trajectory. The conservative minimum obligation is **48,000** transitions; the cap is **240,000**. Across all fits, logical TRAIN route forwards range from at least120,000 to600,000, before activation-checkpoint recomputation. The two-collection residual control gives at least66,000 and at most330,000 production reverse collections. Fit VALID route forwards have the same120,000–600,000 range plus60 selected-finalization forwards. Twelve shared-bank selected route collections add48 full-graph VALID forwards; native M1/I4 use their saved selected exports. Charge actual recomputation/diagnostic/failed work separately. A cap is unresolved convergence, never an automatic extension or seed replacement. No wall-time estimate or four-hour transfer-fit cap is inferred from MolBACE.

Only release scientific comparisons after all24 terminal fits and required selected collections complete. Preserve every signed seed contrast, mean/sample SD over exactly three seeds, failures, traces, selected/completed epochs, stop reasons and resource/I/O costs. Primary bounded signal: positive three-seed mean structured-minus-uniform **and** structured-minus-permuted served accuracy; report permuted-minus-uniform and native M1/I4/G0 comparisons separately. Keep signed CE/NLL and individual/mean/worst member accuracy/NLL. No fixed practical margin, NLL veto or significance claim is invented.

Usable specialization must be measured in native decision space: exact pooled node repairs/harms, new/lost correct-member alternatives, alternatives still lost in pooling, all-member wrong events and strict common-false-rival witnesses. In multiclass prediction, all-member wrong alone does not imply a common rival. Report both, tied-score cases, observed-class support and per-class harms. Do not substitute binary pair AUC diagnostics, train a VALID gate or treat nodes/routes as independent replications. Any degree-regime diagnostic uses fixed TRAIN-derived boundaries and does not affect selection. An accuracy gain without the usable-difference signature remains a result for the whole procedure.

## Small provider gate and engineering required before fit admission

1. **Provider gate completed by root; reuse it:** `provider_pubmed.py` verified the existing processed-data/source-utils receipts and reused the paid twice-materialized native split0 TRAIN/VALID-only role cache. It reconstructed and compared the saved TRAIN/VALID mask hashes without regenerating a full-label split. It deserialized the existing native provider container and indexed only proved TRAIN/VALID label IDs; no TEST label was indexed, and no TEST-specific tensor or score was loaded. It exported TRAIN/VALID class counts and role/label/ID digests with zero model, task forward, optimizer, CUDA context, feature transform or basis generation. PubMed passes the three-class support condition; **72-node class0 support remains a material limitation**. Do not repeat this check. CS's six zero supports remain disclosed from its source exhaustion proof; CS is not the selected next source setting.
2. **Source composition:** implement only the explicit node residual block-credit bridge and saved selected-node route collector, using existing native callbacks, portable residual/table/updater and full-epoch runner. Preserve source identity and state. Do not change frozen MolBACE code. Root reviews exact bytes and source/provider pins before numerical access.
3. **CPU lifecycle on each exact uniform/structured/permuted composition in the one selected setting:** initial fresh-native eval equality; nonzero up-gradient / zero down-gradient on the first update; shared-F/private-W derivatives at one old state; all native Parameters and Adam group options/defaults preserved; checkpoint after one update; fresh constructor/restore; exact full VALID replay; one resumed second update; changed-table/method rejection. This is six candidate optimizer transitions in total, plus every diagnostic derivative/replay. No engineering state initializes science.
4. **One-worker CUDA qualification of each exact composition:** complete full-graph TRAIN update/VALID, selected-state save/fresh restore and second update, finite failure cleanup; record actual constructions, route forwards including activation-checkpoint recomputations, reverse collections, six total candidate Adam transitions, TRAIN/VALID times, peak allocated/reserved GPU and host memory, checkpoint/export bytes and collection overhead. Existing M1/G0 engineering can be reused within its exact source scope; the new long controller still needs source/selector verification. If more new workers are proposed, first measure that exact concurrency; otherwise keep one serial new worker under root's slot lock/admission. Stop on failure and retain all work.
5. **Freeze resource bounds after measurement:** root sets positive GPU/host reservations, live allowed peer identity, finite per-fit wall/output limits, original `.venv`/publication and no-retry rules. Unknown processes block admission. The old CS21.7GiB private-credit peak and PubMed3.48GiB peak cannot certify the new updater, full-fit speed or a concurrent allocation. No scientific fit or campaign GO is granted here.

## Reuse boundary

Reuse source presets/operators, provider identity/mask proofs, native body0 within the new I4 cohort, tensor-level residual/table/updater primitives, original full-epoch runner/selector, and existing CPU-only saved-output arithmetic after task adaptation. Retain old engineering costs and exposure facts. No paid scientific CS control is established; MolBACE control scores, molecular/private-BN checkpoints, other CS backbones/controllers, old PubMed Mono/amended-split/internal-factor predictions, old probability pools and source-reference horizons cannot substitute for new exact matched controls.

`PLAN.json` gives the finite count, source identities, completed provider authority and work arithmetic; `SOURCES.json` pins the inspected local source/metadata. This protocol agent performed no remote/Git operation, local tensor/checkpoint load, new score calculation or scientific execution. Root's separate6.412-second provider read is explicitly receipted above.
