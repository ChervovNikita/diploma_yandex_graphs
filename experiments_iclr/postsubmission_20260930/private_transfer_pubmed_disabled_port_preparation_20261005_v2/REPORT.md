# Disabled Pubmed port: current NCN private transfer and native NCNC controls

Source preparation, 5 October 2026. All work was local source reading, copying, AST parsing, JSON bookkeeping and hashes. No model/module import, fixture, sampler, numerical execution, feature/data/checkpoint/prediction/outcome payload access, server or MacLink contact occurred. Every execution recipe is disabled. The immutable v1 packet and independent review were preserved. No original source, study, gate, canonical state or ledger was changed.

## Implemented source

The candidate remains **nonrecursive NCN**, width256, one-layer puregcn, input projection, **JK=True**, xdropout .4, encoder/head dropout .3, no edge dropout, the existing private rows and unframed initialization. Only its input width changes 3703→500; data custody changes to Pubmed's declared 19,717 nodes, 37,676 TRAIN pairs and 2,216 VALID positives with fixed500 negatives. No NCNC completion, Pubmed native dropout or completion/clamp parameter is substituted into this candidate.

`private_adam.py`, `transfer_step.py`, `heads.py`, `episode_geometry.py`, `native_episode_cycle.py`, `recursive_adjoint.py`, `qualify_training_step.py` and all vendor numerical bodies are byte-identical to their bound saved sources. The candidate `models.py` differs only at input width and the two audited outer-count assertions. The v2 runner changes gate/control-flow custody only: it accepts the FP32 receipt only after comparing its recorded TRAIN/feature members and runtime with the actual current loader/runtime, and applies the complete candidate/native cohort cost and budget contract before fitting. Its metric and numerical training/serving operations are preserved. `custody.py` uses the allocation PHASE, local copies of its dependencies, Pubmed input geometry and exact adopted TRAIN-member pin. Source execution needs separately authenticated root evidence, the singleton GPU identity, reviewed owned supervision, a fresh phase output and exact packet bytes before numerical imports.

The original current-gradient Adam rule, old-moment freezing, virtual→shared→recomputed-private schedule, replayed four dropout streams, half aggregate/half member BCE, raw-logit pooling and serving without adaptation are unchanged. The ordinary F4 control retains three native joint Adam commits. Candidate fit/cost authorization accepts only the three proposed F4 endpoint rules; extra architectures remain in the inherited discarded qualification, not in the proposed scientific study.

## Parameter assertions genuinely recalculated

The bound native GCN source creates `Linear(input_width,256)` and one JK scalar. One puregcn layer and its dropout carry no parameters. At width500 this is `500*256+256+1 = 128,257` encoder coordinates.

The unchanged `CNLinkPredictor`/head flags produce seven 256→256 linears and one 256→1 linear: dense weights/biases total `7*(256*256+256)+(256+1) = 460,801`. The factor rows total `7*(256+256)+(256+1) = 3,841`; five private LayerNorm weight/bias pairs add2,560 and beta adds1. Thus:

| Architecture/role | Expected source count |
| --- | ---: |
| F4 shared encoder + dense bases/biases | **589,058** |
| One private row | 6,402 |
| F4 private total | 25,608 |
| Untied4 outer, qualification only | 2,356,232 |
| Capable single outer/private, qualification only | 128,257 / 463,362 |

`PARAMETER_COUNT_DERIVATION.json` records shapes and formulas. The constructor assertions now use589,058 and its untied multiple. These are source-derived expectations, not a model construction result; exhaustive actual role/shape/count checks remain part of fresh qualification. The old1,409,026 assertion is absent from executable port source.

## What actual geometry adoption establishes

Root's bound `ROOT_ADOPTION.json` and report establish exact allocation TRAIN reauthentication and feasibility of all589 paired episodes in **seed0/cycle0**, including the44-row tail, with no redraw/drop/fallback. Mean common-mask removal was5.1334%, versus the retained roughly40–42% Citeseer geometry. This packet reads only that compact source/design adoption. It fetches no raw census or query payload.

The sampler still forms both endpoint and matched-random rows to preserve the same paired union support mask, although only endpoint candidate cells are proposed. Endpoint exclusion constrains inner queries; shared graph context and historical fitting remain correlated. Other seeds/cycles are unqualified. This geometry result admits no feature loader, model, derivative, cost, evaluator or fit.

## Dataset and runtime custody contract

`AVAILABLE_INPUT_CONTRACT.json` specifies a future **new allocation** manifest with exactly TRAIN, VALID, `heart_valid_samples.npy` and `gnn_feature`, under `available/pubmed/`; TEST is absent. `custody.load_inputs` permits only TRAIN+features in discarded candidate/native qualification and candidate cost. It checks the exact known TRAIN hash/bytes, finite float32 `(19717,500)` feature tensor, positive format/count/range/unique undirected facts, TRAIN/VALID disjointness and complete integer `(2216,500,2)` VALID pool with original row/candidate association. It preserves duplicates and other-VALID-positive collisions; no pool repair, regeneration or filtering is implemented.

| Evidence | Status |
| --- | --- |
| Allocation exact TRAIN bytes/count and seed0/cycle0 geometry | Root-adopted design evidence, bound here |
| Allocation Pubmed features/VALID/pool acquisition and safe loader | **Pending**; no values accessed or imaginary available manifest produced |
| Feature provenance | Exporter unresolved in saved notes; earlier source references a raw-feature equivalence receipt, but its exact bytes and applicability to new allocation inputs must be independently bound before admission |
| Sampler runtime | Geometry adoption measured Torch2.1.2+cu118, NumPy1.26.4, PyG2.7.0; recipes propose the current candidate's complete sparse/scatter versions; no new model-runtime compatibility claim |
| Current NCN native Adam/mixed derivatives on Pubmed | Pending fresh exact-source FP32 gate; Citeseer receipts cannot admit this manifest/input width |
| Native NCNC on allocation runtime | Pending; prior source/engineering custody from GPU77 is not imported as allocation qualification |
| Full589-episode candidate costs, native epochs, complete VALID/replay, memory/output and practical horizons | Pending; geometry elapsed/RSS cannot replace model/evaluator costs |
| Evaluator/selected-state agreement | Pending fresh allocation engineering; saved formula and evaluator bodies are source evidence only |

The recipes contain real source hashes and prospective fresh output paths, plus disabled authority fields and null unmeasured bounds. Those paths are proposals, not receipts or existence claims. Root must bind actual data/runtime/feature/pool evidence, reviewed ownership/caps and fresh outputs before any program can execute. This source supplies no new supervisor or launch adapter. `OWNED_NATIVE_FAMILY_CUSTODY_CONTRACT.json` defines the prospective metadata projection that a reviewed owned supervisor must truthfully produce; actual terminal/family evidence remains absent.

## Concrete separate disabled qualification recipes

1. **`QUALIFICATION_JOB_DISABLED.json` → `qualify_training_step.py`.** Fixed seed0/original block0 factor seed, full unchanged sampler cycle, first episode only; current source's sharedF4/capable-single/untied4 gates, zero plus two reachable private histories, native Adam parameter/moment parity, all-coordinate direct+mixed derivative oracle, live/detached virtual-value parity, recomputed and stale discarded commitments, complete committed serving and alias/mode/RNG restoration. Existing tolerances and episode sizes64/256 are byte-preserved. No VALID/TEST values, selected checkpoint or fit. All states discarded. Bounds and root exact-source admission are pending.
2. **`NATIVE_QUALIFICATION_JOB_DISABLED.json` → `native_ncnc.py`.** Separate fresh seed0 source-native NCNC factory and **two full native TRAIN epochs**, own optimizer, batch1024 and native shuffled dropped-tail behavior;36 commits per epoch. No VALID, private meta-gradient, engineering donor or fit. This is a discarded wrapper/TRAIN accounting and cost check, not reference numerical parity, evaluator admission or selected-state replay. Those still require separate allocation qualification.
3. **`COST_JOB_DISABLED.json` → `run.py`.** After the new candidate FP32 gate, one complete discarded endpoint-live TRAIN cycle, no VALID/early shortening. Detached/ordinary costs also require complete reviewed measurements before a future cohort freeze; the illustrative live job does not authorize a cost grid or fits.

The native source has candidate/reference NCNC training bodies with identical AST bodies, and preserved factory/validation projections. Static equality is not a substitute for real-dataset/runtime parity. Later evaluator qualification must check complete original-order logits, exact per-query tie-aware metrics and rounded4 dictionaries. Candidate versus author evaluator equality and selected-state replay remain unexecuted. No new tolerance, grid, early stopping rule or accuracy threshold is introduced.

## Distinct strong native NCNC wrapper and true independent four

`native_bodies.py` is copied byte-for-byte from the saved source-v2 native module. Its Pubmed factory retains native one-layer puregcn/JK, .1 encoder/head dropout, .3 input dropout, `IncompleteCN1Predictor` depth1 completion, `twolayerlin=False`, native scale/offset/alpha/pt/degree/splitsize settings and own .001 encoder/head Adam groups. Its masking, negative sampling, PermIterator, loss and validation bodies are unchanged. This differs deliberately from the NCN candidate.

`native_ncnc.py` runs **one independently trained member per fresh process/output**, with its own complete encoder/head/optimizer and RNG initialization. Each receives its own normalized native loss with no1/4 scaling. It preserves maximum9999 epochs, VALID every5, eleven misses, first maximum of rounded4 complete MRR, no initial VALID serve, and the native dropped shuffled tail. It never imports an engineering or prior selected state to fit. `native_state.py` reuses the saved complete-state/RNG helper bodies exactly.

`native_pool_replay.py` is a separately gated zero-update stage. It first checks a root-approved metadata-only complete-four inventory, every distinct member job/output, every final artifact inventory and successful owned supervisor terminal relationship, and all selected artifact hashes before opening any selected freeze containing quality. It then checks all four freeze identities and selected-file bindings before runtime/data setup or any replay payload deserialization. The actual loaded inputs/runtime must match that metadata. It checks each member's exact declared NCNC fit and runtime identity, reconstructs each in a verifier, restores full serialized pre/post-VALID states/RNG, checks the inherited128*float32-epsilon score tolerance, exact per-query/rounded metrics and selected-MRR identity, and exact post-serve state/RNG, then pools **raw logits**. Native setup checks that all five numerical dependencies resolve to the packet's pinned files. Member0 supplies the prospectively disclosed single comparison. There is no ensemble checkpoint selection or retraining. This is source preparation of a true independent ensemble, not a jointly trained bank; replay is unexecuted and cannot currently admit a result.

`NATIVE_WRAPPER_CONTRACT.json` records that distinction and the pending source/runtime/evaluator/resource/owned-custody gates. `NATIVE_FIT_JOB_DISABLED.json` and `NATIVE_POOL_REPLAY_JOB_DISABLED.json` remain disabled.

## Fixed future proposal, no fitting admission

`COHORT_PROPOSAL_DISABLED.json` contains only the requested conditions across base seeds0,1,2: F4 endpoint live/detached/ordinary joint, source-native NCNC single and true independent four. Candidate factor seeds are copied from the original three blocks; native member seeds are `s,s+5,s+10,s+15`. Member0 is reused for the single. This yields15 served comparisons and **21 unique fits if later adopted**:9 candidate/control plus12 native fits.

The candidate proposal retains60 complete cycles, VALID every5, eleven misses and the first rounded4 maximum. Native baselines retain their competent native schedule. These have different fitting/exposure/selection budgets; all work and failures must be charged and disclosed. No F1/random/untied fit, width/rate grid, threshold, weaker baseline or shortened horizon is added. The unchanged original30 quality/promotion and attribution decision remains a prerequisite. The complete fixed39 family supplies descriptive mechanism analysis, including companion9 contrasts, with no new companion acceptance threshold; unavailable diagnostics remain unavailable. Successful geometry does not replace either documented prerequisite.

Static verification covers syntax, exact copied bytes/function bodies, task-only edits, manifest/recipe pins, explicit parameter formulas and all disabled flags. It executes no numerical source. Real input acquisition, fresh source/runtime/numerical/evaluator qualification, measured model/serving costs, representative horizon practicality, owned bounds and a separately reviewed prospective scientific release remain outstanding. No predictive or novelty claim follows from this packet.

## v2 custody repairs

`REPAIR_REPORT.md` maps the four independent P2 findings to the exact source-only repairs. Candidate and native fits now share checks for enabled prospective cohort scope, byte-pinned complete candidate cycles, native epoch/complete VALID/replay/pooling costs, explicit selection budgets and authenticated horizon practicality. Every recipe remains disabled with unresolved real evidence and null bounds. No model schedule, seed, numerical operation, selector, metric or tolerance changed.
