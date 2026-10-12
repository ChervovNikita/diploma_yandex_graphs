# Native NCN/NCNC link packet

Status: complete official TRAIN/SELECT and separately sealed TEST packs are prepared. Endpoint/compiler CPU QA passes. `qa/link_materialized_source_controls_v3` is a materialized, disabled 1,250-fit source-controls packet with 1,276 run records. Required original-GNNM link adaptation, CUDA resource qualification, root plan gates and human GO remain blockers. No scientific fits or official TEST scores were produced.

## Source and scope

The adapter loads unchanged `GCN`, `CNLinkPredictor`, `IncompleteCN1Predictor`, and `PermIterator` from GraphPKU/NeuralCommonNeighbor commit `11d597013750da17ce7468e344bec756a7af39a4`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `graph_link_competing_family_source_20261003_v1/source/ncn__model.py` | 39,783 | `5b6cfe26074c83d57900f589b37a4ef57fc7b8eb6dff3179fc52e3f6943aa5e3` |
| `graph_link_competing_family_source_20261003_v1/source/ncn__utils.py` | 9,208 | `29df3b9cf82ae5885ece0fac010b81383f1fb5391068649c88b7d2bc98ad86be` |

Source receipts are enforced against those fixed pins, independently of the caller's claimed commit. This packet admits native single scorers and separately fitted I4 bodies, with NCN or NCNC. `compile_link_matrix.py` prepares disabled original-source single/I4 domains and prospective study rows. `adapters/link_search.py` ranks only complete declared own-selected TRAIN/SELECT search records. `adapters/link_endpoints.py` exports label-free raw banks and evaluates them against separately sealed targets after closure. Factorized NCN/NCNC variants are outside this packet; CAL is absent.

`adapters.link:make_adapter` has the common engine factory interface. Module import is stdlib-only; numerical providers are loaded inside the factory. Runtime dependencies are Torch, NumPy, PyG, torch_sparse and OGB. Data preparation additionally requires pandas. The packet performs no dependency installation.

## Native collab recipe

The caller must freeze every constructor, optimizer, support, sampler, metric and tail setting in its run config. The recorded native collab recipe is:

- GCN: one layer, hidden width 64 at the native anchor, dropout 0.1, LayerNorm and residual enabled, `gcn`, no JK, edge dropout 0.25, feature dropout 0.25, tail dropout 0.05, `max_x=-1`.
- Predictor: constructor `num_layers=3`, dropout 0.3, no edge dropout, LayerNorm enabled, all common neighbors (`cndeg=-1`), `use_xlin=True`, `tailact=True`, `twolayerlin=False`, beta initialized to 1.
- NCNC additionally: alpha 1.05, scale 2.5, offset 6, unsampled residual neighbors in train and inference, prior threshold 0.1, `learnablept=False`, recursion depth 1, `splitsize=-1`.
- Adam: encoder LR 0.0082 and predictor LR 0.0037 in separate parameter groups, weight decay 0, betas (0.9, 0.999), eps 1e-8, amsgrad false.
- Full 100 epochs; TRAIN batch 65,536; query batch 131,072; strict first-tie SELECT Hits@50 selection. No early stopping.

The predictor source accepts `num_layers` but its NCN/NCNC architecture does not use that argument to vary depth. The configuration records it for exact constructor custody, without making a depth claim. The GCN's actual use of tail dropout also follows unchanged source behavior.

## Support, sampler and objective

Each update removes the supervised TRAIN record indices from the positive record list and constructs the native symmetric sparse support graph from the remaining records. `native_record_multiset` preserves source behavior if repeated canonical pairs occur: removing one record may leave the same pair through another record. `require_unique_pairs` rejects this situation before fitting. The manifest declares the selected rule and counts unique pairs.

The epoch sampler is called exactly as:

```python
negative_sampling(raw_TRAIN_edge_index, num_nodes)
```

No negative-count, method or force-undirected override is supplied. The actual provider file and function are hashed, and required defaults are checked. Raw graph edges remain in authenticated OGB interleaved reverse-edge order. Sampler exclusion uses only complete TRAIN raw edges. Heldout positive identities are never used to reject TRAIN negatives. The adapter checks returned endpoint ranges, nonself edges, and exclusion of directed TRAIN identities.

The native loss is `mean(-logsigmoid(positive)) + mean(-logsigmoid(-negative))`. Each minibatch has the same positive and negative record count. All terms are averaged over the entire declared epoch with record weights; positive and negative denominators are explicit. This fixes last-minibatch logging while preserving the update objective.

`native_drop_tail` preserves author's floor `PermIterator`: 1,179,052 records with batch 65,536 means 17 updates, 1,114,112 used records and 64,940 dropped records per epoch. `complete_tail` includes the remaining records and is an explicitly declared adaptation. Both are full epoch budgets under their declared coverage law. An empty native-floor epoch fails.

SELECT always includes every query tail. Scores are raw native logits using one full TRAIN support encoder result. Frozen OGB shared-pool Hits is crosschecked against the strict `positive_score > kth_negative_score` rule; balanced positive/negative BCE can be an explicitly selected alternative metric.

## Protected role protocol

Fit opens exactly three hash-bound plain tensor dictionaries:

| Pack | Tensor fields |
| --- | --- |
| GRAPH | float32 `x[N,F]`; int64 `raw_edge_index[2,2E]` |
| TRAIN | int64 `positive[E,2]` |
| SELECT | int64 `positive[P,2]`, `negative[Q,2]` |

The graph must equal the exact bidirectional TRAIN record multiset. Features preserve the full node count, including isolated nodes. The default `select_overlap=reject` contract requires canonical SELECT pairs to be disjoint from TRAIN. The official collab temporal packet explicitly uses `select_overlap=official_temporal_records`, restricted to `ogbl-collab`: roles identify occurrences in different years, and existing pair identities can recur. The preparer never filters targets or changes role membership. It records 967,632 unique canonical pairs among 1,179,052 TRAIN records, 28,072 SELECT positive records and eight negative records whose pair identities already occur in TRAIN, and zero positive/negative SELECT pool intersection.

The official negative pool also contains one self pair. `select_negative_self_policy=preserve_official_records` preserves that exact query; `reject` remains available for pair-disjoint protocols. TRAIN/raw and SELECT positive self pairs are always rejected. The audit states `heldout_records_added_to_support=false`, while separately reporting pair identities already known from TRAIN. No SELECT occurrence or year is used to augment support.

`prepare_link_data.py` reads exactly cached `split/time/train.pt`, `split/time/valid.pt`, `raw/node-feat.csv.gz`, and `raw/edge.csv.gz`, using the frozen `graph_ncNC_valid_data_authority_root_20261003_v1/DATA_AUTHORITY.json`. Pass a current explicit cache root: the archived authority's old absolute `dataset_root` is ignored. All four files must match fixed official bytes/hashes before legacy NumPy pickle deserialization. Array shapes, dtypes and digests, raw graph/features and temporal years are then checked. Outputs use plain tensors loaded by the fit factory with `weights_only=True`. `--admission` binds a fresh archive admission receipt when the cache was downloaded for this packet.

Official temporal TRAIN is intact, and entire official VALID becomes SELECT. Source weight/year fields are authenticated but excluded from the unweighted model. Fit preparation creates no CAL or evaluate authority. Separate endpoint preparation supplies TEST custody; CAL remains absent. The study supports raw ranking only, with no confidence/calibration claim. SELECT labels cannot also become CAL labels. The schema explicitly uses `raw_only_absent` CAL authority; the preparer does not fabricate a role.

This is the protected TRAIN-only adaptation. Published author execution inspects TEST while fitting and augments final support with VALID edges. Author replication with that graph policy belongs to a separate track; this packet makes no author-protocol-replication claim.

Example preparation command, executed only by the parent allocation owner after route/custody checks:

```sh
python training_pipeline_v2_20261012/prepare_link_data.py \
  --project-root /current/authorized/research/root \
  --cache-root /current/authorized/cache/ogbl_collab \
  --authority /current/authorized/research/root/graph_ncNC_valid_data_authority_root_20261003_v1/DATA_AUTHORITY.json \
  --output /current/authorized/research/root/link_fit_data_fresh
```

The fresh `MANIFEST.json` supplies `link_inputs`, `link_source`, `sampler`, `metric`, and `support` config fragments plus source and provider custody. It refuses an existing output directory. The tensor preparer uses no dataset constructor, `get_idx_split`, TEST accessor, download, model, optimizer or predictive metric.

The authorized archive admission helpers separately verified installed OGB 1.3.6 metadata `master.csv` (SHA `8c317b29892fa41c021446858a7fc6761c30d0f1af651f7942af8d85799a7054`). Its URL is `http://snap.stanford.edu/ogb/data/linkproppred/collab.zip`, retrieved using HTTPS at the same host/path. The official archive is 121,625,147 bytes, SHA `c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc`. Initial fit admission opened only the four permitted TRAIN/VALID/features/raw members. The later explicitly isolated endpoint preparation decoded authenticated TEST once into separate query and target/index authorities. Newly downloaded local bulk was removed after server hashes verified; metadata receipts remain.

Retained preparation evidence under `training_pipeline_v2_20261012/qa`:

- `link_source_stage_v1`: exact model/utils source staging with hostname/sole UUID receipt.
- `cpu_link_packet_v9`: preserved failed QA from missing source fixtures.
- `cpu_link_packet_sources_v10`: 21 tests pass after source staging.
- `link_official_cache_v1`: preserved allocation HTTPS handshake failure.
- `link_admitted_transfer_v1`: official archive/member/provider receipts, server custody and local bulk cleanup.
- `link_official_cache_v2` and `v3`: preserved guard failures identifying official negative self and temporal recurrence semantics.
- `link_official_cache_v4/fit_data`: successful complete GRAPH/TRAIN/SELECT output on allocation, with local metadata copy in `link_official_cache_v4/RESULT.json`.
- `cpu_link_temporal_complete_v11`: all 30 targeted schema/source/numerical/compiler tests pass with zero skips after the final role-policy fix.
- `cpu_link_compiler_current_v12`: five metadata-only tests pass for current compiler and frozen I4 HPO ancestry; current dispatcher included. No further numerical refresh was performed.
- `link_disabled_matrix_v2`: actual custody-bound disabled default domain, zero materialized model-fit rows; current compiler/roster snapshot. Prior `v1` is preserved.
- `link_sealed_archive_v1`: exact official archive reacquired and transferred after hostname/sole GPU UUID verification; archive SHA verified on allocation.
- `link_sealed_evaluate_prep_v1/sealed_endpoint`: complete official TEST query and separate sealed target/index packs, no model/optimizer/score.
- `cpu_link_sealed_endpoints_v13`: all 15 targeted endpoint/compiler tests pass on allocation CPU, zero skips. Immutable snapshot includes the current dispatcher namespace and validation-cost fixes.
- `link_materialized_source_controls_v3`: actual source/data-bound disabled source-controls study, 1,250 fits and 1,276 total run records. Older 1,256 declarations remain preserved.

## Sealed TEST and raw endpoints

`prepare_link_endpoints.py` reads only authenticated `collab/split/time/test.pt` from the exact archived provider artifact in isolated preparation. It preserves all 46,329 positives and 100,000 shared negatives, with no filtering. Its preparation manifest binds original arrays, counts, year 2019, archive/program/fit receipts and the unchanged metric source.

The 146,329 query records are shuffled with a secret 256-bit nonce; only the nonce SHA is retained. The public query pack contains exactly CPU int64 `{ids,pairs}`, with opaque sequential IDs attached after shuffling. It exposes no labels, original concatenation position, public shuffle seed or within-pool index. A separate target pack contains exactly `{ids,is_positive,original_index}`, where indices refer to the original order within each positive/negative pool. Training, configuration selection and export never decode that target pack.

`export_bank` reads the query-only pack, restores each complete full-budget parent's own selected checkpoint, and scores all queries with the unchanged native encoder/decoder and full TRAIN-only support. A genuine I4 bank contains four separate selected raw score rows. Serving averages raw scores. The matching singleton copies the prospectively fixed body-0 row; it requires identical query/support ancestry and current selected epoch, checkpoint receipt, resolved configuration hash and initialization seed. It constructs no model and predicts no queries again.

The endpoint requires the complete declared search, search-selection and comparative roster to be terminal. The current dispatcher additionally requires the approved root distributed-plan phase closure. Evaluation restores original within-pool ordering from sealed indices, applies pinned OGB Hits@50, and crosschecks the strict `positive > 50th_negative` tie law. There is no sigmoid conversion, CE/Brier endpoint, CAL reuse, confidence claim or author-protocol-replication claim. Failures retain attempted/completed costs and known partial inference counts as lower bounds.

## Prospective search and disabled compiler

Default `source_controls` contains 302 NCN and 311 NCNC configurations. A 288-point joint base covers width (64/128), encoder depth (1/2), encoder and predictor dropout (0.1/0.3), independent group LR neighborhoods (0.5×/1×/2× native), and weight decay (0/0.0001). Declared one-control native-anchor neighbors add edge/input dropout, beta, LayerNorm/JK, residual at effective width 128, and source predictor linear/activation flags. NCNC additionally has nine neighbors for alpha, scale, offset, threshold and learnable threshold. Stable complete-recipe hashes deduplicate points.

`legacy_widths` preserves the prior two-point width (64/128) domain with all native constructor and optimizer settings fixed. Both profiles include the exact native width-64 anchor. The ignored predictor layer argument and the unused nonpure-GCN tail dropout are locked, rather than advertised as effective HPO knobs. NCNC2 depth, sampled degrees, operator changes, extensions and altered tail laws are outside these profiles.

With two search seeds (`71101,71103`) and three comparative seeds (`71201,71203,71207`), the exact prospective model-fit formula is:

`S*(C_NCN+C_NCNC) + R*2*4`.

Each I4 has four separate paid fits with separate seeds and selectors. The matching singleton is prospectively fixed to I4 body 0 and reuses its fit/checkpoint/prediction row; there is no fifth comparative fit or duplicate prediction cost. I4 uses the corresponding member-equivalent single search configuration. Each paid fit runs all 100 native epochs, 17 updates per epoch.

The dispatcher checks configuration-choice membership in the frozen search domain. I4 roster validation permits only the resolved encoder, predictor and optimizer controls to differ from its prospective placeholder; source, role, seed, graph/sampler protocol, body identity, shared selection ancestry, budget and own selector remain fixed.

| Profile | Search fits | Comparative fits | Total fits | Epochs | Optimizer updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `source_controls` | 1,226 | 24 | 1,250 | 125,000 | 2,125,000 |
| `legacy_widths` | 8 | 24 | 32 | 3,200 | 54,400 |

These are declarations, not measurements or adopted launch budgets. The actual source-controls packet contains 1,226 search fits, 24 comparative fits, two search selections, 12 exports (six four-member forwards plus six member-0 copies), and 12 raw evaluation jobs: 1,276 run records. There are zero CAL fits. The CLI hashes actual data/source/query/target/decision receipts without deserializing role tensors. All rows remain `scientific_launch_enabled=false`, with human GO required. The old 1,256-fit formula is reconstructible by `fit_formula(member0_reuse=False)` and its immutable metadata is preserved.

`RESOURCE_PROVISIONAL.json` declares 32,768 MiB and deliberately loose finite wall placeholders: 1,800 seconds per epoch, 7,200 seconds per four-member export and 250,000,000 seconds for the packet. These are unmeasured scheduling reservations, not rate estimates supported by CUDA evidence or feasibility claims. `EXPOSURE.json` explicitly recognizes historical collab development exposure and rejects protected-confirmation claims. The current study source points to the passing `cpu_link_sealed_endpoints_v13` engine/dispatcher snapshot.

Author final TRAIN+VALID support remains a separate blocked track. Its graph, fit-time diagnostics policy, export/checkpoint ancestry and source protocol need complete independent verification; this packet makes no author fit-count claim.

The next master snapshot must use root's current dispatcher namespace fix and validation cost accounting fix. Earlier immutable standalone QA/domain packets remain preserved. This link packet does not activate an older runtime for scientific work.

## Genuine I4 and cost custody

`native_independent4_body` requires exactly four paid model-fit rows, body IDs 0–3, distinct run IDs and distinct uint32 initialization seeds. Their source, roles, architecture, optimizer, epoch budget and selector recipes must agree. Each factory owns separate parameters, Adam, private Python/NumPy/Torch CPU/CUDA RNG and its own selected checkpoint. I4 serving averages the four own-selected raw scores. Same-epoch bank rescue and aliasing member 0 as another paid fit are not admitted. Fixed member-0 singleton reuse is explicit and charged as reuse.

Counters record sampler attempts/completions and drawn/unused negative rows, positive/negative supervision, dropped positive tail, sparse graph rebuild attempts/completions and input records, encoder attempts/completions, top-level decoder passes, backward attempts/completions, Adam attempts/completions, and evaluator attempts/completions. An instance wrapper calls the unchanged source decoder and also counts NCNC recursive calls and query rows. Failed adapter operations preserve known partial costs through `AdapterStepError`. These counters are completed/attempted source operation counts, not FLOPs; failed internal operations can only give a lower bound. The fixed full-support graph build and input tensor bytes at adapter initialization are recorded in provenance and charged once in the first training call; the checkpoint retains whether that setup charge occurred.

Checkpoint state includes model/optimizer, mixed module modes, gradients' trainability flags, input/source ancestry, private and outer RNG state, selector and epoch. The engine's strict restore verification compares the complete saved state. `predict_queries` opens no role files and uses only its explicit query tensors and frozen full TRAIN support.

## Engineering checks

`tests/test_link.py` includes stdlib config/source/role/sampler/selector/I4 contracts, weighted epoch metric checks, tensor overlap/multiset guards, and an exact pinned-source synthetic CPU fixture for NCN and NCNC. The latter covers a full native epoch with a dropped tail, complete SELECT query coverage, strict selected checkpoint replay, the complete-tail alternative, four disjoint body storages/private RNG, and failure cost reporting. It uses only a twelve-node synthetic graph. Missing source files fail when the runtime providers are available; they are never a skip condition. `GNNM_REFERENCE_PROJECT` can point allocation QA at the mirrored research root.

`tests/test_link_endpoints.py` adds synthetic tests for label-free queries, complete separate target/index coverage, closure before decoding, four own checkpoint selectors, raw mean fusion, exact member-0 no-forward reuse, strict Hits50 ties, complete query tails and retained failure costs. No official TEST scores are used in these checks. All 15 current endpoint/compiler checks pass on allocation CPU with zero skips; the earlier 30 exact-source/temporal tests remain preserved.

## Remaining required comparisons and CUDA qualification

`ORIGINAL_GNNM_LINK_BLOCKER.json` binds the pinned original source (`47c4af6228c52aa450ab8316efe5c7cfc33119a4`, model SHA `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`) and current node bridge. `TABMModel(graph,x,tabm_seed)` returns node-wise outputs from live input/output BEBlock factors and its shared native message trunk. The node adapter consumes node IDs and multiclass CE. There is no implemented original-GNNM query-pair decoder or source/role-bound link training/sampler/SELECT/restore adapter. An explicitly named boundary-encoder/query-decoder link adaptation in separate source is required before the comprehensive comparison can be ready. NCN/NCNC shared banks are never original GNNM. No original-GNNM link fits are counted. The original paper's five-dataset rerun exclusion remains intact.

Exact NCN/NCNC source, complete GRAPH/TRAIN data and numerical dependencies are prepared for a zero-update CUDA qualifier. It must use exact constructors and native support/masking, TRAIN queries with synthetic labels, sparse forward plus backward, and retained peak allocated/reserved memory and operation-rate/failure evidence. It must create no optimizer, apply no optimizer update, perform no SELECT scoring and open no TEST query/target pack. CPU checks cannot establish CUDA sparse-kernel behavior. No such qualifier was executed here. Original-GNNM link adaptation, link CUDA qualification, root global closures and human GO remain launch blockers; this is not a ready comprehensive-study claim.
