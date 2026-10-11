# Official ogbn arxiv provider and SAGE confirmation source audit

The pinned OGB author example supplies a concrete ordinary SAGE reference and symmetric graph support without extra self loops. Keep the current E core at width 128 and two residual blocks; add the one author SAGE single at its fixed published source point across the same ten fresh seed blocks. This gives 60 banks and 90 native fits, including the unchanged 50-bank/80-fit core. The source specification and recommendation remain inactive. Root owns input admission, numeric freeze, resource qualification and execution.

## Pinned provider and role authority

OGB is pinned to author repository commit `61e9784ca76edeaa6e259ba0f836099608ff0586`, recovered from an existing saved source receipt. Exact public source copies and retrieval hashes are under `primary_sources/`. The node master records dataset version 1, `arxiv` download name, `time` split, 40 classes and accuracy. Its exact published download URL is **http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip**. This audit did not request that archive or any data, role-index or target payload.

Official metadata specifies 169,343 nodes, 1,166,243 raw directed citation edges and 128-dimensional precomputed paper features. Preserve those original features. Node years guide the official split, not an extra model input; raw text, GIANT features, Node2Vec and label propagation are outside this fixed information contract.

`PygNodePropPredDataset.get_idx_split()` obtains official indices from `split/time/train.csv.gz`, `valid.csv.gz` and `test.csv.gz`, or the declared cached `split_dict.pt` shortcut. The temporal rule is TRAIN through 2017, VALID in 2018 and TEST from 2019 onward. Expected released sizes are **90,941 TRAIN, 29,799 VALID and 48,603 TEST**, totaling all 169,343 nodes. These are explicit preparation expectations. The inspected primary source and official page confirm the paths and temporal rule but do not print the exact counts; no payload was opened to certify them. Root must bind the actual authoritative index identities, counts and hashes during later authorized provider preparation. An unexpected count is a provider discrepancy, not permission to generate a replacement split.

Sort official VALID IDs, permute that ordered vector with MT19937 `RandomState(11709)`, take the first floor-half as CAL and the remainder as SELECT. Expected sizes are **14,899 CAL and 14,900 SELECT**. Every arm and author context uses the same roles. This separation changes neither official TRAIN nor official TEST. CAL truth is excluded from gradients and checkpoint selection.

The stock provider reads all labels into `data.y`, including TEST, during processing or processed-file loading. A future custodian must export the complete transductive features/support with model truth confined to TRAIN, and separate SELECT, CAL and sealed scorer-only CONFIRM views. Record custodial parsing separately from scientific heldout scoring. The two current local status files contain no literal ogbn-arxiv name; that name search establishes no whole-history unused-input or unused-TEST certificate. Root's target-specific access audit remains pending.

## Published graph support

The author driver constructs `PygNodePropPredDataset(..., transform=T.ToSparseTensor())` and calls `data.adj_t.to_symmetric()`. The provider has `add_inverse_edge=False`, so the raw directed graph is symmetrized once by the driver. It does not call `add_self_loops`, `fill_diag` or a diagonal transform. The audited SAGEConv also adds no self loops: it averages neighbor messages and adds a separate learned root map. Preserve any raw diagonal already present, while adding none; actual diagonal counts remain a provider receipt.

PyG 2.7.0 source is independently pinned at `76ff9c2ce18c8cebf52122b57e2aeadce9793d10`. `ToSparseTensor` stores transposed adjacency with row=receiver and column=source. Its runtime backend must be explicit: with torch-sparse installed it produces `SparseTensor`, while the fallback can produce a Torch CSR tensor that lacks the author's `.to_symmetric()` method. Pin the installed PyG/torch-sparse source and binary versions before adopting that path. Code availability does not qualify the runtime.

All core arms and the author context must use the same symmetric, coalesced support and node order. For E's generic COO path, recover source=adjacency column and target=adjacency row. The reciprocal-entry bound is **2,332,486**, before actual coalescing; it is not an observed edge count. The earlier 2,501,829 scenario included an added loop for every node and is not the selected author support. Disable the legacy WikiCS loop insertion when preparing this graph. Do not equate added root transformations with explicit neighbor self edges.

## One fixed author SAGE context

| Item | Pinned official author point |
| --- | --- |
| Operator | Three ordinary SAGEConv layers, 128→256→256→40 |
| Hidden treatment | BatchNorm after the first two layers, ReLU, dropout 0.5 |
| Optimizer | Adam, learning rate 0.01, default zero weight decay |
| Objective | Ordinary full-batch own NLL on log-softmax outputs |
| Training cap | Exactly 500 epochs; no early-stopping patience or scheduler |
| Metric and prediction | OGB accuracy; integer argmax of class log probabilities |
| Support and features | Common published symmetric graph, no added loops, original 128 features |
| Repeats | The ten explicitly seeded paired blocks adopted by root |

The current E recipe differs: two residual SAGE blocks, width 128, LayerNorm/GELU and FFN maps, dropout 0.2, AdamW 0.001, mean member CE and a 1,000-update/300-stale ceiling. The ordinary author model supplies competence context under a different fixed complete procedure. It is not a replacement architecture for E or a sharing intervention. No official SAGE I4, hyperparameter grid or backbone expansion is included.

Symbolic counting from the pinned SAGEConv and BatchNorm definitions gives **218,664 trainable parameters**: 65,792 + 131,328 + 20,520 convolution parameters, plus 1,024 BatchNorm affine parameters. The historical public OGB row also lists 218,664. Root's proposed current-E count is 218,760, only 96 higher. Neither count has been certified by a new actual constructor in this task; verify the port's registered parameters and buffers before any storage claim. BatchNorm running means, variances and batch counters are checkpoint state, not trainable parameters. Compact E storage provides no claimed advantage over this particular ordinary single.

The stock author `test()` reads TEST every epoch, and its logger selects the first maximum on full official VALID without saving the selected model. That driver is unsuitable for closed confirmation. The prospective adaptation uses SELECT-only scoring each epoch, saves the earliest strict accuracy maximum, and restores the complete model including BatchNorm buffers, optimizer and RNG state after all 500 epochs. It uses CAL only after native closure and scores TEST once after the entire family closes. This is an explicitly modified access/selection protocol, not an exact replication of the author's reported score.

## Competence context and limits

The OGB paper's arxiv reference table reports ordinary GraphSAGE **71.49±0.27 percent**, GCN **71.74±0.29** and feature-only MLP **55.50±0.23**. These published values motivate a credible ordinary SAGE reference; they do not predict this new SELECT/CAL protocol's outcome or certify the h128 E controls locally.

The saved public leaderboard and Luo et al.'s 2024 table report **GraphSAGE+ 72.95±0.31 percent** on arxiv. Its pinned author source uses original OGB node features and official roles, but a modified residual/batch-normalized model, a different optimizer point and 2,000 epochs; its driver also inserts loops. It is materially different from the ordinary OGB author SAGE point. This stronger context limits any claim of competitiveness against modern classic GNNs. It remains source context only, with no extra fit proposed and no latest-SOTA assertion. A leaderboard “No external data” label alone does not establish identical architecture, preprocessing, label exposure or cost.

Only specific public table cells and source passages were inspected. Full paper reads are not claimed. The primary scope record identifies every passage and its permitted claim.

## Disabled confirmation recommendation

Core: BASE, E4, N4, equipped M1 and genuine equipped I4, ten blocks, **50 banks/80 body fits**. Add ten ordinary author SAGE singles: **60 banks/90 fits total**. Root's fresh base seeds are **8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403 and 9623**. Retain the current member stride 1,000,003, factor offset 2,000,003 and dropout offset 3,000,007. A source-only arithmetic check finds 120 distinct block/member/role identifiers, with no accidental collisions; this does not certify stochastic independence or realized stream state.

The proposed primary family is E4 minus BASE and equipped I4. Use all ten paired raw accuracy differences, Holm correction over the two one-sided paired-t positive-null tests at alpha 0.05, plus the separate practical requirement that each observed mean gain is at least 0.2 percentage points. This does not prove the true gain exceeds 0.2. Do not add an all-ten-positive veto. Conditional on passing that primary family, a fixed-sequence E4-minus-N4 endpoint test at alpha 0.05 with an observed 0.1-point screen is the proposed secondary attribution family. Root finalizes this hierarchy before fitting. Equipped M1 and author SAGE comparisons remain fully reported descriptive references, not a silently added third primary test. If the author SAGE wins, retain that result and narrow the conclusion to the matched current recipe.

After all 90 selected body states and 60 bank prediction archives close, fit one common-policy positive temperature per bank on CAL: **60 fixed fits, 30,000 CPU-float64 scalar updates**, no grid, retry or full-VALID refit. Preserve the E operator of tempering member log probabilities before probability averaging; the ordinary single is its one-member case. Report raw/calibrated accuracy, NLL, Brier and uncertainty for every reference. Keep primary-reference mean/per-seed calibrated NLL caps 0.02/0.05 as a separate practical screen, not statistical noninferiority or an implicit raw-accuracy veto.

One scorer-only release evaluates all 60 banks on the complete unchanged official TEST population after sources, native states, CAL endpoints, reporting and runtime receipts are locked. Retain failures and missing pairs; missing pairs prevent the complete ten-block confirmation for that contrast. There is no silent complete-case subset, substituted seed, TEST-selected top-up, poor-reference rescue or replacement population. Ten blocks quantify conditional optimizer variation on one graph, not performance across graphs.

## Bounded costs and estimates

At the ceilings, the complete proposal has **85,000 logical Adam updates**, **175,000 TRAIN and 175,000 SELECT native trajectories**, **180 final selected trajectories** and **710,370 SAGE graph-block calls**. Sequential own-CE backward changes the core autograd-call ceiling to 170,000 while preserving 80,000 logical updates; the ordinary context adds 5,000 calls. Charge data preparation, symmetry/coalescing, source equivalence, checkpoint/optimizer/BatchNorm buffers, restoration, selected forward/readout, scalar calibration and collection separately.

The closed WikiCS core owner took 4,315.771 seconds with concurrent Penn94 work. A crude edge-only analogy to ten blocks and the new reciprocal-entry bound gives about **21 hours if the prior stopping pattern repeats**. Normalizing that analogy to all 80,000 core updates gives about **55 hours**. These are scenario estimates, not a runtime range or admission basis. The larger node population and TRAIN-label volume, different stopping, sequential retention, generic E message path and sparse ordinary context make extrapolation uncertain; the extra 5,000 author epochs remain unmeasured. Do not promise completion under the old 24-hour cap.

Scaling the 16.120 GB WikiCS E4 allocated peak only by the selected reciprocal-edge bound gives approximately **84.89 GB** for unchanged four-tape retention, before actual coalescing and with node/allocator uncertainty. Sequential own-CE backward may reduce retained tapes, subject to source equivalence and actual complete-graph resource evidence. Root must choose finite wall/RSS/CUDA caps and scheduling after those permitted checks; none was performed here. No current Penn94 or E fit was changed.

## Deliverables and remaining root bindings

`PROVIDER_PREPARATION_SPECIFICATION.json` pins the public provider, expected roles, support and safe custodial export contract. `FIXED_AUTHOR_SAGE_CONTEXT_SPECIFICATION.json` pins the one published source point and its declared protocol changes. `PROSPECTIVE_CONFIRMATION_RECOMMENDATION_DISABLED.json` contains the full 60-bank/90-fit recommendation with every admission disabled. Primary source copies, line/table scopes, public retrieval receipts, local name-search snapshots, seed arithmetic and resource scenarios are saved alongside them.

Input and TEST access history, actual archive/role/target/support hashes and counts, installed runtime sources, dimension/loop guards, constructors and registered states, sequential-gradient equivalence, scorer/calibrator failure policy, finite resource caps and the final inference/reporting freeze remain root bindings. No data, target or prediction payload, numerical model, remote scientific runtime, qualification, fit, desktop or reviewer was used in this source audit.
