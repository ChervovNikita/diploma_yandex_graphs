# Upstream OGB BatchNorm author-reference port plan — 12 October 2026

**Disabled source preparation; no scientific execution.** Eight recipe cells are prepared on the verified allocation: MolHIV/MolBACE × GIN, GIN-virtual, GCN, GCN-virtual. No author groups are registered, no model/provider constructor or model forward ran, no optimizer step ran, and no GO exists.

Allocation plan: `ogb_batchnorm_author_reference_preparation_20261012_v2/DISABLED_AUTHOR_REFERENCE_PLAN.json`, SHA256 `4a4977198b4035930b41177bcb119f5af74de3d65f0a6fcb86d3c51861196cb9`. Source repository `snap-stanford/ogb`, commit `61e9784ca76edeaa6e259ba0f836099608ff0586`. Locally pinned `main_pyg.py`, `gnn.py`, `conv.py`, `README.md` and retrieval receipts were staged byte-for-byte into a fresh allocation preparation root.

## Authentic source recipe

| Setting | Pinned value / behavior |
|---|---|
| Batch / width / layers | 32 / 300 / 5 |
| Optimizer | Adam, learning rate0.001, weight decay0 |
| Dropout / budget | 0.5 / 100 full epochs, no early stop |
| Features | Full native categorical atom/bond features |
| Residual / JK / graph pooling | false / last / mean |
| Train loss | Native observed-label BCEWithLogits, mask `y==y` |
| Native provider | `PygGraphPropPredDataset(name=dataset)`, default `root=dataset`, no transform/pre_transform |
| Roles | Official scaffold TRAIN, whole VALID, TEST; no invented CAL |
| Selection | Earliest `np.argmax` whole VALID ROC-AUC after100 epochs; no selected model checkpoint/restore |
| Diagnostics | TRAIN, VALID, TEST each epoch in source order; all raw logs quarantined |
| RNG / repeats | No native seed call, seed argument or repeat argument; one process/invocation |

The source uses BatchNorm in GIN MLPs and every node layer, plus virtual-node MLP BatchNorm. GCN keeps source bond/degree/root-embedding semantics. Current standardized LayerNorm residual GINE+virtual-node is a separate recipe. It cannot serve as the official OGB author-original track.

The train loop skips a batch containing one total node or only one graph; evaluation skips one total node. TRAIN evaluation reuses the shuffled TRAIN loader, advancing process RNG before the next epoch. Do not replace its order with a deterministic standardized epoch shuffle. Epoch counts alone cannot certify exact optimizer calls.

### Disabled recipe cells

| Dataset | Source model | Workdir |
|---|---|---|
| ogbg-molhiv | gin | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molhiv-gin` |
| ogbg-molhiv | gin-virtual | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molhiv-gin-virtual` |
| ogbg-molhiv | gcn | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molhiv-gcn` |
| ogbg-molhiv | gcn-virtual | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molhiv-gcn-virtual` |
| ogbg-molbace | gin | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molbace-gin` |
| ogbg-molbace | gin-virtual | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molbace-gin-virtual` |
| ogbg-molbace | gcn | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molbace-gcn` |
| ogbg-molbace | gcn-virtual | `ogb_batchnorm_author_reference_preparation_20261012_v2/workdirs/ogbg-molbace-gcn-virtual` |

Each workdir has exact source bytes, a private native `dataset/ogbg_mol*` raw/mapping/scaffold-split cache and a fresh empty processed directory. Each CLI keeps the pinned numeric arguments and empty `--filename`; its native TEST-bearing result pickle is not created. The full argv and file hashes are in the JSON. No fit command is executed by this preparation.

## Source/data custody and engineering result

Raw provider files were rehashed against existing custody receipts, copied byte-for-byte and rehashed at each destination. Labels, graph arrays, scaffold indices, processed tensors and outcomes were not decoded. Existing standardized processed caches were neither copied nor decoded; fresh native processing provenance must be recorded when processing is admitted. Native whole VALID differs from the standardized SELECT/CAL division. Prior exposure and confirmation status remain root-owned.

**CPU-visible native CLI help/import passed, exit0, empty stderr.** `CUDA_VISIBLE_DEVICES=""`; native Torch/PyG/OGB and source modules imported before argparse help. Native loss objects are module globals; no GNN/dataset/optimizer constructor or numerical forward ran. This proves import/parser compatibility only, not training/resource qualification.

| Runtime package | Actual version |
|---|---|
| numpy | 1.26.4 |
| ogb | 1.3.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| torch | 2.1.2+cu118 |
| torch-geometric | 2.7.0 |
| torch-scatter | 2.1.2+pt21cu118 |
| torch-sparse | 0.6.18+pt21cu118 |
| tqdm | 4.67.1 |

Exact upstream environment and published-score replication are not claimed. The first attempt verified allocation hostname/UUID, then failed because the historical source directory was absent on allocation. The empty v1 root and failed transport receipt remain retained. The successful v2 stages the locally pinned bytes and leaves source originals/frozen packets untouched.

## Family extension of the author runtime

The current `author_cli_spec.py` and `run_author_cli.py` explicitly encode PolyFormer: seed42, ten sequential repeats, fixed random split seeds, maximum2000/patience250, strict accuracy, `training.py`, DGL overlay and PolyFormer log grammar. OGB cannot be admitted unchanged under that contract. A family-discriminated successor is feasible; the existing validator must remain fail-closed.

- Fresh author_family-discriminated spec/validator; family-specific budget/selection/CLI semantics; closed allowlist of source bytes and dataset/raw receipts.
- Separate OGB progress observer reading original Train/Validation fields while ignoring Test fields, enforcing contiguous1..100 epoch headers, strict numeric/log grammar, and no rounded-accuracy PolyFormer parser reuse.
- Actual batch/update counters cannot be claimed from epoch logs; native singleton skipping and shuffled TRAIN eval make total calls data/RNG dependent. Keep counts unknown/lower-bound unless source-preserving instrumentation is independently admitted.
- Separate one-process native cost contract; explicit externally seeded repetition, if chosen, must be labeled a controlled extension of unseeded native CLI.
- Family-specific fresh processed cache/workdir validation instead of PolyFormer bases/split-seed/DGL overlay assumptions.
- Preserve global whole-search/whole-author quarantine gates, allocation UUID checks, real registry claims, per-child PID/PGID/birth custody, immutable source/plan receipts, GO and retained failure/no-restart behavior in fresh successor runtime.
- Add fresh disabled master study/group compilation only after provider/runtime/protocol/resource/seed-repeat gates. Existing frozen master/controllers must remain unchanged.

**Engineering effort:** approximately3–6 person-days after provider/RNG policy decisions; this is not training time, a GPU ETA or a schedule promise. No large code or frozen-controller mutation was made here.

## Remaining launch blockers

- Current author_cli_spec.py hard-codes PolyFormer seed42/ten repeats/2000max/patience250/accuracy/source training.py and fixed split seeds.
- Current run_author_cli.py Progress parser/costs require PolyFormer log grammar and ten repeats; OGB epochs/minibatch skip/Train-Validation-Test dictionary grammar differs.
- Decide explicit repeat count and RNG initialization policy; source provides no seeds or repeated-run interface.
- Add family-discriminated validator and separate observer in a fresh successor runtime; preserve existing validator fail-closed.
- OGB actual optimizer calls depend on singleton-batch skip, graph-size condition and native shuffle. Help/import proves no training resource/cost evidence.
- Bind fresh provider-processing provenance and complete role/prior-exposure authority; native whole VALID is distinct from standardized SELECT/CAL packs.
- Complete resource/protocol qualification, compatible original-GNNM/M1/genuine-I4 comparator scope and fresh disabled master approval/GO before any fit.

Repeat count and seed list are deliberately `null`: the native source provides neither. A externally seeded repeated process can be a controlled source-code path extension after explicit labeling, but cannot be claimed as a source-native published seed protocol. The eight prepared rows are recipe cells, not eight admitted author process groups or eighty fit trajectories.
