# Complete public node-graph acquisition

This packet prepares complete Reddit2 and ogbn-products graphs for the prospective propagation-cost study. Both tasks were adopted before new model/test outcomes in `propagation_cost_scout_v1/TASK_ADOPTION_ROOT_before_outcomes_v1.json`. It does not run a model, compute scores, choose a method or recalculate earlier results.

## Execution scope

`acquire.py` uses the exact official `torch_geometric.datasets.Reddit2` and `ogb.nodeproppred.PygNodePropPredDataset(name="ogbn-products")` providers. All provider roots, original archive copies, exported data, source snapshots and receipts live in a new directory below `propagation_data_v2`. Repository cache/temp paths come from the current `protocols/repo_env.sh`. Existing outputs and symlink path components are rejected. Failures are preserved; the script does not resume or overwrite a failed acquisition.

The current wrapper places the remote phase at:

`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930`

The directory name differs from the local source phase. Source-relative bindings support that approved relocation. The live guard requires the actual phase, repository cwd, corrected SSH destination, all confined cache/temp values, current supervisor/entrypoint hashes and the visible single A100. It queries `nvidia-smi` but does not require an idle GPU or initialize CUDA. Guards and source checks precede Torch/provider imports and output creation.

## Label access

Public provider construction internally loads all node labels and retains raw/processed files containing held-out labels. Original archives also contain them. This is disclosed in the receipts; this packet does not claim cryptographic isolation. Whole public files are hashed opaquely without target indexing or label summaries.

Study code accesses label values once: `data.y.index_select(0, official_train_ids)`. Only training label values are checked, summarized and exported. Validation/test IDs and masks are public and exported; their label values are never indexed, summarized, scored or exported. Official class counts are fixed as Reddit2=41 and ogbn-products=47. The provider `num_classes` property is never used because it can scan all targets.

The provider features and `edge_index` are preserved exactly, including order, direction and duplicates. No normalization, sampling, edge modification, transform or node subset is applied. Node/feature dimensions are checked against the complete official tasks: Reddit2 232,965/602; ogbn-products 2,449,029/100. The edge count is reported for the exact installed PyG representation, which can differ from a dataset's count of undirected edges. Official split ID order is preserved for products. Reddit2 IDs are the ascending node IDs derived from its official masks.

## Archive preservation and OGB confirmation

For the duration of constructing the named provider only, the script intercepts the download function referenced in that exact class's `download` globals (`download_url` or `download_google_url`). It calls the original function, streams a byte-identical archive copy into `archives/` before the provider deletes it, then restores the original globals in `finally`. Unsupported provider APIs fail closed. This is a process-local hook; installed files are not edited.

For explicitly authorized **ogbn-products only**, the exact `decide_download` global is temporarily replaced with a recorded affirmative answer for `http[s]://snap.stanford.edu/ogb/data/nodeproppred/products.zip`. Other URLs, datasets or call signatures are refused. This bypasses an interactive confirmation for the named public download, not authentication or access controls. The explicit `--authorize-public-download ogbn-products` flag is required and tied to the prior adoption receipt. Reddit2 uses four separately downloaded public files. This version pins the observed installed provider module SHA256 and accepts only its exact `adj_full_id`, `feats_id`, `class_map_id` and `role_id` declarations. The unchanged original provider performs processing; each returned source file is preserved before processing. Unsupported provider identities are refused.

## Export contract

`graph_training_targets.pt` is a `torch.save` dictionary with:

- `schema`: `gnnm-complete-node-graph-training-target-export-v1`;
- `public_dataset`, `declared_official_class_count`;
- full CPU `x`, full CPU int64 `edge_index` with shape `[2,E]`;
- `official_split_node_ids` and `official_split_masks`, each keyed `train`, `valid`, `test`;
- `training_target_node_ids` and `training_targets`, preserving the provider training-target dtype and `[T]` or `[T,1]` shape;
- optional exact `edge_attr`/`edge_weight` when present.

There is no full `y`, validation label tensor or test label tensor in this export. The ID vectors and masks are checked for uniqueness, disjointness and complete node coverage. `TENSOR_FINGERPRINTS.json` records shape, dtype, native byte order, byte count, raw tensor bytes SHA256 and a canonical metadata-plus-bytes SHA256. Hashing streams views of contiguous CPU tensors in 8 MiB chunks; it does not make multi-GB tensor copies. Noncontiguous provider tensors fail closed rather than silently changing their layout.

Other receipts: `PRE_PROVIDER.json` includes package versions and installed provider source; `DOWNLOAD_INTERCEPTION.json`, individual archive/OGB confirmation receipts, `GRAPH_AND_SPLITS.json`, `TRAINING_TARGET_CHECK.json`, `PROVIDER_STORAGE.json`, `ACQUISITION.json` and an output artifact manifest. Raw/processed/archive files have exact byte hashes. Live supervisor references remain pending until the authorization wrapper exits. Root must audit the completed chain and exact export before scientific admission.

## Exact acquisition child CLI

After the root uploads the sealed packet and dependencies to the current approved remote phase, run from the exact repository cwd with the repository interpreter already used by supervision. Let `P` denote the current absolute phase above, `PY` that existing repository interpreter, and `S` a **new** supervisor output under `$P/protocols/logged_runs/`. These are CLI argument values; do not create new runtime installations for acquisition.

```bash
"$PY" "$P/propagation_data_v2/acquire.py" \
  --task reddit2 \
  --authorize-public-download Reddit2 \
  --output "$P/propagation_data_v2/reddit2_authorized1gpu_run01" \
  --supervisor "$S"

"$PY" "$P/propagation_data_v2/acquire.py" \
  --task products \
  --authorize-public-download ogbn-products \
  --output "$P/propagation_data_v2/products_authorized1gpu_run01" \
  --supervisor "$S"
```

Each child must be started through `protocols/run_authorized_v2.py --output "$S" -- "$PY" ...`, with the exact matching `--supervisor "$S"` child argument. Source `protocols/repo_env.sh` first, set `GNNM_SSH_DESTINATION=anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru` and `CUDA_VISIBLE_DEVICES=0`. `PYTHONDONTWRITEBYTECODE=1` is provided by the wrapper. Omit `--require-idle-gpu`: this acquisition is CPU-only and may proceed while GPU engineering work runs. Use distinct new `S` values for the two commands.

## Verification status

Local verification is stdlib-only compilation, source/dependency hashes, AST label-scope checks and early refusal before third-party imports/output creation. No Torch, PyG or OGB runtime is available locally; provider construction and downloads have not been executed by this packet's author. An unsupported provider-version API may cause a preserved failure; changing it requires a new sealed source generation before another acquisition.

## Version 2 technical amendment

Version 1 refused Reddit2 before any download because it expected a single `url` declaration. The installed official provider instead declares four Google IDs. The original sealed packet and failed supervisor remain unchanged. This version changes only that acquisition interface and its source pin; complete graph topology, features, official splits, label scope, model recipe and scientific gates are unchanged. Products version 1 remains live and is not restarted.
