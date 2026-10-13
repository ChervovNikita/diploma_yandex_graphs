# Exact final v29 handoff

The disabled final master has a completed census, source manifest, actual source-loading audit and transfer bundle. No GO receipt was created and no scientific fit or server77 operation was performed.

- Plan canonical SHA256: `0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b`
- Roster: 98 studies, 51 packets, 80,850 atomic model fit groups, 88 author CLI groups and 212,328 records.
- Concrete declared IDs: 668 method/control IDs across all kinds; 459 model-fit IDs; 619 served-bank IDs. These differ from the 263 canonical coverage rows.
- Source-context catalog: 6,173 source groups and 155,060 recipe records under the exact documented config/budget/selector definition; no marginal-domain cross product is inferred.
- Dataset labels: 16 raw identifiers mapped explicitly to 12 underlying datasets. Named architecture labels: 35 raw labels mapped to 21 declared families; frozen-feature and unspecified labels remain separate.

## Verification

- Manifest contains 6,824 files / 14,655,207,950 opaque bytes. Maximum opaque manifest hashes per unique input: one.
- Actual frozen launcher audit loaded all 98 studies / 212,328 records, opened 5,084 required source files and kept every phase read within the manifest union.
- Tensor/target deserializations, CUDA initialization, phase writes, scientific children and fits: zero.
- The original source hash validator may reread shared bindings across studies; its file-open counts are preserved. The whole workflow is not described as a single pass.
- First manifest attempt failed at the phase-only author-provider cache and is retained. The immutable v4 correction accepts only the exact declared repository `.venv/lib` provider receipts; arbitrary roots remain rejected.

## Transfer artifact

- Allocation archive: `final_sync_master_v29_20261013_v1/V29_EXACT_SOURCE_TRANSFER.tar`
- Bytes: 14,666,782,720; SHA256: `b2c58938fb75cdba247c343eb8dca792a597ea15b6b670c14b75335342ada3e3`
- Members: 6,826 files plus 96 empty directories = 6,922 entries.
- Each input was checked while streaming once; the completed archive was hashed once.
- The data tar remains on the allocation. Publication uses source, compact receipts and compressed metadata.

## Readable files

- `final_census_master_v29_20261013_v1/RUN_CENSUS.md`: dataset/backbone/stage/kind table and stage totals.
- `METHOD_STAGE_COUNTS.csv`, `DATASET_BACKBONE_STAGE_COUNTS.csv`, `WORKER_STAGE_COUNTS.csv`, `SOURCE_RECIPE_COUNTS.csv`: exact concrete rosters/counts.
- `HYPERPARAMETER_DOMAINS.json.gz`: every declared config/budget/selector domain and source group.
- `SOURCE_RECIPES.json.gz`: exact source-context recipes and explicit identity/scheduling omissions.
- `ARCHITECTURE_FAMILY_MAP.json`, `DATASET_FAMILY_MAP.json`: explicit alias definitions and source-factory evidence.
- `GPU_COMMANDS.md`, `run_gpu.sh`: one command per GPU, bound to this master and an existing exact explicit-GO receipt. The wrapper is outside the immutable controller and data tar.

Two slots per GPU is a ceiling. The worst registered pair on allocation/GPU0 and server77/GPU0 exceeds declared VRAM, so admission selects one large job or two fitting jobs. No measured efficient balance or ETA is claimed from placeholder weights. Destination runtime/ABI qualification and the scientific resource gate remain required. Unknown/bounded optimizer totals retain their original meaning.
