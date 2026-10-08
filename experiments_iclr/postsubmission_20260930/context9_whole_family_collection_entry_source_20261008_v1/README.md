# Disabled Context9 whole-family collection entry

`collect_closed9.py` is glue around the sealed closed9 restore/readout hooks and the existing Wiki12 collector/analysis. Frozen packages are not edited. There is no new inference implementation, selector, fit, calibration procedure or qualification fixture.

## Root admission

Copy `RELEASE_TEMPLATE_DISABLED.json` into a separate root activation directory. Populate its future bindings and resource limits only after all original nine fits and their owner have terminated. Keep this source directory and its sealed templates unchanged. Bind `MANIFEST.json` by its exact SHA-256 and pass the separate release's exact hash to the entry.

Create a separate root snapshot following `TERMINAL_EVIDENCE_TEMPLATE_DISABLED.json`. Bind the actual original `FAMILY_CLOSURE.json`, `PARENT_OWNER.json`, and each `logs/<cell>.OWNER.json` / `.EXIT.json`. Preserve the recorded lowercase `pid`, `start_ticks`, `state` format and the original child argv. The gate checks the original parent **526200 / 6021896966**, all nine successful `subprocess.Popen.wait` receipts and child reaps, exact selected checkpoint hashes, and root's parent/child absence and no-CUDA-row observations. The entry consumes saved evidence; it does not poll the original jobs.

Root must authorize trusted checkpoint deserialization, charge collection and readout costs, confirm current resource readiness, and provide an externally owned finite active/cleanup/hard deadline. The entry checks the arithmetic of that envelope; **the external root supervisor enforces its wall deadline and terminal cleanup**. Resource limits are unset in the template. Admission requires an owned Torch allocation cap <= 20 GiB, fresh GPU headroom >= 24 GiB and >= cap + 4 GiB, and a positive root-selected fresh disk threshold. These bounds are admission limits, not a new measured collection-cost claim or an ETA.

Use the bound allocation Python and PYTHONPATH, repository cwd, and sole GPU UUID. The CLI accepts `--release <separate-root-release>` and `--release-sha256 <exact-hash>`; it rejects the shipped disabled template before accessing endpoint metadata or importing numerical providers. The chosen output directory must be fresh, inside the allocation research phase and outside all source/original family directories.

## Collection and readout

All three COMMON prediction archives are collected before any COMMON cohort is frozen. Every COMMON cohort must then be durably sealed before any ROUTE or PERMUTED prediction call. ROUTE3 precedes PERMUTED3. A fresh legacy collector module receives the existing closed9 restore callback and a 36-attempt budget. Original condition labels are retained.

The unchanged readout produces per-cell/per-seed summaries, both fixed contrasts, paired error flows and `STAGE1_FIXED_GATE.json`. All three fixed seed slots are required for means and intervals; there is no survivor-only summary. Read `PREREAD_CONTROL_NOTE.md` before interpreting the gate or considering later root admission of Stage2.

`compact/GATE.json`, `COLLECTION.json`, `COMMON_COHORTS.json`, `COST.json` and progress/failure records retain custody and costs. Raw logits, representations, labels, IDs, partial archives and cohort masks remain in `raw/` on the server. Collection status and training-family status are distinct: failed inference never becomes a failed or excluded training seed.

## Preparation verification

Only local stdlib syntax/AST checks and JSON/source seals are used for this package. No package import, numerical fixture, checkpoint or data load, server call, prediction collection or fit was performed. Admission remains disabled pending root's later complete terminal evidence.
