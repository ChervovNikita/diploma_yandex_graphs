# Independent static assessment: disabled retrospective collector v2

## Assessment

**V2 needs one metadata closure revision.** The collector adds the complete final owned-root inventory and repo ancestry/confinement checks requested by root, and it preserves the original failed workflow. It does not reauthenticate the current retained acquisition manifest outside the execution root. No collection, retrospective acceptance or qualification/fit admission is granted by this source review.

The reviewed source is `private_transfer_pubmed_fp32_retrospective_collection_preparation_20261005_v2/collect_existing.py.txt`, SHA-256 `9feb5b5f7d0930c7997fb4a03df6724ac900e3a812906737454eb8b5418a93b3`. Its preparation manifest is `4ac72b0e74f708bbb04ad8c22ab26100e21f5401090878f0e75241a574713c81`; preparation seal is `2e632f3ea748c458e503eb2919d32945c5697d8540c07a865458cd165f71ab3a`.

## Remaining prerequisites from the prior independent review

| Prerequisite | V2 source assessment |
|---|---|
| Current retained control/evidence and acquisition metadata hashes match staged/admitted refs | Control/evidence pins are checked. Acquisition closure remains incomplete: the exact current remote ACQUISITION_MANIFEST is not read or hashed. |
| Final complete root inventory/bytes and ordinary nonsymlink confinement, including late files | Implemented as bounded full enumeration, known-file allowlist, complete path/type/byte/hash inventory, repeated file fingerprints and path-set stability; final byte total uses actual hashed bytes and must remain within 64 MiB. |
| Separate retrospective provenance, exact result/diagnostic/source bindings and preservation of the failure | Implemented in the disabled receipt's classification, exact result/input/failure pins and explicit false original-success/admission flags. Any root adoption and actual collector transport/source binding remain separate root work. |

The original supervisor authenticates the external acquisition manifest at the fixed phase-relative path. V2's local INPUT_BINDINGS record authenticates the saved local manifest and the staged `INPUT_EXECUTION_MANIFEST.json`, but neither substitutes for a new read-only hash of that current remote metadata file. The focused successor should check that exact confined ordinary nonsymlink metadata path, its existing byte/hash pin, JOB `available_manifest_relative`/`available_manifest_sha256`, and its TRAIN/feature identities. It need not read any input values or introduce a new qualification.

## Verified static behavior

The literal `COLLECTION_ENABLED=False` guard precedes imports and reads. Assertion-disabled execution is rejected. Imports are limited to pathlib, datetime, hashlib and json. No reviewed code was imported or executed. No network, subprocess, process/GPU probing, signal, model run, numerical import, root write or rerun appears in the collector.

`confined()` checks exact strict resolution of REPO and ROOT, the fixed ROOT parent beneath the expected repo, full path containment and nonsymlink ancestors. `owned_root_inventory()` enumerates all root paths (maximum 512), confines every entry, accepts ordinary files only from the fixed source/control/log/result allowlist, records directories without payload, fingerprints allowed ordinary files, checks total actual bytes against 67,108,864, rehashes every inventory file and repeats the path set. The output is compact metadata, capped at 65,536 bytes. This is sequential filesystem checking; it does not claim an atomic filesystem snapshot.

All six authority metadata mappings and all 22 numerical source names match the original launcher. All 47 preparation input bindings match exact local bytes and hashes. Existing `collect()` assertions are AST-identical to preserved v1. Expected START, terminal and execution-failure semantic hashes independently match the authorized saved diagnostic; RESULT, JOB and input identities are pinned.

The receipt retains `classification=retrospective_existing_evidence`, `original_workflow_success=false`, the original `physical_ownership_observation` terminal failure, the original local transport exit 1, no original post-child success path, no original final success-response checks, and an unresolved physical-observation gap. Original success receipts must remain absent. The original terminal elapsed/output bounds are kept separate from current collection time/root bytes. Qualification and fit admission remain false.

## Scope and preservation

Only bounded local source/AST/JSON/hash review was performed. No raw RESULT, input/model tensor, history, checkpoint, score or prediction was opened; source bytes were used only for static review/hash authentication. No endpoint, allocation, 18.77 or MacLink contact, collection, staging, run, signal, test, fixture or admission occurred.

All seven v1 preparation files and all six files in the prior sealed independent review are preserved and reauthenticated in `PRESERVED_BASELINE_BINDINGS.json`. V2 and original sources/metadata were not edited. The acquisition gap was promptly reported to root; the source agent is preserving v1/v2 and preparing a focused v3 successor.
