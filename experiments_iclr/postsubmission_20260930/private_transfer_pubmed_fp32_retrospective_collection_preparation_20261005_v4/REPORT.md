# Focused source repair: retrospective collector v4 metadata allowlist

## Repair and status

This is a **focused source repair with static checks**, not an independent review of this agent's changes. Root will assess the exact delta against independently reviewed v3 before collection. V4 remains disabled before imports or reads; it was not imported, executed, transported, staged or admitted.

The collector changes only ten exact metadata path strings in `owned_root_inventory()`'s file allowlist and the preparation `INPUT_BINDINGS_SHA`. Every numerical, resource, custody, acquisition, confinement, inventory-size and failure-preservation predicate remains unchanged. No existing allowed path is removed; no wildcard or general metadata directory exemption is added.

V4 disabled collector SHA-256: `d75f044d5fe4ca1c7668b2f0a1f9099684afbaeab2c4aee3dc2a6c74e2d10f62`.

## Recorded V3 failure and scope

The actual v3 read-only collection returned `collection_completed=false`, `retrospective_assertions_passed=false` and transport exit 1. The root's exception-frame diagnostic locates the failed assertion at original v3 `owned_root_inventory`, line 87: ordinary-file/allowlist membership. Its assertion ASTs are identical to v3; only activation and exception frame metadata were added by root.

The saved metadata-only listing reports `files_opened=0`, 58 ordinary file paths and 4,709,294 total file bytes. Comparing those file names with the exact v3 allowlist yields precisely the ten added sidecars below. This repairs a collector allowlist omission. It does not alter numerical predicates or treat the failed collection as an admission. The listing's byte total is saved metadata, not the still-pending complete hashed v4 inventory.

The original qualification workflow remains failed with `physical_ownership_observation`, original transport exit 1, no original success receipt and an unresolved exact `/proc` cause. The v3 collection failure and diagnostic are retained separately. No numerical rerun is required for this source repair.

## Exact added paths

These paths are relative to the existing qualification execution root:

- `LOCAL_INTENT.json`
- `MONITOR_REMOTE_CODE.py.txt`
- `RESOURCE_ADMISSION.json`
- `RESOURCE_TRANSPORT.json`
- `ROOT_ADMISSION.json`
- `TRANSPORT.json`
- `metadata_20261005T131659Z/OBSERVATION.json`
- `metadata_20261005T131659Z/TRANSPORT.json`
- `metadata_20261005T132219Z/OBSERVATION.json`
- `metadata_20261005T132219Z/TRANSPORT.json`

The existing inventory mechanism will confine these files to the fixed repo/root, require ordinary nonsymlink files and ancestors, fingerprint their bytes, include them in the complete inventory/64 MiB total, repeat fingerprints and path-set checks, and export only path/type/byte/hash metadata. All prior rejection and compact-output limits remain.

## Static delta checks

- Reversing the ten literal additions and restoring only the previous input-binding hash reproduces v3 source bytes exactly.
- The complete module AST is identical after the same normalization.
- Every assertion AST and the entire `collect()` function AST are identical to v3. Acquisition reauthentication, exact RESULT/JOB/source/input/runtime/tolerance checks, all three histories/commit/guards, numerical/resource caps and retrospective failure flags are preserved.
- The ten additions equal all listing file paths excluded by v3; all 58 listed file paths are covered by v4.
- All 84 permitted source/metadata/evidence bindings authenticate unchanged, including the prior 50 inputs, full v3 packet/review, actual failed collection/transport, diagnostic source/receipt/transport and metadata listing/source/transport. The ten saved sidecars have local byte sizes matching the listing.

Only local source/AST/JSON/hash checks were performed. No raw RESULT, dataset/model values, history, checkpoint, score or prediction payload was opened. No target/numerical or collector import/execution, model run, test, fixture, signal, rerun, collection, endpoint/allocation/18.77/MacLink contact, original-file edit or admission occurred. Only this new v4 preparation directory was written.

## Key preserved evidence bindings

| Evidence | SHA-256 |
|---|---|
| V3 disabled collector | `5ae182e2ba1ef054455269b0878d2e8b5d0add1cf1a1a69c725c8ea8328ee47b` |
| V3 preparation manifest | `6f9037bac58ddf35bdb1741720001ec9462f8f5b2eed80d4e5f3a25bd2181ea3` |
| V3 independent review seal | `c25cd84558509952c98194483f80255c8e4c5e2a07fdfe4a04e4bd1e510ff7da` |
| Original failed V3 collection | `25fc11c1aa045c8152abeb8542a4941fc2335feed6a5cc2713bb3ced159a8bb1` |
| Original failed V3 transport | `c0f6527c2da937e6e7f710c0763fc28ff023af35b6c2e6aec48d2a4b579d27ae` |
| Exception-frame diagnostic | `91e28aa3f02e14e2410734a87487798a80a7654cd76be04b56b7f8b68b18a1b8` |
| Metadata-only listing | `30028cb9162f58f1d188c0305790bb451930560778d6162be3d999f9eaec3d42` |

Exact paths, sizes and all remaining pins are in `INPUT_BINDINGS.json`; focused checks are in `STATIC_VERIFICATION.json`. V1–v3 preparation packets, prior reviews, original qualification failure and failed v3 collection remain preserved. Root review and actual read-only recollection are pending. Qualification and fit admissions remain false; this packet requests no outcome.
