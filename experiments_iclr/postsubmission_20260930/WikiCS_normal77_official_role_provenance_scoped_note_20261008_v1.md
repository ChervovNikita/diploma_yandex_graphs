# WikiCS normal77 official source and role provenance

Date: 2026-10-08. Scope: the exact TRAIN/VALID NPZ archives pinned by the reviewed Wiki12/SupCon15 aggregate collector and graph-relation source v2.

## Finding

**Existing historical normal77 evidence supplies the official source, mask and ordered-tensor chain for these exact archives.** The collector directly pins the archive hashes and successful safe-role preparation receipt; its source bindings do not directly pin the earlier official acquisition record and manifest. The remaining evidence gap is to bind those existing records into the review packet. A new raw-role computation is not needed to supply the missing historical provenance.

All archive and payload paths below are relative to the normal77 phase root:

`/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930`

| Archive | Recorded bytes | SHA256 |
| --- | ---: | --- |
| `wikics_unit_contrastive_attribution_gpu77_data_20261007_v1/train.npz` | 21,137,972 | `dce6c4a982027604dad198533352d397c5d1b7a0a6d6fa4868468c872457f9e2` |
| `wikics_unit_contrastive_attribution_gpu77_data_20261007_v1/valid.npz` | 84,878 | `99114cb1d50b876e9cdb2d4a1c116e217f0a8aa2583a2561d0491461c0c0ec48` |

## Existing chain

1. **Official acquisition, 2026-10-07 00:34 UTC.** The saved successful receipt targets `shmelev@192.168.18.77`. Its embedded normal77 program, decoded and inspected without execution, requires hostname `peptide`, the known repository/runtime and CPU-only execution. It uses PyG 2.7.0 `WikiCS(is_undirected=True)` without transforms, and checks the downloaded raw file and installed loader source against their recorded hashes. The receipt includes the official download message and exact preparation-record bytes.

   Official URL: <https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset/data.json>. Raw file: `wikics_official_acquisition_gpu77_root_20261007_v1/official/raw/data.json`, 82,533,647 bytes, SHA256 `9bf8cb3ef8eeae81b25e6ccbe0ea195600c205d7edf63ce04f2ec8d9c7dcb3d8`. Installed loader source SHA256: `d5a10b29d64aaaa422764c1148d089513b400826eb9590a13ae6316a40f77d17`.

2. **Official roles and safe projection.** The actual normal77 source selects split 0: TRAIN is `data.train_mask[:,0]` (580 nodes); VALID/development is `(data.val_mask | data.stopping_mask)[:,0]` (5,274 nodes). VALID therefore combines official validation and stopping masks. It asserts mutual disjointness with the official test mask, derives IDs with `nonzero().flatten()`, and writes only `x`, `edge_index`, `train_ids`, `train_y`, `valid_ids`, `valid_y`. Graph processing is `to_undirected -> remove_self_loops -> add_self_loops`, producing 11,701 nodes, 300 features, 442,907 directed edges and 11,701 self-loops.

   Safe source: `wikics_official_acquisition_gpu77_root_20261007_v1/available/wikics_split0.pt`, 21,223,748 bytes, SHA256 `74868c33350221f3a38f26f156a72003f6ee943f6275fb87c205f6bc1cf775a5`. The successful record reports `complete: true`, verified actual hashes, and byte identity to the pinned safe payload. Official raw public labels were loaded during CPU acquisition; only TRAIN/development labels entered the safe trainer payload. A claim that raw TEST labels were never read would exceed this evidence.

3. **Exact NPZ conversion, 2026-10-07 16:11 UTC.** The later successful receipt hash is already pinned by the collector. Its command hash-checks that safe source and public data pins, verifies all six tensor shapes/dtypes/raw-byte hashes, creates TRAIN NPZ keys `x, edge_index, ids, y` and VALID NPZ keys `ids, y`, then loads them through the public interface and asserts `torch.equal` against all six safe tensors. Its record reports `ordered_array_equality_with_safe_source: true`, the exact archive hashes above, and no TEST-label access or training/GPU work in this conversion step.

   `official_source_verified_by_interface: false` describes the public interface's archive-domain checks. The separate acquisition evidence provides official provenance; the interface flag itself does not establish that provenance.

## Local verification and evidence to bind

This review verified local file hashes, equality of the receipt-carried preparation bytes with the saved record, both saved command hashes against their receipts, and the decoded acquisition-program hash (`c5bda1db4801bdb1f1cb1c6b6eac9814915b3168b914fb25e0a41087e347a470`). All six fingerprints agree across acquisition, NPZ preparation, collector bindings and public data pins. Both current binding files agree on the exact archive hashes.

The local NPZ paths are absent. This is a review of saved historical evidence, not a new verification of current remote raw/payload/archive bytes. No remote operation, numerical outcome, checkpoint, benchmark, fixture or GPU work was performed.

Paths in this table are relative to this note's local directory, `postsubmission_research_20260930`.

| Evidence | Locally verified SHA256 |
| --- | --- |
| `wikics_official_acquisition_gpu77_root_20261007_v1/PREPARATION_RECORD.json` | `bf8a9a43ed7beda2d2a4167bef60d7aa1205fcf214426c06d93550c6f306ae0e` |
| `wikics_official_acquisition_gpu77_root_20261007_v1/AVAILABLE_MANIFEST.json` | `4c2fe1e36b3b0729917f2a4845b986bae07bcde8b74b568e5802ee311d7f0564` |
| `gpu77_connection_recovery_v1/commands/wikics_gpu77_cpu_acquisition_20261007_v1/RECEIPT.json` | `58bdfb8a971a214762bdc341fc3a9ffb512c259b5173ba3334ac404acc9519e3` |
| Same acquisition directory, `COMMAND.txt` (contains the actual program) | `66b0137b6322f7e98720fd9825bb8cbee65e4ef1ba5c2f7a453f5b8dc4e01647` |
| `gpu77_connection_recovery_v1/commands/wiki_attribution_complete_safe_role_preparation_20261007_v1/RECEIPT.json` | `9ccfa9f75dd31d858037124ff6d939ee0bafc181a0b68bac957e9113ae939791` |
| Same safe-role preparation directory, `COMMAND.txt` | `7ec54b956489e2eef63a667a96db639a8f2809b0071bbd6440af9fe45cbdb0bf` |
| `portable_internal_be_public_interface_20261007_v2/PUBLIC_DATA_PINS.json` | `37ba470c09d55c2f7c6b68177423b430353b8966bd6402685001fffa8d9da0c8` |
| `Wiki12_SupCon15_union_collection_source_20261008_v1/SOURCE_BINDINGS.json` | `06fedacad449e6deac3d93f609eb865055b6e3fddb1cfd71fc3c110551bc5527` |
| `graph_relation_private_credit_source_20261008_v2/SOURCE_BINDINGS.json` | `ca03d95bb78d4cc95dd796675104798961abe5110aa35d96f360706c4d17c80d` |

## Recommended closure

Bind the existing official acquisition record, manifest, acquisition receipt/command and already-pinned NPZ preparation receipt/command to the review packet, preserving the combined VALID-role definition and historical-verification scope.

If an independent fresh check is required, the cheapest follow-up is CPU-only on normal77: hash the existing raw file, safe payload and both NPZs; reconstruct the same split-0 masks/projection from the existing official source; compare all six ordered tensors with the safe source and NPZs. Use the existing pinned loader/runtime, with no download, fitting, scoring or GPU work. That follow-up was not performed here.
