# Safe 77 public Git transport and sealed F source deployment

## Exact Git sync

Verified host `peptide` and both authorized physical GPU UUIDs before repository reads or writes. Fast-forwarded the repository from `a7c87fcbfa1c6d4093fabb542eb2240c49686ae1` to exact `ab8419011cebd09beedc35f3e756baab95995b7e` after verifying the incremental public bundle and its prerequisite. Final tracked edit count: zero.

A bounded public HTTPS fetch timed out. The fallback bundle contains 21 public commits, is 7,878,716 bytes, and has SHA256 `b10bdda80b65ce61e510714fd162f5d89b5d0096457d6a2eb598160f11317083`. Every one of its 107 transferred chunks and the assembled bundle were verified. No network or mount settings changed.

All 50,169 preexisting untracked/ignored files were accounted for. Fourteen colliding files were preserved intact by atomic rename under the repository's `.git/codex_public_transport_20261010_v1/preserved_untracked/`, retaining original relative paths. The remaining 50,155 stayed in place. No missing or changed-inode untracked paths were found. Preservation mapping and original metadata are in `EXACT_SYNC_RECEIPT.json`; the remote journal is `.git/codex_public_transport_20261010_v1/PRESERVATION_TRANSACTION.json`.

## Follow-on F source transfer

The root separately committed the sealed F acquisition source, its independent source review, and the complete F reader at allocation commit `63b643f7e4c75492a7ae2ebc07aa2766f8e65723`. The named transfer comprises exactly 23 top-level `.py`, `.json`, `.md`, and `.patch` files from those three directories. Only equal-or-absent destinations are permitted. `F_SOURCE_TRANSFER_PLAN.json` pins each file and both supplied source/reader manifests. Deployment completed with 23 created files, zero conflicting destinations, and all source hashes verified (`F_SOURCE_TRANSFER_RECEIPT.json`). Git HEAD remains `ab841901`; these newly deployed files are available for the root's subsequent qualification and publication.

## Existing runtime metadata

`EXISTING_RUNTIME_METADATA.json` selectively records historical saved project receipts. The normal project provider is `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python`. Its 6 October successful import receipt records Python 3.11.14, Torch 2.1.2+cu118, PyG 2.7.0, torch-scatter 2.1.2+pt21cu118, and torch-sparse 0.6.18+pt21cu118; CUDA remained uninitialized in that historical receipt. The earlier base rapids provider and isolated dependency setup are separately identified. These saved observations do not establish a new runtime qualification.

No scientific job, native model, numerical calculation, dataset, TEST payload, or quality outcome was opened by this transport task. No credential value was copied or recorded. Root owns runtime qualification, scientific execution, canonical research state, and publication.

## Publication scope

Keep the opaque bundle, encoded chunk commands, and per-chunk transport artifacts as local operational transport material. The concise decisions, hash inventory, final sync mapping, runtime metadata, and sealed F deployment receipt provide reviewable provenance without publishing redundant payloads.
