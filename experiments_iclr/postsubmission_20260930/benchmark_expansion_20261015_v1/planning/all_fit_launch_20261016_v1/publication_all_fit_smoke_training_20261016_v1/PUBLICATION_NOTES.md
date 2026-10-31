# All-fit smoke and training source publication

Branch: `codex/all-fit-smoke-training-20261016`.
Commit parent: `2e6ed736554f23901ab6fa74a3bdda061fd719bc`.
Expected unchanged live HEAD: `48329b19941ea869c439e5a220a3df0e30aa1237`.

The user has explicitly authorized one or two real iterations for every one of the 6,460 fits, technical repairs, then full training under tmux. USER_GO.json and the updated AGENTS.md preserve this conditional authorization. Publication itself starts no training and does not assert that the smoke condition is satisfied.

## Included

- Exact frozen all-fit smoke v1 source and manifest.
- Exact frozen all-fit resources v1 manifest and members only; the separately proposed resource changes and other unmanifested records are excluded.
- Executed campaign execution v1 source, preserved alongside its launch records and cgroup-view failure/repair provenance.
- Campaign execution v2, fixing duplicate cgroup mount views while retaining effective ancestor limits.
- Root launch source, authorization and compact current launch/staging/preparation records.
- Updated project AGENTS.md, the new publication helper and its stdlib source-check receipt.

A separate v1 terminal receipt was unavailable at selection time; no terminal result is invented. Root reports that four initial v2 smoke attempts failed before optimizer updates because server51 lacked a pinned NCN README excerpt in the link-policy source-binding closure. Root is restoring that closure and preserving the failed attempts. At this snapshot there are zero all-fit smoke PASS records and no full training starts. The included immutable launch receipts are observations of their own times.

## Stage and publish

Every selected byte is copied into this bundle's immutable `snapshot/` tree. STAGING_MAP.json gives `source_absolute` for the local snapshot, `stage_relative` for its fresh remote repository-relative staging path, and `target_relative` for the intended Git path. All snapshots are below 120,000 bytes and work with the existing stage_source.py. Stage the snapshots and control files; do not overwrite existing live source or evidence paths.

SELECTION.json uses `files[].source_path` for the snapshot and `files[].path` for its Git destination. Run the staged snapshot of publish_source_branch.py with `--root`, `--selection`, and a fresh `--receipt`. It reads the explicit parent into a temporary index, hashes the verified in-memory file bytes, creates a commit with that parent, checks the separate live HEAD and index, and updates only the new branch. Root owns remote publication and push.

The helper records conditional user GO as true. It performs no checkout, reset, live-index add, push or model launch. Its two tests used mocked Git subprocesses; no actual Git command ran during local preparation.

## Excluded

Datasets, provider caches, result payloads, checkpoints, run directories, raw logs, large inventories, execution plans, generated artifacts, private keys and tokens are outside the selection. The existing SSH key path mentioned in AGENTS.md is an instruction reference only; credential files and contents are never read or selected. Control files and the staging map are not self-included in the commit, avoiding circular hashes.
