# Campaign51 readiness publication selection

This is a source and compact-record publication for branch `codex/campaign51-readiness-20261015`, based on `48329b19941ea869c439e5a220a3df0e30aa1237`. It conveys no full-campaign GO or runtime/resource admission. No remote operation, Git write, model execution, or scientific launch was performed while creating this selection.

## Staging and publication

`SELECTION.json` follows `publish_preparation_branch.py`: every `files[].path` is repository-relative and binds exact bytes and SHA256. `STAGING_MAP.json` maps each selected local file to that target. Root stages those exact bytes and separately stages the control files under this directory, verifies destination identities, and invokes the publisher with the repository root. The publisher uses a fresh temporary index and preserves the live HEAD and live index. Root owns publication and push.

The final local dependency-planning package is mapped byte-for-byte from `planning/campaign_dependencies_concurrency_20261015_v1` to `planning/campaign_dependencies_concurrency_20261015_final_v1`. All 26 members and its unchanged manifest are selected together. The original remote `dependency_plan.py` draft and its historical execution location are outside this selection. Compact receipts and reviews are copied into `publication/campaign51_readiness_20261015_v1/records/<original expansion-relative path>`. Their original local source and original repository-relative identity are retained in the staging map. Existing remote evidence at the original paths is preserved, including formatting/newline differences. Continuation and the publisher helper retain their normal fresh planning paths. The final package uses local sibling imports or explicit path arguments; relocation changes no source bytes.

## Selected source packages

- Frozen server51 campaign port: all 66 members plus manifest.
- Metadata projection cache wrapper: all 3 members plus manifest.
- Final dependency/storage/concurrency planning: all 26 members plus manifest, at the distinct target above.
- Remaining-runtime v2: all 15 members plus manifest and both v1/v2 reviews.
- Server51 concurrency v3: all 9 members plus manifest and all three compact review records. Current v3 source review is clear; actual resource admission and launch remain separate.
- Campaign applicability v1: all 4 members plus manifest. Generated class-map receipt covers exactly 6,730 tasks; no runtime PASS gate is created.

Current compact projection, deployment, source/data transfer and applicability receipts are included, together with the finalized continuation, corrected observation record, environment probes, allocation dependency inventory and publisher/source check. Historical failures remain records. Receipts describe their saved observations; they do not establish a current capacity reservation.

## Exclusions and limits

Generated releases, deployment archives, descriptors, queues and application-class maps are excluded, as are dataset payloads, provider caches, checkpoints, result arrays, raw logs and credentials. No file exceeds the publisher's 2,000,000-byte limit. The complete generated artifacts remain represented by compact path/size/SHA receipts. The core input transfer may still be in progress; an absent final receipt is not treated as success. The full 6,460-fit plus 270-export campaign still requires user GO.

`CHECKS.json` records local byte/manifest/schema/AST/target checks. This v2 control bundle supersedes the earlier v1 selection for publication. The selection and staging controls are not self-included in the selected commit contents, avoiding circular hashes.
