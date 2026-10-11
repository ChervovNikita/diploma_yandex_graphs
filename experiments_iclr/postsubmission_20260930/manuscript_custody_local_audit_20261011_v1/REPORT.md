# GNNM manuscript custody — local audit

## Latest preserved submission record

The latest manuscript-specific custody receipt found in the permitted workspace is `desktop_submission_storage_cleanup_20261002_v1/OVERLEAF_PRESERVATION.json`, snapshot **2026-10-02 20:45 UTC**. It records:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `overleaf/main.tex` | 27,885 | `fcf44541f6af547a77face3a765a8b02a8e762dc7d6d1765c41c49fb94c853e3` |
| `overleaf/appendix.tex` | 45,325 | `c3dae7170525d65aea287dbd6eedbb945a80d14ecc2c344e57bceb208160b3d1` |
| `overleaf/references.bib` | 20,451 | `a18b7b70ae213cdae30386c3d774a9220e24b957634ccdbab2ad87e3b3170729` |
| `GNNM_ICLR_2027_Overleaf.zip` | 375,965 | `d50728d8476a5b08375fed88bb52460b200438c37ae5f131884b402dabbc3bd6` |

These are recorded identities. The receipt's original Desktop location was not accessed. No later manuscript-specific custody receipt or platform submission confirmation was found locally.

## Exact recoverable server archive

Repository: `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs`.

Archive: `author_archives/mac_cleanup_20261002_v1/desktop_evidence.objects.tar.gz`, **5,206,338,271 bytes**, SHA256 `a06e58505baf4b2e033d45ad929da4c49204eef7ec4bc39f3d1a89b578482a95`.

The existing allocation receipt records complete member/object verification on **2026-10-02 20:51 UTC**. Its manifest SHA256 is `73f73778f64b25f0ef4a0f2c954f63f7f52c136409fd0be7be69258e2741a487`; that local manifest hash was checked in this audit.

The latest numbered manuscript review snapshot in this manifest is `reviews_round18_snapshot/overleaf/`. Its main source is archive member `objects/0ecb135e22ca4952c70e26081aa19a177b8c10cf53f23d3bdeef7576d2f5f0b8` (**27,788 bytes**). Appendix: `objects/3e7d4e632c963e0270fa8a326f1771d44c791e9e822714ab3eefc4645b252b18` (**45,266 bytes**). References: `objects/a18b7b70ae213cdae30386c3d774a9220e24b957634ccdbab2ad87e3b3170729`.

The archive also locates `FINAL_DELIVERY_SOURCE_BINDING.json` as `objects/a6407eca8229743a6da040639a40be53f52bd8f3f3a651935b0523417d5ed103` (**152 bytes**). Its contents were not recovered. This is a precise next locator for the final delivery's source revision.

The cleanup explicitly kept the live Overleaf folder and final ZIP outside the removed evidence tree. **Round18's main hash therefore does not identify the final preserved main above.** Remote availability was not rechecked.

## Git and user style prompt

`second_pass_inventory_v1/REMOTE_GIT_CUSTODY_METADATA.json` records server branch `codex/postsubmission-research-20260930` at commit `4c7b8e02e4b61e6c0a2b65980b54fbc619a8d8f1`. This identifies that repository snapshot, but no record maps this commit or a tracked Git path to the final manuscript. The local workspace branch has no commits.

No named copy of the user's writing-style prompt was found in the permitted local records or historical manifest names. The archive contains round18 reviewer prompt copies (`prompt_alfa.txt`, `prompt_bravo.txt`, `prompt_charlie.txt`); their exact object locators are in `AUDIT.json`. They are not established as the author style prompt. `FINAL_OFFICIAL_STYLE_AUDIT.json` concerns the conference-style check and likewise does not establish a user prompt copy.

## Scope

Local records only; no Desktop or remote access, archive recovery, PDF compilation, fits or canonical-state edits. GENLINK, BioHub and unrelated project content supplied no custody evidence. `AUDIT.json` includes source-file hashes and exact archive object locators.
