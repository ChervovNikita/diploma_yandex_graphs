# Exact literature raw asset offload

Status: complete. User-authorized cleanup was restricted to inactive raw assets in the two named literature packets. Both collaborating agents confirmed that these raw assets were not actively needed.

| Packet | Removed raw files | Removed raw bytes |
| --- | ---: | ---: |
| graph_count_conditioned_masked_link_auxiliary_prior_assessment_preparation_20261004_v1 | 29 | 21,545,443 |
| graph_count_conditioned_pattern_literature_assessment_preparation_20261004_v1 | 9 | 1,110,440 |
| Total | 38 | 22,655,883 |

All 67 original payloads were present and checked against their unchanged manifests before offload. Full exact raw assets and original small provenance were archived through one direct SSH binary transfer to the verified anogena-2 account, sole GPU UUID GPU-44039938-fd82-41d2-fefd-de71514e2fac. The remote archive contains 73 files including its manifest; all 72 declared files were rehashed in a separate post-transfer call, along with the compressed archive and durable archive seal. Local raw deletion occurred only after that verification and a durable local resolver write.

All 29 original nonraw payloads plus three existing manifest/seal files remain byte-identical locally. Conclusions, read scopes, algebra, comparisons, code, retrieval receipts, decisions, reviews and original sealed manifests/history were not rewritten. The pattern packet originally has no standalone SEAL.json; its existing sealed manifest remains its original authority. Indices and active-run files were untouched.

## Resolution and rereads

RESOLVER.json maps every removed asset to its exact remote archive path, original sealed path, bytes and SHA256. PREOFFLOAD_LOCAL_FULL_CUSTODY.json records the original complete local custody. REMOTE_FINAL_REVERIFICATION.json records current full remote custody. DELETION_RECEIPT.json records the deliberate local absence and retained local checks.

Original whole-packet local raw hash checks now require rehydration or use of this resolver; they must not be described as currently complete local payload custody. A named reread restores exact bytes without modifying any sealed manifest. The helper rechecks the one-GPU route, archive manifest/seal and named asset identity before placing the file at its original path. It refuses to overwrite a different existing file.

Example named reread:

    python3 '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/literature_raw_asset_offload_20261004_v1/rehydrate_assets.py' --path graph_count_conditioned_masked_link_auxiliary_prior_assessment_preparation_20261004_v1/primary/cpc.pdf

To restore a whole original packet before running its original full local checks:

    python3 '/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/literature_raw_asset_offload_20261004_v1/rehydrate_assets.py' --packet graph_count_conditioned_pattern_literature_assessment_preparation_20261004_v1

The helper was statically parsed; no raw reread or new research execution was performed during cleanup. Only the named archive directory on the authorized one-GPU repository was created. No seven-GPU scientific route, sudo, dataset/model/checkpoint access, active-run mutation or broader deletion occurred.
