# V2 source repairs

Preserved predecessor: `graph_ncNC_collab_resource_epoch_preparation_20261003_v1`, manifest SHA `8661011fe87a9980acf806678483e7ed6f903628024813dff07c8f00b1eec5f8`. Every predecessor payload remains unchanged. The immutable NCNC prototype remains under manifest SHA `ac04bc7f6f6f37b86c50d38436aee111e987195c5f992816c4a5ef9d175887ae`.

## Fail-closed accounting and status

V1 marked a twin complete before final accounting and caught accounting exceptions without clearing its successful status. Its entry point could similarly retain GPU parity success after a final accounting error. V2 adds stdlib `qualification_status.py` and moves commits behind validated complete accounting and final CUDA synchronization. Stage-local numerical success is isolated from the final qualification Boolean. Required accounting values must be present, finite, nonnegative, and state that the final synchronization completed.

Every caught runtime/accounting error leaves qualification false, records its primary/secondary conditions, and invalidates successful family/twin statuses. Successfully completed work and receipts remain available as evidence, without a qualified status. Diagnostic retry receipts cannot rescue a prior error. Exit success requires the exact final pass status and both qualification/accounting flags. Final metadata-write failure also clears success and exits nonzero.

Eight stdlib-only fault-injection cases exercise final-sync failure, missing accounting, invalid/nonfinite accounting, top-level GPU failure after numerical checks, family failure after both twins commit, diagnostic retry refusal, retained primary/secondary errors, and exact-status exit semantics. Fake dictionaries/meters are used; no numerical library, model, tensor, dataset or GPU is imported or executed.

## Existing binary continuity

Root's `graph_ncNC_77_runtime_qualification_root_20261003_v1/LIVE_RUNTIME_METADATA.json`, SHA `5bf8d386bbcbfdfbad75e4a71e5b9619204fa2cf48880113f913708e1feb1668`, observed 32 existing torch-sparse/scatter CPU/CUDA `.so` files. V2 binds that exact list, byte counts and hashes in admissions and checks every file before numerical imports. Runtime also rechecks them, verifies actually loaded extension paths belong to the admitted package directories/list, and carries the binary pins/loaded paths in `runtime_identity`. Resource execution must match the GPU parity identity. No binary was acquired, copied, installed, built, loaded or changed during this preparation.

The saved installed PyG sampler source SHA remains `c04beecc5331144a2e10fdc3c66fbd8b4ac495f9dbdeb1649ae3cf047237b2f9`; the root-observed function SHA is `0f26dd305f8c4443d0231a9ae09f591bd741d671c3e9ddb68f0debc07218112b`. Native default sampler semantics were independently source-audited and remain unchanged.

`gpu_parity.py` and `train_only_data.py` are byte-identical to v1. All scientific/native/prototype semantics, full scope, tolerances, RNG use, initialization, dropout, losses, optimizer, candidate schedule, twins and serving pool remain unchanged. No new seed, budget, predictive gate, scientific fitting run, data access, runtime probe or remote action is introduced.
