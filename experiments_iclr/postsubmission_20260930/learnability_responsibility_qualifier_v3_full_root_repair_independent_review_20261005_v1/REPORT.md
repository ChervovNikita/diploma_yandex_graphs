# Independent qualifier V3 full-mode root repair review

6 October 2026. **Static delta PASS; the full-mode root mismatch is repaired.** No concrete blocker was found in this narrow source/plan review. No imports/runs, source edits, SSH, data/result/mask payload access or fits. This review supplies no full-native numerical or scientific qualification.

Reviewed worker SHA256 `baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688`; manifest `418e71539b9ad437f738abe22dec0d45bdd2663130b9a514971ba92963516c1a`; seal `dedf9c8869d6539bf33a912d33be9c10a83db8e545397687b1c0ea649573c639`. All seven packet files are readonly. Manifest entries and listed source/metadata bindings match; described data and engineering-receipt payloads were not opened or hashed in this review.

The entire V3 worker equals reviewed V2 with exactly one line changed in full mode:

```python
payload = accessor.load_public_b(source_root, Path(args.public_b_dir), device=str(device))
```

`source_root` was already resolved from `--source-root` and used for the immutable phase sources. Passing it to the accessor correctly resolves its public artifact, native recipe, producer and PyG helper descriptors relative to PHASE. The main repository root remains bound to the interpreter/site-packages checks. The explicitly supplied public+B projection directory and device argument are unchanged. The actual one-line difference matches the sealed patch.

The reviewed enabled accessor remains SHA256 `9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85`. Its source-hash-bound projection, W/S/R roles, A-ID-only reader, exact CPU edge recipe/helper pins and full native context are unchanged. This call performs no custody projection, A evaluation or scientific warmup/arm execution.

Every other V2 byte is preserved: synthetic fixture/path/direction, local float64 scales and original tolerance, float32 scales, all later checks, source/entry guards, fixed180/900-second deadlines, resource/failure receipts, complete public episode and original-phi recompute/restoration behavior. A genuine V2 `PASS_SYNTHETIC_ENGINEERING_ONLY` receipt with the same immutable PINS satisfies the unchanged prerequisite predicate. Root reports such a V2 result; its payload was not independently reviewed here. The original coarse V1 failure remains preserved.

PLAN now correctly says native activations are nonsmooth, fine FD supports the tested local direction, and neither a base-state kink nor the cause of the coarse failure was proved. It claims no full-graph FD result, actual full execution or fit release. Default full mode still leaves optional full FD unperformed and must report that limitation. Actual full-native context, higher-order/public-episode and resource evidence must come from the separately authorized run under these exact bindings.

Exact reviewed identities/read scopes and closure evidence are in `PROVENANCE.json`. No predecessor source or review was changed.
