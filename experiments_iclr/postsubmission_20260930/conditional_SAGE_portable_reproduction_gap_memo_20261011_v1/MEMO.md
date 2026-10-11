# Portable reproduction of conditional SAGE

The current anonymous V3 source explains the E equations, five arms, optimizer, selector and calibration, and carries six compact evidence files. Its reproducibility statement names an accompanying code and evidence supplement, but this local package supplies no E acquisition or scoring entrypoint. `verify_sources.py` checks manuscript source and transcription. A small portable release of the existing numerical source is needed before an external reader can rerun the new method. Portability has not been verified.

## Existing numerical source and execution wrappers

| Layer | Existing source | Portable boundary needed |
|---|---|---|
| Canonical E acquisition | E V2 `run_family.py` | Retain native construction, factor initialization, route RNG, five-arm acquisition, own CE, strict selector, restoration and archive writer. Separate local binding and run records from cluster admission. |
| Conditional values | E V2 `value_transport.py` and `value_installer.py` | Reuse the exact equations and native SAGE replacement; preserve self messages, mean aggregation, root maps and gradients. |
| Native body and factor routes | `common_routes.py`, `core/factors.py`, original `models.py`, `run_common.set_seed` | Bind modules through package paths with source identities. E never calls the imported `run_base.train_step`; separating this unused import avoids its unrelated dataset/grid dependencies. |
| Sequential Physics execution | V5 `own_ce.update`, `port.render_sources` and role activation | Preserve four quarter-scaled route backwards followed by one AdamW step; retain the explicit source derivation and Physics profile. Keep release/PASS checks outside the public numerical API. |
| Selected prediction scoring | E `analysis.py` and its retained F math reader | Expose numerical scoring/calibration with a portable archive loader and local completion manifest, independent of original host/owner custody. |

E's current `Family` calls `contracts.validate` before numerical imports. It requires `HOST_GPU_SOURCE_DATA_ADMITTED`, the current hostname, a GPU UUID, runtime fingerprint and source/data/config receipts. Its reader also checks the original runtime and owner closure. WikiCS `acquire.py` additionally hardcodes the cluster repository, hostname and GPU UUID. These are author execution constraints, not part of the conditional-value equation.

V5 already exposes the useful separation `port.bind(e_directory, cfg, root_validate, release)` and source-only `render_sources`. However, `bind` still requires root release names, prior engine/resource PASS records and a capable-reference decision. An external release must introduce a genuine local validator and public run record; it must not manufacture those admissions or modify the sealed author wrappers.

## Smallest artifact set

1. **Numerical package.** Bind the immutable E owner and value modules, common route wrapper, factor implementation and original native model. Include the seed routine and a narrow provenance-bound extraction of E protocol/count constants. Add V5's sequential hook and profiled source derivation only for the corresponding prospective recipe. The existing complete `models.py` is about 11 KB; a large framework copy is unnecessary. Record hashes of original carriers and every extracted or generated source, with an explicit diff/AST mapping.
2. **Fixed recipes and portable roles.** Publish the exact E WikiCS recipe and ordered seeds, and the separate V5 Physics recipe. Provide a portable data preparation function preserving WikiCS raw/tensor fingerprints, split 0 TRAIN and `(val_mask | stopping_mask)` development roles, and its prepared support. For Physics, bind the public provider/preprocessing source, raw features, already-looped support, ten paired custom role ID blocks and TRAIN/SELECT/CAL/CONFIRM policy. Neither trainer may add loops or open later-role truths implicitly.
3. **Environment and ordinary runtime checks.** Supply a pinned Python/package/wheel recipe and licenses. Record the local runtime, dtype, device, source versions, support/role hashes and finite resource limits prospectively. Check native/builder parameter ownership, optimizer coverage, factual support and checkpoint/RNG restoration. Local checks must produce their own evidence; recorded author PASS results are provenance, not a foreign-host admission or a promise of bitwise output equivalence.
4. **Output contract and scorer.** Write config/runtime/source manifests, complete validation traces, selected model/optimizer/route-stream states and per-arm selection records, plus `selected_VALID.npz` with `ids,y,raw_logits,probability_mean,member_errors,pooled_errors`. Raw decisions retain archived float32 error authority; stable log scoring must not replace it. Reproduce E's fixed five-fold, 500-update scalar-temperature procedure and paired repair/harm accounting from the existing math. That reader hardcodes ten classes, 5,274 nodes and three seeds; it is not a ready generic Physics scorer. V5 supplies SELECT archives only. Faithful later-role reproduction also needs a selected-state label-free exporter and the separately frozen common CAL/CONFIRM scorer, rather than silently applying E development calibration.
5. **One short README and evidence manifest.** State the acquisition and scoring commands, expected arm/member/body counts, selection and role semantics, failure retention and output schemas. Supply hashes for the closed E arrays/records so a reader can inspect the reported scoring before paying for a fresh acquisition. Selected checkpoints enable output replay; fresh acquisition remains necessary to reproduce training. Keep historical benchmark replay limits distinct from the new E release.

## Minimal implementation action

Create a small public module with explicit `prepare`, `acquire`, `export` and `score` entrypoints. Its binding accepts local source/data paths, a fixed recipe, device and resource limits; it returns the preserved numerical Family plus a local run manifest. A package-local validator checks recipe/source/data/runtime facts and never requires credentials, a literal hostname/UUID or inherited qualification receipts. Preserve the sealed E/V5 author sources and wrappers as provenance carriers. Extract only the binding, loader and reporting boundary; preserve numerical functions and their ordering.

After that source work, verify installation and the declared representative native/update/restoration paths in a clean external-style environment before calling the release portable. This memo authorizes no implementation, training, new method, review or outcome opening. The current Physics and Penn94 result state is outside its scope.

## Source locations

All paths below are relative to the research workspace. Exact file hashes and line anchors are in `SOURCE_AUDIT.json`.

- `complete_anonymous_quality_draft_20261011_v3`: README, reproducibility statement at `main.tex:146`, and E acquisition/calibration description at `sections/conditional_value_E_PE_appendix.tex:4`.
- `native_SAGE_endpoint_value_transport_source_20261011_v2`: canonical E owner, contracts, value modules and reader.
- `native_SAGE_sequential_own_CE_Physics_source_20261011_v5`: `port.py`, `own_ce.py` and current API README.
- `shared_fast_graph_model_interface_20261010_v1/common_routes.py`, `portable_internal_be_public_interface_20261007_v2/core/factors.py`, and `common_wrapper_native_source_20261010_v1`.
- `common_wrapper_WikiCS_allocation_data_source_20261010_v1/acquire.py` and `SAGE_matched_joint_reference_77_complete_reader_source_20261010_v1/analysis.py`.
