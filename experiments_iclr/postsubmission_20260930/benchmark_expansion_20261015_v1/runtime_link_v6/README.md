# Runtime link v6: authentic native receipt annotations

Narrow successor of frozen runtime_link_v5 (manifest bfb796eef480d20662ad66db4f56ce6fce241f6f561e25e2dfff59fbca695831). The v5 source-only blocker is preserved at qualification/runtime_link_v5_NATIVE_BLOCKER.json.

Only adapters.py::_link_source_dependencies changes among the seven runtime/policy producers. It verifies exact path, byte length and SHA256 identity; requires the native manifest annotation adapter_hashes_verified to be the boolean true; preserves that annotation in dependency_annotations and leaves authentic native metadata and checkpoint annotations untouched. Unknown or misplaced annotations, changed bytes and conflicting identities fail.

The original v5 worker, adapter lifecycle/export methods, native objectives and finite engineering policy are byte-identical. No source/checkpoint migration is provided. Production keeps original link_policy_v1.

check_receipt_contract.py runs the actual dispatcher function extracted from adapters.py with the annotated receipt from a completed native NCNC check. Twelve source-only checks pass, including refusals and producer preservation. The 29 v5 tensor/authority checks are inherited coverage of unchanged mechanisms, not rerun here. No native model, forward, optimizer update or TEST access occurred for v6 preparation. Native lifecycle qualification remains pending publication and explicit admission.
