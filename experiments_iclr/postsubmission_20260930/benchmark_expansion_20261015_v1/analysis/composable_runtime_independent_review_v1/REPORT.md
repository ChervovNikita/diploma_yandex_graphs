# Independent review: composed normalization runtime V1

**Result: no actionable source blocker found within the requested scope.**

Reviewed frozen manifest `518784a9e83f4ba92c4f7bff55705999a787c7a67ff69c0cc9c09b5530550d38` and all 28 listed files. The package was not edited. This is a source and scalar-protocol review; neither native case has been executed by this reviewer.

## Findings

- **Producer/helper identity:** `runtime_bridge.py:88–111,132–154` binds the actual new producer separately from the pinned unchanged helper. The helper receives its original authenticated runtime view; the public export retains the new producer identity. The producer binding is inside the complete saved normalization envelope.
- **Restore and selection:** V2 `adapter.py:196–214,234–267,284–307` addresses both earlier reproduced faults. Its scoped post-load verification export admits the original parent's exact read; full parent-plus-bank state is checked before passing contained parent state to the original selected binder. The new bridge retains this envelope. Resume uses the same genuine construction and envelope restore; stopped finalization runs no TRAIN loop.
- **Scorer ownership:** `final_scoring/role_serving.py:237–243` enters the outer inference owner before either molecular or BUDDY role execution. `runtime_bridge.py:156–167` fingerprints model state plus external BN banks under the export owner. The existing molecular `run_member` and BUDDY `_route` calls reach the installed bank contexts; the scorer keeps optimizer refusal and checks the combined fingerprint.
- **Typed migration:** `descriptor_binding.py:74–118` classifies the exact parent receipt pointer; `host_binding.py:145–161` authenticates the parent and reconstructs the exact V2 migration. Generic rebinding refuses composed descriptors. The engineering qualifier's shortened controller is explicit and separate from the retained compiler parent.
- **Qualifier interface:** `qualify_native.py:91–174` runs the complete parent epoch, so molecular batches use OGBAdapter's owned iterator rather than the previously unusable fresh direct-update path. Its session-update wrapper reaches the actual molecular and BUDDY update calls. The declared MolBACE GIN/dropout0 seed1 and BUDDY seed1 bounds are 38 and 1,585 updates. The selected restore, stopped resume and fresh scorer VALID path keep the existing native loss/update methods and route seams.

## Focused evidence

The package's stdlib source check passed. An additional combined scalar check executed the actual pinned parent load, selected binder, mean exporter and `_check_runtime` bodies through the real V2 owner and runtime bridge. It passed complete-envelope restoration, public/helper identity retention, unchanged caller payload, zero added forwards and stale-export refusal after restore. Details are in `EVIDENCE.json`.

Actual tensor gradients, BN numerics and native serving compatibility remain the purpose of the two already prepared finite qualifiers. No extra native test matrix is proposed. No Torch, model/data/checkpoint read, remote operation, Git operation or launch occurred. The inherited `convergence_established` field remains a validation stopping flag; this review makes no stationarity claim.
