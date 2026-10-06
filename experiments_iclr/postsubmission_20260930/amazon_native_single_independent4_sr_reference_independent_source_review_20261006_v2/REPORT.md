# Independent narrow re-review: native reference V2

## Verdict

**PASS for static source scope; new native qualification, root resource admission and external whole-child closure remain pending.** Reviewed source is 37,430 bytes, SHA256 `0d8e677b4e2c9b3616a0795cc8f1b55758cc63d7048b2d42f614a295a1c1a6ef`. The V1 blocker is preserved verbatim in this review; the original readonly review is unchanged.

## Repair

The three worker guards close NATIVE_REF_INPUT_BINDING_01. Lines 335–336 require the actual preflight manifest digest to match the fixed input pin before W-label access. Lines 345–347 require actual W-reader provenance to match that pin before data_w or W fitting. Lines 383–387 require actual B/S/R-reader provenance to match that pin before data_sr or S/R fitting. The pinned accessor already verifies label archive bytes against the accepted manifest descriptors. The self-consistent changed frozen manifest/label pair admitted by V1 cannot now be accepted for training while asserting the original identity.

Timing is precise: the B provenance guard follows label decoding and precedes training admission. V1 suggested a separate pre-S/R filesystem recheck before decoding; V2's actual-reader guard supplies the accepted-input binding under the unchanged frozen-input contract. This pass does not claim that a changed B manifest could never be transiently decoded on a failing path or provide concurrent-file-mutation security. All four W400 images still freeze before the B/S/R reader.

## Unchanged semantics

Only run changes. Removing the exact three repairs and their messages makes its AST identical to V1. Every other function/class, top-level import/constant and source pin is unchanged. The prior static conclusions therefore remain: four independently fresh complete native models and empty Adam; exact V6 defaults; own model/Adam/RNG histories and dormant clocks; all W400 frozen before S/R; fixed member0 SINGLE alias; full 2700 update horizons; no selector/rewind/retry; checkpoint and work accounting; safe early-failure output ownership.

Root design and cost-plan files are byte-identical. Protocol adds only repair provenance/schema; qualification/admission source and protocol hash bindings are current. The packet manifest, seal, readonly rows and anticipated flag-only release hash `c76bff8e5f4298ad61ac6f7f4f9dbc17b8ff1c3ad2b4c9cd6541b9b289473909` verify. SOURCE_RELEASED=False and unauthorized templates remain preserved.

## Scope

Only the V2/V1 source delta, metadata and preserved finding were examined with stdlib AST/JSON/SHA checks. No prepared source was imported/executed, no data/label/checkpoint/prediction payload was accessed, no server/SSH/GPU action occurred, and no source or old packet was changed. New full-native qualification, actual measured root fit caps and reviewed owned-child/watchdog/wait4 closure remain necessary. Serving/held scoring requires a separate evaluator/admission. No numerical, predictive, efficiency, manuscript or novelty verdict is made.
