# BUDDY shared-cache source v2 repair

The sealed source v1 packet is preserved. This packet repairs the independent audit's B1/B2/B3/G1/G2 without changing CONFIG.json, SOURCE_PINS.json, the four vendor snapshots, arms, seeds, learning recipe, objectives, parameter counts or deterministic cache semantics. It prepares sources only; no training or numerical qualification occurred on this Mac.

| Finding | Concrete repair | Verification |
|---|---|---|
| B1 native cast | Factorized feature branch uses the native `features.to(torch.float)` before concatenation. | Static cast check; all seven CPU tests retained with unchanged 1e-12/1e-10 parity tolerances, numerical execution pending. |
| B2 history/selection lock | Shared guard requires exactly 100 consecutive finite/ranged epoch records; derives earliest maximum Hits@50; binds identity, full ledger, selected record and checkpoint hashes. Selected checkpoint carries source/cache/arm/seed/epoch/record metadata. Both lock producer and consumer require real CPU envelope/state validation with strict keys/shapes/dtypes/finiteness. | Eight stdlib fixture tests pass, including missing/short/repeated/reversed history, wrong/tied selection, nonfinite/ranged values and altered history/checkpoint bytes. Real checkpoint assertions are prepared inside the seven CPU tests; pending execution. |
| B3 actual vendor bytes | Shared implementation identity verifies every pin and hashes actual vendor bytes plus every top-level Python file, config/pins and dependency extras. It is used by certificates/cache/run/locks and checked before data opening. | Copied vendor mutation fails immediately; source and lock fixture checks pass. |
| G1 loader ordering | Test split reads require the cached loader source SHA; convention and exact source are checked before `torch.load`. | Static ordering check and synthetic loader mismatch/missing-digest failures before any Torch import. |
| G2 finite losses/state | Existing logit/prediction guards preserved. Additional checks reject nonfinite BCE, gradients, parameters/BN state, Adam state, accumulated train BCE, validation BCE and emitted record values; `hits50` also guards scores. | Static inspection and strict JSON/record fixtures pass. CPU loss/gradient/Adam assertions prepared, pending execution. |

## Local evidence and retained gates

`LOCAL_STATIC_CHECK.json` records stdlib parse/pin/formula/ordering checks. `LOCAL_GUARD_CHECK.json` records eight successful stdlib synthetic tests. Neither is a numerical certificate. Test fixtures were removed after execution. No Torch/native model source was executed; no real dataset/model/checkpoint, GPU, remote host or network was accessed by this repair work. No paper acquisition or new primary reads occurred.

Run in root's authorized compatible runtime:

```sh
python check_source.py
python test_guards.py
python test_cpu.py --output CPU_QUALIFICATION.json
```

The seven tests must all pass without skips or tolerance relaxation. Root manages isolated datasketch/torch-sparse/torch-scatter additions and records exact runtime compatibility. Installed OGB source/per-file compatibility, complete official training-topology/data provenance, full cache correctness, all-arm resource qualification and resource admission remain required. No actual fitted family or final-test result is asserted.

Checkpoint/ledger binding establishes internally consistent recorded evidence, not independent proof that training happened. The processed graph's correspondence to official train edges/weights remains a complete-data qualification requirement. Factorized private gradients remain mean-scaled versus own-member independent BCE; Adam epsilon prevents an exact trajectory-equivalence claim. Dropout layouts remain reproducible without exact paired masks. Native1024 remains the disclosed full-node/year-0/training-only-test competence anchor rather than a published tuned-score reproduction.

V2 intentionally rejects v1 source certificates/cache/run/lock formats. Fresh v2 source qualification and artifacts are required. Finite-state guard overhead must be included in resource timing. Root will perform the bounded independent v2 recheck after this authoring slot returns.
