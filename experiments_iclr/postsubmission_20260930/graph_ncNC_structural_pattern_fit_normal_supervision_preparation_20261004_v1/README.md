# V5 ordinary supervision for fixed J/F fits

Source preparation only. Each disabled release example authorizes no invocation. Root reviews exact source and the proposed 24 h per-fit wall ceiling against actual fullgraph timing, then binds both actual V5 qualification PASS receipts and their matching normal-supervision terminals before releasing either fit.

The supervisor accepts exactly one fresh J or F seed0 invocation. Separate per-arm output/session handles support GPU0/GPU1 dispatch. Each original fit runs 100 epochs, 1,700 native optimizer updates, 100 complete VALID selector candidates and two complete selected-serving/serialized-replay VALID traversals. The sealed V5 driver, native Adam, objectives/lambda, initialization, selector, batches, VALID pools, thresholds and checkpoint memory policy are unchanged.

Memory caps remain 32 GiB host RSS, 70 GiB CUDA allocated and 75 GiB CUDA reserved. The ordinary process/session supervision, own-group signals, wall/RSS/wait4 loop and persistent CUDA observation are reused. Process cuBLAS workspace configuration precedes Torch import, and completion requires the same measured V5 transition/final profile.

The driver preserves complete current and selected model/Adam/RNG/module-flag snapshots in its own journals, both state slots and selected artifact. The supervisor binds owned artifacts by streamed hashes, checks original completion and typed replay receipts, and records inclusive costs and failures. It does not deserialize scientific state or private selector payloads. A cap failure preserves partial artifacts and paid costs; there is no automatic retry/resume or TEST.

SOURCE_DIFF.patch is the complete adaptation from the sealed fullgraph supervisor/child. FIT_WORK_PLAN.json and BINDING_ADAPTATION.json document existing work and stdlib source checks. No driver or historical evidence trees are copied.
