# Inactive full-input BSNN engineering qualifier

This separate packet prepares one engineering check of the sealed authentic BSNN Cayley d2/f32/L2 wrapper in the recorded existing runtime. It is inactive: no source/model import, role-array read, numerical run, server action or installation was performed during preparation. The sealed baseline wrapper and prior packets are unchanged.

## Fixed work

The prospective qualifier uses paired seed1103 and the exact sealed starting configuration. It calls the actual sealed BSNN CPU-constructor/explicit-placement factory, including its edge-weight index transfer, and the unchanged actual NSD V2 role/RNG helpers. Its model is the original `BayesBundleSheafDiffusion` at author commit `73b15dd1e7bb737d54dfe222b098a7618619cbb9`; source/math bytes are untouched. No author experiment driver is called.

There is exactly one full native TRAIN forward, one backward and one Adam step, using every official TRAIN label and the complete Tolokers graph/features. The sole update preserves mean TRAIN NLL plus native KL and the original epoch0 coefficient `sigmoid((0%40)/2-10)`. Loss, outputs and gradients must be finite, and at least one parameter must change. No repeated update, seed/config replacement, training continuation or competence metric is included.

After that update, the model makes four genuine full native eval calls, averages probabilities and takes the log. The only checkpoint is the sole post-update state; there is no validation selection. It contains model/optimizer state, all Python/NumPy/SciPy/Torch streams and the full-graph pooled output. The trained model is released. A second fresh original constructor loads exact parameters/buffers and optimizer state, restores all owned streams, then makes four genuine native reconstruction calls on the same recorded evaluation stream. Total work is **two original constructors, one TRAIN update, eight eval calls, nine full model calls and eighteen native per-layer SO samples/array transfers**.

The exact six-key V2 loader validates the supplied official TRAIN/VALID archive and its hashes. VALID arrays are immediately removed, never transferred to GPU or scored. There are no VALID metrics, AUROC calls, TEST truth inputs, full y, mask permutations or checkpoint ranking. Full-graph output differences and prediction-change counts are label-free engineering diagnostics.

## Practical gates

Exact model/optimizer/stream restoration is required. Full-graph maximum absolute log-probability and probability replay differences are recorded and fail only above the fixed practical gross limits0.001. Prediction changes and first/last native draw differences are recorded without a bitwise-output gate. There are no orthogonality/gradient microscopic parity limits, direct-device twin update, statistical campaign, repeated update or retry. Passing establishes this one full-input engineering path; it does not establish competence, accuracy or permission for a baseline screen.

## Costs and failures

The qualifier records actual imported module paths/file hashes and all eight package versions, original CPU constructor timings, input-role tensor transfer bytes/time, placement time after constructor return, actual SciPy CPU sample shape/dtype/bytes/time, native NumPy sample-to-GPU tensor shape/dtype/bytes/time, synchronized full-forward/backward/Adam/checkpoint/reconstruction costs, total CPU/wall time, cumulative RSS and CUDA allocation/reservation peaks. It records whether the native sampler RNG aliases global NumPy and exercises the sealed wrapper's RandomState/Generator API and exact restoration.

For cost observation only, it temporarily wraps the frozen sampler's `rvs` and NumPy-array `torch.tensor(..., device=cuda:0)` calls. Each observer calls the original API exactly once and returns the original value; original hooks are restored in `finally`. No sample, map, gradient, model forward, KL or objective is replaced. Synchronization and timing overhead are included. Scoped times overlap and must not be summed as disjoint components; they are engineering cost evidence, not a baseline timing estimate. The placement residual includes transfer and the sealed factory's remaining checks. Static topology and edge-weight index storage are reported separately.

Per-operation COST_EVENTS, early/partial RESULT, owned checkpoint, FAILURES and COMPLETE preserve successful work and exceptions. Cleanup or unavailable CUDA cost finalization prevents a passed qualification. Hard process kills cannot run Python finalization; root owns ordinary subprocess supervision/resource limits and must retain the last partial records and external terminal/resource evidence. No supervisor is added here.

## Release

The disabled release template has no execution authority. Root must supply an external enabled release bound to this packet's exact seal, the authentic wrapper seal, committed execution-source identity, exact role archive/metadata hashes, one visible cuda:0, all eight recorded dependency versions including SciPy, explicit deterministic policy and a fresh normal-phase output directory. The existing Torch2.1.2+cu118/NumPy1.26.4 runtime is required. Prior BSNN work qualification is not required because this check is intended to produce engineering evidence for that review. Root must review actual imported providers, sampler behavior, diagnostics and cost/failure records before adopting qualification.

`static_verify.py` parses sources and checks pinned byte hashes only. It never imports the qualifier, sealed wrapper, author model, numerical providers or data. See `CONTROL_ASSESSMENT.md` for the separate deterministic bundle control recommendation; no control runner or new baseline screen is implemented or authorized here.
