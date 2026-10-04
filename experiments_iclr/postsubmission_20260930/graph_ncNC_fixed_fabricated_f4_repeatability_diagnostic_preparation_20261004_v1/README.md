# Fixed fabricated F4 repeatability diagnostic

Source preparation for one diagnostic on the preserved failed qualification's existing `factor_private4`, seed 3, selected snapshot. This packet has not run numerical work. Its example release is disabled and has five missing remote fixture descriptors.

## Fixed work

Exactly four original `pilot_evaluate.score_valid` calls, in order: `False1`, `False2`, `True1`, `True2`. Every call uses `warn_only=False`, the original `pilot_model.make_factorized`, and the same authenticated complete selected model/Adam/RNG/module-flags snapshot restored through `pilot_state.restore_snapshot`. Full fabricated query pools have 60,084 positives and 100,000 negatives. There is no training, fixture generation, seed search, tolerance, fallback, or replacement call.

Original runtime admission remains deterministic False, FP32, TF32 False, and mixed precision False. True is an explicit diagnostic transition after admission. All four calls share `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before interpreter startup. This is a **new diagnostic process setting**: the failed qualification did not bind its ambient cuBLAS value.

## Report

`DIAGNOSTIC.json` retains per-call profile and failures, both actual raw-pool digests, historical reference-digest equality, original Hits50 and exact selected-Hits50 equality, restoration and after-call state/RNG/flags receipts. All six call pairs receive per-pool exact digest equality, numeric and bitwise changed query-row counts, maximum absolute and symmetric relative error, and maximum FP32 ULP distance. Signed-zero changes count as bitwise changes; ±0 share one ULP rank. Arrays stay in process memory and no tensor bodies are exported.

The historical reference digests come from the authenticated journal. Neither historical reference arrays nor the failed replay arrays were retained, so historical numeric-error measurements are unavailable. The old failed cell's Hits50 comparison was not reached and remains unknown.

`DIAGNOSTIC_COMPLETE` means all four calls and metrics were collected. It does not require matching digests, issue qualification PASS, change production replay admission, or authorize scientific replay. Scorer failures are retained. Remaining predeclared calls continue only while snapshot/RNG/tensor/profile custody holds; a setup or custody failure stops further calls.

## Root completion before launch

1. Authenticate the existing remote COMPLETE, JOURNAL, STATE_SLOT_0, SELECTED, and FABRICATED_TENSORS descriptors. Confirm the actual journal slot and selected checkpoint metadata agree with the fixed paths. The fabricated family metadata descriptor is already bound to the failed baseline receipt.
2. Copy `ROOT_DIAGNOSTIC_RELEASE_TEMPLATE.json` to a separate unsealed root release. Fill those five descriptors, the final diagnostic MANIFEST SHA, authorization reference, and exactly one fresh output invocation. Set its execution flag only for this diagnostic.
3. Use GPU0 UUID `GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998`, current ordinary availability, and the existing root own-session/wait4 supervision. Retain physical process wall, peak RSS, exit/signal and logs; internal intervals are subsets. Preserve GPU1 and the failed run.
4. Set the release's original PYTHONPATH and OMP/MKL thread environment plus CUDA_VISIBLE_DEVICES and the common cuBLAS setting before startup. Invoke the entry with `--execute`, the separate root release, and its fresh output.

`SOURCE_BINDINGS.json` pins original packet manifests, API source bytes, runtime authority and failed scalar evidence. `DIAGNOSTIC_PLAN.json` fixes work and report semantics. `STDLIB_PREPARATION_CHECK.json` records local source/metadata verification only. `MANIFEST.json` and `SEAL.json` seal this packet.
