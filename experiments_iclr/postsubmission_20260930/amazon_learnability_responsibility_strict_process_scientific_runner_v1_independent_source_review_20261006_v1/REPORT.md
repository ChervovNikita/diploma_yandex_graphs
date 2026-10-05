# Independent source review: strict scientific runner V1

**BLOCKED: one concrete admission-interface defect.** Reviewed run_scientific.py SHA189ceefe61ca06bbe81a9fd8c69ce81bd69524d2beade7bc03a90b28c29e849b, manifest d922587a86a1c0eb0bff93fe14adaecbbb4eb2255db7e466e80d28957a8022d4 and seal1071b668b73d330cc6ea8ef4c0672d813b224ca9172c5f4690e74a68bea50e05. The packet's hashes/sizes, AST, read-only modes and bound source metadata were verified. This review covers the new runner only; the reviewer's authored scientific V2 was not re-reviewed.

## Blocking finding

At line219, limits is assigned admission["scientific_resource_limits"]. Line220 then reads limits["external_watchdog_seconds"]. The exact released PREREQUISITES.json SHA3b1cc42aad205fd9783096bf13d6d5d33d660bfc9520d2c8af46c3e895199cc7 has no watchdog key in that nested dictionary: external_watchdog_seconds=14520 is at admission top level, while nested max_elapsed_seconds=14400. The runner therefore raises KeyError before its numerical setup or worker call with the actual admitted packet.

Use the existing top-level admission["external_watchdog_seconds"] in a sealed successor and correct the corresponding interface metadata. No cap, admission, mathematical source, worker or qualification outcome change is needed. V1 remains preserved and unexecuted.

## Other focused checks

No further blocker was identified in admission-before-Torch/W ordering, strict CUBLAS/Torch setup, matching qualified backend/runtime, intra1/interop1 fresh-child setup, the single unchanged released-worker run_six call, preserved5740-forward scientific recipe/closed A, primary failure and restoration handling, whole-process accounting or the root watchdog contract. The accepted certificate/scope metadata and exact released-worker hash were inspected without importing or executing prepared source. No SSH, numerical/data/checkpoint access or fit occurred. This finding supplies no execution/fit authority and does not alter any prior failure.
