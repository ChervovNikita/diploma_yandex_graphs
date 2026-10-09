# Independent assembler V3 legacy-binding delta review

**Ready for root adoption; no new blocking source defect found.** Reviewed assembler seal `833ec6a76c44ecb05309a8dfef426b6415e425e56471698e6d1bdce710f8d5e1`, manifest `58cb8f3b4c9556342b0ec449098676c1698e30954e93b69456aa5df6b4eb0dc4`, entry `5345a184074e09c74365da850b0bc49f22ccc5ccac1ceed1ef6ae799417542b7`.

`ASSEMBLE.py:54–59` always requires a phase-contained existing file and exact SHA256. It validates size when the original row contains `bytes`. A legacy path+sha256 pointer can therefore pass without an invented size; a missing/wrong hash still fails, and an incorrect recorded size still fails.

Only `bound()` has changed executable AST. Reversing that exact predicate change reconstructs V2 source byte for byte. `binding()` at lines48–51 still emits actual size and SHA256. The old10 EXIT call site at135 still consumes the original legacy pointer; newly generated terminal source receipts at141 use `binding(exit_path)` and contain complete new bindings. No old pointer or receipt is normalized or rewritten.

SOURCE_BINDINGS and every disabled template are byte-identical to V2. Full snapshot, child wait/exit, source/release/adoption, metadata consume, numerical, scientific, resource, fresh-scope and failure handling therefore remain unchanged. Reader V3 and observer V2 seals match the prior independent review exactly. Verified all32 payloads across these three packets, their manifest/seal/sidecar chains, the superseded assembler seal and exact reader dependencies. No source entry was run.

Root must bind the V3 entry/manifest in a fresh assembly admission while retaining the existing observation facts and every old failure. The prior fresh-sibling placement instruction remains in force. This review verifies source behavior; actual legacy EXIT records and prior failed attempts were not opened or independently rehashed.

No runtime, entry, consume gate, server, numerical provider, outcome, checkpoint or array access occurred. No existing source, seal or failure was modified. This compatibility repair introduces no scientific policy or execution authorization.
