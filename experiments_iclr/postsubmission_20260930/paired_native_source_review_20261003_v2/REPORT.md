# Independent native-assay v2 delta review

**The earlier wrapped-allocator defect is fixed, but a newly confirmed P2 returned-failure path still requires an amendment. The complete resource-deferral contract is not endorsed.** This is a targeted source review; native execution and scientific quality remain unassessed. The v1 review is preserved.

Reviewed script SHA-256: `489ef0d0976ec7e58644afea17f807a4da3976fc35952448da9fc72aba19923e`. Manifest SHA-256: `07665a4d75613dcfc6c7f6b7582982d5a297de0a7e98150cefe549ab12d2bde8`.

## Closed: v1 B1, raised allocator exceptions

V2 adds a cycle-safe traversal of both `__cause__` and `__context__`, including suppressed contexts. It recognizes builtin `MemoryError` subclasses, `OutOfMemoryError` classes/subclasses, and case-insensitive legacy CUDA OOM text. A matched receipt identifies the reason, exception and chain link. The outer traceback and charged paired geometry report remain intact; nonresource exceptions retain `qualification_failed`.

Independent stdlib checks cover direct/subclass/wrapped/implicit/deep/shared/cyclic cases. Using the actual sealed helper's extracted exception class, mocked main receipts classify wrapped host and opaque device allocator errors as `resource_deferred`, while wrapped invalid geometry stays `qualification_failed`. Original fake source bytes and geometry receipts survive. This closes the specific v1 B1 defect for exceptions reaching the outer handler.

## Required: B2, allocator exceptions caught by the paired helper

The actual native observer at `prototype/qualify_squirrel17.py:251` records a failed closure's type/message and rethrows, without recording resource classification. The sealed paired helper catches every candidate-forward exception at `graph_full_node_cotangent_paired_alpha_v1/prototype/paired_shared_alpha_initializer.py:223`, stores a per-member string receipt, and continues its shared grid. It can return `None` plus `joint_failure` after those failures; a later successful trial can also return slices. Neither return reaches v2's outer exception handler.

V2 then assigns `joint_failure` at line 301 or proceeds into installation and `native_paired_source_qualified` at line 323. Thus allocator exhaustion during a candidate forward need not become `resource_deferred`. Direct allocator types remain visible in trial strings, but an opaque outer `RuntimeError` with an allocator cause loses that cause in the strings; rescanning returned strings is insufficient for the new chain semantics.

A targeted stdlib mock executes the exact v2 observer and exact sealed candidate try/except AST snippets. Both a direct mocked `MemoryError` and an opaque error with a `MemoryError` cause are caught without propagation, with one started/failed closure call and a `None` output. The classifier would recognize the live exception, but neither stored observer event nor candidate error retains that classification or chain. No tensor method or native operation is reached.

The observed closure should persist an allocator receipt/sentinel before rethrowing. After the helper returns, the assay must check it before any arm installation or success status, retain the paired attempt/counter/error records, and report `resource_deferred` if it fired. A separate sealed amendment should test resource failures followed by both joint failure and apparent later acceptance, alongside nonresource candidate failures. This P2 issue affects failure/status correctness; it does not assess scientific merit.

## Delta preservation and remaining note

`BOUND_INPUTS.json` and `RESOURCE_EVIDENCE.json` are byte identical to v1. There are still 151 descriptors and exactly seven numeric originals; runtime labels remain TRAIN only. All pre-existing helper functions have identical ASTs. After removing only the reviewed classifier replacement and normalizing the output schema string, the complete `main` AST is identical to v1. Therefore the previously reviewed loader/restore, native FP32 measurement, canonical output mapping, actual paired invocation, factor-only K4 installation, preservation and accounting paths have no normal-path semantic regression in this delta.

The unchanged topology receipt still saves seed/hash/equality rather than the permutation vector. The v1 nonblocking audit-retention note remains: saving the vector would directly fulfill the sealed paired caller's saved-permutation instruction.

Read-only manifest verification passed for eight v2 payload files, all 144 nonnumeric original mirrors, and the sealed v1 packet. **62 targeted review checks passed**, including reproductions of the outstanding defect; these are not a native qualification pass. The full normal-path fixture suite was not rerun. `REVIEW_CHECKS.json` records hashes, chain tests, main mocks and swallowed-candidate mocks; `CONCERNS.json` records B1 closure and B2. No numeric original, validation/final label bytes, Torch, model, dataset, checkpoint, logit, GPU, remote execution, source mutation, or new agent was used.
