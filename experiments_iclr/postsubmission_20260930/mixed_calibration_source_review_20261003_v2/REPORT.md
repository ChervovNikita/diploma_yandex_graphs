# V2 bounded source amendment review

This separate review covers the two qualification-comparator expressions, unchanged Python sources, preserved v1 failure, and v2 custody. It does not repeat the implementation review or execute the qualification. The v1 source review missed this comparator defect; the authorized runtime qualification exposed it. Both v1 review files remain byte-for-byte unchanged.

**Resolved harness finding H1 / P1.** The preserved v1 result records a qualification-only failure at `2026-10-03T04:33:52.208679+00:00`: an `OrderedDict` from model state skipped exact-`dict` dispatch, reached tensor-containing scalar equality, and raised `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. The failure occurred in the synthetic witness before real graph access; all eight resource cases were blocked, no validation/test scoring or scientific training began, and the result remained `unqualified`. This is a harness failure, with no scientific outcome inferred.

V2 `qualify_policy.py:30–31` changes exactly two expressions:

```diff
-    elif type(left) is dict:
-        c.require(type(right) is dict and left.keys() == right.keys(), 'Replay state keys differ')
+    elif isinstance(left, dict):
+        c.require(type(right) is type(left) and left.keys() == right.keys(), 'Replay state keys differ')
```

`OrderedDict` now enters recursive mapping comparison. Matching concrete container type and key checks remain required. Tensor comparisons, exact/nonfloating behavior, floating tolerances, sequence checks and recursion are unchanged. The entire v2 qualifier equals v1 after only these substitutions; the independently generated diff matches both declared diff records exactly.

The four other Python files (`block_policy.py`, `common.py`, `train_policies.py`, `verify_source.py`) are byte-identical to v1. Module bindings, native model/loss/gradient/fit/selector/OneCycle/RNG source records, fit custody and executable freeze are unchanged. The scientific authority remains `739160acd4ffb0380348f79496e88624db6163f4a3a93902b043d58eaacc18c5`. The previous implementation review and its stated limits therefore carry forward; the native suite was not repeated.

V2 manifest SHA256 is `ee2cc8623c8752e1fc5a56dde5968c624d2e9e9b735f9534672fae6549c592d4`. Its live seal, all payload/provenance hashes and lengths, and static receipt bind these exact bytes. Both release templates remain false. V1 manifest/seal and every v1 payload still verify. The original failure and its packet copy are byte-identical at SHA256 `5f2ab277dfbbf82b6d818288c5d62561f1aeac8ce0a9caf7adcdaf10db6e9fca` (4,997 bytes); the correction/provenance records bind that failure.

No remaining source defect was found in this narrowly scoped amendment. V2 numerical qualification and the existing full-graph resource/replay requirements remain unverified by this review. No new fixture, Torch import/install, real data/label read, fitted-state/scientific-outcome inspection, remote execution, Git operation, source edit, or manuscript acceptance decision was performed.
