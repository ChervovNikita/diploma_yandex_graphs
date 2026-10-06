# Independent V5 joint-factor oracle / V3 supervisor source and math review

**Verdict:** PASS_SOURCE for qualifier `fcf201ab3ba49f36949639a884bae0007819abb82f4a3c4c65a444a4c7eb9fbe` (55,070 bytes) and supervisor `d86205ab07b54af6b82b3f8465cbe9fdca8e778d586e3d0c524d6cd7a1335641` (22,392 bytes). No new source blocker. This reviewer proposed the mathematical reformulation but did not author either source. No execution or fit authority is granted.

## Exact preservation and independence

Both saved patches equal the complete predecessor-to-successor source diffs. All manifest entries, seals and declared source/text descriptors match. Every qualifier literal, original g/h/scalar U construction and fixed_t block is AST-identical to V4. The complete episode, original-phi recommit, FP32 comparator, runtime/backend/cold fixture, observers, restoration and main atomic-save definitions remain AST-identical. Original full scalar derivative remains returned; neither the joint result nor a two-branch sum replaces the candidate reference.

For x=(theta,phi), U=-eta<g(x),h(x)> has gradient J_g^T(-eta h)+J_h^T(-eta g). The added same-graph joint VJP lists g outputs then h outputs in original phi order and uses the correct detached analytical seeds. Constant/unused AD outputs are excluded solely by requires_grad=False, with no magnitude-based pruning; any nonzero constant factor remains in the opposite seed. Unused input derivatives receive exact shape/dtype/device-matched zero padding. Existing shared-branch nontriviality rejects an all-constant accepted path.

The new self-gate uses original compare(full,joint); candidate still uses original compare(candidate,full), at ATOL2e-6/RTOL2e-5. Each finite h-only/g-only omission must fail that original coordinate tolerance on an active shared/private coordinate, while original shared >1e-12 support remains. Detectability must actually pass; no numerical support is predicted. Oracle construction never invokes candidate credit/helpers or candidate factor trees.

## Diagnostics and attempted work

First-failing tensor/worst normalized coordinate, actual/expected/original threshold, scales and branch cancellation values are atomically saved before strict gates. Collection errors or gate failures cannot publish success. Current split residual has only diagnostic labels; historical14ebb stays FAILED_PRESERVED. Host-float branch_sum/condition values describe the detached FP32 branch entries; the actual GPU FP32 split residual remains separately recorded. Failure attempt totals survive into restoration.

| Stage | Native forwards | Native reverses | Grad APIs |
|---|---:|---:|---:|
|Complete episode|44|60|71|
|Fresh recommit|12|16|16|
|Candidate fixed credit|2|5|5|
|Isolated scalar/branches/joint|1|6|6|
|Total|59|87|98|

The isolated bill is2 private partials+3 original scalar/branch VJPs+1 joint VJP. Diagnostics add no native forward/autograd API, but their tensor reductions, transfers and serialization are real work inside whole-process caps.

## Owned supervisor and remaining boundary

V3 changes only qualifier pin/path and87/98 gates. Physical ownership/sole wait4/watchdog/TERM-KILL/bounded cleanup APIs are AST-identical to reviewedf65. Root review and worker scope are exactly source-bound; interpreter binary, GPU/argv/cwd and result execution-scope SHA join. Successful worker exit0, restoration flags, support and counts precede whole-child wall/kernel-RSS/CUDA cap closure. Worker final publication/exit is covered; supervisor own final terminal write remains outside its sample.

A full/joint match can coexist with failure of b55's separately rounded candidate against full. That remains a strict blocker if observed. No old failure is relabeled and no native/resource/H16 PASS is claimed. Root still needs one distinct prospectively frozen reviewed invocation and actual successful external closure. H16 V3 remains bound to old850/f65/59-86-97 and requires a separate reviewed caller successor plus fit admission before it could use V5. Original83967 restoration is a prospective separately reviewed fallback only; no alternate run is authorized here.

No subject source was authored/modified/imported/executed. Only stdlib AST/SHA/arithmetic and source/text metadata were read; no scientific payload, SSH, numerical retry, held scoring, native reference fit or utilityH16 occurred. Predecessors and failure packets remain preserved.
