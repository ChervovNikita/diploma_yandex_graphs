# Calls, memory and unchanged failure handling

Counts below are for one completed initializer invocation, without interface qualification, warm fitting, graph acquisition, factor installation or continuation.

| Operation | Graph or random tangent | Common-only |
|---|---:|---:|
| VJP primal closure forwards | 1 | 1 |
| Pullbacks | 5: one TRAIN CE plus four bands | 1 |
| Graph sparse matrix products | 3 | 0 |
| JVP closure forwards | 0–4 | 0 |
| Existing trial forwards | Up to 48: six contrast trials plus six fallback trials, four per trial | Up to 24 |
| Extra same-alpha common diagnostic forwards | Up to 6 | 0 |
| Total closure forwards | Up to 59 | Up to 25 |

The support change itself adds no pullback, JVP or graph-product count relative to the remasked mode. Wider root cotangents can increase actual reverse work and memory. The full-output diagnostics retain JVP entries that the original source sliced to TRAIN. A successful first contrast trial ordinarily costs one VJP primal, four JVP primals, four candidate forwards and one common diagnostic forward: ten closure forwards. Zero-gradient/null/fallback paths can cost less. Counts do not measure elapsed time, rematerialization, kernels, communication or native model resources.

The extra common diagnostic call is counted before execution, including an exception. Existing four-candidate bundles are counted after completion, following the pinned source convention; a candidate closure exception propagates and may leave that incomplete bundle uncounted. A future driver must retain the exception and charge work rather than treating this successful-return counter as a complete failed-call receipt.

## Dense memory

The code never forms a full Jacobian or NTK. A model-dtype `N*C` output/error field uses `element_size*N*C` bytes; a double field uses `8*N*C`. The source keeps dense fields and intentionally does not claim streaming support:

- Base logits, residual and full graph residual: up to three fields.
- Four bands and four shared full-q cotangents: eight fields.
- Four stored JVP list entries and the four-field stacked JVP tensor: up to eight fields when the tangent is nonzero.
- Candidate logits: four fields per trial. The local common-logit cache can retain six further fields.
- Filter powers, output centering, stacks, double Gram/slope operands and finite signed changes add temporary allocations. The signed helper computes double full-output changes and q; product/Gram temporaries may also be full-sized.

These field counts are an allocation inventory, not a measured peak bound. Python lifetimes, autograd tape, model activations, sparse graph storage and allocator reuse matter. The VJP tape is released before fresh JVP/trial calls. Model/private-parameter buffers are additional. The full-node semantic contract does not imply affordable native memory, and the small private vector does not bound activation memory.

## Acceptance and failures

All pinned constants, TRAIN mean CE, task-gradient projection, double centering, common cap, radius, directions, Armijo bound, TRAIN pair thresholds, member/pooled TRAIN finite-CE conditions and `quality and useful` acceptance are unchanged. The original finite trial test inspects full logits and is retained. Failure to construct TRAIN-distinct tangents falls back to safeguarded common descent; failed common descent returns identical warm factors.

Off-TRAIN JVP nonfiniteness makes full JVP diagnostics unavailable while preserving the original TRAIN finiteness guard. Extra common-forward exceptions, signed diagnostic failures and nonfinite diagnostics are recorded without adding an acceptance gate. Closure exceptions in original candidate/VJP/JVP paths remain explicit exceptions. Positive signed contrast, broad output separation and graph-band algebra are construction evidence only.

A direction detectable only at unlabeled outputs can be screened out by the preserved TRAIN separation condition. That limitation belongs to this conservative source amendment. A wider acceptance rule would be a separate operation requiring review. A paired support study also requires the separately reviewed common-warm/shared-alpha integration; this packet cannot certify independently chosen alphas as matched.
