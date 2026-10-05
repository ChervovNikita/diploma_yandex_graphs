# One full-network float64 dense oracle: source and decision plan

**Source prepared; no import, model construction, numerical execution or host action occurred for this packet. No fit is admitted.**

The substantive next gate is one sequential recursive-native versus independent-dense **analytic** comparison. `qualify_full_network_float64.py` uses the authenticated full Citeseer TRAIN/features, unchanged donor/47-name partition, seed/factor seed `20261005`, native four-member network and the existing private SGD step. `JOB_TEMPLATE.json` is disabled. The previously sealed native failure (`22de2538…`) and repaired-kernel/FP32-FD result (`251b8638…`) remain unchanged.

## Exactly what is compared

The native FP32 initialization and full input values are generated/loaded exactly as before, then cast to float64; no new float64 initialization or feature reordering occurs. Parameters, features, buffers and constant support values use float64. Support values are explicitly double so encoder normalization also has the declared precision. The unchanged CN utility creates default-valued supports; both diagnostic aliases cast those same exact constant values to float64 at the operator boundary.

The recursive branch uses the unchanged wrapper and saved native `spmm_add`. The independent branch assembles each dense K directly from COO indices/values and evaluates `K @ Z` with ordinary PyTorch autograd. Sparse values must be constant and supports unique/coalesced. Donor code, graph population, loss and query shapes remain unchanged. The only alias replacement is the donor module's local `native.spmm_add` inside this diagnostic, restored in `finally`; no package/global patch is made.

Both branches replay identical CPU RNG and start from identical θ,φ values. Each independently evaluates the complete live virtual private map `φ'=φ−4×0.001∂φ L_inner` and the mean/own outer objective. Compare:

- Full inner route logits and all 47 inner first-gradient tensors at identical initial values/RNG.
- Full outer logits and all 47 outer first-gradient tensors at an **identical** point: the recursive-native branch's detached virtual private-step values.
- All private fast-state/cotangent coordinates and outer live logits from each independent evaluation of the same SGD map; tiny arithmetic differences are recorded and gated.
- **Every one of the 1,409,026 shared mixed-pullback coordinates**, using the same fixed native-branch outer private cotangent in both branches.
- **Every one of the 1,409,026 shared meta-gradient coordinates**, differentiating each branch's full live private step and outer objective.

No qualifying live private map is detached. Detachment is confined to comparison outputs, the fixed verification cotangent and the independent outer first-gradient parity point. Missing/unused/nonfinite derivatives fail; no coordinates are omitted or filled with zeros. There are no finite differences in this oracle.

The predeclared symmetric coordinate limit for every comparison is `1e-10 + 1e-7 × max(abs(native), abs(dense))`. Every coordinate must satisfy it. Log only counts, maximum errors and error/limit ratios, never model weights or logits. Tolerances are fixed before any numerical outputs and will not be adjusted afterward.

## Constant-K adjoint argument

For fixed sparse K, define `F_K(Z)=KZ`. For any perturbation ΔZ, `DF_K[ΔZ]=KΔZ`. Under the ordinary tensor inner product,

\[
\langle q,K\Delta Z\rangle=\langle K^Tq,\Delta Z\rangle,
\]

so the dense-input pullback is exactly `Kᵀq`. K's structure and values receive no derivatives. The wrapper forward calls the saved native `F_K`; its backward calls the **same recorded wrapper** for `F_(Kᵀ)(q)`. Hence differentiating the pullback with respect to q records `K`, instead of cutting that dependency at a raw kernel. Repeated transposition supplies the higher-order cotangent path; the operator itself is linear and has zero Hessian, while a composed nonlinear loss correctly gives, for example, `H_(||KZ||²/2)d=KᵀKd`.

The prior FP32 execution already verified invariant full-network forward/47 first gradients and zero-error quadratic HVPs on four active supports. The dense oracle tests the complete nonlinear mixed/meta composition independently of unstable finite-difference steps. This is implementation verification, not method novelty or predictive evidence.

## Bound and precision limits

Run the recursive branch to completion and retain detached comparison tensors only; release its higher-order graph before the dense branch. The source has **six full bundle forwards** total: each branch's inner/live outer plus a separate outer first-gradient parity call. No shared update, private recomputation at θ⁺ or serving call is made. Require two CPU threads, one inter-op thread, a 300-second soft check and external 360-second hard timeout, one attempt with no fallback/retry.

Five cached float64 constant matrices occupy at most 197,570,568 bytes at the declared full shapes (88,551,432 for the encoder graph; 27,254,784 per CN matrix). A dense bundle forward entails about 9.81 billion multiply-accumulate terms; higher-order backward adds work. The prior FP32 repaired diagnostic took 14.65 seconds and 3.03 GB RSS. Float64 activation/derivative storage and dense compute are unmeasured; no time/memory guarantee is inferred from that FP32 result. The actual diagnostic must preserve timing, peak RSS and timeout evidence.

A pass establishes float64 full-network derivative equivalence on this fixed fixture. It **does not pass the previous FP32 FD gates**, change inference precision/operator, admit an Adam recipe or justify predictive training by itself. The dense/float64 branch is a verification oracle and never a serving replacement.

## Minimum extra evidence for an FP32 decision

To justify the FP32 implementation despite unstable FD, combine the exact constant-K argument and existing FP32 forward/first-gradient/HVP evidence with a **prospectively declared direct FP32 derivative comparison** or explicitly matched-mask cross-precision comparison. It must inspect the complete shared mixed/meta vectors and identify arithmetic/ReLU-branch discrepancies, rather than infer correctness from noisy FD. A float64 pass alone is not a numerical FP32 error bound.

Then verify the previously unreached practical step: the discarded FP32 shared update, private recomputation at θ⁺ with exact replay, and actual committed-copy mean-raw-logit parity using the original native serving forward. Keep the historical FD result failed/inconclusive; record any new direct-verification acceptance criterion separately before execution. This is the small decision package needed before representative method training, without an expanded FD campaign. The current packet authorizes none of those later executions or scientific fits.
