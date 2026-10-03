# Independent NCNC v2 dropout source review

Target: `graph_ncNC_member_completion_qualification_preparation_20261003_v2`.
Manifest SHA256: `a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9`.

**Decision: no concrete source blocker found in the two dropout changes or their requested interactions.** This is a source review, not a numerical certificate or full-model CUDA parity result.

## Exact successor and native mapping

Independent stdlib hashing matched all 25 v2 payloads and all 25 immutable v1 payloads. The predecessor manifest matches `ac04bc7f6f6f37b86c50d38436aee111e987195c5f992816c4a5ef9d175887ae`. Direct source comparison confirms exactly two `inplace=True` additions in `prototype.py:75,120`. Other executable modules are unchanged. Input dropout remains out of place at line 63; probabilities, shape/call ordering, graph, sampler, loss, tolerance, capacity and recipe values are unchanged.

The additions match preserved native `model.py:162` for encoder post-normalization dropout and `model.py:547,549,553,555,559,563` for decoder MLP dropout. The native input dropout at line 131 remains out of place. Portable ReLU remains its existing out-of-place operation; this successor does not claim complete storage-alias equivalence with native code.

## Intermediate reuse and autograd

Encoder dropout mutates the fresh LayerNorm result, followed by a fresh ReLU result. It does not mutate the input-dropout tensor used by the dimension-conditional residual or the projected/aggregated tensors upstream (`prototype.py:62–76`).

Every dropout in the instantiated decoder layouts follows `FactorLinear` or `PrivateNorm` (`prototype.py:87–99,149–155`). `FactorLinear` forms a new affine result, including a final bias addition; `PrivateNorm` returns a new layer-normalized result. No supported layout applies the new mutation directly to the shared encoder output, an outer transformed feature bank, a leaf, a sigmoid result, or a retained ReLU output. The following ReLU is out of place. No definite saved-tensor/version conflict is apparent from these paths; composed backward remains for execution to verify.

Outer `hm = h + xlin_m(h)` is separately retained for differentiable feature aggregation. Recursive scoring remains under `no_grad` and computes another new `x + xlin_m(x)`, leaving the supplied `hm` intact (`prototype.py:168–185,197–215`). Thus the new dropout does not erase the intended outer feature-gradient path or mutate features used by other members. Positive and negative decoder calls continue to reuse the same encoder result without applying dropout to that result directly (`prototype.py:241–250`).

## Replay and twin schedules

Selected state retains model/Adam values, training flags, Python/NumPy/CPU/CUDA RNG, recipe, mode, code hashes and the exact prototype manifest. Restore checks both source bindings, restores model/optimizer/flags before restoring RNG, and does not preserve ephemeral activations (`selected_state.py:38–76`). The v2 code/manifest hashes automatically prevent accepting a v1-bound selected state. The existing replay test executes another forward/backward/Adam step for both modes at restored RNG (`qualify.py:224–255`); it must receive a fresh v2-bound numerical certificate.

Both twins execute the same changed dropout sites and member/candidate shapes. The mode switch still occurs only after all member scores and clamps, with no random operation in `route_weights`; final decoding follows the same member sequence (`prototype.py:135–140,187–221`). The disclosed standardized cross-member interleaving is unchanged. The existing twin test compares raw clamped weights and final CPU RNG, while the native identity test covers train/eval outputs, feature gradients and corresponding native parameter gradients (`qualify.py:156–208,258–279`). Source equivalence of call schedules does not itself certify CUDA masks or full-model parity.

## Saved probe and limits

The saved `PROBE.json` SHA256 independently matches `059356d94ddb988299461d915aabe6bc6f1b23abcdbb6f6d6246c21b20ad2643`; its bound probe source matches `8a9746b217411da63de318e6482620403773b8f2ffcd2029e79f054025c145d6`. Source inspection confirms isolated float32-ones, nonleaf activations and identical restored CUDA RNG per branch, without model/data imports.

At the three affected shape/probability cases, the saved receipt reports old-mask mismatch counts 2,716,862, 6,338,513 and 1,761,087 despite equal final RNG states. Matched functional inplace dropout reports exactly equal masks, outputs and input gradients to native `nn.Dropout` in all four cases, including the unchanged input control. This supports the specific dropout diagnosis on the recorded Torch 2.7.1/CUDA 12.6 A100 runtime. It does not identify the sole cause of the previous full-model failure, certify linear/normalization/sparse-aggregation backward, or establish current/full-model GPU parity.

Fresh root-run CPU qualification, current runtime/resource binding and complete native GPU parity at the unchanged tolerance remain the already declared admission steps. No additional source repair is requested by this review.

## Scope

Read complete portable prototype, graph primitives, native loader and selected-state helpers; qualification test bodies; selected preserved native encoder/decoder/recursion ranges; successor/plan/pin/mapping metadata; and the saved probe source plus structured receipt fields. Only stdlib hash/JSON/file metadata code was executed. No project/old verifier or numerical module was imported or executed, no tensor payload/dataset/label/checkpoint/training/quality artifact was accessed, and no remote command was run. All sealed inputs and old timestamps were preserved. `HASH_VERIFICATION.json` records the independent custody evidence.
