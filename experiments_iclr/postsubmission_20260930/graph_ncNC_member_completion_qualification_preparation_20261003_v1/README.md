# NCNC member completion: engineering preparation

3 October 2026. **Sealed source preparation only.** Root adopted the candidate for implementation qualification, without adopting a scientific method, a study freeze or a fit. No PyTorch/native model was imported, numerical test executed, environment installed, dataset/label/tensor or live outcome opened, or training launched in this preparation.

## Prepared operation

The two modes are `private` and `pooled_after_clamp`. Both have one shared node encoder and four native-shaped factorized NCNC decoder branches. Each branch computes its own depth-0 residual-neighbor scorer under `torch.no_grad`, then applies the existing native clamp. The private mode uses those weights with the same member's aggregate and decoder. The pooled mode averages the four **already clamped** weights and supplies that mean to each member's own aggregate/decoder. The serving method averages **raw logits** in both modes. No sigmoid/probability pool, router, uncertainty score, learned pt or auxiliary loss is added.

The public code was written as a portable PyTorch implementation of the attributed operations. Original author files are preserved only under Git-ignored `private_evidence/native`, with exact hashes and their original paths in `PRIVATE_SOURCE_PINS.json`. The pinned author tree had no LICENSE/COPYING path; an explicit reuse grant remains unresolved. This packet does not redistribute the author implementation as the public prototype or certify legal clearance. The private copies are reference evidence for root's independent correspondence review.

## Files

| File | Role |
| --- | --- |
| `graph_ops.py` | Complete unweighted symmetric TRAIN graph, record masking before symmetrization, CSR row expansion, exact common/exclusive-neighbor enumeration and differentiable weighted feature sums. |
| `prototype.py` | One-layer native-shaped GCN; native-shaped xlin/endpoint/CN/final maps; rank-one sharing and private LayerNorm; exact depth-1 recursion; private/post-clamp pooled twins; native-scaled loss helper. |
| `selected_state.py` | Source/recipe/mode/pool binding, model/Adam state, all module training flags, Python/NumPy/Torch/CUDA RNG, and synthetic selected-state restoration. No dataset accessor. |
| `native_reference.py` | Hash-verified private native source loader for later mandatory QA. Missing native dependencies are not installed or skipped. |
| `qualify.py` | Eight mandatory engineering checks. It has synthetic fixtures only, no benchmark loader, score selection or predictive success gate. |
| `source_check.py` | Standard-library hash/JSON/AST/compile check without importing model code or PyTorch. |
| `SEMANTIC_MAPPING.md`, `RESOURCE_EPOCH_PLAN.md` | Operation-by-operation provenance, declared composition changes and the prospective complete-collab resource qualification plan. |

## Exact boundaries and disclosed composition changes

The prepared native profile is collab's GCN with one layer, hidden64, input dropout .25, encoder dropout .1, encoder edge dropout .25, LayerNorm, dimension-conditional residual; decoder dropout .3, predictor adjacency dropout0, use_xlin/tailact/LayerNorm, depth1, fixed pt .1, alpha1.05, scale2.5, offset6, unsampled common/residual neighborhoods, and splitsize -1. The recorded author learning rates are .0082 for the encoder and .0037 for the decoder. These values are source evidence, **not newly selected hyperparameters**. Final experiment seeds, initial member factors, budgets and quality gates remain unselected. The `Recipe` permits native positive scorer splitsize values as an explicitly identified future variant; the initial native parity check covers -1 only.

The one shared learned encoder and rank-one tied decoder matrices are BatchEnsemble-style composition changes relative to independent native NCNC. Biases are shared, LayerNorm affine parameters and native beta are private. Native unused ptlin parameters remain in state and optimizer accounting. Private maps and nonlinear feature paths remain; this is not four scalar heads on a shared completed representation. Default unit factors are an engineering equality limit; a final random sign or other factor initialization is not adopted. Numerical identity is established by copying a native single member's exact values into the portable unit-factor reference, rather than claiming identical constructor RNG consumption.

Native ordering is retained **inside each member**: endpoint product before outer xlin; left and right recursive scoring; xijlin before xcnlin/final decoding. The recursive scorer receives the already outer-transformed node features, then applies xlin again, as the source does. Its entire forward stays under no_grad, including active training-mode dropout. Outer transformed features and their weighted sums retain gradients. Empty candidate queries do not silently bypass full native scorer work/RNG when splitsize=-1.

Both twins use the same cross-member schedule: all member outer-transform/recursive-score paths, then all member final decode paths. This standardizes a cross-member interleaving that is absent in a single native model. It keeps the number/order/shapes of dropout calls identical across the twins; it is not a claim of exact dropout pairing with a four-model independent launcher. The common/exclusive sets for the outer query are enumerated once because predictor adjacency dropout and sampling are both absent. Recursive candidate-query enumeration remains charged for every native member scorer call.

The portable primitive has no torch-sparse dependency. Its scatter accumulation order may differ numerically from the author sparse implementation. Native parity is therefore a required numerical gate, including train-mode RNG-reset logits and gradients. A passing CPU harness does not qualify GPU numerical behavior, complete-collab memory, sampler identities, dataset processing or predictive utility.

## Later root execution

The following is a **prepared command for root's independent engineering QA**, not a command executed here or a scientific fit authorization:

```text
python source_check.py --output /absolute/new/path/source_check.json
python qualify.py --output /absolute/new/path/engineering_qualification.json
```

Run from this package directory, in root's qualified PyTorch/NumPy environment. The mandatory native reference additionally needs its existing PyG/torch-sparse/torch-scatter dependencies. The package performs no installations. The harness verifies the complete sealed manifest, refuses to replace the output, reports every required check, and fails if native imports or any numerical assertion fail. Source fixes require a preserved successor package and new review; a failed test cannot be silently skipped.

Required checks are: exact record masking/enumeration versus a set oracle; identical-member evaluation equality; nonzero affine weight/decoder association and its identical-response reduction; post-clamp rather than pre-clamp pooling; simultaneous member permutation equivariance and a weight-only association counterexample; detached completion scores with surviving downstream feature/xlin gradients; private native unit-member eval/train outputs, input/all corresponding parameter gradients and candidate sets; selected model/Adam/RNG replay for both modes; and identical twin training dropout schedules with one shared encoder pass. The permutation checks apply in evaluation; train-mode global dropout order is not claimed permutation-equivariant without permuting randomness too.

The selected-state test performs one artificial numerical update only when root later runs the harness, to populate Adam moments and test exact replay. Its fixture has no benchmark labels, accuracy, selection sweep or scientific result. This preparation performed zero such updates. Engineering tolerances derive from the tested floating dtype's machine epsilon; they are not experiment success thresholds.

## Contribution boundary

NCNC already supplies the graph-specific completion operation. Independent NCNC predictors already retain private completion/decoder pairing. Rank-one decoder sharing is faithful BatchEnsemble composition. The proposed research contrast is **whether pooling that member association before decoding loses useful ranking information under this sharing**. Forward differences can exist, but have no favorable-sign or generalization guarantee. The whole member bank can be expressed as one deterministic grouped network. No new function-class principle, verified missing links, calibrated completion probabilities, new graph primitive, complete-method novelty, speedup, predictive gain or manuscript claim follows from this engineering package.

Root owns independent numerical qualification, data-only resource admission, any representative pilot and strong single/independent/BUDDY/recent-PENCIL comparisons. The active frozen studies and previous auditor packages remain unchanged.

Parent coordination during preparation supplied a bounded native15 status summary (all15 replays passed; no GNNM advantage over native methods). No underlying native15 artifact was opened, and it did not change the source operation, fixtures or resource plan. This preparation does not claim total outcome-summary blindness.
