# Independent engineering review of the v2 FoRDE graph Gram source

This is a source and implementation review, with no manuscript/conference verdict or requested disposition. The subject is `conditional_graph_response_dice_forde_source_recipes_20261003_v2`, requested manifest SHA256 `26f45903d29ab1f9988a3eab426e9923010a70168cebccf99307c09b2b78a3a7`. The requested manifest, all eleven payloads and all eighteen explicit source-binding records match their declared bytes and hashes.

The corrected author bandwidth, stop-gradients, target-inside-log order, positive repulsion sign and whole-logical-batch force scale are consistent with the pinned FoRDE author source. The proposed fixed-operator Gram contraction is mathematically correct for the saved node-local PolyFormer composition. No new algebraic defect was found in those formulas. The remaining concrete issues concern the absent integration, cache custody, numerical conditioning, warm symmetry and execution qualification described below. These are not numerical failures observed in a run.

No model/tensor libraries were imported, source modules executed, numerical oracles run, or data, outcomes or checkpoints read. Only local source/metadata and standard-library hashing, JSON and AST parsing were used. No remote commands, installations, predecessor edits, sudo, GENLINK or PDF compilation occurred. Inspection scopes and actual hashes are in `INSPECTED_HASHES.json`; hash-only verification of prior review files is explicitly distinguished from independent content inspection.

## Author force checked directly against saved code

The authority is the local `sources/forde_train_forde.py` bound to AaltoPML/FoRDE commit `e8f9d7418be43064ffdc247ba7e62a61149da542`, especially lines 278–318. The nested JAX vmaps at line 293 yield axes `[live i, reference j, target b]`. `jnp.median(stop_gradient(sqdist), 0)` at line 294 removes only the live-member axis, so the bandwidth is `[j,b]`, not a full-matrix median and not a live-row bandwidth. Prepared `core/forde_identity.py:96–107` sorts the detached distance tensor along axis 0 and averages the two middle values for even M. It retains self distances. This resolves the predecessor's full-MxM median defect.

The reference coefficient detach at prepared lines 81–82 and bandwidth detach at lines 102/116 implement the author first-argument derivative. The live coefficient q is produced with `create_graph=True` at line 146 and is not detached before normalization. The norm is the full-X norm, with additive squared-norm `1e-24`. This epsilon is supported by the explicit author README replication command; the saved author file default at line 68 is `1e-12`, so the README convention should continue to be identified as the chosen source setting.

For one complete logical batch, define

`D[i,j,b] = ||s[i,b] - stop(s[j,b])||²`,

`h[j,b] = stop(median_i D[i,j,b])/log(M) + 1e-12`,

`S[i] = sum_(j,b) exp(-D[i,j,b]/h[j,b])`.

The prepared expression is exactly

`R_sum = (1/B) sum_i (log S[i] - log B)`.

Thus it is `log(sum_j mean_b kernel)`, with target averaging inside the log. Flattening axes j,b preserves that order. It includes self terms in the denominator and retains the author's extra `1/B`. CE is a sum of mean-per-member CE, with no extra `1/M`. With references/h fixed,

`d R_sum/d s[i,b] = -2/(B*S[i]) sum_j exp(-D/h) * (s[i,b]-s_ref[j,b])/h[j,b]`.

Gradient descent therefore gives repulsion. Self terms contribute kernel one and zero distance derivative at the base point. An additional candidate diversity coefficient, full-matrix median, symmetric second-argument derivative, targetwise average of logs or removal of self would change this source force. None appears in the inspected primitive.

Ordinary Torch SGD with fresh momentum, momentum 0.9, Nesterov, zero dampening and coupled decay matches the pinned optimizer recurrence: the initial velocity is zero, then `v=0.9*v+g` and displacement is `-lr*(g+0.9*v)`. Source decay is added once outside the author optimizer; its internal decay is zero. Applying this only to existing R/S factors remains a restricted graph adaptation, because the author trains all network parameters.

The author source uses all-gather at line 307, a local `batch_size` denominator at line 297, local cotangent slicing at line 309 and parameter pmean at line 314. The prepared helper should be treated as one complete logical-batch objective until a distributed path is separately qualified. Its expression is directly source-consistent for a single device. Blindly reusing it independently on shards and averaging shard losses changes both the kernel mixture and potentially the absolute force.

## Gram identity and locality checked against actual code

The saved mono token construction at `native_source/tokens.py:107–111` is `T_k=P^k X` with fixed P. For target v and token derivative `q[b,k,f]`, node locality implies

`g_X[b,n,f] = sum_k row_v(P^k)[n] * q[b,k,f]`.

Consequently,

`<g_X_i[b], g_X_j[b]> = sum_(k,l,f) G[b,k,l] q_i[b,k,f] q_j[b,l,f]`,

where `G[b,k,l]=<row_v(P^k),row_v(P^l)>`.

The Gram has no feature dimension because the fixed mono operator acts on nodes identically for every original feature. Cross-order terms k,l are necessary and present. The prepared quadratic form and normalization use this full-X metric; Euclidean token normalization would be a different comparator.

At `core/forde_identity.py:42–48`, current starts as target basis columns E. Repeated multiplication by `P.transpose(0,1)` yields `(P^T)^k E`; transposing produces the required selected rows of `P^k`, including k=0. This orientation is correct for nonsymmetric P. The routine does not mutate or explicitly coalesce P and does not differentiate P. Duplicate COO arithmetic and rounding behavior still need their declared runtime oracle.

Actual `native_polyformer.py:39–60` uses per-hidden-vector LayerNorm, per-order linear/MLP operations and attention across order/head dimensions inside each node. FFN at lines 89–110 is also per node. The outer source at `native_polyformer_outer.py:29–43` sums order, then applies per-node readout/ReLU/classifier. Eval disables all dropout. The BE affine wrapper at `adapter.py:105–109` is per row, and `forward_member` restores every active-member value in `finally` at lines 163–166. No saved operation mixes the node axis in this eval composition.

It follows that a summed true-class score VJP on distinct selected *token rows* yields separate q coefficients for every target. Overlap of graph receptive fields is handled afterwards by G. It would be incorrect to replace this with a single full-X VJP of summed target scores and interpret its result as per-target gradients. The prepared code avoids that error. Detached native token leaves are appropriate for a private-parameter force at fixed X/P: q retains its parameter graph, and fixed common affine maps are applied without detaching their inputs.

This locality claim is conditional on the exact audited composition. The author FoRDE ResNet runs `is_training=True` and has BatchNorm, so its summed raw-score input VJP is not generally an independently computed per-example Jacobian. The graph's eval row-local gradient construction is an explicit adaptation. It does not reproduce the whole author training program.

## Remaining defects, risks and concrete remedies

### F01 — Missing all-private source-style integration

The sole inspected adapter continuation helper (`adapter.py:171–182`) enables only `intermediate_sites`; its custody names at lines 208–216 have the same restriction. The proposed source-style JSON instead includes boundary lin1/lin2/lin3 R/S as well as intermediates. The objective relies on its caller for eval mode and ownership. Therefore the advertised wider partition is a prospective specification without an implemented, qualified entrypoint. This gap is already disclosed by the packet and should remain explicit.

Prepare a separate setup wrapper that clears gradients, freezes all parameters, enables only every existing `site_names` R/S, calls eval and emits exact trainable names/optimizer ownership. Qualify that every complete member trajectory uses the intended factors, that common maps stay frozen while input derivatives stay live, and that selection flags/modes are restored. Reusing `begin_private_continuation` supplies the narrower family; leaving a fresh adapter untouched supplies training mode and common trainable parameters.

### F02 — Unbound positional Gram input

`native_row_grams` returns only a tensor. `source_style_objective` has no ordered target IDs, operator fingerprint or token-bank identity. Its shape check cannot detect a same-shaped Gram cache from a different node ordering, P or polynomial token bank. Such a mismatch silently changes norms, distances, bandwidths and force. `requires_grad=False` does not make a Tensor immutable.

Implement an orchestration descriptor binding G to exact ordered target IDs, K/base, native COO representation/operator identity, feature/token-bank identity, dtype/device and precision receipt. Use those same IDs to gather tokens and labels; verify dependencies before each use, and invalidate G if any dependency changes. A synthetic swapped-order/operator witness should demonstrate rejection. The primitive's algebra is correct when the caller satisfies this custody contract; no actual mismatched cache was read.

### F03 — Unqualified float32 cancellation and mixed derivatives

Mathematical PSD of G does not guarantee nonnegative float32 `einsum` results. Correlated polynomial rows and signed q coefficients can nearly cancel in the represented full-X gradient while the individual contraction terms remain large. Identity P supplies a simple singular Gram case. Near such cancellation, the `1e-24` norm stabilizer makes normalized direction and mixed force highly sensitive to arithmetic error. Small/zero column medians near the `1e-12` bandwidth floor can further magnify differences. A negative quadratic deliberately stops the prepared routine; the rate and cost of that path are unknown.

O5 already asks for identity/nonsymmetric/duplicate COO, zero/small gradients and mixed-private checks. Make its cancellation and runtime precision coverage explicit: include dependent/near-dependent row powers with opposing coefficients, norm regimes around the stabilizer, duplicate directions, median ties/floor bandwidths and native ReLU cases away from kinks. Freeze tolerances first, then record direct-versus-Gram errors in gradients/norms, normalized distances, h, log-KDE and mixed-private force, including cancellation ratios. Bind autocast, float32 matmul/TF32 settings, sparse transpose semantics, versions and device; float32 Tensor dtype alone does not specify all multiplication arithmetic. A qualified direct pullback remains the reference if Gram cannot satisfy the adopted tolerance. No numerical failure is inferred from these risks.

### F04 — Warm symmetry remains unexcluded

The BE constructor initializes all member R/S to one (`adapter.py:100–101`). The author FoRDE uses separate initialization subkeys (`forde_train_forde.py:247–250`). If a deterministic shared warm procedure leaves every member identical, normalized input-gradient directions coincide. Their first-argument distance gradients are all zero, so FoRDE cannot leave exact symmetry through repulsion; identical CE and decay can preserve it. DICE's instance noise can generate asymmetric stochastic updates. This can handicap the FoRDE control even when the warm predictor has competent CE.

No warm checkpoint or values were read, so collapse is not an observed defect. Bind the prospective warm construction/member symmetry-breaking policy and copy the same completed bank into every arm. Before comparison, diagnose member equality, raw-gradient norm/direction collapse and mixed-force sensitivity at the intended private sites. If collapse is intentional, label its interpretation explicitly. A general FoRDE conclusion would not follow from a symmetry-collapsed restricted start.

### F05 — Logical B, chunks, devices and final partial batch

The current whole-batch expression has correct source scaling: algebraically duplicating targets leaves K unchanged and halves R_sum while mean CE stays unchanged. Independent objectives on smaller memory chunks do not preserve that force or the inside-log mixture. The predecessor explicitly required single-device qualification, whereas the new execution JSON does not state a global/local B device policy. The suggested two-pass cotangent method is unimplemented and needs its own direct oracle.

Restate single-device whole-logical-batch scope until another backend qualifies. Chunks must share the global logical-batch references/h/log-KDE cotangent while recomputing the live q path. A distributed implementation must separately derive and test gather/VJP/reduction scale. Freeze the last-batch policy and use its actual B. The final-partial-batch 1/B effect may be large; the author loader itself is not bound, so the original run's final-partial-batch emission is not established by the extracted batch-size line.

### F06 — Runtime, competence and resource gates are still open

AST/JSON/hash checks are source checks. No native all-private composition or warm receipt, row-locality/mixed-autograd result, DICE learning/state receipt, graph-qualified learner schedule/duration, checkpoint selector or resource profile exists in this packet. The prepared FoRDE objective has no full learner runner, optimizer setup or checkpoint integration. These omissions are already listed in the packet and are execution prerequisites.

Complete integration/custody first; then obtain the separately authorized numerical receipts and predeclared ordinary runtime, precise ownership, warm, schedule/duration and validation-selector contracts. Resource formulas are reasonable leading-order estimates, including the approximately doubled A residency from row-list plus stacked A. Qualify actual shape and full TRAIN cold/reused Gram preparation, cache custody, mixed-backward memory/time, and DICE detached partner/auxiliary cost. A fictional shape calculation is not a feasibility receipt. If Gram fails and direct pullback is the fallback, that fallback also needs a viable resource receipt.

## Interpretation of the controls

The proposed arms are frozen-common, all-private BE PolyFormer graph controls. All-affine R/S expands the predecessor's intermediate-only family, but shared weights, LayerNorm and shared attention bias scale remain fixed. Full source DICE/FoRDE networks are independently trainable and do not have this restriction.

DICE uses the paper-supported Appendix F.2 deterministic encoder plus unit noise CR ablation. It omits the full VCEB scale heads, backward embeddings and compression/sampled classification objective. No DICE author implementation was retrieved; the explicit graph finite-sampling/noise/state policy is an implementation choice, not an author-code reproduction. FoRDE uses the Identity/raw-logit author-force choice, omitting PCA/data-lengthscale choices. Column-specific bandwidth makes the author K generally asymmetric; that behavior should be retained as source code fidelity without a symmetric-PSD-kernel or posterior guarantee.

The primary `softmax(mean raw logits)` serving endpoint differs from author mean probabilities. The proposed same-checkpoint secondary `mean softmax(logits)` appropriately discloses this difference. The separate candidate-guarded continuation imposes additional partition/CE/feasibility restrictions and remains a separately named scientific control. Neither restricted protocol licenses a full-source superiority conclusion.

`REVIEW.json` gives structured evidence and closure actions for F01–F06. This review creates no scientific adoption or execution authority and changes no predecessor.
