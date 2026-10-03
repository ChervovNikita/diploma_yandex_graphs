# Hidden separation can change without a decision changing

This is an analytic counterexample and an executed implementation witness in the qualified NCNC decoder. It is not an accuracy experiment, evidence of a new method, or a general statement that embedding losses cannot help.

## Exact operation

The final NCNC decoder block contains a member-private affine LayerNorm, dropout, ReLU, and a final factorized linear map. Let its post-ReLU hidden vector be `h_m` and the final linear input factor be `r_m`. For any positive diagonal vector `c_m`, make three changes:

- Multiply that member's LayerNorm scale by `c_m`.
- Multiply its LayerNorm bias by `c_m`.
- Divide its final linear input factor by `c_m`.

LayerNorm's normalized input is unchanged. Its affine output is multiplied coordinatewise by `c_m`. Dropout with the same mask and ReLU commute with positive coordinatewise scaling. Consequently the final linear input product is unchanged: `(c_m * h_m) * (r_m / c_m) = h_m * r_m`. The final output factor, shared weight and bias stay fixed.

This argument applies to each call of the block, including recursive completion scoring. Its raw logits and clamped completion weights therefore remain unchanged, as do the final member logits and their serving average. It holds in exact arithmetic on every graph/query input supported by this architecture. The execution below is a floating-point implementation check, not the proof of universal invariance.

## Executed witness

The fixed, data-free CPU fixture used the sealed width64/M4 decoder, one nine-node graph and two predeclared edge-removal variants, six queries, and both private and pooled-after-clamp routing. No dataset, checkpoint, label or predictive metric was accessed. The original unit factors intentionally make the four functions identical. A fixed positive scaling emphasized a different16-coordinate block in each member, using only powers of two:32 and1/32.

On the native graph, mean squared hidden cosine changed from approximately1 to0.000005734. **All member logits were bitwise identical before and after the transformation.** The same statement held on both graph variants and under both routing modes. Sigmoid responses to the two graph variants were also bitwise identical. `WITNESS.json`, the exact remote fetch receipt and the execution command preserve the actual values, source identity and scope.

## Research consequence and limit

Hidden cosine separation can be achieved by a change of internal coordinates that creates no predictive complementarity. Therefore hidden separation, a smaller contrastive loss or larger embedding distances cannot be used as the successful endpoint of our exploration. Those are diagnostics; representative served prediction quality remains necessary.

Prediction-based responses are invariant to this transformation because the underlying functions are unchanged. Functional input-gradient methods such as FoRDE also already pursue function-level diversity. This witness does not establish novelty for our response objective, superiority over FoRDE, useful graph specialization, or a predictive benefit. It supports retaining a hidden-repulsion control alongside function-level and graph-response controls. Any method claim still requires competent baselines, the frozen representative comparison and heldout confirmation.

The proof is scoped to the identified private-affine/final-factor block. It does not assert that every intermediate layer, arbitrary backbone or heterogeneous architecture admits this same independent member transformation. Positive rescaling symmetry is an established property of suitable neural-network compositions; this project-specific witness is explanatory analysis, not a new rescaling theorem.
