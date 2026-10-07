# Cardinality-channel witness through the exact native Wiki update

An algebraic existence witness only. No model, numerical fixture, labels, graph payload, forward, training or score was executed. It does not identify the cause of the actual common errors or guarantee that optimization learns these parameters.

## Restrictions and current-model invariance

Use the saved pinned Polynormer source with hidden dimension D≥3 (the source recipe uses512), dropout off, no edge features, identical input features at all nodes and every effective incoming neighborhood nonempty. Per-member factors are global node-independent parameters. Under the standard normalized GAT operation, all transformed neighbor values and attention logits are identical within a member, so the neighborhood output is the same vector regardless of degree. Each local root/gating/LayerNorm operation is node-identical. The global normalized attention on node-identical states is node-identical too. Induction therefore gives degree-independent final logits within each member, for arbitrary current weights/factors. This does not concern TRAIN-time independent node dropout or actual nonconstant Wiki features.

## Explicit first-layer parameter choice

Let identical post-stem states be e0. Choose shared/fast linear maps so the normalized local neighbor output is N=e1, the root map is R=e2 and `h_lins[0]=0`. Set local beta logits to zero, so beta=1/2. Set local LayerNorm affine bias to zero. These are admissible shared parameter choices; all fast factors may be one.

Insert the proposed channel at this first aggregation only:

`N'=N+a S(d)N`, with `t(d)=1+a S(d)>0`.

The source executes `x=ReLU(N'+R)`, then `x=(1−beta)LN(h*x)+beta*x`. Hence

`x1(d)=(1/2)[t(d)e1+e2]`.

LayerNorm sees zero in the first summand and its affine bias is zero. The beta residual carries the unnormalized, non-collinear state exactly. This is why a pure-amplitude argument would have been insufficient.

For each later local layer, choose zero neighbor maps, root map2I, h map0, beta1/2 and LayerNorm bias0. ReLU is the identity on this positive state and the same formula returns x1. Thus native `x_local=sum_l x_l` is a positive constant multiple of `t(d)e1+e2`.

## LayerNorm does not erase the degree ratio

Before the global stage, choose `body.ln` affine scale1/bias0. Let `u(d)=LN(x_local(d))`. The same centering and scale is applied to every coordinate, including its positive epsilon. For coordinate3, whose unnormalized value is zero,

`(u1−u3)/(u2−u3)=t(d)`.

The denominator is nonzero. Unequal S(d), with a≠0, therefore gives different u(d). A D=2 pure centered-direction example would not suffice; the native D=512 admits the third coordinate. Scalar magnitude suppression is not assumed away.

## Carry the distinction through both global attention layers

In each global layer choose key projection0/bias0, so q=k=sigmoid(0)=1/2 and the denominator is positive. Choose value projection0/bias0, so the normalized attention output is0. Set that layer's attention LayerNorm affine bias to the all-ones vector. Choose h projection I/bias0 and global beta logits0. The product `LN(attention)*(h+beta)` is therefore the incoming state plus a constant1/2.

The native global `lin_out` is shared across its layers. Choose its weight I and a sufficiently large constant bias C·1, so ReLU is positive at both layers. Dropout is off. With the source's two global layers, the result is `u(d)+(2C+1)1`, which retains the degree distinction.

Finally, for two unequal degrees dA,dB let `w=u(dB)−u(dA)`. A final class margin with weight w and the midpoint bias gives margins `−||w||²/2` and `+||w||²/2` on the two states. The common additive global constant can be included in the midpoint bias; since u is centered, w also has zero coordinate sum. This uses the source's ordinary private-factorized classifier with factors one and an admissible shared bias.

Thus the proposed cardinality observation can produce separated final class margins through the literal native root/ReLU/local-LN/residual/global-LN/global-attention sequence. Other parameters can suppress it, and a shared cardinality coefficient already has this basic capacity. Member-specific coefficients require an empirical advantage over that shared/backbone change, a same-access single and a same-access untied ensemble.

## Prior and limits

PNA `2004.05718v1` Section2.2 Eq6 supplies log-degree scaling; Section2.3 supplies combined aggregators/scalers. The witness above is a direct source-level specialization, not adoption or reproduction of PNA's injectivity theorem. It assumes the standard PyG normalized GAT semantics; no runtime/dependency equivalence was checked. Degree must count the effective message multiset after the source's graph preprocessing. Neither real data relevance, member competence nor pooled utility follows from this witness.
