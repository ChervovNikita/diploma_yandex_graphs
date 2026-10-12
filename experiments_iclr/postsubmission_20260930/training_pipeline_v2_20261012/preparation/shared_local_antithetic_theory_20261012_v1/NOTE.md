# Shared local encoder and paired global kernels: scoped theoretical assessment

Status: conceptual assessment only. No fit, score, CUDA operation or new study
registration. This note does not change a frozen packet. It examines a possible
successor to native Polynormer, which is distinct from PolyFormer.

## Construction and exact scope

Let a common local encoder produce H. In one global layer, both routes use the
same Z=HW+b and value projection, but route + has q+=sigmoid(Z+delta) and
k+=sigmoid(Z-delta). Route - swaps the signs. Define the unnormalized positive
kernel K=q+ k+^T. With the same input H, K-=K^T and the two kernels have the
same diagonal. Thus the routes exchange who sends to whom without requiring
independent large Q/K weights. This statement applies to one layer with common
inputs; it is not an accuracy guarantee.

Native Polynormer normalizes each row. With Dr=diag(K1) and
Dc=diag(K^T1), A+=Dr^-1 K and A-=Dc^-1 K^T. Consequently,

    A- = Dc^-1 A+^T Dr.

In general A- is not A+^T. Even the normalized self weights can differ because
row and column sums differ. In the second global layer, route outputs differ;
the common-input premise no longer holds, so a layerwise transpose claim does
not extend automatically through the network. Reusing a common H at every
layer would be a different architecture and needs its own controls.

## What symmetric prediction pooling does and does not ensure

If all other parameters and stochastic masks are shared and every paired
perturbation flips sign, the pooled predictor [f(delta)+f(-delta)]/2 is even
in delta. If f is twice differentiable near zero, its first-order perturbation
cancels and the pooled departure from f(0) is O(||delta||^2). This only bounds
a local function change, not task risk. ReLU kinks can give O(||delta||), and
independent dropout prevents samplewise cancellation. Probability pooling and
logit pooling are different functions; the corresponding local claim must be
stated for the one actually implemented.

There is also an optimization obstruction: a symmetric mean of own-route
cross entropies has gradient zero with respect to delta at delta=0 when the
routes use identical stochastic masks. Zero-initialized delta remains inactive
under exact symmetry. Small nonzero initialization gives an O(delta) gradient
under smoothness and can still return toward zero. Antithetic initialization
alone is therefore not a complete mechanism for useful specialization.

## A possible task-linked activation mechanism

One testable alternative is fixed complementary TRAIN weights
w_i+=1+rho*u_i and w_i-=1-rho*u_i, where |u_i|<=1, 0<rho<1,
and the mean of u within each TRAIN class is zero. Use the complete scalar
weighted loss

    L = (1/(2N)) sum_i [w_i+ ell(f(delta)_i,y_i)
                         + w_i- ell(f(-delta)_i,y_i)].

At delta=0, with common stochastic masks, the shared-parameter gradient equals
the ordinary single-route CE gradient, while the private gradient is

    grad_delta L = (rho/N) sum_i u_i J_delta,i^T grad_f ell_i.

This is generally nonzero. Complementary weights can activate task-dependent
private directions without weakening shared learning at the exact starting
point. After delta moves, the shared gradient is no longer guaranteed to equal
the single-model gradient. This mechanism is an untested hypothesis. A useful
u must be fixed from TRAIN-only graph properties or a preregistered rule;
selection, calibration and heldout outcomes cannot determine it. The loss
must remain a single conservative objective rather than a custom gradient
claim. Comparable controls include independent weights without pairing,
paired kernels with ordinary CE, an ordinary shared-trunk/two-head model and
the matched original GNNM baseline on new data.

## Native transition and cost limitations

Native global prediction has its own initially untrained pred_global. Keeping
a competent common local encoder does not preserve competent local logits at
the local-to-global switch. Any residual local predictor or local-logit anchor
would be a new method component and must receive its own control and explicit
cost accounting. It cannot be attributed to the kernel identity.

A common local pass can reduce work only if route gradients are accumulated at
the shared H and the local encoder is backpropagated once. Calling each route's
backward with retain_graph=True still repeats local backward computation.
The current source-compatible Q/K adapter does two complete own-CE views per
member and does not implement this shared-local optimization. It should not
receive a compute-saving claim from this conceptual construction.

## Prior-work assessment

This note reused saved primary-source passages, not a new literature search:

- Wu, Goodman and Ermon, *Differentiable Antithetic Sampling for Variance
  Reduction in Stochastic Variational Inference*: title/abstract and sections
  2.1-2.2 in `antithetic_v2_blocks.json`. Their construction concerns
  marginal-preserving correlated samples and variance reduction in Monte
  Carlo gradients. Deterministic paired GNN routes are not automatically
  marginal-preserving Monte Carlo estimators, so their unbiasedness guarantee
  cannot be borrowed.
- Ren, Zhao and Ermon, *Adaptive Antithetic Sampling for Variance Reduction*,
  ICML 2019: saved pages 1-3 in `adaptive_antithetic_ren19b_pages.json`.
  Positive/negative pairing helps odd components but can harm even components;
  adaptation is problem dependent. This rules out a general accuracy or
  variance-reduction promise based on opposite route signs alone.

These were scoped rereads of two existing records; no additional full-paper
reading count is claimed. A shared trunk with several global heads and tied
Q/K weights is close to established multi-head ensembling. Antithetic
constraints alone do not establish methodological novelty. A credible
extension would need an explicit task-linked optimization mechanism, genuine
new-data paired comparisons, and a broader closest-prior search before a
novelty claim. Reinforcement learning is not justified by this derivation.

## Decision

Do not register an additional fit now. Complete the native Q/K source
qualification and retain this note as a potential successor hypothesis. No
current result supports saying that this construction improves accuracy or
deserves acceptance.
