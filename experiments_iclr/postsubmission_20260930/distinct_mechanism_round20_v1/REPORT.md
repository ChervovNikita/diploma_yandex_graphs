# Round20: curvature and residual-geometry closest priors

2 October 2026. Bounded literature and symbolic interpretation only. Current study outcomes, scientific arrays, labels and checkpoints were not opened. Round19, the running study, earlier failures, original scores and literature dispositions remain unchanged. No method, experiment grid or execution packet is promoted.

## Conclusion

Two newly scoped primary papers materially strengthen the ancestry: **Splitting Steepest Descent** already derives balanced-copy motion from residual-weighted curvature, and **BGNN** already transfers a graph-dependent prediction-loss gradient into an additive ensemble. Cached **LoRA-GA** and functional repulsive ensembles further establish supervised gradient-based adapter initialization and output-cotangent-to-parameter projection. The existing initializer is an operational combination whose useful graph-alignment premise remains empirical. This search supplies no evidence for a new negative-curvature mechanism inside it.

The concrete useful question is: **do correctly aligned graph-error bands yield directions with useful residual-weighted logit curvature, or useful subsequent trajectories, beyond ordinary random diversification and topology-null filtering?** The current five-arm screen can test complete-operation utility. It cannot identify directional curvature signs because it does not measure them and accepts different step sizes. No new measurement or grid is proposed before that screen finishes.

## New primary evidence

### 1. Balanced copies and residual-weighted curvature are close prior

**Qiang Liu, Lemeng Wu and Dilin Wang, [Splitting Steepest Descent for Growing Neural Architectures](https://arxiv.org/html/1910.02366v2), arXiv:1910.02366v2, 28 October 2019.** The v1 source was read first; v2 was then inspected to check a sign inconsistency. The abstract-page submission history identifies a later v3 (4 November 2019), which was not inspected. Exact versioned source hashes, blocks, algorithm text and reading limits are saved in `EVIDENCE.json` and `INPUT_BINDINGS.json`.

Sections 2.1–2.2 replace one neuron \(\sigma(\theta,x)\) with a positive weighted average of copies. At identical parameters the old and expanded functions agree. Copy motion is decomposed into common displacement \(\mu\) and centered displacements \(\delta_i\), with \(\sum_i w_i\delta_i=0\). Eq. 3 defines the splitting matrix

\[
S_{\rm split}(\theta)=\mathbb E[\Phi'(\sigma(\theta,x))\nabla^2_\theta\sigma(\theta,x)].
\]

Eq. 4 separates ordinary common descent from the splitting term \(\epsilon^2\sum_iw_i\delta_i^\top S_{\rm split}\delta_i/2\). Theorem 2.3 and Algorithm 1 choose the minimum splitting eigenvalue and its eigenvector, split into two equal-weight copies with opposite perturbations, and allow no split when the criterion is not favorable. The algorithm alternates ordinary optimization to a point where parameter updates no longer improve the loss with curvature-selected growth. Appendix B.3 states that ReLU is replaced by Softplus for its curvature computation; this is a material smoothness/architecture choice.

This is close ancestry for Round19's pooled-curvature explanation. The loss is applied **after** averaging neuron outputs. For whole-logit copies with pooled CE, the analogous residual-curvature term is \(C_\phi=\sum_j s_j\nabla^2_\phi z_j\). The current continuation instead minimizes **mean member CE**, whose local contrast curvature is \(J^\top WJ+C_\phi\). It also introduces four fixed graph-filtered factor routes after a fixed warm schedule rather than selecting neuron growth from negative splitting eigenvalues at a local optimum. The prior's splitting matrix is not the graph adjacency/Laplacian matrix, and its negative-curvature criterion cannot be attributed to the current first-order filter construction.

**Source discrepancy retained:** both retrieved HTML versions print the loss change after the opposite eigenvector split as \(-\epsilon^2\lambda_{\min}/2<0\) under \(\lambda_{\min}<0\). That displayed sign is inconsistent with their own Eq. 4 and the chosen eigenvector. Substitution into Eq. 4 gives \(+\epsilon^2\lambda_{\min}/2<0\). The inconsistent passage is preserved verbatim; this report uses the latter algebraic substitution, not a silently corrected quotation. The PDF and native code were not inspected, and the HTML discrepancy is not asserted to occur in every publication format. This does not remove the clear balanced-copy/curvature-selection ancestry.

### 2. Graph-dependent gradient residuals already train an ensemble

**Sergei Ivanov and Liudmila Prokhorenkova, [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://arxiv.org/html/2101.08543v1), arXiv:2101.08543v1, 21 January 2021.** Scoped primary reading covers the graph/boosting definitions, Section 3 and Algorithm 1. Native code and figures were not inspected.

BGNN first builds a GBDT on TRAIN targets, passes its predictions as graph-node features, and optimizes both the GNN weights and those features. New trees fit the difference between optimized features and original GBDT predictions. Section 3 explicitly shows that with one inner gradient step this target is

\[
-\eta\,\partial_{X'}L(g_\theta(G,X'),Y).
\]

Thus a graph-dependent supervised error signal already determines new additive ensemble components. The mechanism is input-feature-gradient fitting by nondifferentiable trees with repeated GNN/tree co-training; it is not an ensemble of copied private-factor predictors or fixed Bernstein filtering of logit cotangents. No Hessian selection or zero-mean private motion is specified in the inspected method. This is a scoped method distinction, not an exhaustive absence claim. Paper performance/efficiency claims are not transferred to the study backbones.

## Cached evidence reused

| Saved primary/conclusion | Established ingredient | Consequence for the present premise |
|---|---|---|
| LoRA-GA, **arXiv:2407.05000v2**; cached Sections 3.2/3.4 reread, prior URL correction retained | Initial low-rank factors use SVD of the supervised weight gradient to approximate the first full-weight update. The frozen base is compensated by subtracting the initial adapter product, preserving the original effective weight. | Gradient-selected adapter coordinates and function-preserving nonzero initialization are prior. R15 uses diagonal private multipliers and graph-filtered cotangents, not this low-rank update-matching objective. |
| Repulsive ensembles, arXiv:2106.11642v1; cached Section 2.2/Eq. 5 passages | Output-space likelihood/repulsion cotangents are projected into parameters with the network Jacobian transpose. | Cotangent-to-private-parameter VJP is not by itself a missing principle. R15's graph filter, one-time schedule and safeguards are its specific operation. |
| StarSSE / PreGS / TabM | Warm-copy exploration, graph-supervised expert transfer, and efficient shared/private member training/init are prior. | Preserve the warm-copy and ordinary diversification explanations. |
| BernNet / C&S | The Bernstein operator bank/all-pass identity and graph diffusion of masked TRAIN residuals are prior. | Graph residual spectral filtering needs evidence of added utility, not attribution as a new primitive. |
| FoRDE / B³F-GNN | Continuous supervised input-gradient diversity and error-guided sequential local graph-module learning are prior. | Changing the derivative space/schedule can distinguish operations, but does not establish useful complementarity or novelty. B³F-GNN's saved timing/weighting ambiguities remain. |
| GraphMoRE / G-Adapter / HG-Adapter | Manifold-curvature expert initialization and graph-aware compact adapters are prior. | Negative manifold curvature is distinct from negative **loss-Hessian** curvature. Topology-aware adapter capacity alone is not the missing mechanism. |

MORGAN's scoped author-code conclusion and inaccessible full primary, and FAGEL's staged diversification ancestry plus unresolved chapter/prose-code issues, retain their earlier dispositions. No repeated access loop was run for them. The newly discovered HesGCN title/DOI is metadata-only here; its precise relationship to optimizer curvature was not resolved. Broad discovery results and uninspected leads supply no originality or absence evidence.

## What graph filtering actually leaves unexplained

In the homogeneous notation, let \(q_b=M_T H_b s\), \(g=J^\top s\), and \(K=JJ^\top\). Ignoring only numerical reprojection, the capped direction has first-order function motion

\[
Ja_b=\lambda\left[Kq_b-\frac{s^\top Kq_b}{s^\top Ks}Ks\right].
\]

The graph filter bank chooses node-axis cotangents, but the private-factor Jacobian kernel \(K\) transforms them. No commutation of \(K\) with the graph operator is established. Therefore a graph-frequency label on the residual does not imply that the resulting predictor is restricted to that band, or that parameter-norm matching matches prediction spread. The operation contains neither a kernel inverse/whitening criterion nor a negative loss-curvature criterion.

There is also a direct limit to any sign guarantee based on these inputs. Two smooth logit maps can share the same warm logits, Jacobian, residual and topology, while having opposite second derivatives: locally take

\[
z_{\pm}(\phi)=z_0+J\Delta\phi\ \pm\ \tfrac12v(u^\top\Delta\phi)^2,
\quad s^\top v\ne0.
\]

Their first-order graph cotangents/VJPs choose identical directions, while their residual-curvature matrices differ by sign, \(C_\phi=\pm(s^\top v)uu^\top\). A finite TRAIN Armijo check can safeguard descent dominated by the common first-order term; it does not turn that direction construction into a negative-curvature finder. This is ordinary local calculus, not a new proposed method or a claim about the measured study Hessians.

The actual gap is consequently **a missing justification for useful graph-to-optimization alignment**, not a demonstrated missing component that should now be added. If current controls support the operation, one may later ask whether its benefit comes from curvature, functional perturbation magnitude, accepted-step differences or later optimization. If they do not support it, curvature language should not supply a favorable explanation in place of evidence.

## Reading and retrieval limits

The search used seven bounded OpenAlex discovery queries, then inspected two distinct new primary papers through versioned arXiv HTML. Splitting v2 was a targeted version follow-up to v1's sign issue; its v3 remains uninspected. BGNN v1 is a version-bound scoped read; its later v2 remains uninspected. Saved cached sources were checked first and reused.

A mistaken presumed LoRA-GA identifier, **2406.08480**, was requested before the local search returned the correct cached **2407.05000v2**. The retrieved title is an unrelated group-theory paper. Both mistaken response bytes remain in the log with an excluded-identity disposition; no method conclusion or new-primary-read credit comes from them. Discovery rankings were often broad and irrelevant. The search is bounded and nonexhaustive; no missing-access or missing-result argument is used for novelty.

`EVIDENCE.json` retains exact inspected passages and read scopes; `INPUT_BINDINGS.json` binds their sources and the cached conclusions. All work was public text retrieval, standard-library parsing/hashing and symbolic reasoning. No scientific code was imported or executed, no SSH/GPU work occurred, and no study/source/selector/manuscript was changed.
