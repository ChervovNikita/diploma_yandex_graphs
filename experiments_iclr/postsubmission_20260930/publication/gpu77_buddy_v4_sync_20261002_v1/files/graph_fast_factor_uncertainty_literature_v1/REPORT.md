# Graph-correlated uncertainty in a compact fast-factor model

2 October 2026. Bounded literature and author-source review. Three new primary methods were read in recorded scopes; **zero new full-paper reads**. The saved index was consulted first. No model implementation, scientific execution, data or checkpoint acquisition, SSH, GPU use, score recalculation or active-study change occurred.

## Decision

**No-go for promoting this direction as a distinct new learner, and no full-data pilot is recommended on the current evidence.** A frozen graph encoder with Bayesian output weights, distributions over rank-one factors, posthoc Laplace uncertainty over compact adapters, and graph-correlated Gaussian priors are established operations. Combining them by adding a graph-Laplacian quadratic to local factor precision gives a well-defined conditional approximation under explicit assumptions. The construction supplies neither a new uncertainty principle nor a selective-risk guarantee.

The finite-model gap is precise: no inspected source establishes that a graph prior confined to a supervised, frozen fast-factor Jacobian captures omitted shared-backbone uncertainty or improves held-out selective risk on a complete modern node-classification task at useful total cost. No candidate correction for those omissions was identified. This is an unresolved quality question, not evidence that the proposed combination is globally novel. An exact published BatchEnsemble-plus-this-quadratic formula was not established in this bounded review; absence from three readings is not an absence-of-prior result.

## Established sources and what they actually cover

| Source | Relevant operation | Scope and limit |
|---|---|---|
| Borovitskiy et al., [Matérn Gaussian Processes on Graphs](https://arxiv.org/html/2010.15538v4), v4, 15 May 2026; original AISTATS 2021 method | Graph covariance proportional to \((2\nu/\kappa^2 I+L)^{-\nu}\); small integer \(\nu\) gives a sparse Gaussian Markov precision. Graph Fourier features give a Bayesian linear model; output cross-covariances and categorical variational inference are described. | Sections 2–3, classification paragraphs and Appendix A recipe were inspected. The finite feature approximation explicitly risks variance starvation. Graph-only Cora uses 2,485 nodes, 5,069 edges, 140 training labels and 500 eigenpairs; categorical robust-max likelihood and SVGP, not a modern attributed-graph learner. v4 corrects mislabeled/scaled regression RMSE; it is not a newly introduced 2026 method. |
| Yang et al., [Bayesian Low-rank Adaptation for Large Language Models](https://arxiv.org/html/2308.13111v5), v5, 5 February 2024 | Frozen base, adapter-only Laplace approximation, low-rank KFAC curvature, linearized logit covariance and training-evidence prior selection. Bayesian prediction averages probabilities after softmax of sampled logits. | Sections 3–4 and selected setup/Appendix A/E/F passages were inspected. Compact adapter count alone does not make dense covariance affordable. Language tasks do not establish graph calibration or selective-risk behavior. |
| Zhang et al., [Graph Random Features for Scalable Gaussian Processes](https://arxiv.org/pdf/2509.03691v2), v2, 25 September 2025, preprint under review | Random-walk graph features, coherent graph-function prior samples, implicit feature operators, CG solves and pathwise posterior conditioning for Gaussian observations. | PDF pages 1–6, Section 4.4, reproducibility paragraph and Appendix C.7 were inspected. The \(O(N^{3/2})\) result depends on fixed walker/modulation/noise/solver assumptions. The paper explicitly defers a full scalable classification treatment and does not extend that guarantee to its classifier. |

The recent GRF classifier uses the same graph-only Cora largest connected component, an 80/20 train/test split, softmax SVGP, up to 1,000 Adam iterations and five seeds. Its reported setting uses 16,384 walkers per node and 22.17% nonzero kernel entries. The classification example therefore does not demonstrate a compact full-data industrial graph uncertainty service. Large-graph Bayesian-optimization results cannot supply that classification result. The PDF promises public code after double-blind review; this pass verified no official GRF implementation.

The following conclusions are reused from saved packets, with their original scopes and limitations retained in REUSED_CONCLUSIONS.json:

- **Rank-1 Bayesian Neural Networks, 2005.07186v2:** distributions on all-layer rank-one r/s factors, point-estimated common W, expected likelihood plus factor KL and W regularization, initialization/KL schedule, four mixture components and coherent samples. Occasional noise around deterministic factors does not reproduce that inference.
- **GVBLL, 2609.13655v1:** deterministic graph encoder, variational Bayesian final layer, frozen encoder online, diagonal Laplace/power-prior forgetting and anchor. Prediction precedes receipt of each batch's labels; updates use those later labels. MAP and MC metrics differ, and head-update complexity excludes encoding.
- **How to Train a Shallow Ensemble, 2602.15747v1:** common graph/atomistic representation with final-layer Laplace or rigidity committees. The saved source-access failures remain unresolved. Its energy/force task is not a node-classification uncertainty guarantee.
- **BLoB, 2406.11675v5:** frozen base and variational Bayesian low-rank adapters already supply compact posterior perturbations.
- **He et al., 2007.05864v2:** frozen Jacobian/JVP ensemble prior and output-scale matching are prior. Its infinite-width, squared-loss NTK posterior does not justify finite shared-factor cross-entropy posterior claims. It was not reacquired or counted as new.
- **Graph uncertainty collapse, 2605.22593v1:** the saved GCN/GATv2 uncertainty-collapse finding motivates a competent uncertainty comparison; it excludes graph transformers and retains functional convexity as a hypothesis.

## The conditional finite-model construction

Fix a shared backbone W, an undirected graph with a symmetric PSD Laplacian L, and a local point \(\phi_*\) for p private fast-factor parameters. Let \(\ell(\phi)\in\mathbb R^{NC}\) stack node logits, \(J=\partial\ell/\partial\phi|_{\phi_*}\), and \(\Pi=I_C-\mathbf1\mathbf1^\top/C\). Set \(B=I_N\otimes\Pi\). Centering removes the class-common logit shift that cannot affect softmax probabilities.

For a perturbation \(\delta\phi\), the local graph energy is

\[
\frac{\tau}{2}(J\delta\phi)^\top(L\otimes\Pi)(J\delta\phi)
=\frac{\tau}{2}\delta\phi^\top Q\,\delta\phi,\qquad
Q=J^\top(L\otimes\Pi)J\succeq0.
\]

With a proper base precision \(P_0\succ0\), PSD local negative-log-likelihood curvature H, and \(\tau\ge0\), one can define

\[
P_\tau=P_0+H+\tau Q,\qquad
\delta\phi\sim\mathcal N(0,P_\tau^{-1}),\qquad
K_\tau=BJP_\tau^{-1}J^\top B.
\]

This is a graph Gaussian energy restricted to a finite feature/Jacobian basis. It is not generally the full graph Matérn kernel. For the same fixed J, H and \(P_0\), the PSD added precision implies \(K_\tau\preceq K_0\). That matrix-order statement proves contraction of modeled perturbation covariance. It proves no monotonic improvement in softmax calibration, error ranking, selective risk or deployment utility.

Required interpretations and qualifications:

1. A Gaussian posterior is exact for the declared linear-Gaussian observation model. Categorical observations require Laplace/variational or another approximation. A Fisher/GGN approximation must be distinguished from the full parameter Hessian, especially for nonlinear, multiplicative factors.
2. A claimed Laplace center must be stationary for the corresponding posterior objective. Changing a precision at an arbitrary warm point is not automatically the Laplace posterior of a fitted graph model. A penalty on perturbations around a fitted point is a local choice; a penalty on actual nonlinear logits can also change the center.
3. Conditioning on supervised W, J, the feature basis and any selected prior hyperparameters gives a conditional plug-in approximation. It excludes uncertainty in those choices and in W. Freezing the backbone does not make that excluded uncertainty vanish.
4. The graph Laplacian has null modes. For the combinatorial Laplacian, componentwise constant centered-logit perturbations are unpenalized; normalized Laplacians have the corresponding degree-weighted null modes. Systematic component/class errors can remain. Heterophily or incorrect neighborhood relations can make smoothness shrink uncertainty in useful directions.
5. Four deterministic ensemble members are not posterior draws. Their empirical covariance has rank at most three in output-function space and cannot be substituted for a full factor posterior without an explicitly different approximation.

The algebra uses the **negative-log-posterior precision convention**. The Laplace-LoRA main printed sign/Fisher notation is not copied as an unqualified definition; its Appendix E precision-addition construction is the relevant consistent reference.

## Latent correlation does not deliver selective-risk control

A common sampled factor vector produces correlated latent logits across nodes. Under a declared categorical observation model, integrating that vector can also induce posterior predictive label dependence. Those are properties of the chosen model distribution. They do not establish the true dependence or calibration of graph-node errors.

For a selection set S fixed conditional on the fitted predictor and graph, let \(E_i\) be the error indicator and \(p_i\) its true conditional marginal error probability. Then

\[
\mathbb E[R_S]=|S|^{-1}\sum_{i\in S}p_i.
\]

Pairwise dependence does not alter this mean identity. It enters the uncertainty of the aggregate:

\[
\operatorname{Var}(R_S)=|S|^{-2}
\left[\sum_{i\in S}p_i(1-p_i)+2\sum_{i<j;\ i,j\in S}
\operatorname{Cov}(E_i,E_j)\right].
\]

A joint latent covariance is therefore relevant to a model-based variance or tail-risk claim, but replacing true error covariances by posterior logit covariance supplies no frequentist guarantee. A data-dependent selection set also needs its conditioning/selection rule stated. No read source establishes a graph-dependence-valid selective-risk bound for this finite frozen-factor construction.

Independent per-node Gaussian logit draws preserve individual marginals but destroy the graph-wide joint covariance. A joint sampling claim requires coherent graph-wide factor/function draws. For ordinary pointwise marginal probability prediction, separate marginal draws can be sufficient; they cannot then support a joint error-tail claim. Probability averaging is also different from softmax of averaged logits. Neither pooling operation repairs omitted W uncertainty.

## Author-source availability and real costs

The graph Matérn author repository was pinned at **781be70c8913b96c365475889f1f39629a5c9d0f** (17 November 2025). README lines 1–64 mark it deprecated and point to GeometricKernels. The inspected kernel scales spectral weights to an average-variance normalization and can multiply a point-feature kernel; its SVGP subclass/source was read only. No notebook, dataset or runtime was used.

The newer Laplace-LoRA author repository was pinned at **4e99db8df6860eaf0e4bc1a168fb3faa45a094f0** (22 June 2024). README lines 1–148, package metadata lines 1–96 and main.py lines 1–312 were read as text. Its variance routine accumulates a tensor shaped (batch, classes, classes), not cross-example/node blocks; a graph-wide joint sampler needs a separate design. Its README example averages sampled logits, whereas the paper specifies probability averaging. Default output helpers contain unresolved external configuration assumptions. These are reuse/audit limitations, not tested runtime failures.

For frozen final-layer features, common graph encoding can be amortized and feature/Jacobian work can be cached under a fixed recipe. Fast factors inside nonlinear graph layers or attention still alter member trajectories; a frozen W does not make those states shared or reusable after a nonlinear factor update. Linearized cached samples instead approximate those nonlinear members. Materializing J or a dense factor precision can be expensive; implicit solves, diagonal/block approximations and truncation change the model or its numerical approximation and need to be charged.

Any later utility claim must include upstream encoder acquisition/training, posterior-curvature or feature construction, precision approximation/solves, prior selection, coherent sample count, full-node output coverage, graph/feature loads, private nonlinear states and probability pooling. The retained code and papers do not qualify these costs on the current host.

## Why this stops without a pilot

The proposed operation is a combination of existing graph-prior and compact Bayesian approximation components. Its direct algebraic effect is modeled covariance contraction. The requested finite-model selective-risk benefit requires evidence or a specific correction beyond that contraction; none was found. A full-data run at this point would primarily tune the strengths and approximations of an established prior while leaving the claimed uncertainty mechanism unsupported.

Reopen only with a concrete failure of capable conditional-head/adapter uncertainty methods on the same complete task and a specific mechanism that addresses it—for example, a justified correction for omitted backbone uncertainty or a graph-dependence-valid risk procedure with its assumptions. A known-component empirical comparison could still be valuable under that separate aim. This packet supplies no admission, pilot or active-study amendment.

Discovery is bounded. The metadata-only title **Conformalized Gaussian processes for online uncertainty quantification over graphs, 2510.06181v1** is an unresolved lead for dependence/risk work, not assessed method evidence. The metadata-only graph-transformer GP-limit lead 2603.17569v1 is likewise not a finite-model posterior result in this review. The three-new-primary cap was exhausted; no claim that all graph risk theory or all combinations have been surveyed follows.

## Evidence and integrity

READ_SCOPES.json and INSPECTED_PASSAGES.json bind every scoped passage to saved primary/source bytes and exact zero-based HTML block indices, one-based PDF pages or source lines. INPUT_BINDINGS.json hashes the saved index and reused packet files. Retrieval logs preserve endpoints, times, statuses and body hashes, including the GRF HTML 404. DISCOVERY_DISPOSITIONS.json retains metadata-only leads separately. MANIFEST.json and SEAL.json seal the new packet; predecessor packets are unchanged.
