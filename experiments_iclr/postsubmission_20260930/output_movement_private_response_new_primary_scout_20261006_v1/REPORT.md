# New scoped priors for output movement and transferable private correction

6 October 2026. **Three genuinely new bounded primary method scopes; no full-paper/code, prediction or novelty certification.** ACTIVE_SUPPLEMENTS_20261006_v5, index72 and saved literature custody were checked before selecting primary methods. Correct-and-Smooth was already scoped and was reused rather than reopened. The fixed pilot, sources and budgets remain unchanged. No model/Torch/native execution, scientific/data/result/checkpoint payload read, SSH, GPU work, fit or scoring occurred.

## Bounded search and custody

Nine OpenAlex searches returned59 ranked metadata rows, followed by one exact GEENI metadata lookup. Broad searches were noisy and were narrowed to exact method titles; this is not a field census. Exact IDs/titles for MetaFun, WarpGrad and ProMP had no hits in the checked index/supplements and selected saved literature/scout/prior custody. The first two additionally underwent a527-file bounded exact-ID/title screen. Scope and discovery limits are saved, not converted into novelty evidence.

Three version-specific arXiv HTML requests succeeded once each, with no primary refetch:1912.02738v1,1909.00025v1,1810.06784v1. Receipts preserve URLs/timestamps/status/body lengths/full-response SHA256; final artifacts retain only navigation and selected method text/equations. No full HTML/PDF mirror remains. Full headings and selected caption/prose exposures are disclosed. There were13 HTTP requests:10 discovery/metadata and3 primary. No HTTP retrieval failed. The block parser did not recover WarpGrad's algorithm bodies; that extraction limitation is preserved and no complete algorithm/code scope is claimed for WarpGrad. No failed historical DERG/GENNN route was repeated.

## 1. MetaFun: functional corrections and learned transfer already exist

**MetaFun: Meta-Learning with Iterative Functional Updates**, [arXiv1912.02738v1](https://arxiv.org/html/1912.02738v1). Read§§2–3 and AppendixA text/equations. Full-response SHA483a39175cb87cf6871e572a6ec338d62c6bc140b3f76d025cdc6ac1af3b9c0c. No empirical-result tables, author implementation or complete-paper audit.

Its starting algorithm performs RKHS functional descent:

    delta f(x) = -(alpha/|C|) sum_i k(x,x_i) loss'(f(x_i),y_i).

AppendixA derives this from bounded evaluation/reproducing-kernel inner products and the chain rule; for class logits the local CE correction is p-y. This explicitly propagates supervised correction from context to target locations using an input-similarity operator. The function values at context suffice to construct subsequent classical kernel updates and evaluate the final function elsewhere.

MetaFun then replaces the local derivative by a learned update u, the kernel by a learned deep kernel or attention, and the predictor by a decoder of an iteratively updated latent function r(x). Shared update/aggregation modules are trained on target loss after context-conditioned updates. Its class-specific positive/negative local update heads are motivated by the CE derivative. Thus a broad claim of *learning output/function corrections that transfer beyond support* is already occupied.

The learned attention need not be symmetric/PSD, and a learned u need not be an actual loss gradient. The classical RKHS derivation does not supply a descent or correction theorem for arbitrary MetaFun modules. RKHS/kernel choice itself supplies geometry; it is not a unique intrinsic metric, nor an exact proof of invariance to every neural parameter gauge. There is no native private R/S/B probe, item/member balanced allocation, original-phi commit or complete M4 probability pool in this read scope. Those are operational differences, not novelty clearance.

## 2. WarpGrad: shared geometry for task/private adaptation is direct ancestry

**Meta-Learning with Warped Gradient Descent**, [arXiv1909.00025v1](https://arxiv.org/html/1909.00025v1). Read complete main§2, AppendixA design prose and AppendixB training prose/captions; algorithm step bodies were not captured/read. Full-response SHA333c29974eb9270ebff16ad9662e0ac5d0bb0629990b9d410a51b9c1b770d0cf. No AppendixG natural-gradient experiments, empirical tables or author code audit.

Task-adaptable layers coexist with shared warp layers held fixed during task learning. Backpropagation through these warps changes task update directions. The shared warp parameters are meta-learned across sampled task states/trajectories to improve a post-update objective that may use different data or a different learning goal. Main Eq7 is an expectation/sum of one-step adapted task losses with respect to the shared geometry. Its live form needs second-order effects; Eq8 separately declares a stop-gradient approximation. This is concrete prior for *learning a shared body so private/task SGD updates transfer*, not just a name for natural gradients.

The geometric account assumes an explicit map Omega and nondegeneracy. G^-1=D Omega D Omega^T is SPD only under the required rank condition; nonlinear finite updates are related only to first order (Eq5, smooth Taylor setting). The general architecture discussion does not prove that every nonlinear warp admits the stated exact reparameterization. Our native factor redundancy has null gauge directions, so an unrestricted SPD inverse/coordinate-free finite-step guarantee cannot be imported from that nondegenerate account. Quote no unqualified convergence guarantee for the native nonsmooth/redundant model.

Neither metric learning, shared/private partitioning nor a one-step transfer objective is a distinct new principle. The current finite-response responsibility solver and original-phi recommit still differ operationally, but require empirical isolation from these known learning operations and matched first-order utility.

## 3. ProMP: adaptation with statistical-distance control is established; soft control is not a hard budget

**ProMP: Proximal Meta-Policy Search**, [arXiv1810.06784v1](https://arxiv.org/html/1810.06784v1). Read§3,§6 including complete Algorithm1 text, and AppendixC. Full-response SHA1f12e9c3cf01b343a365a69c46593123c1ac42e7668cc17eabcfd85fae3326b8. Sampling-credit§4, low-variance-curvature§5/AppendicesA–B proofs, author code and empirical tables were not audited.

The method adapts a pre-update policy per task, evaluates post-adaptation performance, and performs meta-updates using a clipped post-update objective plus KL penalty between the current and reference pre-update policies (Eq13). A likelihood-ratio surrogate accounts for changes in sampled pre-update actions. Algorithm1 refreshes pre/post trajectories when opening a new batch of several meta-gradient steps. This establishes an actual algorithm combining gradient adaptation, transfer and pre/post output-distribution control.

The KL term is a **soft penalty** and clipping is a surrogate; neither certifies a hard equal per-member predictive-displacement budget. AppendixC's RL improvement discussion assumes discounted infinite-horizon MDPs and controls changes in state visitation; it is not a theorem for supervised correlated graph episodes. Its descriptions of TRPO/PPO are secondary scopes inside this paper, not new primary TRPO/PPO reads. Policy action distributions/sampled trajectories differ from fixed graph-node classification inputs and complete ensemble serving.

Preserve one printed-formula caveat: AppendixC Eq107 scales F^-1 g using a denominator g^T F g. For a quadratic KL constraint and direction F^-1 g, algebra instead gives direction norm g^T F^-1 g. Therefore the displayed formula must not be copied as a verified exact budget implementation. No executable natural-gradient recipe is adopted from it, and ProMP's actual Eq13 soft/clipped objective remains the verified method operation.

## Graph common-error boundary: reuse, not another reading credit

Saved Correct-and-Smooth v2/code uses allowed-label residual Y-probability, topology diffusion and addition to a base predictor, followed by a separate smoothing step. It can change predictions through supervised graph correction; the unanimity impossibility for nonnegative pooling does not rule out that operation. Graph-correlated supervised residuals themselves are already prior. The saved v1/v2 prose/sign discrepancy stays disclosed; no old version or code was re-fetched.

Previously saved GAR/Meta-Weight-Net/Ren cover gradient utility and differentiated learner updates; MCL/Hellsemble cover supervised specialization; E2GNN/FAGEL and modern shared ensembles remain graph/sharing references at their existing scopes. The new methods add function-update/learned-geometry/KL-control obstacles to a broad contribution claim. None was ported or fitted.

**Efficient ensembles of graph neural networks (GEENI)**, DOI10.1145/3489517.3530416, was already an E2GNN bibliography locator. This scout confirms OpenAlex DOI/title/authors and an OA-flagged ACM PDF URL only; its primary method was not fetched/read, so overlap remains unresolved. An OA flag is not inspected full text. DERG/GENNN and other unresolved methods remain unresolved too.

## One falsifiable extension, with a clear rejection rule

The plausible narrow question is: **At matched current predictions and a predeclared actual private output-change budget, does member-specific learning response predict and cause useful correction on disjoint graph nodes beyond ordinary supervised correction using the same geometry?** The possible value is a reliable forecast of which private correction transfers, rather than functional descent, a learned metric, KL regularization or gauge covariance itself.

Freeze the S/R split and all budget/metric/context choices before labels/results are used; no R label enters probe/allocation. A future output/KL budget may use unlabeled context inputs, but both reference and response learners must use identical allowed information, geometry and actual movement accounting. Use the same complete mean-probability pool and own-CE anchor. The already disabled one-update canonical-gauge diagnostic is the cheaper rejection test; its RMS balancing does not itself match output/KL movement. No output-budget solver or scientific arm is authored here.

A member can agree on a wrong prediction yet have a different reachable supervised correction direction, especially because its private stem precedes graph propagation. If response detects a direction that transfers from S to R, it could allocate extra corrective training productively. However, equal reachable function responses center to zero; lost information cannot be recreated; and a support-only gain can harm R. A smaller KL/more stable optimizer can also explain apparent gains without useful allocation.

Reject the proposed allocation contribution if the same-geometry ordinary learner gets the same complete R-pool gain, if gauge artifacts explain the rankings, if response rankings do not predict transferable correction, or if improvement appears only on S/oracle members while pool quality/competence fails. Initially unanimous-wrong items are a prospectively defined mechanism description, not a subgroup that rescues the aggregate endpoint. A favorable tiny test would still require the saved competent ordinary, matched live first-order, capable single and untied/full-context comparison obligations plus untouched confirmation. It would not certify novelty.

**Accounting:** three new version-specific bounded primary method documents; zero full papers or author-code scopes; zero result-table/performance adoption. Discovery metadata and reused graph/scoped conclusions add no primary reading credit. Base index72 and active supplements are unchanged; no new cumulative full-paper total, training authority or desired reviewer verdict is claimed.
