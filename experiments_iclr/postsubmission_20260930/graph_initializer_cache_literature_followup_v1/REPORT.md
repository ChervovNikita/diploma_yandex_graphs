# Graph-conditioned initialization: what the nearest sources leave unresolved

## Decision

**No distinct new initializer is established by this review.** There is one useful unresolved attribution question: does the *orientation* selected by graph-filtered training errors improve later task decisions beyond a random perturbation with the same local prediction geometry and the same common descent step? A future control can answer that question more precisely than the current factor-norm control. It requires its own freeze; the active five-arm study is unchanged.

The current initializer could address the practical limitation that private factor randomness, warm graph transfer and a fixed graph cache do not ensure complementary task errors under shared dense weights. Its training-loss safeguards and nonzero logit checks address immediate failure, but they do not establish useful complementarity after training. This is an empirical utility question, not evidence of an unoccupied literature gap or a superiority claim.

This packet contains literature and static source inspection only. No dataset, labels, checkpoint, remote host, scientific runtime or GPU was accessed. The historical 18.77 paths are prospective metadata, not current runtime qualification.

## Index first; reuse the closest established ingredients

`literature_memory/index_v16` was consulted before acquisition. It has 81 conclusion records, 37 normalized paper identities and two software-documentation identities. Those counts are not full-paper reads. Targeted conclusion hashes were checked; exact inherited scopes remain in `REUSED_CONCLUSIONS.json`.

| Reused primary/source | Closest established ingredient and boundary |
|---|---|
| BatchEnsemble / TabM | Shared dense weights, private multiplicative factors, member losses and collective selection. First-factor random initialization already exists. Shared weights still require private learned transforms for every member. |
| PreGS | Warm supervised graph-head transfer into multiple experts. Transfer and graph supervision are established; transfer dimensions alone do not guarantee complementary predictions. |
| Correct & Smooth / BernNet | Graph-correlated supervised errors and Bernstein spectral bands are established. The current detached CE cotangent VJP into private factors is a particular composition of those ingredients. |
| BUDDY / ELPH | A deterministic structural/feature cache is reusable by four independent predictors as well as a factorized family. Shared-cache savings cannot be attributed to factor sharing. Shared sketches also share their approximation errors. |
| Link-MoE | Structure-conditioned expert combination, independently trained expert scores and global/uniform ensemble controls are direct prior. Its collab gate uses 80% of official validation for fitting; its scores cannot silently become a supervision-matched comparison. |
| PENCIL v4 | A recent query-structure competence reference, retained at §§3.1–3.2 only. Query-dependent trainable computation is not a fixed-cache drop-in; reported benchmark superiority is not transferred here. |

SIGN and GAMLP already appear in earlier project reports. They were not acquired again or counted as new papers. The prior cached-LP review already judged an edge-space residual VJP implementable but unsupported as a distinct learner; this review does not reopen it as a proposed LP method.

## Two new scoped method reads; zero full-paper reads

1. **Fort, Hu & Lakshminarayanan, [Deep Ensembles: A Loss Landscape Perspective, 1912.02757v2](https://arxiv.org/html/1912.02757v2).** Read §3 setup, §§4.1–4.3 perturbation/similarity methods, §5 controlled comparisons and Appendix B. Weight cosine distance and prediction disagreement are separate measurements. Random-direction, dropout and diagonal/low-rank Gaussian perturbations around a trained solution are explicit controls. §5 rejects samples using validation accuracy; that recipe is not adopted for the training-only initializer. These experiments concern vision architectures, not shared-factor graph models. Their results motivate the control question but do not prove its outcome on graphs. No author repository for this paper was verified.

2. **He, Lakshminarayanan & Teh, [Bayesian Deep Ensembles via the Neural Tangent Kernel, 2007.05864v2](https://arxiv.org/html/2007.05864v2).** Read §2, relevant §3 methods including Eq.10 and Algorithm 1, §§3.4–3.5, Appendices F–I at the exact scopes recorded in `READ_SCOPES.json`. The method adds a fixed function `delta(x)=J(theta0,x) theta*` to each independently initialized learner, with the auxiliary readout parameters zeroed. Jacobian-based ensemble construction is therefore prior. Its exact posterior statement assumes infinite-width NTK dynamics with squared loss; it does not apply to the finite shared-factor CE initializer. Classification is one-hot regression with validation temperature scaling and target-scale tuning. Appendix H already matches initial output second moments to task scale. A future Gram-matched control cannot be sold as inventing output-space scaling.

The second paper links official code. Commit **91487abc7f5192fe140dc4a3ed0d25aa97593857** (12 March 2021) is pinned. Source verifies the additive frozen JVP, readout zeroing and anchored regularization. The supplied model has one scalar readout; the documented notebook is a toy 1D regression example, with JAX 0.1.70, jaxlib 0.1.47 and Neural Tangents 0.2.0. This is methodological evidence, not a drop-in graph comparator or a currently qualified dependency set. The inspected `new_predict_fn` calls its fixed-primal JVP during forwards; it does not itself build the fixed-data JVP cache described as possible in Appendix G.

Public HTTP discovery was recorded. Google returned redirect shells; Bing returned no result or unrelated token matches. These searches do not establish absence of prior. Direct arXiv metadata resolved both papers to v2; attempted v3 URLs returned 404 and remain in the retrieval log. **Two genuinely new papers of the allowed three were read at scoped method level.** A third broad paper would not resolve the identified control issue, so acquisition stopped.

## The precise missing control

Static inspection of `initialize_four_routes` confirms that `random_tangent` matches the graph tangent's total **parameter** Frobenius norm after centering and projection. It does not match the Gram matrix of the class-centered logit JVPs. The finite-step test only requires their pairwise differences to exceed a threshold. Different Jacobian gains, per-route norms and accepted step lengths can therefore explain a complete-operation difference without demonstrating an advantage of graph-selected error orientation.

`graph_init_mechanism_analysis_root_v1/ANALYSIS_v2.md` already records first-order cancellation, the same-alpha caveat and the Hessian/Jensen bounds. Those are reused, not new theory. The new literature reinforces two attribution limits: parameter distance is not prediction distance, and frozen-Jacobian ensemble construction/output scaling already have direct prior.

## One falsifiable future control, using the existing PolyFormer boundary

Use a separate, small prospective pair on the retained PolyFormer-Mono/SquirrelFiltered implementation, its existing cache and its **512-dimensional** active private slice. Fix the warm schedule, graph, training pack, pooling, continuation, seeds and selection contract before fitting. Reuse the existing three screen seeds 17/29/43 only if they remain appropriate for that new freeze. No new dataset or NTKGP dependency is needed. This already bound small task is an attribution screen, with limited modern dataset coverage. The indexed GraphLand/RelBench v2 work supplies more current industrial/relational context, but this review does not establish a compatible private-slice interface or resources there. It cannot support an industrial or modern-scale utility claim.

Let the four graph tangents be rows of `T`, and let `U` contain four fixed-seed random rows with the existing zero-mean and `g`-orthogonal projection. Let `Z(T)` flatten the training-row class-centered JVP of each row. Set `G_T=Z(T)Z(T)^T/(|TRAIN| C)` and similarly `G_U`. With a fixed orthonormal basis `Q` for the three-dimensional route-contrast space, form `A=Q^T G_T Q`, `B=Q^T G_U Q`. When `B` is numerically positive definite under a predeclared tolerance, define

`U_match = Q A^(1/2) B^(-1/2) Q^T U`.

This ordinary three-dimensional whitening/coloring matches the full training centered-logit Gram matrix, hence all six linearized pair distances. It preserves route centering and gradient orthogonality. The control uses labels through `g` and the graph Gram target; it does not optimize an additional supervised objective or select random draws using validation. A common scalar shrinks **both** tangent sets if necessary to preserve the existing factor cap. The compared graph arm is consequently a separately declared paired variant, not a reinterpretation of the active arm.

Use one initial radius determined by the largest direction norm across both arms, and one shared alpha chosen by the bounded training-only line search. Require both arms' four member CEs and pooled CE to pass at that alpha. A fixed random draw that cannot match the Gram matrix, a rank failure or failure to find a common finite step is an unmatched pair, retained in the denominator; do not resample or change tolerance after seeing results. Report nonlinear endpoint Gram mismatch separately. First-order matching does not guarantee exact finite-logit matching.

**Falsifiable question:** at equal common descent displacement and matched source prediction spread, does graph-selected orientation lower later pooled validation NLL, with accuracy, member quality and correct held-out decision changes as secondary checks? Fix a practically relevant NLL difference and the analysis of paired failures before the future run. No benefit over this matched random control, or persistent inability to construct matched pairs, undermines the narrower orientation account. Three seeds supply an exploratory paired screen, not a definitive inference. A benefit would support that account only under the declared graph/backbone/slice/continuation; it would not establish novelty or universal usefulness. Training Gram equality does not equate Fisher curvature, held-out prediction geometry, decisions, parameter norms or later trajectories.

## Practical cost and 18.77 boundary

The pair needs four graph and four random JVPs in total, small 3×3 matrix roots, and streaming accumulation of a 4×4 Gram matrix. It avoids a full example-by-example NTK and a full parameter Jacobian. A shared six-attempt line search can still require up to **48 complete member forwards** for the pair; common fallback, if separately retained, adds its actual passes. Charge warm fitting, graph filtering, all VJPs/JVPs, matching, failed trials, continuation, cache construction/storage, every member inference and optimizer state. Report standalone and any amortized reuse costs separately. No speed or memory saving is measured here.

This is source-level feasibility, conditional on native AD and complete-model execution working on the eventual host. The historical repository `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning` and Python path are not current host, device or runtime evidence. Existing connectivity and device restrictions remain with root. No runner, experiment admission framework, new model arm or deployment machinery was built.

## Deliverables and accounting

`PAPER_CONCLUSIONS.json` stores the two new scoped conclusions; `REUSED_CONCLUSIONS.json` preserves inherited scopes; `READ_SCOPES.json` gives exact HTML IDs and source lines; retrieval logs preserve URLs, timestamps, byte counts, hashes and failures. `INPUT_BINDINGS.json` binds local evidence. The seal verifies files only and is not scientific qualification. No sealed predecessor packet was edited.
