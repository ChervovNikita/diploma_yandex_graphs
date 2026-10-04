# Closest-prior check for graph-conditioned head covariance initialization

## Finding

The current proposal has a narrow, testable distinction, but its main ingredients are established. In particular, a newly inspected primary paper already studies **a shared backbone with centered, curvature-aware last-layer ensemble initialization**. It also distinguishes the orientation of head covariance from its overall amount. Mean preservation, data-dependent initialization, and useful covariance orientation cannot be presented as new learning principles.

This does not establish equivalence to the complete graph selector. The remaining candidate is a **graph-conditioned restriction of the initialization search**, selected for the finite effect of the next native learning step. Its value remains untested. Better later predictive accuracy, rather than parameter savings or a lower TRAIN surrogate, is the relevant outcome.

## Exact candidate considered

This assessment binds `graph_curvature_selector_source_preparation_20261004_v3/PROTOCOL.json` and `ELIGIBILITY.md`; it does not independently verify their numerical claims.

At one common warm head center, four degree-three Bernstein filters act on TRAIN-injected classification errors. TRAIN-remasked output cotangents are pulled back into the private head-factor coordinates. After projection against the common head TRAIN gradient, an ordered rank-three basis defines three possible pairs. Each pair gives four antithetic offsets, `(+s vi, -s vi, +s vj, -s vj)`. All candidates preserve the initial mean raw logits, and the radius is matched to one finite mean-member versus pooled CE gap under a shared cap.

Each eligible initialization is evaluated after one actual, full-model coupled-Adam update on mean own-member TRAIN CE, with dropout off. The selector compares the resulting pooled TRAIN CE with the common trial. It discards all trial states and returns only the initialization, with the pretrial optimizer and continuation RNG. Native continuation still minimizes mean own-member CE. The deployed predictor is `softmax(mean raw member logits)`, and checkpoint choice remains pooled VALIDATION NLL.

All thirteen new numerical constants are null in the bound protocol, and numerical/source qualification is unfinished. This report adopts no constants, experiment outcomes, new runs, or gain claim.

## Newly inspected primary scope

**Schäfer, Kellner, Kästner and Ceriotti, _How to Train a Shallow Ensemble_, arXiv:2602.15747v1 (17 February 2026).** The retained version-pinned abstract and HTML both resolve successfully. `PRIMARY_PASSAGES.json` preserves exact source text, TeX, block indices and HTML IDs.

- II.3 describes DPOSE: an end-to-end shared-backbone ensemble with multiple output units in its readout, optimized by a Gaussian NLL using ensemble mean/error and variance. All parameters are jointly optimized.
- II.3 also describes LLPR: a Gaussian last-layer posterior whose covariance is a regularized inverse generalized Gauss–Newton Hessian. Samples form a shallow ensemble. **The sample mean is constrained to the MAP head exactly** (`S2.SS3.p14.1`).
- IV compares isotropic centered Gaussian initialization, subsampled trained heads, Laplace posterior samples, and shallow-ensemble pretraining; the following stage either fine-tunes just the head or the whole model.
- S10 isolates head covariance on a shared frozen backbone. The authors report that isotropic-head fine-tuning can increase variance without aligning it with useful posterior directions, while the Laplace initialization has an informative orientation (`S10.p2.1`–`S10.p4.1`). This is a reported finding within atomistic regression/UQ, not a classification-accuracy result or a reproduced experiment.

The paper targets energy and force uncertainty in machine-learning interatomic potentials, including derivative predictions. Its objective is Gaussian NLL, not the candidate's classification CE; it does not specify the candidate's Bernstein-error VJP construction, matched Jensen-CE-gap comparison, or discarded finite coupled-Adam selection in the inspected passages. Author code, the full proof/figure/table set, and all related work were not audited. An exact complete predecessor has not been established by this bounded inspection; originality has not been cleared.

## Retained ancestry

| Ingredient | Inspected prior and retained scope | What is still different in the candidate |
|---|---|---|
| Multiplicative shared-weight ensemble factors and own-member losses | BatchEnsemble; TabM, including first-factor randomization and identity later factors | A one-time supervised graph/head initialization search |
| Data/gradient-dependent compact initialization with preserved initial map | LoRA-GA, arXiv:2407.05000v2, Sections 3.2/3.4 | Diagonal private head factors and graph-error spans, rather than a low-rank update approximating the full-weight gradient |
| Graph spectral band operators | BernNet, arXiv:2106.10994v1 | The borrowed basis filters a supervised output cotangent for head initialization rather than replacing propagation with learned spectral filters |
| Supervised graph-error transport | Correct and Smooth, arXiv:2010.13993 | A parameter VJP and initialization choice rather than postprocessing predictions |
| Supervised graph expert transfer; topology-aware expert initialization | PreGS, arXiv:2609.26310v1; GraphMoRE, arXiv:2412.11085v1 | A common affine-head function with antithetic factors and a fixed native pool; no new expert gate or manifold claim |
| Graph-spectrum expert construction | MORGAN, DOI:10.1609/aaai.v40i28.39553, scoped author source | Full-primary access is unresolved. The inspected implementation filters features and fuses learned spectral experts; it does not clear whole-paper absence |
| Centered copy perturbations and curvature-sensitive splitting | Splitting Steepest Descent, arXiv:1910.02366v2 | The affine head has no instantaneous output-curvature term. The candidate evaluates a later own-CE update, not immediate neuron splitting; retained sign/version caveats stand |
| Mean/member loss interpolation and ensemble loss curvature | GNCL, correct arXiv:2011.02952v2 | Native mean own-member CE is retained; a new objective is not claimed |
| Initialization evaluated through a learning step | MAML, arXiv:1703.03400v1 | A discrete same-TRAIN three-candidate choice, not cross-task meta-learning or a learned meta-gradient |
| Weight perturbations that alter the next training gradient | SAM, arXiv:2010.01412v1 | Selection minimizes post-own-CE-step pooled loss over antithetic ensemble initializations; no SAM rule or guarantee is transferred |
| Covariance samples and warm ensemble branches | Deep Ensembles: A Loss Landscape Perspective, arXiv:1912.02757v2; StarSSE, arXiv:2303.03374v3 | Native graph continuation and transported warm Adam differ from validation-filtered samples or fresh-optimizer cyclic branches |
| Centered, curvature-aware head covariance followed by full-model continuation | How to Train a Shallow Ensemble, arXiv:2602.15747v1 | Finite matched Jensen gap and a graph-restricted one-step own-CE selector, rather than Gaussian posterior sampling/probabilistic regression training |

The exact CE identity `mean CE(z_m,y) = CE(mean z_m,y) + mean KL(p_geometric || p_m)` is background algebra already retained in this research. GNCL supplies general member/pool and loss-curvature ancestry; this report does **not** attribute that particular exact logit identity to an equation printed in GNCL. Equal mean logits and equal finite Jensen gap do not equalize the full response geometry or the next optimizer step. That difference is a hypothesis to test, not a predictive guarantee.

## Evidence needed for a useful contribution

1. **Separate graph restriction from selection.** Keep the prospective five arms: common-only, fixed first graph pair, selected graph pair, equally selected permuted-topology span, and equally selected random span. Match candidate counts, failures/nulls, initial mean logits, finite Jensen gap, cap, optimizer history and continuation RNG. The graph-selected arm must improve later pooled quality over both selected controls; a TRAIN-trial win alone is insufficient.
2. **Separate initialization benefit from ordinary ensembling.** Use a competent modern native single, ordinary native GNNM/BE initialization, and a fully trained same-width independent ensemble of four. Retain a warm-copy/identity-head continuation if claiming that private covariance initialization itself improves a warmed predictor. A common-only extra supervised displacement is not automatically that warm-copy control.
3. **Address covariance-aware ancestry at the scope of the claim.** A broad curvature/covariance initialization claim calls for a centered classification head Laplace/GGN initialization with subsequent identical continuation and accounting. At a short nonstationary warm state, a MAP/Laplace interpretation is not licensed; a regularized GGN-inspired span may be a useful engineering control but must be named accordingly. A plain supervised gradient/SVD span also tests whether graph filtering adds value beyond LoRA-GA-like data/gradient initialization. These are baseline proposals, not admitted runs or frozen recipes.
4. **Evaluate quality, not spread.** Retain later accuracy, pooled NLL, Brier score, member competence and paired blocks with uncertainty. More parameter variance, larger embedding distances, or lower TRAIN CE do not demonstrate complementary correct decisions. Mean-logit CE and probability-average CE are different served predictors and should not be silently interchanged.
5. **Freeze before scoring.** Qualify the actual warm states and finite head behavior, freeze the currently null constants and practical decision gates, preserve all null/abstention and failed arms, and account for fresh warm acquisition plus every candidate trial and reference. Do not use heldout results to choose bands, pairs, radii, or extra backbones.

## Scoped disposition

Continue this as a bounded accuracy-first hypothesis if the prospective qualification succeeds. A defensible potential contribution is **a useful graph-conditioned candidate span for private-head ensemble initialization, demonstrated beyond equal-budget non-graph selection and competent ensemble baselines**. None of the consulted papers by itself proves the complete candidate equivalent. None of this inspection establishes novelty, a gain, acceptance, or a reason to broaden experiments before the mechanism clears its representative comparison.

## Reading and integrity accounting

One newly retrieved paper was read in selected primary scopes: 2602.15747v1. The HTML locator also exposed selected results/discussion paragraphs; they are recorded in `READ_SCOPES.json` and are not counted as a full paper. One retained, previously unread scope was incidentally exposed: the abstract and first introductory anecdote of **Gradient Starvation**, arXiv:2011.09468v1. It is **not GNCL**; its method, results and guarantees are not used here. EVA (arXiv:2410.07170), Embedded Ensembles (arXiv:2202.12297), and arXiv:2606.00442 remain discovery metadata only.

The index v54 consultations reuse saved conclusions and do not increase the global paper count. This packet edits no index, manuscript, ledger, array/result, or server state. Exact receipts and bindings are retained; the seal establishes file integrity, not scientific correctness or search exhaustiveness.
