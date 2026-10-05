# Scientific rationale for private endpoint frames with shared learning

## Conclusion

The proposal is a plausible **finite-sample learning bias**, with no predictive evidence yet. A common node encoder supplies graph evidence to separately supervised native routes; each route learns a compact private reflection before the outer endpoint Hadamard product. The useful hypothesis is that shared representation learning restrains unstable graph shortcuts while private endpoint/context routes preserve complementary compatibility evidence.

Neither geometry nor sharing guarantees this outcome. An independent four-member native ensemble with the same reflections can represent the candidate by tying its encoders and effective dense maps, and has additional freedom. A win against that control must therefore concern estimation, regularization, or optimization. The capable single with the same four reflected products already receives the interaction features; beating it requires useful consequences of the separate native routes. The two mechanisms below would need to work together to explain superiority to both controls.

## 1. Regularizing evidence reused across graph endpoints

**Hypothesis.** Link examples repeatedly reuse nodes and neighborhoods, so their evidence is correlated. Joint updates to one encoder, combined with small private endpoint bases, may constrain member-specific representations that fit incidental node or neighborhood associations. Private routes could retain alternative compatibility judgments without giving every member a freely varying encoder. The expected advantage over independent four-member learning is more stable served predictions where endpoint evidence is limited or training runs differ.

This creates no additional labels: independent members also see the training data. Their freedom may help, and conflicting route gradients may damage the shared encoder. A single already shares its encoder across all edges, so this mechanism alone does not explain beating the capable single.

**Closest prior.** CAMERO (arXiv:2204.06625) explicitly studies quality from a shared encoder and private, separately supervised classifiers, with perturbations and consistency regularization. BatchEnsemble (2002.06715), TabM (2410.24210), and graph shallow ensembles (2504.12627) also precede the shared/private construction. The graph-specific hypothesis here concerns repeated endpoint evidence under the native link-prediction objective; those precedents do not establish its benefit.

**Falsifying diagnostic.** Before inspecting outcomes, freeze endpoint-support strata from the training graph, such as bins of the smaller endpoint training degree. Across fixed seeds, compare the candidate and same-operation independent ensemble using the served ranking metric and, where meaningful, log loss. No replicated overall gain, together with no predicted low-support or stability benefit, would undermine this explanation. Gains confined to well-supported endpoints would require a different account. Stratum patterns alone cannot attribute an effect causally to encoder sharing.

## 2. Coupling endpoint compatibility to private graph context

**Hypothesis.** A reflection before the Hadamard product exposes symmetric cross-coordinate endpoint interactions that post-product diagonal scaling cannot recover. Each native route can combine this compatibility view with its private nonlinear common-neighbor/residual context. Routes might resolve different edge regimes—for example, common-neighbor-supported closure and sparse compatibility—and their equally averaged raw logits could improve the served prediction when their corrections survive averaging.

The capable four-frame single has one context and prediction route. It may learn this combination equally well or better. The independent reflected ensemble already has separate endpoint/context routes, so mechanism 1 is still needed to explain beating it. There is no gate or diversity objective: route disagreement can be redundant or harmful. Moreover, an isotropic dot readout is reflection-invariant, and averaging linear reflected readouts reduces to one symmetric bilinear score. Useful route effects must be earned in the native nonlinear computation.

**Closest prior.** NCN/NCNC supplies the endpoint/common-neighbor fusion. Link-MoE (2402.08583) already combines link experts using structural and feature evidence, although its pair-specific learned gate and supervision differ from equal raw-logit serving. HousE (2202.07919) and GoldE (2405.08540) precede Householder-based endpoint geometry. No new orthogonal or bilinear principle follows here.

**Falsifying diagnostic.** Freeze zero-versus-positive training common-neighbor strata. Check whether competent routes make complementary correct predictions and whether their corrections improve the served mean against the four-frame single. At frozen inference, replace each reflected outer endpoint branch with its unreflected counterpart while holding cached context fixed. A final-decoder interaction check can use

\[
I=\frac1M\sum_m\{f_m(c_m+e_m)-f_m(c_m+e^0_m)-f_m(e_m)+f_m(e^0_m)\},
\]

where \(e_m\) is the reflected endpoint branch, \(e^0_m\) its unreflected replacement, and \(c_m\) the fixed context contribution. If replacement preserves served quality and interaction differences do not accompany correct recoveries, this explanation lacks support. Nonzero interaction or disagreement alone is insufficient. These perturbations can leave the training distribution and omit context-mediated frame effects; they do not prove the training mechanism.

## What a positive result would establish

Replicated gains over **both** the capable same-four-frame single and the same-operation independent ensemble, under matched data, supervision, serving, and declared selection budgets, would establish bounded empirical utility of this constrained graph-learning recipe: share node evidence, untie compact endpoint interaction bases before pair compression, and retain separately supervised nonlinear context routes. Report complete measured training and serving costs alongside quality.

Those comparisons identify the composite recipe. They do not isolate encoder tying as the cause, establish universal superiority, or establish novelty. A shared-frame four-route control would help test whether private outer frames contribute. Quality parity with lower cost would support an efficiency result, not the requested better-quality claim.

The completed CPU checks establish executable controls and accessible gradients only. The strongest next scientific decision is a prespecified quality test against both indispensable controls; there is presently no basis for claiming shared-learning benefit or publication acceptance.

## Reused evidence

This rationale reuses the current synthesis in `quality_method_synthesis_20261005_v1/REPORT.md`, the CAMERO/graph-ensemble assessment in `accuracy_ensemble_quality_prior_gap_20261004_v1/REPORT.md`, and retained paper records in `literature_memory/index_v60/LITERATURE_INDEX.json`. No new literature review, fit, server work, or canonical edit was performed.
