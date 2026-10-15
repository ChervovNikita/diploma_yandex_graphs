# Useful routes require ranking and probability quality

15 October 2026. Separate prospective synthesis. The running six-fit atom-quartile pilot, its criteria and serving remain unchanged. No experiment, tensor inspection, source modification or launch is performed here.

**One secondary probability-quality hypothesis: own-risk plus log loss on the uniform probability mixture, trained through all existing parameters.** This is an attributed objective/readout adaptation, kept separate from structural responsibility. An NLL-only gain would not meet the main goal of better task accuracy/ranking than genuine I4.

## 1. What the completed evidence establishes

The residual-only/private-BN bank has mean member AUC **74.560%** and pool AUC **74.518%**, versus I4's **74.847%**. Pooling adds −.043 points on average, compared with +1.438 for I4. Of its pooled ranking reversals, **95.54% are wrong in every route**. These findings support strong individual ranking and insufficient useful ranking complementarity. They do not establish general predictive competence: pooled BCE is **1.952 versus I4 .890**, worse in every seed. BCE measures overall probability quality; its degradation alone does not identify calibration error, overconfidence, resolution or a specific cause. [E1]

All three residual fits trained ordinary F through all learned parameters. Their BCE result therefore **cannot be attributed to shared-F/private-W separation**, which is introduced by the new pilot. AUC checkpoint selection also supplies no probability-loss guarantee. The common native update averages four stochastic dropout forwards; stronger ranking than M1 does not isolate private residual capacity from that training law. The proposed objective comparison keeps the same four forwards in every arm, so it needs no extra zero-residual fit. Such a control would be justified only by a later specific capacity-attribution claim. [E1,E2]

Prior completed findings limit easy remedies: private BN improved ranking with worse BCE; probability averaging on older correction banks reduced BCE while slightly lowering AUC; WikiCS own/pool interpolation did not improve accuracy. GNCL, learner-collusion, ACE and the other retained scopes already establish relevant ancestry and warnings. [E3–E5]

## 2. The structural pilot is useful, but is not a probability-quality safeguard

Its fixed positive within-class atom-count weights, exact whole-vector permutation and uniform residual control can identify whether the structural association adds value beyond generic persistent reweighting. This is a cleaner question than interpreting arbitrary spread as skill. Atom count nevertheless need not describe the missed chemical prediction tasks. Equal class totals and positive supervision do not preserve the conditional probability target. [E2]

For a population illustration, let `eta(x)=P(Y=1|x)` and train a private predictor with positive class-dependent weights `w1(x), w0(x)`. Its unrestricted weighted-BCE optimum is

```text
p_w*(x) = w1(x) eta(x) / [w1(x) eta(x) + w0(x)(1-eta(x))].
logit p_w*(x) = logit eta(x) + log[w1(x)/w0(x)].
```

Within-class normalization preserves total mass, not this pointwise ratio. Even when weights sum to four for every labeled example, averaging their log-odds shifts need not cancel. This is an illustration of the changed target, not an exact solution for the finite coupled GNN. Shared F can constrain the common parameters, but private states change later shared gradients; F/W separation has no representation or calibration preservation theorem. It may create useful specialists, probability bias, or both. The frozen pilot correctly reports signed BCE without retrospectively adding a loss veto. [E2,E5]

A private-only holdout is not literal cross-fitting when the shared encoder trains on those labels. Bootstrap/private reweighting also cannot by itself repair a serving-loss mismatch. No teacher acquisition or bootstrap grid is nominated.

## 3. Probability pooling, mean logits and proper losses are distinct

Let `p_m` be a route's sigmoid/softmax probabilities, `p_m(y)` its true-label probability, and `F = mean_m[-log p_m(y)]` with the unchanged native reduction.

| Forecast | Definition | Interpretation |
| --- | --- | --- |
| Current mean logits | `p_L = softmax(mean_m z_m)`; binary `sigmoid(mean z)` | Normalized geometric opinion pool; binary odds are the geometric mean of route odds. |
| Probability mixture | `p_P = mean_m p_m` | Arithmetic mixture of predictive probabilities. |
| Proper mixture loss | `L_P = -log p_P(y)` | Scores the actual mixture probability, not a mean of member losses. |

Both `L_P <= F` and `L_L <= F` hold pointwise by Jensen's inequality. **L_P and L_L have no universal loss or AUC ordering.** For route probabilities .99 and .60, the probability mean is .795 and the mean-logit probability approximately .924; which has lower loss depends on the label. Identical routes give identical pools. Probability averaging is not merely temperature scaling or new task information.

Gneiting and Raftery establish the categorical logarithmic and Brier scores as strictly proper. This concerns expected scores for quoted distributions, not guaranteed calibration after finite neural optimization or validation selection. Gneiting and Ranjan distinguish linear from generalized linear pools and show, under their stated forecast-space conditions, that pooling can disrupt calibration/dispersion. Their continuous generalized-pool theorem explicitly does not cover Bernoulli forecasts; no blanket theorem about our mean-logit binary bank is imported. [P1,P2]

## 4. One secondary objective and its diagnostic falsifier

Keep the existing residual-only parameterization, native operators, inputs, optimizer, regularization, BN policy where applicable and four-route stochastic forward law. Add no learned coordinates, teacher, temperature or gate. Train **all shared and private parameters** with the fixed scalar objective

```text
J_P = .5 F + .5 L_P,
serve p_P = mean_m p_m.
```

Use one ordinary backward collection and stable `logsumexp` over route log-probabilities. Shared body training changes explicitly; this is an all-parameter objective, not shared-F/private-W credit. It needs exact integration/resource qualification.

For each record define `r_m = p_m(y)/sum_j p_j(y)`. Then

```text
gradient_z_m L_P = r_m [p_m - one_hot(y)],
gradient_z_m J_P = [.5/M + .5 r_m] [p_m - one_hot(y)].
```

The mixture component gives more credit to a route already assigning useful probability to the observed target; the own-risk half retains a gradient floor. This is standard mixture-likelihood responsibility, not new routing. It can reinforce early winners and permit compensating weak members—the retained collusion warning remains. At identical predictions it reduces to own-risk gradients and creates no symmetry breaking. At an unrestricted population optimum, the positive own-risk term favors every member predicting the true conditional distribution; useful diversity must arise from finite-data/capacity learning, not an artificial requirement to disagree. [P1,E3]

**Prospective controlled experiment:** compare the exact same-capacity F bank, `J_L=.5F+.5L_L`, and J_P. Both new objectives update all parameters. Freeze probability mean as the primary readout for all three; also report each selected bank under the original mean-logit readout to separate training from serving. J_L controls generic own/pool interpolation; F under probability mean controls a readout-only gain. Give genuine I4 the same two fixed readouts, retain native M1/G0 and report all signed member/pool task-metric and NLL differences. This is six new fits for two objectives across three paired seeds **only if exact F/reference controls already exist**; the proposed PubMed/Cheb identity currently lacks those completed controls. No structural-plus-objective combination is proposed.

Select each bank once by the unchanged mean-logit native task metric; evaluate both readouts at that checkpoint. Do not switch to a BCE-selected epoch, fit a calibrator or choose a pool from the exposed VALID results. Probability-based selection would be a separate protocol.

**Falsifier:** J_P must improve pooled NLL beyond both the F probability readout and the matched J_L objective. If it does not, this probability-objective explanation fails. If task ranking/accuracy or member loss worsens, report a trade-off rather than preserved quality. A separate complementarity claim additionally requires new correct alternatives and net served repairs; lower NLL, larger disagreement or reduced common-error fraction alone is insufficient. These are prospective descriptive contrasts, not significance or noninferiority tests.

Advancement toward the main research goal additionally requires improved native task metric against the exact shared control **and genuine I4**, with NLL trade-offs retained. Passing the probability-loss falsifier alone does not establish progress on accuracy over I4.

## 5. Transfer and disposition

Root now selects **PubMed/PolyFormer-Cheb for source preparation only**, following the completed provider check: TRAIN [3943,3943,3943], VALID **[72,1919,1952]**. All three classes have support, but class0 is only 1.83% of VALID; whole accuracy can hide its harms. Report per-class support/accuracy/NLL. Native multiclass CE and no BN make this objective transferable. Exact controls, residual composition and cost remain unqualified; the matched minimum 2,000 / patience 100 / cap 10,000 remains. Its old private-credit smoke memory is lower than CS, without proving residual runtime. PubMed has substantial prior development exposure, not protected confirmation. CS remains the earlier alternative with six classes absent from VALID. No new fit is authorized. [E6]

Do not transfer a calibrated Bernoulli interpretation directly to link prediction with sampled negatives: its loss depends on the negative sampler and is not automatically a probability for real-world edge prevalence. A query-ranking objective needs its own definition.

**Disposition:** finish the running pilot under its freeze; retain this separate probability-quality question. No new launch or frozen-package change follows. Two scoped primary reads, no full-paper/code audits or novelty clearance. Receipts: `READ_SCOPE.json`, `LOCAL_SOURCES.json`.

### Citations

- **P1:** Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA102:359–378, [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437). Scoped PDF pp4–7, especially categorical Example3 and Table1; limited introductory/incidental excerpts disclosed in the receipt.
- **P2:** Gneiting & Ranjan, *Combining Predictive Distributions*, [arXiv:1106.1638v1](https://arxiv.org/html/1106.1638v1). Selected framework/calibration definitions and §§3.1–3.2; theorem assumptions and Bernoulli limitation retained. Exact equivalence to a later journal version was not audited.
- **E1:** `pilot_20261015_residual_only_private_bn_v1/complete_results_v1/INTERPRETATION.md`.
- **E2:** `pilot_20261015_structural_private_risk_v1/PROTOCOL.md`; running study receives no new outcome read.
- **E3:** `literature/quality_preserving_diversity_followup_v1/REPORT.md`: retained GNCL, learner-collusion and Wood et al. scopes; prior fixed-pool results and F/L identities.
- **E4:** `research/next_quality_mechanism_20261015_v1/RECOMMENDATION.md`: completed cross-setting evidence; no new historical payload inspection.
- **E5:** `literature/structural_private_credit_followup_v1/NOTE.md` and `literature/shared_competence_primary_followup_v1/REPORT.md`: ACE, decoupling, RIDE and SADE scopes, reused without primary rereads.
- **E6:** `PUBMED_PROVIDER_CLASS_SUPPORT_22.json` and `research/second_setting_protocol_v1/DECISION.md`: root-selected PubMed source preparation, measured support, imbalance, exposure and missing controls. Earlier CS decision retained as history.
