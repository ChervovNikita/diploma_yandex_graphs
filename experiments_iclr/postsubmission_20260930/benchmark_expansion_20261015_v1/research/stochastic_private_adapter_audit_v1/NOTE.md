# Recommend one learned-noise acquisition test

15 October 2026. **Retain a small stochastic private-adapter comparison as a genuinely different, testable quality hypothesis.** It should not be excluded because Bayesian adapters already exist or because uncertainty estimation is not the main goal. This is a research recommendation, with no code or fit authorized by this packet. PubMed transfer remains the conditional default; this alternative does not automatically add work or combine with structural/pairwise objectives.

## Was the earlier deferral justified?

The saved E/Rank-1 proposal already recommends stochastic acquisition for stronger deterministic mean-factor predictions. The graph-fast-factor note stops a different claim: adding graph precision contracts modeled covariance but does not establish selective-risk control. Its no-go must not be applied to learning adapter means/scales from scratch for accuracy. Neither document establishes that stochastic acquisition is empirically useful on the current native wrapper; neither warrants rejecting that empirical question for lack of novelty. [H1,H2]

Current residual-only MolBACE has respectable mean member AUC **74.560%**, but pool AUC **74.518% versus native I4 74.847%**, nearly no pooling gain and **95.54% common pool reversals**. BCE is **1.952 versus .890**; its excess loss is widespread and class-specific. These motivate checking acquisition robustness, without diagnosing sharp minima, epistemic uncertainty or a noise-repairable cause. [E1,E2]

**Accuracy mechanism:** expected own-task loss under small private weight perturbations can discourage adapter solutions whose predictions depend on fragile local parameter choices. Learning scales can reduce harmful perturbations in sensitive directions while retaining tolerance elsewhere. Route-specific distributions give different learning trajectories, but supply no new class information or explicit complementarity guarantee. A fixed-noise control can explain the benefit as ordinary perturbation/anchoring regularization. The hypothesis is useful precisely because that alternative is testable.

## One concrete adaptation of the current wrapper

Keep the current native GIN, full AtomEncoder output, four rank-eight residuals, private BN state, native dropout/sampler/Adam/controller and raw-logit mean. All native weights remain trainable. For each route:

```text
h_m = h + U_m GELU(V_m h)
V_m = mu_m + sigma_m * epsilon_m,  epsilon_m ~ N(0,I)
sigma_m = exp(rho_m)
```

Only the **down matrix V** is stochastic. U remains an ordinary learned matrix initialized at zero; mu starts at the existing data-free orthonormal down frame V0. Each route draws one complete V per native TRAIN minibatch, reused across all atoms and graphs in that route's batch. Draws are independent across routes/updates. No per-atom Flipout or changed graph is introduced.

This asymmetric choice has a useful construction property: **every sampled residual is initially zero**, preserving the point wrapper's initial native function under matched stochastic/state conditions. The U gradient can learn immediately; task gradients to mu/rho start at zero and become available after U moves. It does not add the dead second-zero gate described in the retained initialization note. [P1,P3]

Use one fixed local Gaussian prior `p_m=N(V0_m,s0²I)` and initialize `q_m=p_m`. A concrete prospective scale is `s0=.1/sqrt(300)`: expected squared perturbation norm per down row is `.01`, versus the initial unit row norm. This is a **new fixed port choice**, not a BLoB default, optimized value or guarantee of small logit change. Use the native labeled TRAIN count N=1210 and one scalar objective through all learned parameters:

```text
J = mean_m [native minibatch BCE_m(V_m) + KL(q_m || p_m)/N]
KL = .5 sum_coordinates [(sigma²+(mu-V0)²)/s0² - 1 + 2 log(s0/sigma)]
```

Here each BCE is the unchanged native mean over the eligible labeled minibatch; each KL is a **sum over that route’s 2,400 down-matrix coordinates**. The total bank regularizer is therefore `sum_m KL_m/(4*1210)`, added once per eligible update. It is not divided again by minibatch count, minibatch size or parameter count. Native masking/remainder conventions remain unchanged; the KL normalization uses the fixed global TRAIN label count. This estimates an expected **own-route** risk, not the negative log likelihood of a probability mixture.

No KL annealing, scale sweep, prior fitting, extra loss or altered selector is proposed. Keep the native optimizer options for all original parameters; new rho uses the existing adapter group. Save separate noise RNG state. This is a Gaussian-KL regularized stochastic acquisition rule with point-estimated native weights/U and native dropout/BN conventions; it is not certified full-network Bayesian inference.

## Small, discriminating comparison

Use the same complete MolBACE protocol and seeds 101/202/303. Reuse exact authenticated completed controls only after matching residual placement/rank/initialization, private BN ownership/update policy, source data/split, native schedule/selector, sampler, optimizer groups and RNG ownership. New factor draws need independent saved streams that do not consume native construction/dropout/sampler RNG. If those identities do not match, this six-fit count is invalid; do not silently substitute equipped or differently trained controls.

| Procedure | Status/purpose |
| --- | --- |
| Current deterministic residual-only F bank | Paid point baseline; unchanged objective and saved predictions. |
| Fixed-noise bank | **Three new fits** with sigma=s0 frozen and the same prior/mean penalty. |
| Learned-scale bank | **Three new fits** with the same initialization/prior and trained rho. |
| Unchanged native M1/I4 and original GNNM | Retained quality references, with M1 exact I4 body0 and I4 own selectors. |

At fixed sigma=s0, KL is exactly `||mu-V0||²/(2s0²)`. Therefore the two new arms share the same mean anchor; their difference tests **the ability to adapt scales under this scale-KL penalty and optimizer**, including its effects on subsequent mean/shared updates and BN histories. It does not separately identify Bayesian posterior correctness, an entropy benefit, noise magnitude versus allocation, or a general advantage over other adaptive-noise rules. Their contrast with the original point bank changes both perturbation and anchoring. It cannot isolate noise from anchoring, and this packet does not claim that attribution or add another arm for it.

Evaluate/select both new arms using **mu only**, with the original mean-logit AUC selector, four routes, native eval mode and private BN buffers. This directly tests better deterministic acquisition at unchanged primary serving cost. `f(mu)` is generally not `E[f(V)]`; no posterior-predictive integration claim follows. There is no sampled-serving panel, temperature fit, threshold fit or inherited E calibration panel.

The unchanged native references are quality bars, not objective-matched stochastic-adapter models. A favorable result supports this complete low-overhead shared procedure and the learned-versus-fixed scale increment. It does not isolate a causal benefit from sharing; that stronger claim would require separately declared independent models with the same adapter/acquisition opportunity. Do not relabel equipped E controls as native M1/I4.

## Falsifier, risks and cost

**Learned-scale screen:** its mean served AUC must exceed both the point bank and fixed-noise bank; every paired seed and BCE difference stays visible. A nonpositive mean contrast fails that fixed hypothesis. To advance toward the user's goal it must also exceed unchanged native M1 and I4. Better NLL alone, nonzero learned sigma or larger disagreement is insufficient. If fixed noise matches/wins, learned scales have no supported incremental value; a useful fixed-noise result remains honestly reportable as that control's result.

Report mean/worst member AUC/BCE, classwise losses, absolute common-reversal counts and net pool repairs/harms. Improved competence without more pooling benefit is still a competence result if served ranking improves; do not require invented specialization. Worse probability quality must be disclosed as a trade-off. A complementary-error explanation additionally needs new correctly ranked alternatives and net served repairs. Three exposed-VALID seeds remain exploratory.

The main risks are a poorly chosen prior, excessive shrinkage of V, redundant distributions, added gradient variance, U growth compensating for V regularization, and failure to learn information absent from the native inputs/capacity. **Noisy training versus mean evaluation also changes the BN activation distribution.** Both arms retain the same private-BN update law, but different learned scales can yield different buffer mismatch. A gain therefore does not by itself prove curvature regularization or a Bayesian mechanism. No recalibration pass or BN-policy rescue is added; BN-free transfer would be a later question.

There are **9,600 additional learned scale coordinates**, 38,400 FP32 parameter bytes and 76,800 dense Adam-moment bytes, beyond the existing 19,200 residual coordinates. Total private learned overhead is about **1.53% of native parameters**, versus 1.0183% currently; gradients, prior constants, draws and state also consume memory. Training still uses four native route forwards and one backward collection per eligible update, plus small sampling/KL work. Actual runtime/memory requires qualification.

The paid three point fits took 4,094.57 worker-seconds over 484 epochs. Six fits with similar stopping would suggest roughly **2.27 worker-hours before added overhead**; extrapolating that old average epoch cost to six 1,000-epoch caps gives roughly **14.1 hours**, not a measured bound or ETA. Noise can change stopping. This is a modest *incremental fit count*, not a claim of free or already-admitted compute. Charge failures and actual work if implemented; this audit does not request another resource probe or resource gate. [E1]

## Conditions for implementing this follow-up

After the planned PubMed transfer decision, root can choose this acquisition question on its empirical merits; no new-paper, novelty or proof-of-sharpness prerequisite is imposed. Implementation is justified if root accepts this **single fixed six-fit contrast**, its mean-serving/KL contract and the limited attribution above, and can bind the exact existing point/native references and RNG ownership. Freeze the one prior/scale choice, unchanged controller and complete readout/falsifier before work. If a required reference does not match, return a corrected finite count for decision rather than adding controls silently. No structural/pairwise/noise combination, scale retry or calibration panel is included. The current running and queued studies remain unchanged.

## Actual prior sources and scope

**BLoB**, [arXiv:2406.11675v5](https://arxiv.org/html/2406.11675v5), §§3.1–3.5/Algorithm1, is the closest asymmetric-adapter source: deterministic zero-start B, Gaussian A, variational objective, scale parameterization and low-rank Flipout. It freezes a pretrained linear base and uses squared scale parameters, a zero-centered prior and its own KL schedule. Our live native GNN, nonlinear GELU residual, data-free frame-centered prior, log scales, coherent route draw and mean serving are declared adaptations. Its Gaussian full-linear-weight covariance theorem and language-model performance do not transfer.

**Rank-1 BNN**, retained arXiv:2005.07186v2 §§3.1–3.4/Appendices B–C, supplies live point-shared weights plus stochastic private factors/expected own likelihood. The E note's narrower revisit used an unversioned carrier; that limitation remains. **Laplace-LoRA**, arXiv:2308.13111v5 §§3–4 and saved appendices, is a different post-fit conditional approximation and does not substitute for this acquisition test.

This pass reused the saved reading notes, then made **one scoped BLoB reread/extension from its retained primary text** to resolve initialization, prior and sampling details. Zero new paper identities, author-code reads or network requests. Exact line scopes and hashes are in `READ_SCOPE.json`. No code, model, data/logit/checkpoint/TEST access, GPU work or Git operation occurred.
