# Shared-backbone accuracy strategy: supervised private predictors

10 October 2026. Saved source conclusions only; no new primary reading, scientific execution or current-family outcome access. The frozen distribution study remains unchanged.

## Answer

Among the two close methods compared here, **ONE—Knowledge Distillation by On-the-Fly Native Ensemble—is the strongest accuracy-directed approach to investigate**: train a live shared trunk and private predictive branches, directly supervise every branch with true labels, directly supervise their online ensemble, and use that ensemble as a teacher during the same training run. It requires no independently pretrained teacher. This is a strategic judgment about the learning signal, not a published ranking or a claim of universal improvement.

The more general transferable principle is **direct task supervision of private predictors while the common representation keeps learning**. Its simplest nearby reference is **TabM**, which trains separate member losses and selects the collective predictor. ONE's online teacher supplies an additional hypothesis; the hard-label member losses are the essential competence-oriented signal. Neither source guarantees held-out member competence.

## Two nearest methods

| Method and saved source scope | Explicit prediction supervision | Relevant limit |
|---|---|---|
| **ONE**, arXiv 1806.04606v2, complete §3/Eqs1–7/Algorithm1 and bounded deployment context reused from the 3 October audit | Shared low layers, private final blocks/classifiers; summed hard member CE, hard CE of a positive learned-gate logit ensemble, and temperature-scaled ensemble-to-member KL. All train jointly; no separately pretrained teacher. | Default deployment keeps one live branch. Optional ONE-E retains the learned gate; uniform averaging is another ablation. Author-code loss reductions and teacher-target detachment were not qualified. Its image-task source does not establish graph utility or equality to the current native probability pool. |
| **TabM**, arXiv 2410.24210v3, saved 6 October connected method/ablation/setup rereview | Train each member's own task loss; private first-input adapters act before common feature mixing; shared weights remain trainable; classification serves averaged member probabilities and collective validation selects training state. No teacher. | Individuals can be weak/overfit while the collective helps. Joint untied training/selection already improves TabMpacked, so all gain cannot be attributed to sharing. Dataset-specific tuning, member-count/width/depth dependence and unstudied normalization backbones prevent universal transfer claims. |

ONE is a strong reference for explicit shared/member/ensemble supervision; TabM is the simpler reference for separate member training. Neither is a hidden-repulsion objective. A private branch's true-label CE penalizes its actual mistakes, whereas increasing hidden distance can be achieved by irrelevant or harmful changes. “Prediction residual” here means the supervised error signal; neither comparison invents a new structurally additive residual model.

## Why this has a reason to help across architectures

For class logits, the own-CE residual is `p_m − one_hot(y)`. It can remain substantial for a weak member even when the committee already predicts the right class confidently. Training only a pooled loss can give that private member a small or poorly targeted correction. Direct member CE keeps a route's true-label error visible to its private parameters and the live shared representation; ONE additionally trains the ensemble on true labels rather than assuming the current committee is correct.

This argument depends on prediction/loss interfaces rather than a particular convolution, attention or message-passing operator. That makes it **conceptually portable**. Engineering portability still requires sufficient private predictive capacity, correct shared/private ownership and gradients, actual member normalization, a branchable native interface, stochastic-state handling, deployment choice and full cost accounting. Copying a terminal head is not automatically equivalent to ONE's private high layers, and copying TabM adapters after graph aggregation misses its first-adapter intervention before shared mixing.

The argument is conditional. A hard loss can fit training labels without supplying useful held-out alternatives. An online teacher can spread a shared mistake. Shared optimization can weaken the representation, and probability/logit pooling can lose already correct alternatives. Neither source proves that arbitrary backbones, splits, label budgets or graphs improve.

## Assumption that graphs may violate

The useful-ensemble premise is that **the common representation leaves enough predictive evidence for sufficiently competent private branches to make different, useful errors, and the committee supplies a more reliable learning signal than its members on relevant examples**.

Graph routes may instead inherit the same unavailable or aliased neighborhood evidence, correlated label noise or common strict wrong rival after aggregation. Private supervision cannot reconstruct information absent from every complete branch input. Repeated prediction of the same factual graph also does not create independent observations; scarce, neighboring training labels can make apparent competence reflect fitting shared structure rather than reliable corrections.

There is a precise conditional obstruction: if one wrong class k has `z_m,k > z_m,y` in every member at a node, every positive weighted logit pool ranks k above y. Every positive weighted probability pool also has `p_m,k > p_m,y`, so it cannot repair that node either. A teacher formed from those same outputs has no corrective alternative to distill. This is elementary convex-pooling algebra, not a graph-specific theorem or proof that the full network cannot learn a better state.

The project's **complete** diagnostics already rule out a universal missing-alternative explanation: correct shared alternatives exist on a majority of the diagnosed independent-correct/shared-wrong cases, and some are lost by serving. Strong coherent members can also be nearly redundant. Thus distinguish missing useful alternatives, weakened competence and losses during pooling. Do not label every failure “graph collapse” or infer its cause from correlation alone.

## Implication for this project

Ordinary native member CE and early private factors already instantiate much of the TabM-style principle. The running quartile recipe already directly supervises native and corrected predictions with a live body. Its zero-start identity is not a held-out competence guarantee, but the supervision principle does not need another name or auxiliary repulsion.

Completed private-hop, specialist-CMCL, optimizer-normalization, local/LoRA correction, retrieval and decoder rules failed their respective fixed conjunctions. These failures close the exact recipes, while leaving broader known-method families conditional. ONE's complete online-KD/gated deployment recipe has not been established as executed or falsified by the saved scopes; that fact alone gives no reason to add it now, and it must not be equated to the failed decoder or to a native own-loss baseline.

**New untested mechanism nomination: none.** The available evidence does not yet justify another teacher, gate, loss split or private-capacity proposal during the frozen study. First preserve its complete result. A later source-faithful ONE comparison would be an attributed known-method baseline requiring its complete operation and equally processed capable controls; this memo neither admits that comparison nor chooses any setting. The active distribution hypothesis and all prior closures remain intact.

Sources and exact reused scopes are bound in `SOURCE_BINDINGS.json`. New papers, new method scopes, new model/code/data execution and canonical-state changes: zero. This memo earns no novelty, portable-gain or acceptance claim.
