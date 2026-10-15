# Native initialized private routes after the gradient audit

15 October 2026. **Advance the already specified three-fit residual-only/private-BN comparison as the next scientific question.** The completed audit closes the proposed interference gate. Fresh primary reading supports a precise initialization contract and exposes a gradient trap; it supplies no graph-quality guarantee or new adapter principle.

## What the completed evidence changes

All 456 route/batch observations in the shadow audit have negative predicted shared/joint and finite shared-after-private/joint loss changes. Gate counts are 0/152 in each seed, although negative gradient-pair dots occur 22/228, 11/228 and 1/228 times. The actual Adam displacement improves the measured losses despite those pairwise conflicts. This covers selected private-BN states and one subsequent TRAIN epoch, with 186.40 seconds paid work. It does not measure cold-start history or heldout risk. See `analysis/shared_gradient_shadow_completed_v1/REPORT.md` under the expansion root.

The completed ingredient study also constrains the direction: private BN scores 73.614% AUC; adding geometry gives 73.651%, correction 72.186%, and both 73.028%, versus I4 74.847%. The current input correction was tested on top of signed BE. Its negative increment leaves one concrete missing contrast: that same small additive route around the native initialization, with BE removed. No gain from the earlier ingredients is presumed.

## Two fresh primary scopes

**ReZero, Bachlechner et al., arXiv:2003.04887v2, §3, §3.1 and Appendix D.7.** The method uses `x_next = x + alpha F(x)`, with alpha initially zero. Its toy model analyzes how the gate learns before the initially gated branch weights. This is direct ancestry for preserving an initial map while learning an added path. The paper's whole-network identity and deep-signal-propagation argument do not transfer to a native GNN with one adapter: the native graph network retains its own Jacobian, normalization and optimizer. [Exact version](https://arxiv.org/html/2003.04887v2#S3).

**LLaMA-Adapter author code, one v1 model path.** At commit `521a09da84f70f6913d54b7421afa24010319e47`, `alpaca_finetuning_v1/llama/model.py:84,116–129` initializes a per-head gate at zero and separately normalizes adapter and original-token attention. Only the adapter contribution receives `tanh(gate)`; the original-token softmax retains its own denominator. This is a concrete implementation of an initially silent additive path without diluting the existing path. Lines 199–208 explicitly run the prefix under `no_grad`, which conflicts with our requirement that the whole native backbone keep learning. No prompt, attention or prefix-freezing port is proposed. [Pinned code](https://github.com/OpenGVLab/LLaMA-Adapter/blob/521a09da84f70f6913d54b7421afa24010319e47/alpaca_finetuning_v1/llama/model.py#L84).

### Initialize one zero end

The existing route is `h_m = h + U_m GELU(V_m h)`, with distinct V_m and U_m=0. At this point the adapter Jacobian with respect to h is the identity; the shared native derivative is preserved for the same stochastic view. The U_m gradient can be nonzero, while the V_m gradient is initially zero. Incoming matrices can start learning after U_m moves. A zero V-gradient on the first update alone is therefore expected.

Adding a zero ReZero-style gate to this already zero-up branch would produce `h + alpha_m U_m GELU(V_m h)` with alpha_m=0 and U_m=0. All three private derivatives then vanish: the gate sees zero branch output, U sees a zero gate, and V sees both. Under ordinary task updates that branch stays inactive. Preserve the existing single zero end; do not stack a second zero gate onto it.

Distinct down-projections give different accessible private directions, without guaranteeing different correct predictions. Initial eval equality also does not imply M1's training trajectory: four routes use four native dropout realizations and private running-state histories. Native parameters must remain trainable and in their original optimizer groups. Initial parity, active outgoing gradients and later incoming-gradient exposure are construction measurements, rather than predictive success criteria.

## One next experiment

The condition was the actual harmful-interference diagnostic; it is now closed. Use the existing `pilot_20261015_residual_only_private_bn_v1/SPEC.json`, rather than adding another study or a gradient optimizer:

- Three fresh seeds 101/202/303, one shared native MolBACE/GIN body, four private rank-8 input residuals after the complete AtomEncoder, no BE input/head banks, and the same private-BN buffer law.
- Preserve native mean own BCE, mean-logit serving, Adam .001, full native batches, and the existing 1000-epoch cap/minimum100/patience50 selector. The adapter contributes 19,200 learned coordinates, 76,800 FP32 parameter bytes and 153,600 dense Adam-moment bytes. The shared/native parameter count and actual optimizer storage are reported by the prepared source.
- Primary comparison: residual-only minus the completed private-BN BE-plus-residual condition. Reuse private-BN BE alone, original GNNM, native M1 and genuine I4 as required references. These compare whole procedures; BE removal changes the starting function and private parameterization together.
- Report all three selected-VALID AUC/BCE contrasts, member competence and actual served quality. Better initial parity or stronger members without a useful served comparison cannot establish an ensemble advance. A gain over BE-plus-residual that still loses to I4 establishes a narrower wrapper result. One encountered 151-graph scaffold VALID split gives exploratory evidence only.

Every route retains all factual TRAIN targets. The private coordinates define different learning paths; a context/class specialist interpretation would require a meaningful target and the retained common-target versus assigned-target control. This comparison asks about competence and useful serving without assigning specialist semantics to random subspaces.

This is three new fits with zero new teacher or I4 acquisition. The frozen plan estimates about 75 minutes if stopping resembles its controls; its ceiling is 114,000 optimizer maps and about 8.3 hours at the historical 10-second-per-epoch proxy. Qualification, failures and measured new throughput remain separately charged. The literature pass changes no settings, launches no fit and adds no campaign recipe.

## Objectives and uncertainty remain separate questions

For the same observed binary targets, mean-own BCE satisfies

```text
F = BCEWithLogits(mean_m z_m, y) + mean_m softplus(z_m) - softplus(mean_m z_m).
```

The final two terms form a nonnegative, label-independent Jensen gap. Own-risk learning therefore includes a known disagreement-related term relative to the served mean-logit risk. Every member's same-batch loss decreasing does not establish improved served AUC, calibration or generalization. This identity supplies an interpretation, not evidence that removing the term helps; the retained own/pool controls already cover that objective question. Keep F fixed in the missing-arm contrast so that its result concerns the wrapper package.

The private-BN AUC recovery comes with worse mean BCE. Evaluate probability quality directly with proper losses and the declared serving contract. Small private routes, disagreement or random dropout do not identify a Bayesian posterior. Retained Rank-1 BNN/Laplace-adapter notes already supply uncertainty ancestry; posterior/noise acquisition would require its own likelihood, regularization, sampling/serving and matched fixed-noise controls. No new uncertainty recipe follows from the closed interference gate or the present fresh sources.

## Attribution and read limits

LoRA/LoRA-Ensemble, joint scratch TabLoRA, outgoing-zero adapters and private normalization are established prior ingredients in the retained notes. A defensible result here would show that their explicit native-initialized, jointly trained graph-wrapper combination improves the measured quality/storage operating point. It would not establish a new initialization, gradient or uncertainty principle.

`READ_SCOPE.json` records exact new source bytes and line/block scopes. The fresh scope is one paper's method/toy discussion and one author-code file, with zero full-paper or proof audits. No downloaded code was imported. ReZero result tables, LLaMA training configuration/parameter-freezing policy, published benchmark claims and graph transfer were not audited or adopted. Earlier TabLoRA/ELREA retrievals in `discovery/` were unsuccessful code locators, rather than new method reads; the overlong old TabLoRA locator preview receives no new reading credit. RIDE/SADE/PCGrad method summaries are not repeated.
