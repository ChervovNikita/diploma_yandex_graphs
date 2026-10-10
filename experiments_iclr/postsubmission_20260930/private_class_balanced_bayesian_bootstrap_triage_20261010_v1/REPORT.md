# Fixed positive private bootstrap: scientific triage

10 October2026. **Recommendation: conditionally justify one fixed18-fit exploratory screen for root decision. No implementation or launch is authorized by this memo.** The question is whether persistent supervised risk differences acquire useful private predictions at the existing shared-body operating point. Known bagging ingredients can be useful; no novelty, Bayesian calibration or generality claim follows.

## Saved history and the new premise

The exact Exp(1), class-normalized, private-only rule was not found as a completed recipe in the consulted saved conclusions. Close local ancestors must remain explicit:

- [Oct6 positive graph/IID weighting](../graph_error_guided_branch_growth_method_note_20261006_v1/REPORT.md): unexecuted graph-correlated versus IID-draw weights, bounded in[.5,1.5], applied to the joint supervised risk, including shared gradients. It is a close bagging proposal, not a completed rejection of this rule.
- [Oct8 graph-regime allocation](../contrastive_BE_graph_training_rule_synthesis_20261008_v1.md), following the [Oct3 specialization scout](../graph_structure_conditioned_specialization_scout_20261003_v1/REPORT.md): unexecuted class-balanced degree assignments with a half global anchor, **uniform shared CE and weighted private CE**. This already supplies the proposed block-credit architecture. Random Dirichlet weights replace its structured assignment; they do not create a new principle.
- [Oct10 selective-pool triage](../selective_correct_evidence_pool_and_train_support_triage_20261010_v1/REPORT.md): probability-pool CE is own CE reweighted by current true-class responsibilities; its proposed event gate lacked demonstrated TRAIN support. Fixed bootstrap weights neither depend on current predictions nor disappear because that event gate becomes empty. The saved finite-response correction branch is not reopened.
- Saved pooling studies alter readout of acquired predictions. They supply no equivalent test of persistent private TRAIN weighting. Uniform probability serving remains fixed; a strict common wrong rival cannot be rescued merely by reweighting unchanged probabilities.

Root supplied the fully closed24-fit result and joined report SHA256 `ce1f1f23035de274d99eda192ea700d73fce5581daa6d365f20f57466f874422`: all rotation/LoRA/local transfer flags fail. GAT pair and coherent both serve81.1718%, below ordinaryI4's81.9808%; SAGE pair79.3768% versus coherent79.3642% and ordinaryI4 80.0215%. Members approach ordinary-I4 quality, yet coherent coverage is81.35/79.53% versus I4's85.83/85.37%. Pair adds little; original Rademacher shared has broader coverage with weaker members. This motivates changing supervised acquisition. It proves neither a shared-gradient conflict nor that random weights can recover the missing alternatives.

## Exact proposed learning rule and ancestry

For route m and TRAIN class c containing n_c nodes, independently draw positive e_im~Exp(1) once, using a separate frozen RNG, and set

`w_im = n_c e_im / sum_(j in class c) e_jm`.

Thus every route retains every label and exactly the original class mass: `sum_class w_im=n_c`. Here “class-balanced” controls route-to-route class totals; it does **not** replace the native class frequencies with equal-class loss. Within each class, w/n_c has the Dirichlet(1,…,1) law of the [Bayesian bootstrap, Rubin1981](https://doi.org/10.1214/aos/1176345338). The graph, features, labels, serving and validation selection are unchanged; no outcome chooses draws or concentrations.

With N TRAIN nodes, M=4, existing shared block θ and private factors/adapters φ, define

`F = mean_(i,m) CE_im`, `B = mean_(i,m) w_im CE_im`,

`g_θ = partial_θ F`, `g_φ = partial_φ B`.

Both blocks use the **same pre-update forward/dropout realization** and one native AdamW transition with unchanged reduction, learning rate, moments and clock. All existing shared parameters keep native own-CE credit; all declared private r/s and any private operator coordinates receive the weighted credit. No parameter changes ownership. The shared gradient **law** stays native; its later trajectory changes through the learned private functions.

This is generally **not the gradient of a single scalar potential**. Cross-block derivative equality would require `sum_(i,m)(w_im−1) partial²_(θ,φ) CE_im=0` throughout parameter space, which class-mass normalization does not enforce. A globally weighted scalar loss would also weight θ and would change the proposed rule. A stop-gradient implementation encodes the block field, not a scalar-descent guarantee in the original variables. This distinction already exists in the saved Oct8 proposal.

Closest published collision: [Osband et al., Deep Exploration via Bootstrapped DQN](https://ar5iv.labs.arxiv.org/html/1602.04621), §2 and AppendixB, explicitly combines shared networks/private heads, fixed per-example bootstrap masks and Exp(1) mask weights. §6.1 discusses shared-gradient scaling and notes the computational/diversity tradeoff. Its RL targets and conventional trunk credit differ from this private-only class-normalized CE rule; those differences do not establish originality. [BatchEnsemble](https://arxiv.org/abs/2002.06715), §§3.1–3.2, supplies the shared matrices/private fast factors and member-data allocation ancestry. This is an attributed bootstrap adaptation under an already known mixed-block learning policy, not an exact posterior draw of the whole shared GNN.

## Expected benefit, risk and decisive comparison

The plausible benefit is persistent route-specific **label-grounded** pressure on different TRAIN nodes, creating nonidentical summed private gradients even from coherent functions. Graph-pair capacity might turn those pressures into complementary off-diagonal responses. No new inference information or guaranteed common-error repair is supplied.

Strict positivity is insufficient competence protection. Normalized weights can approach zero or concentrate near n_c; there is no uniform half-anchor lower bound. For a fixed parameter state E[w]=1, but parameters subsequently adapt to the weights, so that identity guarantees neither unbiased learned risk nor VALID competence. With only580 labels, concentration, noise amplification, limited private capacity, shared-body homogenization and uniform-pool harms are credible risks. Record every class/route's weight mass, range and effective sample size descriptively; do not tune, clip or redraw after quality. A half-anchor would be a different future rule, not a rescue of this screen.

Reuse the immutable coherent C, graph-pair A and rank1-LoRA L baselines. Acquire only:

| New arm | Decisive contrast |
| --- | --- |
| Coherent + bootstrap B | B−C: can supervised weighting acquire useful alternatives without new capacity? |
| Graph-pair + bootstrap A+B | (A+B)−A and (A+B)−B: does paired capacity help the changed private task? |
| Graph rank1-LoRA + bootstrap L+B | (L+B)−L and (A+B)−(L+B): capacity/control contrast with identical weight tables |

Three arms ×3 paired seeds ×2 fixed backbones = **18 new fits**. Use the same frozen weight table per seed/route across new arms, existing data/config/seed roles, full native schedule and original earliest maximum pooled-accuracy selector. Close all18 before interpretation; no seed/arm continuation from partial quality.

At max1000 updates this represents at most18000 optimizer transitions and72000 TRAIN plus72000 VALID-selection native route trajectories, with72 selected route readouts. Separate shared/private derivatives add reverse-pass/state-retention work; acquisition time and peak memory cannot be assumed equal to the existing one-backward fits. Weight storage is only4×580 scalars per draw table and changes no serving parameters; root still needs the measured complete-step resource check before adopting execution.

For the C/A/B/A+B comparison, report paired accuracy interaction `(A+B)−A−B+C`, but require actual final accuracy against C and capable ordinary/factorizedI4; positive interaction cannot rescue worse final quality. Keep accuracy transfer and member/NLL protection separate. Require useful coverage to translate into repairs under the exact `pool correct=coverage−lost+aggregation-only` identity. Preserve per-seed/class/member metrics, repairs/harms, A/B-only repairs surviving A+B, common-rival changes, paired descriptive intervals and every acquisition/serving cost. Unchanged private architecture for B is the essential control; LoRA tests whether any conditional capacity benefit is specific to pairs.

**Falsifier:** no served accuracy gain, competence/proper-risk harm, or additional coverage erased by pooling closes this fixed rule. More disagreement or oracle coverage alone fails. An18-fit success would justify same-weight competent untiedI4 and conventional all-parameter-bootstrap controls plus unused graph confirmation; the initial screen cannot establish a distinctive private-routing or sharing advantage over those unmeasured controls.

Scope: saved text/conclusions and root's complete summary only; one new scoped primary HTML read(§2, §6.1, AppendixB), reused BatchEnsemble method passages. Rubin's publisher page returned an access challenge; no full-paper-read or exhaustive duplicate-exclusion claim. No model/data/logit/checkpoint imports, numerical execution, source implementation, remote scientific job or TEST access.
