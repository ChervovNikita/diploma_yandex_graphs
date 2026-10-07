# Live own-risk trunk, internal pool-risk factors: operational prior audit

7 October 2026. Source interpretation and attributed control preparation only. Saved scopes were reused before two genuinely unread decisive scopes were inspected: Ensemble++ v6's relevant method and GNCL's pinned author implementation. No scientific implementation, model/data access, experiment outcomes, training, GPU/server work, or strength choice.

## Finding

**The objective ingredients and block-dependent supervision have direct prior and project ancestry.** GNCL supplies the actual member/pool-risk mixture. Its pinned implementation differentiates one scalar mixture through all parameters; it does not implement the requested three-way permission split. Ensemble++ supplies a particularly clear live shared feature extractor with a detached private branch and different branch targets. Its private uncertainty heads are output heads, and their squared-loss targets are random perturbations, not the actual task pool risk.

The earlier project scout already contains **shared own/private pool** as the reverse control. The present recipe equals that policy at λ=1 when every private block is eligible and its pool, reductions and update contract match. Keeping private input/classifier blocks on own risk while directing the actual pool/GNCL gradient only into internal member factors is the narrower allocation under consideration.

A complete published equivalent was **not established in these inspected scopes**. This is a bounded evidence statement, not a novelty claim, an exhaustive search, or an absence certificate. The testable remaining question is whether this allocation helps the declared pool in the existing graph architecture. Its utility is unmeasured.

## 1. Exact recipe being compared

Partition the existing live model into shared parameters θ, internal member factors φ, and other private parameters ψ, including private input and classifier blocks. Use the declared supervised loss and the **actual task pool**:

```
L_own = mean_m L_m
J_λ = (1−λ)L_own + λL_pool,  λ∈[0,1]
g_θ = ∂_θ L_own
g_ψ = ∂_ψ L_own
g_φ = ∂_φ J_λ
```

The three derivatives use the same pre-update model state and the same supervision/reductions. “Private own risk” retains the slice of the mean-member loss, including its member normalization. No numeric λ is adopted. At λ=0 these supervised paths are own/own; parameter tying remains. At λ=1 internal factors receive actual pool-risk gradients.

The full comparison requires all of the following:

| Requirement | Meaning |
|---|---|
| Live shared trunk | Shared weights keep learning from mean individual risk. No pretrained or frozen trunk is substituted. |
| Internal member adaptation | Eligible existing internal adapters/factors receive pool risk or the GNCL mixture. They are not merely terminal uncertainty heads. |
| Own-risk boundaries | Existing private inputs/classifiers receive mean individual risk, without pool supervision. |
| Actual serving contract | Probability averaging, mean raw logits, target masks and task reductions remain the declared task's choices. A fusion classifier or random sampling predictor is a different pool. |
| True private pullback | Hold θ and ψ fixed for the φ derivative while differentiating through the real shared operations. Detaching an upstream hidden tensor can erase gradients to earlier eligible factors. Include any true cross-member dependencies. |
| Training state | One intended same-state optimizer transition; no teacher, distillation, router or extra frozen feature model. Optimizer/group/state and stochastic-view semantics require later source qualification. |

The existing alignment auxiliary, if retained in a matched control, keeps its separately declared permissions; it does not change the supervised recipe above. This audit does not revise its strength or the running study.

## 2. Reused overlap: the block split already exists

The [3 October four-policy scout](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/graph_mixed_block_objectives_scout_20261003_v1/REPORT.md>) explicitly compares own/own, pool/pool, pool/own, and **own/pool**. In that scout all route-local private factors are eligible and the pool is mean raw logits. Its reverse own/pool row is exactly the supervised λ=1 case above if φ contains all private parameters, ψ is empty, and the same mean-logit pool, reductions and update contract apply. The current proposal's boundary exclusions, task pool contract and internal placements must therefore be stated explicitly; a changed block name does not create a new training principle.

The [complete ONE/PCL method audit](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/mixed_block_objective_closest_priors_20261003_v1/REPORT.md>) is reused without reopening its primary texts. Both have live shared/private feature blocks and individual plus ensemble supervision. PCL's ensemble CE bypasses individual classifiers, reaching shared/private features and a separate fusion classifier. However, shared features also receive ensemble CE, private features receive both hard losses, and distillation/EMA states remain. ONE also gives private branches ensemble CE and includes distillation/gating. Neither retained complete method equals the requested teacher-free allocation.

Saved BE/TabM and graph-factor conclusions provide shared/private architecture ancestry. Existing private nonlinear or attention factors can change graph-evidence dependence; that possibility is not an effect established for this loss allocation.

## 3. New GNCL author-code scope: one global mixture

Buschjäger, Pfahler and Morik, **Generalized Negative Correlation Learning for Deep Ensembling**, `2011.02952v2`. The saved §4.1 Eq.5 and λ range are reused. Its abstract names the [author repository](https://github.com/sbuschjaeger/gncl). This audit pins the main repository to `b76664c9d585964f9dcede00265263cb7a545654` (2021-02-10), then resolves its `.gitmodules` and gitlink to **Pysembles `8858c6466e0897d8c74def7400898594c2a4de70`**. Repository HEAD was only a locator; all decisive code is pinned.

| Pinned code | Direct evidence |
|---|---|
| `GNCLClassifier.py:59–67` | Constructs a `ModuleList` from repeated calls to the base-estimator factory. No internal/boundary permission partition is introduced. |
| `GNCLClassifier.py:80–143` | `mode="upper"` uses `reg_loss = λ f_loss/M + (1−λ)i_loss/M` for every member, then returns `losses.sum(dim=1)` for backward. Summing gives λL_pool+(1−λ)L_own. |
| `Models.py:511–524` | The average combiner directly averages base outputs; no pooled-path detachment appears. Whether outputs are logits follows the supplied estimator/task, not the combiner name alone. |
| `Models.py:322–407` | Optimizer receives `self.parameters()`. Each batch clears gradients, obtains the scalar batch mean, calls one global backward, then one scaler/optimizer step. No loss-specific parameter masks, separate internal derivatives, or filtered shared gradients occur in this path. |
| `FASHION/run.py:176–187,339–350` | The configured factory constructs fresh SimpleResNet objects; GNCL uses `mode="upper"`, `combination_type="average"`, and unreduced cross-entropy. This is static configuration evidence; no data constructor or fit was executed. |
| `E2EEnsembleClassifier.py:41–65` | Uses pool loss for backward. Individual losses are detached metrics, not private own-risk updates. This corroborates the pool-only limiting path; it is not another publication identity. |

Thus the inspected author implementation supports the GNCL scalar objective and its loss reductions. It does **not** implement a live shared trunk on own risk with a different mixture restricted to internal member factors. If a user supplied tied estimator parameters, the same scalar backward would reach those shared parameters wherever its computation graph permits; tying alone would not impose the requested routing.

The exact/Taylor mode was visible in the same class but is not adopted as the desired actual-pool mixture. No claim about its empirical outcomes or code correctness follows. No import, installation or runtime execution was used.

## 4. New Ensemble++ v6 scope: direct live-trunk separation, different purpose

Li, Xu, Wang and Luo, **Scalable Exploration via Ensemble++**, [arXiv `2407.13195v6`](https://arxiv.org/abs/2407.13195v6), revised 28 October 2025. Version 1 has a different title, *Adaptive Foundation Models for Online Decisions: HyperAgent with Fast Incremental Uncertainty Estimation*. Version 1 was used for identity/formulation location only and is not counted as a second method read. The v6 page says NeurIPS 2025; equality to an accepted proceedings version was not audited.

The new relevant scope is §§3.2–3.3 and Appendix B's architecture/objective/gradient passages. Appendix B Eq.12 explicitly uses

```
f++(x,ζ) = ψ(h(x;w);b)
          + ψ(sg(h(x;w)); Σ_m ζ_m θ_m)
          + sg(ψ(h(x;w); Σ_m ζ_m θ0,m)).
```

The feature extractor w and base head b are trainable. θ_m are trainable uncertainty output heads; θ0,m are fixed prior heads. Stop-gradient on the uncertainty features blocks that branch's upstream feature gradient; stopping the complete prior output blocks its feature path too.

Under the symmetric ± perturbation **squared-loss** objective, the paper cancels cross terms and writes a base true-target term plus a detached-feature uncertainty-fitting term (Eqs.13–14). Its text and Eqs.15–17 assign the data gradient for w,b to base risk and θ_m to perturbation residuals. This is strong direct precedent for a live representation and branch-specific supervised/auxiliary gradient exposure. It is not a frozen trunk method.

The complete target differs in decisive ways:

- Private θ_m sit at output, not inside graph feature/message operations.
- Their targets are random uncertainty perturbations after a fixed-prior contribution, not supervised risk of the actual uniform ensemble pool.
- The mean/base predictor, fixed priors and random linear-combination sampling are material architecture/serving choices.
- No separate private input/classifier own-risk partition is specified.

**Scope qualification:** the printed squared-loss normalization is not fully reconciled. Eq.13's expansion gives twice Eq.14's data term as printed, and Eq.17 omits Eq.14's private 1/M. The displayed derivative formulas also omit Φ's regularizer contribution. Use the source for data-path separation, not an exact full optimizer/scaling equivalence.

Appendix B Eq.18 replaces the symmetric squared objective with CE on perturbed full predictions. Its detached uncertainty features remain, but the live base-path CE residual depends on those perturbed predictions. Squared-loss cancellation does not establish base-only unperturbed CE risk. No author code for Ensemble++ was audited, so neither CE update routing beyond the printed graph nor numerical scale is implementation-qualified. Performance and computation claims were not adopted.

## 5. Other locators, access limits and accounting

WELDE, DOI [10.1007/s13721-026-00797-1](https://doi.org/10.1007/s13721-026-00797-1), is an **abstract-only exclusion**: the publisher explicitly says each loss has an adapter/head from a “shared frozen backbone.” It does not meet the live-trunk requirement. Main methods were paywalled; accessible appendix snippets and incidental abstract results are not counted as a complete method or used as performance evidence.

Bits-Ensemble, DOI [10.1109/TCAD.2022.3197986](https://doi.org/10.1109/TCAD.2022.3197986), remains unresolved. The author-deposited PDF returned HTTP403; the exact-title arXiv query had no hit. Metadata about shared/private parameter bits cannot establish loss routing. This access failure supplies no absence evidence.

New decisive scopes: **1 primary-paper method scope + 1 pinned author-code scope = 2**. New whole-paper/proof reads: **0**. No separate credit for Ensemble++ v1, GNCL implementation files, WELDE's abstract, search records, or failed Bits access. Previously read GNCL paper/ONE/PCL/BE/TabM methods were reused, not recounted. Incidental source-level result prose was not adopted. New performance claims, scientific executions and outcome inspections: **0**. No additional third scope was forced after the two decisive scopes answered the routing question.

## 6. Remaining graph/internal adaptation delta

The defensible next object is an **attributed, fully specified allocation control**: existing live shared graph weights and private boundaries keep mean individual supervised risk; only existing internal factors receive the actual-pool/GNCL pullback. Its question is whether that permission choice improves the declared served prediction compared with controls using the same permissions and architecture. The matched hidden-private control remains necessary because the running hidden-full arm also exposes shared weights to auxiliaries.

Graph-specific interpretation requires the actual internal placements: for example, factors before nonlinear gates or recomputed attention can change message/evidence dependence, while terminal classifier changes alone cannot demonstrate that mechanism. The earlier operator audit supplies that architectural distinction; this packet establishes no new graph-diversity principle, useful member specialization, generalization or gain. True pullback, same-state update and exact pool/reductions are requirements of the comparison, not guarantees of success.

Builder may prepare attributed source controls. This packet authorizes no experiment admission, coefficient selection, new factor/head/router, scientific code edit, dataset or fitted-state read, canonical-memory edit, or change to the running family. Source evidence, read boundaries, reused byte bindings and the seal accompany this report.
