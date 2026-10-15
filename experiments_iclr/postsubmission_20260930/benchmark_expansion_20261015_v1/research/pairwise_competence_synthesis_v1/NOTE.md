# One complementary hypothesis: specialize on supervised ranking challenges

15 October 2026. Prospective synthesis only. **Test persistent, positive supervision of native positive–negative score differences**, using different bond-composition similarity regimes for the four existing residual routes. This is an attributed combination of pairwise AUC learning and persistent risk weighting. No implementation, fit, teacher, new backbone, HPO, or change to the running structural pilot follows.

## Why this question

Residual-only/private-BN restores mean route AUC to **74.560%**, but mean-logit pooling adds **−.043 percentage points**; **95.54%** of pooled reversals are wrong in every route. Pool AUC is **74.518% versus genuine I4 74.847%**, with I4 winning two of three seeds. BCE is also much worse, **1.952 versus .890**. These are selected results on exposed VALID, not a diagnosis that the model lacks particular chemical features. [E1]

The proposed change directly trains the ordering that AUC evaluates. Different routes emphasize comparisons between molecules with different degrees of similarity in bond composition. A common reversed TRAIN pair gives a nonzero correction to every route and stronger correction to its designated route. It does not reward disagreement, incorrect confidence, or a hidden direction that the native output ignores. The hypothesis is that finite private capacity learns useful conditional ranking skills; bond composition may instead be irrelevant, noisy, or too coarse.

## Concrete target and update

Keep the exact residual-only native GIN, private BN policy, native inputs/operators, optimizer, dropout, sampler, controller and raw-logit mean serving. Let `z_m(i)` be route m's native binary logit. For every ordinary TRAIN minibatch B, form only distinct observed-label positive–negative pairs:

```text
P_B = {(i,j): i,j in B, y_i=1, y_j=0}.
d_mij = z_m(i) - z_m(j).
ell_mij = softplus(-d_mij).
R_w(B) = mean_m [ sum_(i,j in P_B) a_m(i,j) ell_mij / |P_B| ].
J(B) = F(B) + 0.5 R_w(B).
```

F is the unchanged native mean own-route BCE. **All shared and private learned parameters receive the gradient of this one scalar J**, followed by one ordinary optimizer step. The coefficient .5 is one prospective fixed choice, not a derived optimum or a search. If `P_B` is empty, set `R_w=0` and use exactly F; do not resample, carry stale logits, renormalize F, or skip an otherwise eligible native update. Missing labels follow the native mask. With batch size 32 there are at most 256 pairs per route, using the same four dropout forwards and one backward. Extra pair arithmetic, preprocessing and memory must be measured; parameter count and inference computation are unchanged.

**Freeze roles using TRAIN inputs only.** Reuse the already specified per-field native bond-category frequency vector b(i), with actual directed bond records and the native vocabulary [S2]. Normalize each nonzero vector; define similarity `s(i,j)=cos(b(i),b(j))`, with zero similarity when either vector is zero. Over **all unordered distinct TRAIN input pairs, including same-label pairs**, construct four equal-mass similarity quartiles with fractional ties. Freeze the resulting score-to-membership rule q_m(s), without consulting labels, predictions or VALID. Set

```text
a_m(i,j) = 0.5 + 2 q_m(s(i,j)).
```

Thus every pair has positive weight in [.5,2.5], weights are symmetric under swapping inputs, and their mean over routes is exactly one. Equal route mass over all input pairs does **not** imply equal mass over opposite-label pairs or equal per-molecule/class mass; report those masses and pair-support counts. Do not normalize each route by its observed minibatch weight sum. This is a new pair-role definition, not the current within-class atom-count table. No similarity target is required at inference.

The full TRAIN empirical target averages these losses over all observed positive–negative pairs. Implementation would use the within-minibatch reduction above, never full-pair model forwards. Under uniform shuffled batches and fixed deterministic scores, symmetry makes the expectation of the pair term proportional to that full-pair average, with the common factor equal to the chance a batch contains both classes. Actual stochastic dropout/BN scores depend on the batch, so this statement is **not** an exact claim that native SGD optimizes the deterministic full-pair risk. Preserve this distinction and report empty/support distributions; a rare-positive port cannot silently change the sampler.

## Target compatibility, not a calibration guarantee

For two independent population examples, write `A=eta(x)[1-eta(x')]`, `B=eta(x')[1-eta(x)]`. For a fixed positive symmetric input-only pair weight a, the conditional logistic ranking risk is

```text
C(d) = a [A softplus(-d) + B softplus(d)],
C'(d) = a [(A+B) sigmoid(d) - A],
d* = log(A/B) = logit eta(x) - logit eta(x').
```

For interior conditional probabilities, the same score function `z=logit eta` minimizes every such pair risk and ordinary BCE. BCE also fixes the additive offset unidentified by ranking alone. This elementary compatibility differs from observed-class-dependent pointwise weights, which can shift the posterior by `log(w1/w0)` [E2]. It does not certify the fitted GNN, data-derived quartiles, finite shared/private capacity, regularization, native BN/dropout, optimization, or selected-state calibration. Separable empirical pairs can still encourage excessive margins. An AUC gain with worse BCE is a ranking/probability trade-off; an NLL-only gain does not meet the task objective.

Gao and Zhou establish pairwise logistic loss as AUC-consistent over the stated unrestricted population problem [P1]. Their result is not a theorem for this weighted finite shared neural procedure, and consistency does not make it superior to native BCE, whose ideal posterior already gives the correct ranking.

## Necessary comparisons and rejection

Use one prospectively frozen MolBACE/GIN comparison, retaining the existing native-BCE M1/I4 and exact residual-F bank:

| Procedure | Purpose |
| --- | --- |
| Same residual bank, common pair loss (`a=1`) | Whether direct ranking supervision itself helps. |
| Same bank, structured pair roles above | The one candidate. |
| Same bank, one fixed permutation of whole four-weight vectors among observed opposite-label TRAIN pairs | Preserve the exact pair-vector multiset, route totals and every pair's mean weight while disrupting similarity association. This is a finite empirical control, not a new target-preservation theorem. Use fit seed +7030031 once; no redraw. |
| Native M1/I4 trained with the **same common pairwise auxiliary** | Necessary eventual attribution controls: each native body uses its own BCE plus .5 times its own common pair loss, same native minibatch rule/controller/own selector. M1 is I4 body 0. They are controls, never acquired teachers. |

All shared arms retain four route forwards; no zero-residual control is needed for this objective comparison. Select once with the unchanged AUC rule; serve mean logits and report BCE at that checkpoint. Retain original GNNM/native-BCE references. Three seeds would require nine new shared fits; pairwise-native I4 adds twelve independently trained native bodies, with its M1 reused. These are prospective costs, not authorized jobs. A win over BCE-I4 but not pairwise-I4 supports an objective benefit, not wrapper-specific advantage. Even a win over pairwise-I4 compares complete procedures; isolating the causal effect of sharing itself would additionally require independent members with the same structured roles, which is outside this bounded proposal.

**Reject the fixed structural-pair recipe** if its mean served AUC does not improve over residual F, common pair training and the matched permutation; do not retune bins, similarity, coefficient or selector. A tie with permutation leaves the structural explanation unsupported. Advancement toward the main goal additionally requires beating genuine native-BCE I4 and the common-pairwise I4 control, with every paired seed and proper-loss harm visible. Three exposed-VALID seeds supply exploratory evidence, not a superiority certificate.

The stronger **competence plus useful complementarity** explanation needs mean/worst route AUC/BCE, absolute common-reversal rate, acquired/lost correct alternatives, pooling gain, and actual served repairs minus harms. A drop in mean route AUC is a competence trade-off; requiring every route to beat M1 would be unnecessary. Lower common-error *fraction*, greater disagreement, or better TRAIN margins alone do not establish improvement. If only members improve with no pooling benefit, report a competence result rather than complementary skills. Pairs sharing molecules are dependent; they do not enlarge the number of experimental repetitions.

## What this repeats and what it changes

| Retained family | Comparison with this proposal |
| --- | --- |
| Inventory 3, 10–13, 22; current structural pilot | Reuses positive persistent weighting. Changes the supervised unit from an example/query to an actual positive–negative ordering, uses input-only symmetric pair roles, and trains all blocks with a scalar objective. No new weighting principle is claimed. |
| Inventory 4–9, 14–15, 21; completed SupCon/alignment studies | Those supervise embedding geometry, same-query contrast, or factual donor distributions. This loss acts on the existing native task-score difference. The same bond histogram is reused as a role descriptor; it is not a new chemical representation. Old contrastive negative results remain. |
| Inventory 1, 16–17, 25; initialization history | Retains the completed residual-only/private-BN producer. No new start scale, factor, adapter site, or normalization search. The Rademacher-versus-unit competence/coverage trade-off is not reopened. |
| Inventory 2, 18–20, 23–24; pooling/gating work | Neither pooled BCE nor learned serving weights. It attempts to change wrong member predictions before averaging. Completed fixed-readout results and convex-gate bounds motivate this priority; they do not prove every conditional gate fails. |
| Retained sMCL/CMCL, GNCL and staged boosting | No minimum-loss winner assignment, confidence suppression, own/pool interpolation, frozen teacher, or sequential residual learner. Standard assignment/ensemble-risk ancestry remains; calling an ordinary pool gradient a residual would repeat earlier work. |

Closest verified task-loss ancestry is **pairwise logistic AUC optimization** [P1]; closest retained route-training ancestry is **persistent weighted expert supervision**, including ACE and sMCL with their different recipients/assignments [E3,E4]. The proposed composition is untested and receives no novelty claim. AUC is the task-specific reason for this choice: **do not port it unchanged to PubMed's multiclass accuracy or sampled-negative link tasks**. The separate PubMed structural plan remains untouched.

### Sources and scope

- **P1:** Wei Gao and Zhi-Hua Zhou, [*On the Consistency of AUC Pairwise Optimization*, arXiv:1208.0645v4](https://arxiv.org/html/1208.0645v4), §2 and §3.2, especially conditional risk and Corollary 2. One new scoped primary read; no full-paper, code, or proof audit. Exact incidental scope is in `READ_SCOPE.json`.
- **E1:** Residual-only `complete_results_v1/INTERPRETATION.md`; completed scalar/interpretive evidence, including gate-feasibility bounds.
- **E2:** `quality_specialization_synthesis_v1/SYNTHESIS.md` and `WEIGHTED_TARGET_OPTIONS.md`.
- **E3:** Retained ACE/decoupling, RIDE/SADE and quality-preserving-diversity notes; prior scoped findings reused.
- **E4:** v72 record for sMCL, `contrastive_member_geometry_user_clarification_20261010.md`, staged-boosting triage, and `next_quality_mechanism_20261015_v1/RECOMMENDATION.md`.
- **S1:** `benchmark_review_map_v1/README.md`, all 25 recipe definitions.
- **S2:** `context_affinity_molecular_port_v1/README.md` and targeted histogram-source passages; no import/execution or graph records read.

Exact local paths/hashes are in `SOURCES.json`. No partial pilot outcomes, tensors, checkpoints, TEST payloads, scientific remote hosts, Git operations or models were accessed.
