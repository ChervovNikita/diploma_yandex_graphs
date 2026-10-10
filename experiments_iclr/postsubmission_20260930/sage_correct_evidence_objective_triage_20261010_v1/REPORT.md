# Preserve and use correct evidence: SAGE objective triage

**Decision:** retain at most one inactive SAGE adaptation of known pooled-CE credit. Target incorrect private routes only where TRAIN already contains a correct route but the served pool is wrong. No new loss or ensemble principle is claimed. Do not launch shared-harm feedback from this finding.

## Complete evidence changes the target

The authorized SAGE readout is complete:21 groups/39 fits, three seeds on the same5,274-node development role, TEST closed. Unchanged shared4 has80.571356% mean accuracy versus80.084692% for genuine factorized I4, but weaker mean members and worse NLL. Exchange raises mean member accuracy yet falls to79.288333% and fails the frozen quality gates.

Across the three seed readouts, shared4 supplies13,740 correct-node alternatives, loses993 in pooling and has one aggregation-only correct answer; exchange supplies13,032, loses487 and has no aggregation-only rescue. The resulting203 fewer served correct answers decompose as708 fewer covered observations,506 fewer pooling losses, and one lost aggregation-only answer. These are repeated readouts of the same graph, not15,822 independent nodes. Better average members and fewer lost alternatives did not compensate for destroyed coverage. Correct supply, conversion and net final quality must be reported together.

## Closest objective collisions

| Primary scope | What is already established |
|---|---|
| [sMCL1606.07839v1](https://arxiv.org/html/1606.07839v1), retained §3 | Oracle minimum member loss and winner-only gradients explicitly train coverage/specialization. Oracle accuracy is not uniform-pool accuracy. |
| [GNCL2011.02952v2](https://arxiv.org/html/2011.02952v2#S4.E5), retained §4.1/Eq5 | Interpolating own and served-ensemble risk is known. Label-free embedding repulsion is unnecessary for this objective family. |
| [Learner Collusion2301.11323v1](https://arxiv.org/html/2301.11323v1), retained §5/Eq3/AppendixE | Score-average CE and probability-mixture CE differ. Probability mixture gives true-class responsibilities `p_m(y)/Σp_j(y)`; mean-logit CE gives a common pooled residual through each route's Jacobian. Own-risk terms and compensating members are direct prior. |
| Saved [CMCL1706.03475v2](https://arxiv.org/abs/1706.03475v2), method/Algorithm1 | Correct owners plus uniform-to-predictive KL for nonowners address confident wrong specialists overwhelming available evidence. Private-only CMCL and pooled-risk placement are already saved adaptations. |
| **New** [Auxiliary Class Based MCL2108.02949v1](https://arxiv.org/html/2108.02949v1), complete §3/Eqs4–6 | Nonowners predict an auxiliary reject class; remove it before averaging and normalize afterward. Early ownership counts later fix class specialties. A learned fusion of early member features is broadcast back to continuing networks. Coverage conversion, confidence suppression, persistent assignment and continuing ensemble communication all have direct prior. |

The new AMCL source was previously metadata-only here. Its output space/combiner, class-memory schedule and fused feature path differ from our native C-class raw-logit bank. Its image claims are not graph evidence. It also warns that specialists without general information generalize poorly. No source reproduction or complete paper/proof/result audit is supplied.

## One specific conditional adaptation

Use unchanged full-graph shared4, no exchange and no teacher. Let F be the native mean of four own TRAIN CEs, `z̄_i=¼Σ_m z_mi`, and stop the two current TRAIN masks:

```text
A_i = 1[ANY route predicts y_i AND argmax(z̄_i)≠y_i]
R_im = 1[argmax(z_mi)≠y_i].
```

Shared parameters receive only `∂θF`. Every private row retains its native `∂φ_mF` and gets added mean-logit pooled-CE credit only on `A_i R_im`:

```text
g_φm = ∂φ_m F + (λ/T) Σ_i A_i R_im ∂φ_m CE(q_mi,y_i)
q_mi = [z_mi + Σ_(j≠m) stopgrad(z_ji)]/4.
```

Exclude auxiliary shared gradients while retaining the true private pullback through the live current graph. Set **λ=.5 prospectively**, with no coefficient search; use fixed TRAIN count T, not selected-mask size. On a qualifying node an incorrect route receives added cotangent `λ[softmax(z̄_i)−e_y]/(4T)`. Correct suppliers receive no direct auxiliary private gradient. Their original own supervision remains; shared movement and other-node private updates can still change their functions.

This is label-dependent sample/recipient masking of established pool risk, not oracle inference, an independent teacher, new graph information or a hard coverage guarantee. It targets **use of available correct evidence**, not acquisition on all-wrong nodes. If TRAIN rapidly has no covered-but-lost events, it becomes the native rule and supplies no remedy for remaining development errors. Masks from VALID must never enter training.

## Decisive control and falsifier

Compare against the **same private-only GNCL field on all TRAIN nodes/all routes**, `gθ=∂θF`, `gφ=∂φ(F+.5 L_pool)`, with identical pool, coefficient, own-gradient floor, data, initialization and selector. This controls ordinary ensemble supervision; it does not independently attribute event selection versus recipient selection. `F+.5L_pool` is a scaled GNCL mixture—its scale/state convention must be matched explicitly, not declared Adam-invariant. Retain native shared4 and genuine factorized I4 as practical references. Saved literal CMCL/AMCL remain attribution, not extra proposed arms.

Prediction: reduce lost alternatives while retaining useful correct-node coverage, improving net served accuracy/NLL across complete paired seeds. More correct mean members, oracle gain alone, lower entropy or fewer pooling losses accompanied by coverage collapse fails. If all-TRAIN private GNCL matches, selective credit is unnecessary. Record full-population repairs/harms and correct-supplier retention; root locks quantitative criteria before any future fit. This development pattern supplies no unused confirmation.

Training may add up to four private backward trajectories/update. Keeping all native tapes can reuse the four forwards; sequential execution may require up to four recomputed forwards. Holding tapes or recomputing is paid and unqualified. Serving is unchanged. No new source, job, owner, grid or partial GCN/GAT/77 outcome was accessed.

## Reading credit

Saved conclusions first; one new scoped primary method identity (AMCL§3), three retained primary revisits (sMCL, GNCL, collusion), and saved CMCL/placement conclusions reused. Two bounded recent metadata queries yielded no selected new method; no absence inference. Zero full papers, author-code/proof/result audits or numerical executions. AMCL method prose incidentally exposes example/result numbers; they are unused. The only new project outcome read is the explicitly authorized complete SAGE scalar/count analysis. Exact source hashes and scopes accompany this memo.
