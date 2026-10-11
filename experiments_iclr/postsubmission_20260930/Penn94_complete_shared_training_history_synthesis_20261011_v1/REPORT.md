# Penn94 shared training result and history implications

The complete release closes 70 banks/100 body attempts, 90 fixed CAL endpoints and 90 CONFIRM rows, with zero native or calibration failures. The fixed official split contains 19,407 TRAIN, 4,852 SELECT, 4,851 CAL and 9,705 CONFIRM nodes. Ten optimizer blocks repeatedly read the same CONFIRM nodes: 97,050 dependent readouts, not ten independent graphs. Calibration uses a **separate CAL-role scalar fit**, not WikiCS five-fold crossfit. The original paper scores remain unchanged.

## Complete quality and frozen decision

| Primary procedure and fusion | Raw accuracy % | Mean member accuracy % | Raw NLL | Calibrated NLL |
| --- | ---: | ---: | ---: | ---: |
| Internal shared4 / probabilities | 78.153529 | 76.747553 | 1.013331 | 0.466889 |
| Original boundary shared4 / logits | 79.133436 | 77.644513 | 1.674650 | 0.459772 |
| Matched boundary shared4 / probabilities | 77.861927 | 76.576765 | 1.039658 | 0.472374 |
| Ordinary M1 / probabilities | 74.560536 | 74.560536 | 0.878342 | 0.524340 |
| Internal-factor M1 / probabilities | 74.612056 | 74.612056 | 1.071828 | 0.525665 |
| Genuine internal I4 / probabilities | 75.299330 | 74.587326 | 0.560094 | 0.502748 |
| Fixed LINKX / probabilities | 84.685214 | 84.685214 | 0.624994 | 0.344407 |

Internal sharing gains 2.854199 accuracy points over genuine internal I4, with all ten differences positive, Holm one-sided p=8.930737e−9 and ordinary paired 95% interval [2.546435,3.161963]. Its ordinary/factor-single gains are 3.592993/3.541473 points and also pass their frozen individual contrasts. The **full four-contrast primary criterion fails** because original boundary GNNM is better by 0.979907 points: nine negative blocks/one tie, interval [−1.490781,−0.469034]. The original-comparison correct-count vector is −87/−78/−30/−152/−123/−107/−247/−77/0/−50; the I4 vector is +270/+309/+231/+206/+328/+243/+255/+314/+302/+312. All inferential vectors, simultaneous intervals and 1,024-assignment sensitivity remain in the bound complete record; no tests were recalculated.

LINKX reaches 84.685214%, 6.531685 points above the candidate, with better raw Brier and calibrated NLL/Brier. It remains descriptive task context outside the primary family, but rules out a task-wide superiority headline. The three favorable primary contrasts cannot erase the mandatory original-recipe failure.

## Why the original recipe remains stronger in these predictions

Original members average 77.644513% versus internal sharing's 76.747553%: a 0.896960-point advantage. Its pooling increment is also 0.082947 points larger, completing the 0.979907-point served gap. Internal sharing nevertheless has 152 more covered readouts and 152 fewer strict common rivals. It loses 1,103 more covered alternatives in pooling, leaving −951 correct: +152 coverage −1,103 losses. Direct repairs/harms are 5,140/6,091. Thus extra available alternatives and fewer rival predicates do not compensate for weaker members and worse retention.

This is an accounting explanation, not the learning cause. Original BCE/one-logit acquisition, boundary initialization/private biases, global dropout/RNG, selector law, logit serving and greater maximum update allowance differ from the current CE/two-logit procedure. The observed comparison cannot attribute its lead to boundary placement, training duration or pooling alone. The complete confirmation export does not supply selected/completed update counts; elapsed fitting time cannot substitute for them.

Against I4, internal sharing has **stronger** mean/worst members by 2.160227/2.223596 points, plus about 0.693972 points more pooling lift. Coverage increases 5,047 and losses increase 2,277, leaving +2,770 correct; gross repairs/harms are 10,723/7,953. It serves 5,752 old-I4-rival repairs while creating 4,489 new rival predicates. This differs from WikiCS F/E's weaker-member pattern and agrees with a graph/recipe-dependent acquisition account, without a general shared-gradient mechanism.

Factual class recalls are 74.7435/81.2022 for internal sharing, 71.7114/78.5070 for I4, 74.9814/82.8454 for original GNNM and 81.8162/87.2502 for LINKX (class sizes 4,581/5,124). I4 gains improve both classes; original's −951 net consists of −109/−842, concentrated in class 1. The descriptive matched-boundary +283 consists of +597/−314. These marginals introduce no new endpoint or class-specific rule.

## Placement, serving and confidence limits

Internal sharing exceeds the matched common-CE boundary procedure by 0.291602 points, nine positive/one negative. Coverage +589 minus losses +306 gives +283; repairs/harms are 2,632/2,349. The confirmatory secondary gate is **closed**, so this remains descriptive rather than a confirmed placement result.

For identical selected internal states, probability pooling loses 0.148377 points versus logit pooling (547 repairs/691 harms; unchanged member correctness/coverage/rivals). The logit alternative scores 78.301906%, still below original GNNM; it is not a newly selected replacement pipeline. Matched boundary probability pooling also loses 0.121587 points to its same-state logit alternative. Both alternatives improve raw accuracy at the cost of worse raw NLL/Brier; the frozen primary probability rule cannot be replaced after release.

Internal raw NLL is 1.013331 versus I4's 0.560094, despite better accuracy and Brier. CAL temperature reduces internal NLL to 0.466889 versus I4's 0.502748. All four practical mean/per-block protections pass; original still has better calibrated NLL (0.459772), with internal mean harm +0.007117 and maximum +0.020959. Passing caps is not statistical noninferiority or removal of the raw confidence harm. All calibrated/member/Brier endpoints remain in the complete records.

## Paid acquisition and stopping scopes

| Procedure | Unique body records | Fit-wall sum seconds | Selected serving sum seconds | Maximum allocated GB |
| --- | ---: | ---: | ---: | ---: |
| Internal shared4 | 10 | 4461.374 | 1.355801 | 8.474 |
| Original boundary shared4 | 10 | 14144.841 | 1.353448 | 3.430 |
| Matched boundary shared4 | 10 | 4573.694 | 1.340847 | 7.127 |
| Ordinary M1 | 10 | 391.654 | 0.307202 | 2.612 |
| Internal-factor M1 | 10 | 515.699 | 0.346492 | 3.625 |
| Genuine internal I4 | 40 | 1908.580 | 1.387770 | 3.638 |
| Fixed LINKX | 10 | 428.471 | 0.270565 | 1.304 |

These are 100 unique body scopes; alternate pooling reuses states and must not duplicate acquisition cost. The I4 peak describes one separately fitted body, not four concurrent bodies. Fitting includes differing stopping/selector/checkpoint opportunities; I4 constructor time outside fit-wall is separately 2.054163 seconds. Original's longer sum accompanies a greater update allowance and is not a budget-matched causal control. Internal allocated peak is 8.474 GB versus original's 3.430 GB and one I4 body's 3.638 GB. Lifetime RSS peaks are inherited process scopes, not per-arm allocations; stored parameter counts are not supplied in this confirmation export. CAL/CONFIRM owner wall times are 89.644/12.568 seconds. Costs overlap the unchanged Physics work on the GPU and establish no isolated latency, compute or memory-saving claim.

## Three implications

1. **Retain the positive shared-acquisition result and the failed whole criterion together.** Penn supports internal shared SAGE against its genuine I4/singles under this fixed graph/recipe. It does not establish improvement over original GNNM or LINKX, and does not test conditional E.
2. **Keep original GNNM and capable task context mandatory.** Mean-member competence, useful alternatives and served retention must be reported jointly. The original lead, closed placement gate and same-state pooling tradeoff prevent selecting a favorable contrast as the mechanism. No solver, duration, loss, pooling or class rescue grid follows.
3. **Carry this complete evidence into the later synthesis after remaining whole-study releases.** Calibration improves the confidence operating point while costs and stronger references narrow the claim. Shared-training member strength varies across Roman, WikiCS and Penn; no global competence guarantee, absent-information conclusion, new primitive or manuscript verdict is established. V4 and the contingent V5 outline remain sealed and unchanged.

`SUMMARY.json` preserves raw/calibrated means, all dependent bookkeeping, unique costs, frozen inference and safeguards. `evidence/` holds exact complete-release/config/closure copies and root authority tables. `RECORD_REFERENCES.json` binds every source. No prediction/state arrays, server, Physics/FiLM partial scores, new fit, reviewer or paper edit were used.
