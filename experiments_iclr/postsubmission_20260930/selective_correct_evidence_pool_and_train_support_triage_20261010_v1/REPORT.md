# Selective correct-evidence credit: pooling and TRAIN support correction

**Decision: no implementation or fit nomination.** The deployed SAGE bank averages probabilities. With independent private rows, its mixture CE adds only per-node scalar weights to each row's own CE gradient. The proposed TRAIN event gate has no demonstrated support through training. Preserve the earlier sealed proposal as a superseded raw-logit adaptation; it is not a qualified recipe for this bank.

## Reducer correction

The completed SAGE family and its pre-fit `FREEZE.json` both state `serving: probability_mean`. The freeze explicitly supersedes the unfrozen raw-logit proposal. The earlier `sage_correct_evidence_objective_triage_20261010_v1` mistakenly described raw-logit averaging as deployed. Its objective, gate and cost sketch do not transfer unchanged. This correction changes no source, score, saved outcome or sealed packet.

Let `p_m = softmax(z_m)`, `ell_m = -log p_m(y)` and `e_y` be the truth vector. For one node:

| Quantity | Probability mean | Mean raw logits |
|---|---|---|
| Pool | `pbar = (1/M) sum_m p_m` | `s = softmax((1/M) sum_m z_m)` |
| CE | `Lmix = -log pbar(y)` | `Lscore = -log s(y)` |
| Route-m logit gradient | `r_m (p_m - e_y)` | `(s - e_y)/M` |
| Responsibility | `r_m = p_m(y) / sum_j p_j(y)` | No own-CE responsibility identity in general multiclass |

For mixture CE, the factor `1/M` cancels between the derivative of the mean and its likelihood denominator:

```text
dLmix/dz_m = -[1 / sum_j p_j(y)] dp_m(y)/dz_m
           = r_m (p_m - e_y).
```

The raw-logit pool is the normalized geometric mean of member probabilities. It generally changes both the residual direction and the classification gate. A constructed binary example has true-class probabilities `(.8,.8,.8,.001)`: their arithmetic mean is `.60025`, so the deployed pool is correct; the geometric pool has odds `(64/999)^(1/4) < 1`, so it is wrong. Reversing the confidence pattern to `(.999,.2,.2,.2)` makes the arithmetic pool wrong and the geometric pool correct. These are algebraic examples, not project predictions. Binary residuals have one free direction; that fact does not make the two reducers equivalent. In multiclass, even the wrong-class component ratios generally differ, preventing scalar proportionality.

## Actual private law

Use the unchanged bank, with `z_m = z_m(theta, phi_m)` and no dependence of another route on `phi_m`. Hold `theta` fixed when taking the auxiliary private derivative. Write the native own risk as

```text
F = (1/(MT)) sum_i,m ell_im.
A_i = 1[ANY route predicts y_i AND argmax(pbar_i) != y_i].
R_im = 1[argmax(p_mi) != y_i].
```

Stop the masks. The corrected private field would be exactly

```text
g_phi_m = (1/T) sum_i [1/M + lambda A_i R_im r_im] d ell_im/d phi_m,
g_theta = dF/dtheta.
```

Thus the direct auxiliary private contribution has the same direction as own CE at each node. For the earlier prospective `M=4, lambda=.5`, its weight is `.25 + .5 A_i R_im r_im`: a multiplier `1 + 2 A_i R_im r_im` on the native per-node private contribution, between one and three. We do not select that coefficient here. Different node weights can change the summed parameter direction, but no new per-node corrective direction or graph evidence is supplied. A confidently wrong route with tiny true-class probability gets tiny responsibility credit. Correct suppliers receive zero direct auxiliary private credit; shared updates and private updates from other nodes still change their functions.

The original mixture CE already has this derivative. To reproduce it with a hand-built weighted own-CE expression, stop `r_im`; differentiating through the weight adds terms. With identical masks, weights, parameter blocks, scaling and optimizer state, the stopped own-CE field is an algebraic identity, not a separate scientific mechanism. Cross-route private coupling would invalidate the single-row identity. Native shared gradients continue during training; “fixed shared core” here concerns this partial derivative, not a frozen training trajectory.

All-TRAIN/all-route private GNCL has weights `1/M + lambda r_im`. The selective variant changes only which own-CE contributions get these weights. If the shared block also received mixture credit, its derivative would be `sum_m r_m J_theta,m^T(p_m-e_y)` per node; the proposed policy excludes that term.

## A support bound, not a decay measurement

Use natural-log own CE and the same TRAIN nodes, parameters and forward realization as the masks. A wrong route satisfies `p_m(y) <= 1/2`, including a wrong tie under the chosen argmax rule. If `B` counts wrong route-node entries,

```text
B <= MT F / log(2).
```

A stronger bound applies directly to this probability gate. If the pool is wrong, another class has at least its true-class mass, so `pbar(y) <= 1/2`. Jensen gives

```text
(1/M) sum_m ell_im >= -log pbar_i(y) >= log(2).
|A| <= number of pooled-wrong TRAIN nodes <= T F / log(2).
```

Clip at `T` and take the integer floor. No independence assumption is used. Each covered node has at least one correct route, so for four routes the wrong-recipient count is at most `3|A|`. For the task's stated `T=580`, `F=.01` permits at most eight gate nodes; `.005` permits at most four; `F < log(2)/580`, approximately `.00119508`, guarantees an empty gate. These are illustrative thresholds, not observed SAGE CEs. Large CE supplies no positive lower bound on gate occupancy. Small support alone does not bound private parameter movement without Jacobian and step control.

The eligible completed local family has VALID selection/restoration scalars and counts but no TRAIN own-CE trace. The local report says traces remain on the server; this check did not retrieve them. Therefore neither terminal extinction nor rapid extinction is established. VALID NLL cannot substitute for TRAIN CE. A dropout training CE cannot certify an evaluation-mode gate, or a post-update/stale gate, without matching the forward convention. Even an empty TRAIN gate would leave the measured VALID losses in pooling unexplained.

## Revised next decision and source scope

Retain the complete SAGE finding: exchange reduced useful correct-node coverage and final quality despite better mean member accuracy; fewer losses during pooling alone were insufficient. That finding does not establish the availability or usefulness of this TRAIN gate. Do not implement the gated adaptation, nominate another fit, or reactivate shared-harm feedback from this note. Before reconsidering the selection rule, root needs eligible complete scalar TRAIN CE records that constrain its support in the actual mask convention, and a reason selection should help beyond the established all-TRAIN private GNCL/own-CE weighting law. Root owns any source qualification and subsequent decision.

The exact mixture and score-average derivatives are already in [Learner Collusion](https://arxiv.org/html/2301.11323v1), Appendix E, retained equation nodes Ex65/Ex67/Ex69. [GNCL](https://arxiv.org/html/2011.02952v2#S4.E5), Eq5, supplies the own/ensemble risk ancestry. The saved private-pool collision memo already records the stopped-responsibility identity. This is a focused correction/proof triage: no new primary method, full-paper read, result reproduction, model execution, payload access, remote action or TEST access. No global novelty claim follows.
