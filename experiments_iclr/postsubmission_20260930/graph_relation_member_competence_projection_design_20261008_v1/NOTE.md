# Private graph credit with a member-risk safeguard

Inactive theory/design note. The full12 fits remain closed and unchanged. Saved method conclusions and the frozen v2 source are reused; there are no new paper reads, outcome reads, numerical diagnostics or launches.

## What the current rule can learn

For one labelled node and realized view, the frozen source uses mean own CE `F`, CE of the mean member probabilities `L`, and `J=.5F+.5L`. Pools are formed separately in each of the two views, then averaged. With four members, let `rho_m=p_m(y)/sum_k p_k(y)`. The exact logit cotangent is

`grad_z_m J = (.125+.5 rho_m) [p_m-onehot(y)]`.

This is an ordinary true-label CE direction with changed example credit, not a contrastive direction. Relative to F, its multiplier ranges between .5 and 2.5 in the limiting cases. The positive floor does not bound a parameter-gradient norm, Adam step, member accuracy or final member competence. Differentiating a weighted CE scalar through rho would add terms and would not implement the current J derivative.

At an attention aggregation `h_i=sum_j a_ij v_j`, the own-loss derivative through a post-activation edge score is `a_ij <grad_h_i CE, v_j-h_i>`. The remaining scorer chain rule follows. A labelled target can influence several upstream neighborhoods; its rho weights that target's complete pullback, rather than independently selecting an edge. Thus unequal member confidence can change which TRAIN examples most influence private neighbor preferences. Shared feature/value updates, overlapping receptive fields and later layers can reinforce or oppose those changes.

The current recipient set is exactly 14 local scorer banks plus four tied-QK factor banks. It combines local and global relation changes; it cannot identify them separately. phiJ also includes the four QK tensors. GATv1's static local neighbor-ranking restriction remains, while native global q=k kernel attention is query-dependent. Restricting CE recipients does not add a new attention operator or new graph evidence.

## Starvation and agreement remain possible

If all members have equal correct-class probability, rho is uniform and the F/J logit partials are identical, even when wrong-class distributions differ. Copied deterministic paths therefore have no explicit symmetry-breaking force. Distinct dropout realizations can break equality, but useful specialization is not guaranteed.

A weak route can receive half its normal per-example correction through its private relation coordinates. Its other coordinates still receive F, so literal zero supervision is absent; reduced attention learning can nevertheless leave it dependent on the stronger routes. A member that is relatively better but still wrong can receive greater pool credit. None of this is evidence that starvation actually occurs in the closed fits.

The saved identity `F-L=KL(U||rho_y)` explains why the mixture can reward a pooling gap at fixed F. It does not force different useful neighborhoods. Every member can retain the same decisive wrong rival; probability averaging cannot reverse a rival that outranks the truth in every member. Lower J or greater attention variation therefore cannot establish useful complementarity.

## One proposed safeguard: constrain the extra private step

The sole proposed mechanism is a **member-risk-constrained private relation update**. It changes the update rule, not the risk coefficient, serving pool, initialization, backbone, data access or horizon. beta stays .5. No coefficient grid, teacher pretraining, embedding repulsion, router or RL term is proposed.

Let `F_m` be member m's full580 TRAIN CE averaged over the same two realized views; `F=mean_m F_m`. Let R contain the same active private relation coordinates as the running rule, and B their complement. From the same old parameters, Adam state and stochastic views, calculate two counterfactual native steps: `u_F` using F everywhere, and `u_J` using J on R and F on B. Neither counterfactual step is applied during calculation. Define `q=u_J,R-u_F,R` and `g_m,R=grad_R F_m`.

Choose one extra private step by the four-constraint convex problem

`delta = argmin_d (d-q)^T H (d-q)  subject to  g_m,R^T d <= 0  for every m`.

Here H is the positive diagonal native Adam denominator `sqrt(vhat_J)+eps` on the active R coordinates. It is fixed by the existing optimizer's proposed J transition, not a tuned new coefficient. Apply `u_F,B` and `u_F,R+delta` once. Commit the proposed J moments on R and F moments on B, as the current recipient policy does. Inactive native parameters retain their original None-gradient treatment. This explicitly specifies the moment history: the own-F counterfactual is a same-state reference, not an independently trained F trajectory.

Zero extra step is feasible. When q satisfies every constraint, delta=q and the transition is exactly the existing relationJ transition, including its moments. When it conflicts with a member's own-risk gradient, the extra pool-driven movement is reduced or redirected. The elementary Taylor statement is that the extra step has a nonpositive *first-order* contribution to each realized member TRAIN risk, relative to u_F. The finite-step difference also contains second-order terms involving u_F and delta. This is not an Adam descent theorem, an absolute TRAIN-risk invariant, protection of each node, or a heldout accuracy guarantee. The own-F shared step can itself hurt a member.

The current private banks have a useful simplification: member m's R coordinates affect only its own forward path, so the four constraints separate by member. If `g_m^T q_m>0`, its weighted half-space projection is `delta_m=q_m-[g_m^T q_m/(g_m^T H_m^-1 g_m)] H_m^-1 g_m`; otherwise it keeps q_m. A zero gradient imposes no restriction. This standard projection formula removes a general solver from the private-bank candidate. The common-parameter single control must retain the actual four simultaneous constraints and cannot pretend they separate.

The safeguard can also remove helpful specialization when TRAIN CE rewards a shortcut or a temporary increase permits later learning. It cannot create a missing pool-credit signal, break exact deterministic symmetry or reconstruct unavailable graph information. A safeguard that mostly returns zero is not methodological progress without representative quality gains.

The extra cost is material: the existing replay must additionally recover the own-F gradients on R and each member's constraint gradients. One cannot infer these parameter gradients from the J gradients alone. In the present disjoint-bank replay, an additional R/F VJP per member/view would increase reverse collections from16 to24 per update; shadow/replay forwards remain8+8. The matched common-parameter single control may require further separate member constraint VJPs. All counterfactual optimizer arithmetic, constraint solves and diagnostics must also be charged and qualified. No implementation is supplied here.

## One incisive conditional comparison

Freeze one six-cell factorial comparison before scoring: the same **unconstrained relationJ** versus the **constrained rule**, crossed with these three architectures. Keep beta.5, complete data roles, native width/depth/1100epochs, paired original initialization, stream opportunities, probability-pool reductions, one accepted transition and joint stage restore/selector. Use complete paired blocks and open the whole comparison together.

| Architecture | Exact purpose |
| --- | --- |
| Shared four persistent functions | Candidate; ordinary shared maps with private scorer/factor banks. |
| Untied four persistent functions | Four physical copies of the shared condition's fresh maps, retaining its initial member functions and private banks; then maps evolve independently. Same loss scaling, recipient semantics and joint selector. Tests tying, not an optimally trained ordinary ensemble. |
| Joint single function with four stochastic streams | Tie every member bank into one persistent scorer/factor row and one body. Train through the same four stream pairs and per-view pools; apply the same own/pool rule and member-view constraints to that common relation block. Serve one deterministic function. Tests whether eight stochastic evaluations plus joint optimization explain the gain. Its persistent capacity/serving differ deliberately; disclose those counts. |

This is one study, not three alternative mechanisms. The single control's four F_m are stream-specific realized risks, not four persistent classifiers. Its gradients must be accumulated through the actual common parameters; pretending private rows exist would be an invalid control. Its factory and replay remain prospective and require separate source/runtime qualification. No genuinely unused role/seed IDs are asserted here; root must audit exposure and freeze them.

Let Delta_A be constrained-minus-unconstrained quality within architecture A. A similar Delta in untied predictors supports a generic member-risk safeguard rather than a sharing-specific effect. Matching or better constrained joint-single quality removes evidence that persistent private functions were needed. Sharing claims additionally require the direct shared-versus-untied endpoint comparison; an interaction alone is insufficient. If constraints activate but weak-member development competence still degrades, the intended practical safeguard is unsupported. If pool accuracy does not improve, preserving members alone does not meet the research goal.

Retain every seed, failure, cost, final/selected state and both repairs and introduced errors on the full populations. Member mean/worst readouts, any-correct coverage, pooled-only rescues and strict common-rival support distinguish useful evidence changes from confidence changes. Attention differences are explanatory measurements, not outcome-dependent cohort selectors or automatic admission criteria. Competent ordinary single and independently initialized ensembles with their original own selectors are still required before superiority claims. The six-cell factorial is an attribution falsifier, not their replacement.

## Established ancestry and claim boundary

- GNCL, Buschjager et al., [2011.02952v2](https://arxiv.org/abs/2011.02952): saved Eq5/method and author-code scopes establish own/pool mixtures. The loss and responsibility algebra are not new.
- Jeffares et al., [2301.11323v1](https://arxiv.org/abs/2301.11323): saved learner-collusion scope motivates evaluating individual competence; it does not predict this graph recipient rule's result.
- Sagawa et al., [1911.08731](https://arxiv.org/abs/1911.08731): saved GroupDRO conclusions establish robust group-risk training and regularization dependence. Members here are supervised-risk tasks, not data environments; this update is not claimed to implement GroupDRO or inherit its robustness.
- Yu et al., [2001.06782](https://arxiv.org/abs/2001.06782): saved gradient-conflict/PCGrad conclusion establishes projection ancestry. The simultaneous four-constraint step problem above is not claimed to be pairwise PCGrad or a new projection principle.
- Saved shared/private collaborative-learning, GAT/GATv2, Polynormer and BE scopes remain direct ancestry. No exact complete-rule collision or publication priority has been established; the inaccessible closest graph-ensemble body remains unresolved.

A possible contribution is an empirically useful, fully specified competence-constrained allocation of collective feedback to shared-backbone graph decisions. Its component principles are established. This note offers a falsifiable combination, not a novelty certificate, guaranteed benefit or reviewer verdict. Saved scoped conclusions were reused without inflating reading counts. One targeted search in an old saved assessment incidentally exposed historical result prose; none supplies evidence or a design choice here. No current result, tensor, model, dataset or remote machine was accessed.
