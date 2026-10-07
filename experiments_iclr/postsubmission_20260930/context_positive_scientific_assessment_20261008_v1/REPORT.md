# Scientific assessment of route-specific graph-context positives

2026-10-08. Scientific critique of the dedicated contrastive branch, not a code audit or manuscript verdict. Current method/protocol, steering analysis, disabled staged design, saved closest-prior conclusions/scopes and complete Wiki24 interpretation/error summary were read. No fits, model imports, heldout/raw prediction access, source changes, new primary retrievals or extra agents.

## Verdict

**A coherent, falsifiable target-assignment hypothesis; a remedy for weak members and correlated wrong competitors remains unestablished.** Fixed different same-class context targets can supply different private gradients at an otherwise matched unit start. This is more directed than arbitrary member repulsion. Label compatibility does not make those gradients competence-preserving or align them with the wrong class margins that obstruct serving.

Common Qbar and degree-constrained permuted Qm are useful controls for a narrow intervention inside supervised contrastive learning. Together they can test whether persistent differentiated targets and their factual signature/node incidence help. They cannot establish a new contrastive principle, a graph-topology-specific cause, or a benefit unique to shared BE factors. The existing single/untied references remain necessary for those stronger interpretations, with their selection differences disclosed.

No different fallback is proposed. The errors establish a ranking/competence requirement, not a diagnosed missing graph feature or optimizer defect. The saved prediction-response alternatives already have attributed priors and explicit negative limits; renaming them would not produce a supported fallback.

## 1. What the completed evidence actually motivates

Wiki24 is complete: eight arms × three seeds, no excluded prediction cells. These are selected states on one graph/split and the same5,274-node merged validation+stopping development role that selected checkpoints. They are descriptive evidence, not confirmation.

- Unit factors plus the original contrast package improve served accuracy by0.5056 percentage points and mean member accuracy by about0.480 points; extra pool benefit is only about0.025 points. The useful signal is primarily stronger members. That arm still trails ordinary independent4 by0.4108 points.
- Rademacher initialization increases coverage but loses1.199 points of mean member accuracy and0.5878 points of served accuracy relative to unit factors. More disagreement/coverage did not pay for weaker members.
- The initialized-plus-contrast primary comparison produces349 repairs and321 harms over15,822 repeated seed-node instances, net28. Overall pooled NLL worsens by0.004911. On the frozen common-competitor cohort,128/2,503 instances are repaired, while new common errors elsewhere nearly offset the reduction. Favorable wrong-only cohort summaries do not demonstrate full-population utility.
- For initialized factors,2,503/2,528 all-member-wrong instances have a common wrong competitor; with contrast the counts are2,490/2,515. Every cell has zero all-current-members-wrong pool rescues. A common wrong class strictly above truth in every member is an exact barrier for any convex pool of those unchanged probabilities.

This supports protecting competence while acquiring sufficient correct ranking evidence. It does **not** diagnose class collapse, context-specific subclasses, misleading graph neighborhoods, or the inadequacy of ordinary attention. The original package combines alignment and residual repulsion; Wiki24 does not attribute the unit gain to one term. The branch's dependence on the complete Wiki12 attribution is scientifically appropriate; it cannot assume that attribution's result here.

## 2. Does the attraction directly address those errors?

For a row-stochastic target Qm and scaled cross-view cosine scores S_m, one direction of the auxiliary is

\[
\ell_m=-\frac1n\sum_{i,j}Q_{m,ij}\log P_{m,ij},\qquad
P_{m,i:}=\operatorname{softmax}(S_{m,i:}).
\]

Its score gradient is proportional to P_m−Q_m. Qm uses TRAIN class compatibility and fixed raw root/residual/one-hop/two-hop signature similarity. It does not use the identity or margin of a current wrong competitor, whether a member is weak, or whether a prospective change repairs the probability pool. Own CE supplies ordinary label-risk credit separately.

At deterministic equal scores and equal shared Jacobians, linearity in Q makes the mean auxiliary and shared gradient match the Qbar arm; private target cotangents can differ. This is a valid local symmetry-breaking identity. Independent dropout, changing representations/Jacobians and Adam prevent a trajectory identity. The private differences also need not remain purely relative parameter/function changes as training proceeds.

Same-class targets offer a plausible route to task-relevant learning: a member can move toward useful examples of its class through its nonlinear factors. But their relevance is conditional. Cosine attraction is not a convex mixture of positive examples' class margins; denominator repulsion, normalization, shared-weight interactions and finite Adam steps all intervene. For a truth-versus-competitor margin f, an auxiliary-only infinitesimal Euclidean factor step changes f by

\[
\delta f=-\eta\,\nabla_\phi f^\top\nabla_\phi\ell_m.
\]

The overlap can have either sign or be zero. TRAIN label agreement alone fixes none of these cases. This expression is a local diagnostic, not a guarantee for the actual optimizer or selected development objects.

Unselected same-class nodes remain denominator distractors. The intervention deliberately encourages distinctions inside a coarse class. It can learn useful subclass structure, emphasize nuisance degree/topic/neighborhood cues, or split examples whose shared class evidence should stay pooled. Fixed X/PX/P²X signatures add no predictor capacity and do not identify latent semantic subclasses. Near-unit initialization avoids one demonstrated source of competence damage; it does not make the new auxiliary harmless.

### Exact failure: different masks need not give differentiated gradients

Suppose both views map every TRAIN example in class y to a nonzero vector parallel to u_y. Every same-class positive has cosine1, and its derivative with respect to either vector is zero. Because every Qm row assigns total mass1 only to that class, its target-dependent attraction term has the same value and zero gradient for every positive-selection mask. The denominator term is target-independent.

Thus Qm and Qbar can have large target total variation while yielding exactly the same target-dependent embedding gradient. Parameter gradients inherit that equality through the embedding Jacobian. Independent noise or deviations from collapse may break the example's assumptions, but the construction shows that differentiated target masks do not intrinsically eliminate class collapse or guarantee private steering. It is not a claim that the actual WikiCS model occupies this state.

A second failure is prediction-invisible satisfaction: a representation/linear-head interface can vary contrastive features in the classifier's nullspace while retaining identical logits and common errors. No auxiliary projection head is required for a head to have unused directions. The saved two-loss counterexample establishes this general interpretive limit; it does not prove arbitrary nullspace coordinates are reachable by the restricted native factors or are full-objective stationary points. Context positives do not remove the absence of a direct prediction constraint.

## 3. What the controls distinguish

| Contrast | Supported interpretation if favorable | Remaining limit |
|---|---|---|
| Route Qm versus common Qbar | Repartitioning the same aggregate context target into persistent member objectives is useful under this optimizer, architecture and selection rule. | Both arms are supervised multi-view contrastive learning. Per-route target concentration/support/entropy changes; this is not an isolated test of a new learning principle. |
| Factual Qm versus permuted Qm | Factual signature/node incidence helps beyond the chosen class/panel/positive-degree-matched shuffled relation. | This is not an intervention isolating public graph topology or an exhaustive random-target null distribution. |
| Shared route bank versus single Qbar / untied Qm references | Practical quality relative to capable regularizer-matched alternatives. | Joint versus independently own-selected checkpoints and shared/private optimization differ; a gap does not alone identify parameter sharing as the cause. |

### Aggregate mass is matched; per-route entropy is not

Qbar is the average of row-normalized targets after the same panel restriction. It matches aggregate target mass and the denominator sample set. It generally has a larger positive support and smoother weights than any one Qm. Row entropy satisfies

\[
H(\bar Q_i)\ge\frac1M\sum_m H(Q_{m,i}).
\]

Cross-entropy's unrestricted row floor is H(Q_i), although the finite-temperature cosine model need not attain that floor. A lower route contrastive loss can therefore reflect different attainable target geometry rather than better decisions. Equal loss at equal scores remains exactly true; it does not make per-route positive count, entropy, hardness or subsequent updates equal. Describe the control as **aggregate-target matching**, rather than per-route positive-count/entropy matching.

Route beating Qbar can also be consistent with generic heterogeneous target specialization, without factual graph contexts being useful. The permuted arm addresses part of that ambiguity. Fixed context cycling, already excluded from the staged study, would be a different optimization schedule; no extra arm is proposed here.

### The preserved degree is positive-relation degree

The stated permutation preserves every scored anchor's restricted positive count by remaining within class, auxiliary-panel membership and route-specific positive-degree buckets. It is not a public-graph-degree-preserving permutation. It preserves positive labels/self positives and the column-degree multiset, while changing which actual nodes carry column popularity. Such reassignment can affect optimization independently of a semantic-subclass explanation.

A fixed permutation can retain some factual relations or class-correlated nuisance structure. Nontrivial target variation is necessary for a useful control but does not prove removal of all relevant context information. One fixed draw across optimizer seeds is not a randomization distribution over graphs. Matching/exceeding the candidate removes support for the proposed factual-context benefit in this screen; it does not universally prove that all graph contexts are useless.

One of the four signatures is X itself. Consequently, factual-versus-permuted improvement could reflect raw feature similarity or generic within-class structure. The current comparison cannot uniquely attribute improvement to edges, high-pass evidence or multi-hop topology. No additional fitting grid is requested; restrict the eventual claim to the tested signature-positive relation.

## 4. Strongest falsifiable prediction

**Relative to BOTH common targets and permuted route targets, factual route targets should acquire enough correct competitor-margin evidence to produce more net served repairs without the competence damage seen in the randomized-factor family.**

For a competitor c frozen from the common arm and true class y, define the actual probability margins

\[
d_m(v,c)=p_m(y\mid v)-p_m(c\mid v),\qquad
d_{\mathrm{pool}}(v,c)=M^{-1}\sum_m d_m(v,c).
\]

On a reference object where all d_m<0, a strict pool reversal against c requires at least one positive member margin **and sufficient positive mass to offset the remaining negative margins**. Breaking one fixed competitor is necessary for that reversal, but not sufficient for final correct classification; other wrong competitors can remain. Newly introduced common wrong competitors elsewhere must also be counted.

The strongest observation therefore combines: greater acquisition of useful member rankings on the prospectively frozen common-arm cohort; actual served repairs there; fewer newly introduced common errors and positive repairs-minus-harms across the full population; and protected mean/worst member competence plus pooled NLL. Lower hidden loss, nonzero mask variation, distinct private gradients, higher coverage or a better wrong-only cohort mean cannot substitute for it.

If route beats common but not permuted, the evidence favors generic target partitioning rather than the factual-context story. If hidden separation/contrastive loss improves but competitor margins and net serving do not, the intended error mechanism fails. If competence damage cancels repairs, the extension repeats the observed diversity/competence tradeoff. If common or single context alignment explains the benefit, retain the regularizer explanation rather than a persistent-member-specialization claim.

The branch's all-three-seed0.2-point rule and member/NLL tolerances are prospective screening decisions, not statistical guarantees of competence preservation. The development population and graph are already used to motivate the idea and to select checkpoints. Three fresh optimizer seeds remain exploratory; no independent-node inference, graph-general conclusion or heldout claim follows.

## 5. Prior collision and fallback decision

The exact loss operation is weighted multi-positive cross-view contrastive cross-entropy. With uniform weights over all same-class cross-view objects, it recovers the corresponding supervised contrastive positive relation under a matched denominator convention. This is a collision with an established loss family, not a claim that the original SupCon publication uses every detail of this branch's denominator/target construction.

Saved BotSCL already combines supervised same-class graph views, shared maps and channel steering. HLCL already supplies shared high/low-pass contrastive graph encoders. Chen et al.2204.07596 already distinguish meaningful subclass structure from arbitrary class-preserving spread and use weighted class-conditional contrastive mechanisms; its exact denominator and full method must not be silently equated with this branch. Xue et al.2305.16536 provide assumption-specific feature-suppression/class-collapse analysis, not a diagnosis of Wiki24. PMGCL is a direct positive-mining lead, but only its publisher abstract was read; it cannot certify or exclude exact complete-rule duplication. BE/TabM and the saved CDLG/DICE/FoRDE/DIVE/SuGAr family provide sharing, pairing and useful-diversity ancestry.

No inspected saved scope establishes an exact published collision with the whole four-signature/top16/BE assignment rule. That bounded unresolved status supplies no novelty claim. The scope sufficient for this scientific assessment is already saved, so no new primary search/read is essential or made.

No distinct fallback meets the evidence condition now. Moving contrast into finite prediction/graph responses would remove a particular hidden-nullspace blindness, but the saved response proposals already supply that route and show that prediction-visible diversity can preserve common errors or be harmful. Direct member/pool risk is established GNCL territory and is already under separate attribution. A topology intervention needs a diagnosed misleading-context mechanism that these summaries do not contain. Preserve those boundaries rather than inventing a new correction or a fit recommendation.

`CONCLUSIONS.json` and `READ_SCOPES.json` retain this neutral disposition. `INPUT_BINDINGS.json` binds the read branch, saved prior notes and complete descriptive error summaries. `MANIFEST.json` covers only this small new folder. Existing branches, staged permissions, scores and artifacts are unchanged.
