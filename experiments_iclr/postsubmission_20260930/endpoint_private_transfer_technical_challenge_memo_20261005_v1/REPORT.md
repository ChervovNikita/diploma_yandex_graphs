# Technical challenge: endpoint-conditioned private transfer

**Scope:** synthesis of already saved closest-prior conclusions and the sealed v2 source. No new paper read, reading credit, outcome inspection, numerical execution, source change or canonical edit. This is a challenge memo, not a novelty or acceptance verdict.

## Strongest reason it could improve ranking

Ordinary joint training rewards corrections to the same supervised edges that produced an update. In a graph, this can favor endpoint-specific fitting: repeated supervision around a few nodes can dominate corrections that would transfer to other endpoints. The proposed episode instead asks whether learning from edges elsewhere helps predict edges whose **positive and negative endpoints were excluded from the current private update**. The shared core receives credit through what each private route learns. Thus the useful hypothesis is that the core learns features and factor-sensitive directions whose corrections transfer across endpoint sets, rather than merely lowering the inner loss.

Four private routes offer four learning trajectories on the same representation. A tied core must support useful corrections across these trajectories, while the outer mean-logit objective rewards their combined prediction and the individual BCE term keeps each route competent. This could make shared learning more robust than a single correction path. Post-core recomputation then commits a private update computed using the representation that will actually persist. These are plausible mechanisms, not proofs of complementary errors, fewer mistakes or better HeaRT MRR.

The intellectual synthesis is real but its broad ingredients are prior: ANIL supplies head-only adaptation; BMAML/EMAML already combines shared features and private meta-ensemble classifiers; MLDG supplies source-only inner/outer transfer and gradient alignment; OML/MRCL supplies representations trained through correlated head-only online updates; La-MAML and FTML establish ongoing persistent meta-learning. The remaining testable recipe is the graph supervision contract, the declared private-state commitment and their utility for a competent served ensemble. Neither persistence nor differentiating private Adam is a new learning principle.

## Strongest reasons it may fail

**Current-step Adam credit can be weak or concentrated in tiny gradients.** At the first private step, with zero moments and no decay, one scalar coordinate is updated by

\[
\phi'=\phi-\eta\,\frac{g}{|g|+\epsilon},\qquad
\frac{\partial\phi'}{\partial g}=-\eta\,\frac{\epsilon}{(|g|+\epsilon)^2}.
\]

The derivative at zero is the finite limit \(-\eta/\epsilon\). For gradients much larger than epsilon, the update approaches a sign step and its sensitivity shrinks; the meta-credit is therefore not an ordinary SGD gradient-agreement term. Coordinates near zero can receive much greater sensitivity. Positive previous moments change this derivative, so the first-step formula is not a claim about every episode. The analytic native-Adam Jacobian contains cancellation between two terms for zero history; FP32 cancellation can erase a small sensitivity or create relative error. A passed direct derivative gate checks implementation on its fixture; it does not establish a useful or stable learning signal throughout training. Live-versus-detached quality is the decisive practical test. No denominator smoothing, clipping or SGD substitution is justified by this observation alone.

**The episode objective is evaluated on a heavily depleted graph.** The preceding TRAIN-only geometry receipt reports a paired union mask removing about 40–42% of positive support edges. NCN uses common neighbors, each requiring two incident edges; removing targets can destroy structural evidence more strongly than the edge fraction suggests. At validation the complete TRAIN support returns. The rule may learn corrections for this masked/dropout regime that do not improve fixed full-support ranking. Recomputation aligns the current private update with the new core on the episode support; it does not remove this support/distribution shift. The common paired mask is essential for comparisons, but common damage is still damage.

**Endpoint separation does not establish independence.** All nodes/features and other observed incident context remain. Historical weights and moments may already contain supervision about current outer endpoints. Eligible inner queries and repeated route exposure can favor different degree/structure populations; exact full-TRAIN degree/CN matching does not imply equal post-mask strata. This is conditional training regularization on one transductive graph, not cross-fitting or new-node generalization.

**A shared core can preserve the very correlated errors ensembling should remove.** Private factors cannot create every representation available to independent nonlinear encoders. Four trajectories can collapse toward similar predictions or compromise under a common outer objective. BCE improvement also need not improve positive-versus-500-negative ranking. The method pays substantially more training episodes and repeated private work, which can explain an ordinary-baseline gain unless competent references receive explicit training and selection budgets.

## What would distinguish an ensemble contribution from generic adaptation

Let E be shared F4, S the capable single with its **entire nonlinear NCN predictor** adapted, and U the initially function-matched untied four. Assess served complete-query MRR through representative paired seeds and a frozen comparison family; validation findings remain development evidence until the heldout evaluation is prospectively frozen.

| Empirical contrast | What it distinguishes |
|---|---|
| E-live versus E-detached, with identical values/work/support | Usefulness of current private-update credit, rather than exposure or the paid schedule |
| S-live versus S-detached, then compare the E and S live-credit gains | Generic ANIL/OML-style transfer versus evidence that separate routes add useful interaction. A positive difference of gains is suggestive; different private capacities prevent treating it as a theorem |
| E-live versus S-live **and** independently initialized ordinary native4 | Whether the final ensemble actually beats competent single and ordinary ensemble references; a gain only over ordinary single is insufficient |
| E-live versus U-live | Whether tying improves quality under the same operations/adaptation. Untied outer branches receive the declared 1/M scaling; native Adam cannot be presumed to eliminate finite-epsilon/history scaling effects |
| Endpoint versus matched-random E, especially their live-minus-detached gains | Whether graph endpoint task construction makes adaptation credit useful beyond generic random episodic meta-training. Inspect post-mask strata/exposure before causal attribution |
| Recomputed versus paid stale commitment | Whether state recomputation earns a measurable quality benefit beyond a consistency convention |

The strongest pattern would be replicated gains over both competent ordinary references and the capable adapted single, useful live credit specifically strengthened by endpoint episodes, and a favorable sharing/commitment contrast. If the capable single obtains the same benefit, the evidence supports generic graph adaptation. If untied four wins, ensemble meta-training may be useful while sharing remains a capacity compromise. If random episodes or detached credit match, endpoint-conditioned learning has not earned its proposed mechanism. Greater disagreement, lower inner BCE, parameter savings or a correct gradient cannot substitute for these outcomes.

## Reformulation decision

The saved scopes and inspected source do **not** establish a fundamental mathematical flaw: the current shared derivative is coherent for its frozen episode-start private state, and recomputation evaluates the same update map at the new core. Adam does not guarantee monotonic decrease or ranking improvement, but that limitation does not invalidate the rule. Therefore this memo proposes **no automatic reformulation or additional parallel variant**. First complete the finite implementation gate and the already specified representative contrasts. If live credit consistently vanishes or only the capable single benefits, that is evidence against an ensemble-specific claim, not permission to rename generic adaptation as novelty.

## Bound prior scopes

- `shared_backbone_quality_next_hypothesis_20261005_v1/PAPER_CONCLUSIONS.json`: ANIL, BMAML/EMAML and Meta-Graph method overlap and graph-task limits.
- `shared_core_train_only_meta_prior_root_20261005_v1/SCOPED_CONCLUSIONS.json`: MLDG/MetaReg bounded method/analysis conclusions.
- `persistent_private_adaptation_closest_prior_20261005_v1/PAPER_CONCLUSIONS.json` and `REPORT.md`: OML/MRCL, La-MAML and FTML exact-transition comparisons, attribution and falsifiers.
- `private_neighborhood_quantile_quality_hypothesis_20261005_v1/REUSED_CONCLUSIONS.json`: retained BatchEnsemble/TabM and collective/member-objective ancestry.

Exact hashes are in `INPUT_BINDINGS.json`. Existing read-scope limits remain unchanged. The memo adds zero primary or full-paper readings and adopts no local or published score as new empirical evidence.
