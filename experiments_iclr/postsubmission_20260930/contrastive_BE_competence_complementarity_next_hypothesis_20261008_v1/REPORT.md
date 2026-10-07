# Competence and useful complementarity under a shared backbone

8 October2026. Continuing contrastive BE scientific-owner analysis. This uses the retained complete Wiki24 reports and saved method conclusions. No partial context9/Wiki12/Mol18 outcomes, raw predictions, dataset/target/checkpoint tensors, model execution or private-server operations were accessed. No sealed source, targets or gates were changed.

## Concrete result interpretation

The closed Wiki24 study separates two failures that need different remedies.

**Unit factors with the contrastive package produce competent, mostly redundant members.** Mean member accuracy is about81.583%, slightly above the ordinary independent ensemble's81.440%. Served accuracy is81.633%, below ordinary independent4 at82.044%. The improvement over plain unit factors is mostly member improvement: +0.480 percentage points of mean member accuracy and only about+0.025 points of additional pool benefit. Further general regularization could strengthen members, but it does not explain how to acquire different correct alternatives.

The parent reports the complete PER_SEED counts: unit+contrast has4315/4325/4305 any-correct nodes out of5274, versus4327 correctly served by ordinary independent4 in each seed. Thus even an unavailable per-node oracle selecting among the unit+contrast member argmaxes would trail ordinary independent pooling in every seed. Mean oracle coverage is81.8165%, still0.2275 points below ordinary pooled accuracy. Every unit+contrast all-member-wrong node has a strict common wrong rival, and recorded all-member-wrong pool rescues are zero. The exact-arithmetic convex-probability bound therefore applies to these selected states. A better nonnegative probability router cannot supply the missing correct evidence. This does not prohibit a new-logit or hidden-state aggregator from learning a different function.

**Rademacher first factors acquire alternatives at an excessive competence cost.** Relative to plain unit factors, their coverage increases2.667 points while mean member accuracy falls1.199 points and served accuracy falls0.588 points. Pool harm rises from12.0 to183.7 nodes per seed. Initialized factors with contrast retain this problem: relative to initialized factors without contrast,349 pooled repairs are almost cancelled by321 introduced errors across15822 repeated seed-node instances; pooled NLL worsens0.004911. A favorable wrong-only cohort does not establish whole-population utility.

The resulting objective is useful correct alternatives with protected members and sufficient correct confidence to win the actual pool. Higher coverage alone can still leave weak members or excessive wrong confidence. Higher mean competence alone can leave nearly identical rankings. Lower contrastive loss or larger factor/hidden distances establishes neither. These are selected-development observations on one graph/split and three optimizer seeds, not generalization or causal mechanism findings.

## One next hypothesis: separate attention-score and value steering

The current factor adapter changes inputs and outputs of shared dense projections; native attention vectors remain shared. In a standard GAT interface, a projected member state participates both as the transmitted value and as an input to the shared attention scorer. Altering a fast factor can change both. The same parameter change may improve one role while harming the other.

**Hypothesis:** allowing only the attention scoring vectors to be private can let label-compatible context alignment acquire useful alternative neighbor rankings while shared dense value maps retain ordinary competence. This trains from scratch; it acquires no independent teachers and requires no second compression fit. It retains all four routes and the fixed mean-probability readout.

This is a specific utility question inside known attention and shared/private ensemble methods. Generic private attention and conditional graph messages were already assessed locally. They are not newly proposed primitives or an established methodological gap. Current summaries do not diagnose misleading neighbors or prove that score/value coupling caused the common errors.

### What the intervention changes

Use the source-qualified native attention layers. Clone each eligible shared attention vector into four initially identical trainable route-owned vectors. Each route uses its own vector at that site. Dense value projections, all existing BE factors, normalization, residual/global paths, predictor graph, own supervision, context targets and readout remain at their current definitions. Only the attention parameter partition changes. The shared attention-vector objects are replaced by private copies rather than retained as an additional common-plus-private parameterization.

All copies start from the same native vector. Their initial functions therefore match the shared-attention bank under matched realized inputs/views. Native mean-member loss reductions and Adam settings are retained; no separate learning-rate or initialization grid is added. Private attention receives the derivative of that exact joint objective. Changing parameter ownership also changes gradient aggregation and Adam histories, so this is a total parameterization/learning-policy intervention, not a pure forward-capacity causal claim.

At fixed projected values spanning the attention input space, private scoring vectors can change neighbor scores without changing those values. A shared scorer cannot do so at that frozen interface. Existing BE factors can already alter native attention through their changed hidden states; there is no claim that current members use one common operator, that all private attention functions are unrepresentable by the whole factorized network, or that this local observation establishes an expressivity theorem.

The GATv1 static-ranking limitation remains within each head. Merely untying its scoring vector does not make it query-dynamic GATv2. Scalar changes can also act mainly as attention temperature rather than produce useful ordering differences. These interpretations must remain available if the screen improves.

### Closest saved prior

- Saved BatchEnsemble/TabM factor and own-member supervision ancestry; shared vectors are an implementation choice, not a new ensemble principle.
- Saved GAT and GATv2 method scopes: multihead/private scoring under shared projections is established. The saved conditional-message assessment explicitly places sharedB/privatea scoring in this known family.
- Saved GNN-FiLM/ECC and DIVE/SuGAr/attention-disagreement conclusions: condition-dependent messages and diversified graph evidence are established. This candidate adds no per-edge parameter table, mask generator, graph perturbation or disagreement penalty.
- Newly saved AMCL section4.1: shared features, several contrastive heads and adaptive temperatures are established. The candidate remains a fixed-target/temperature auxiliary with internal attention recipients.
- Saved CGCL section3.3: same-graph multiple encoder contrast is established. Unchanged predictor topology is not a novelty argument.

No primary body is reread or retrieved for this plan. Full-rule overlap with any complete modern method is not cleared.

### Expected failures and falsifiers

1. Private score gradients can be zero, mostly common-scale/temperature directions or ineffective after neighborhood softmax. Equal/collinear neighbor features can make changed scores irrelevant.
2. Private attention can pursue nuisance same-class context structure, overfit sparse TRAIN labels, or suppress relevant neighbors; own CE does not guarantee competence.
3. Shared value maps may still omit the information needed by every route. Independent scoring cannot create missing input evidence.
4. New correct member rankings may be too weak to overcome wrong confidence, or may introduce as many errors elsewhere. More attention disagreement is insufficient.
5. COMMON objectives can improve as much as ROUTE objectives. That supports generic private attention capacity/optimization rather than a route-context specialization explanation.
6. Ordinary objective-matched single or untied models can match/exceed the candidate. The practical quality or sharing-advantage claim then fails, even if an internal contrast is positive.

## Minimal representative paired design

Keep one selected-development graph/split initially. The smallest mechanism comparison is the attention ownership by target-assignment2×2 table:

| Attention ownership | COMMON aggregate target | ROUTE persistent targets |
| --- | --- | --- |
| Current shared vectors | Completed context9 COMMON anchors | Completed context9 ROUTE anchors |
| Private initially equal vectors | Three new full fits | Three new full fits |

Use the same frozen seeds8101/8203/8307, graph/labels, target archive, complete1100-epoch horizon, own two-view recipe, local/global transition, checkpoints and selection opportunity. Reuse six completed shared-attention anchors rather than rerun or replace them. Root must qualify source/runtime/seed/initial-function/optimizer/selection matching and custody first. Historical initial-tensor identity must not be asserted unless actually recorded. If valid anchor reuse cannot be qualified, retain the plan inactive rather than fabricate a paired study or silently rerun scores.

This is12 comparison records with6 prospective fits, not12 new fits. The third context9 PERMUTED arm remains retained evidence and is not discarded. The new study is adaptive exploratory development because its design follows inspected Wiki24 results; repeated seed-node instances are not independent evidence. No TEST or original paper score changes occur.

The attention-specific contrast is private-minus-shared within each target policy. The route interaction is the ROUTE attention gain minus the COMMON attention gain. A favorable interaction could support useful target-specific access to private scoring under this exact recipe; it would not establish graph causality or a novel loss. Preserve individual paired seed values, full populations, mean/minimum member accuracy, pooled accuracy/NLL/Brier, any-correct coverage, common rival acquisition, repairs and harms, and costs.

Existing capable ordinary and regularizer-matched single/untied controls remain essential. If context9 Stage2 is admitted, use its frozen singleQbar/untiedQm references when contracts match. If they are not available, this table cannot establish superiority; root must separately admit the required complete capable reference family before any broader claim. A matched strong multihead/query-dynamic backbone is required before presenting a broad attention improvement claim. A compiled single with exactly the same bank/readout/training computation is an identity, not another fitted baseline.

Charge six full fits, all four route computations, actual private attention/gradient/moment storage, all attempts and closure/readout costs. If q is the number of original eligible attention scalars, untying adds(M−1)q stored prediction parameters; this is an accounting formula, not a measured memory/latency saving. There is no inference shortcut, efficiency claim or extra-GPU need established by this plan.

## Admission after the running families close

**The plan is inactive.** It does not change context9/Wiki12/Mol18 conditions, source, targets or gates.

1. Wait for complete context9/Wiki12/Mol18 custody and their existing readouts; do not use partial comparative results to choose this method. Preserve negative as well as positive outcomes.
2. If current route steering already succeeds, finish its prescribed competent references and unused confirmation before expanding. If it fails, preserve the failure; private attention remains a separate capacity/recipient hypothesis, not an automatic rescue of the failed target claim.
3. Root must determine from whole outcomes whether a competence/complementarity deficit still makes this question useful. Favorable hidden separation, targetTV or first-update differences alone do not admit it.
4. Before training, statically qualify the native eligible attention sites/ownership and exact equal initial functions, retain softmax/self-loop/duplicate-edge semantics, and prospectively freeze the complete table, objective, optimizer, readout, costs and quality decision. Source/numerical qualification can use TRAIN-only functional checks; it supplies no quality evidence and cannot be selected from scored pilot cells.
5. Root may admit one fixed6-fit screen with explicit prospective competence/pooled-risk thresholds and resource budget. Existing gates are not copied or altered by this proposal. No tuning grid, coefficient search or favorable-seed selection is permitted.
6. Any positive screen still needs strong objective-matched references and unused graph/split confirmation. A shared-backbone method has no automatic accuracy advantage over the containing untied function class; its finite-data/optimization benefit must be demonstrated.

## Why only one idea is retained

The saved competitor-responsibility maps, graph-response contrast, relative gradient/gauge steering and intermediate member-state mixing already contain direct prediction-facing alternatives. They have attributed priors and known failure limits. They are not reissued as fresh ideas. This plan selects one distinct restricted intervention in attention ownership instead of adding a second undeveloped loss or claiming that combinations of known pieces guarantee acceptance.
