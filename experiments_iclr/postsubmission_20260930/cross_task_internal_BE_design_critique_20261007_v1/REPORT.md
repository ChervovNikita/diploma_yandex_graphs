# Cross-task internal BE design: independent conceptual review

7 October 2026. Source-only review of the three requested files. No other project evidence, raw/current outcomes, data, models, server or GPU was accessed. No searches, execution, coefficient choice or family changes.

**The Wiki O/I/P/G design is a useful attribution comparison:** it fixes eight member/view forwards, alignment permissions, one strength, old-state updates and the joint selector. The initial eight-method families answer a broader auxiliary-package question. Those two experiments should retain their separate claims.

## 1. The closest GNCL controls are specified only for WikiCS

`COMPARISON_DESIGN.json` binds only WikiCS and its mean-probability pool. The cross-task handoff describes the internal-only GNCL adapter but does not freeze corresponding Collab/MolHIV O/I/P/G comparisons. Testing only internal GNCL against own training or hidden contrast would leave a known explanation open: ordinary scalar GNCL or all-private GNCL may perform equally well.

For any Collab/MolHIV claim that the **internal restriction** matters, freeze one task-specific O/I/P/G comparison on one constructor and the same single strength. Preserve alignment on identical internal parameters. Bind the binary mean-logit loss and task reductions rather than copying Wiki's probability-pool contract. O/I/P/G is a fixed attribution roster, not an initialization-by-strength grid; do not append it to the running family.

Wiki probability pooling yields correct-class responsibilities `p_m(y)/sum_j p_j(y)`. Binary mean-logit pooling gives every member the same pool residual `(sigmoid(mean_j z_j)−y)/M`, subsequently pulled through different member Jacobians. Node-dependent winning-member credit therefore cannot explain Collab/MolHIV. For Collab, freeze positive/negative group weighting and sampled negatives; for MolHIV, freeze the finite-target reduction. Official Hits@50 and ROC AUC must decide utility: a better BCE/NLL fit alone does not establish ranking/AUC gains or graph-evidence specialization.

## 2. Initial package attribution lacks a matched shared alignment-only control

Methods8 versus6 add the entire auxiliary package. The eight-method roster has a **single-model** alignment arm, but no shared initialized bank receiving alignment alone with the same stochastic views and parameter permissions. Consequently, a package gain cannot distinguish within-path view regularization from cross-member residual diversity; a single alignment arm does not resolve this in a shared bank. The handoff's deferred single-eight-path controls also do not by themselves isolate those two losses.

If residual diversity or useful internal differentiation is the intended explanation, add just one fixed alignment-only shared comparator on the chosen constructor, matching the package's view count and exact permissions. The prepared private-alignment reference serves this role only for the separately matched private-steering comparison; it is not automatically equivalent to the original full auxiliary family. If the claim remains “the complete package improves predictions,” report it at that scope and do not attribute the gain to residual repulsion. The O/I/P/G comparison already fixes eight forwards, so this is a gap in the initial package attribution, not a view-count defect in O/I/P/G.

## 3. Untied baseline selection changes with its objective

The handoff correctly notes that method3 independently selects each of four checkpoints, while method4 uses coupled training and a different selection rule. Method3 versus4 therefore cannot isolate the auxiliary objective; comparison with a joint-selected shared bank also does not isolate parameter sharing. Their practical baseline results remain valid, with the training/selection difference disclosed.

For a claim about useful sharing or ensemble complementarity beyond selection, the smallest missing comparator is one **own-only untied bank with the same joint pool selector, two-view training and member streams/horizon** as the relevant shared comparison. This is the packed untied control already motivated in the saved synthesis. Match data/dropout draws as far as the architecture permits and report remaining capacity differences. Count actual work: the design already records different reverse collections for P, and equal epochs do not imply equal compute. No new strength, initializer, depth or checkpoint grid is needed.

## Disposition

Keep the current families unchanged. Before funding a mechanism claim on Collab/MolHIV, specify the task-specific GNCL attribution roster and the narrow matched controls needed for that claim. If alignment-only matches the full package, residual diversity adds no demonstrated utility; if all-private/all-parameter GNCL matches internal-only, the restriction adds no demonstrated value; if packed untied explains a gain, sharing-specific attribution weakens. Known components can still support a useful complete method. This review claims neither generic contrastive/initialization novelty nor an exhaustive prior search.
