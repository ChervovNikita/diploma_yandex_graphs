# Scientific scope of the current training studies

8 October 2026. The user requires representative experiments that answer the research problems. This note explains the existing decisions. It changes no active source, target, seed, training budget, selector or continuation gate.

Our aim is better ensemble predictions with shared learned weights. We need each route to remain a competent predictor and to contribute correct decisions that the other routes miss. Different hidden vectors alone do not establish either property.

## Complete task comparisons

Context9 uses the complete WikiCS graph and the native Polynormer recipe. Its nine full fits compare a common contrastive target with four targets derived from different graph contexts and with a permutation that removes their semantic association while preserving the registered target constraints. Every route receives all training labels. The comparison tests whether the factual graph evidence gives useful route differences without damaging member quality.

Wiki12 uses the same complete task and twelve full fits. It separates class alignment from the term that pushes route residuals apart. Its purpose is to distinguish learning competent representations from merely increasing representation differences.

The three new canonical SupCon fits are a published-loss comparison against all three prospectively reserved alignment-only Wiki12 anchors. They retain the full task, 1100 epochs, 512 auxiliary examples, all 580 labels for the ordinary prediction loss, paired seeds, native model, selector and runtime. This tests whether a conventional supervised contrastive objective explains any gain before credit is assigned to a graph-specific rule. It is a joint prediction/contrastive-loss adaptation, not reproduction of SupCon pretraining with a separate projection head.

Mol18 uses all official molecular training and development graphs, the bond-aware GINE recipe, capable single and independent ensemble references, and the registered 100-epoch schedule. It tests transfer of the internal BatchEnsemble training idea to graph classification. A timed-out or failed fit remains a failure. Incomplete training does not become a representative accuracy result.

## Outcomes that decide whether to continue

Read whole registered comparisons. Report every paired seed, pooled quality, mean and worst member quality, newly correct predictions, introduced errors, shared mistakes and training cost. Use the registered actual serving rule. Extra disagreement is useful only when the complete prediction comparison supports it.

A passing Context9 screen still requires its twelve native and objective-matched single/independent reference fits. A gain over a weaker shared model cannot establish an advantage over a capable single or ordinary ensemble. Any broader claim also requires a prospectively frozen unused split or task and an appropriate published ensemble comparator. Three optimizer seeds on repeatedly selected development data remain exploratory, and nodes are not independent experiment replicates.

Small synthetic CPU fixtures and discarded native updates qualify implementation and resource use only. They do not enter scientific accuracy tables. Original paper scores remain unchanged. Preserve failed ideas and measured costs so they are not tried again under a new name.
