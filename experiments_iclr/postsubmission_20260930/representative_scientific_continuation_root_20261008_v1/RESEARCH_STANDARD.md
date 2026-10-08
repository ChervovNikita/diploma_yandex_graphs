# Representative experiments for the current GNNM extension

8 October 2026. This records the user's explicit requirement for representative experiments. It does not change a running recipe, selector, stopping rule or original paper score.

## The problem being tested

The completed Wiki24 study found that the unit-factor contrastive model improved its average member accuracy but did not produce enough different correct decisions. Its any-correct-member counts were 4,315, 4,325 and 4,305 of 5,274 development nodes. The ordinary ensemble's averaged predictions were correct on 4,327 nodes in each paired block. Every node where all contrastive members were wrong also had a strict common wrong competitor. These are selected development predictions on one split. They identify a failure to investigate and establish no general performance claim.

The next method must improve correct alternatives while retaining member quality. Separation of embeddings, lower auxiliary loss, different gradients, or a successful program launch cannot establish that improvement.

## What the current studies answer

- Context9 uses the complete WikiCS graph, the original 580 training nodes, 5,274 development nodes, a capable Polynormer backbone, three paired seeds and complete 1,100-epoch training. Common targets, persistent route targets and shuffled node-incidence targets test whether factual assignments provide useful decisions. Its existing whole-family gates remain binding. Matched single and ordinary ensemble references and unused confirmation are still required after a successful pilot.
- Wiki12 trains the same complete node task in three seeds with plain, alignment-only, residual-only and combined objectives. This separates which instruction produces the existing gain. It cannot establish superiority over an ordinary ensemble by itself.
- Mol18 uses the full official ogbg-molhiv scaffold training/development split, three paired seeds and complete 100-epoch training. Its single, ordinary independent ensemble and four fixed shared variants test whether the internal BE intervention transfers to graph classification. This is a binary ranking task. A multiclass classifier restriction on WikiCS must not be used to explain it without new evidence.
- Direct12 is a proposed diagnostic using the actual complete saved Wiki24 representations, all training/development rows and three paired seeds. It compares factorized and unrestricted prediction layers with identical initial effective maps, common bias, feature scaling and training objective. Refitted ordinary ensembles and singles are stronger references. Its parameters change only the final prediction layers. It can identify useful readout freedom or an optimization problem. It cannot demonstrate an end-to-end method, a strict capacity theorem, or independent validation.

## Scientific admission and reporting

Each new training study must identify the observed weakness, the proposed mechanism, capable baselines, the unchanged elements of a matched comparison, the complete data roles, and the decision it will inform. Freeze these before comparative scoring. Use complete training rather than short illustrative fits as performance evidence. Small semantic and runtime checks qualify code only.

For the direct diagnostic, freeze error cohorts from the untouched native unit+contrast model before fitting. Report correct alternatives, strict repairs, introduced errors, member and pooled accuracy, NLL and the ordinary/single references across the complete population. A gain that also appears in ordinary refits does not establish a specific shared-backbone advantage. All failed fits and convergence limits remain part of the report.

Treat three optimizer seeds on one split as exploration. They do not replace split variation or independent confirmation. State uncertainty at the appropriate unit. Do not treat dependent graph nodes as independent experimental replicates or claim statistical significance from a favorable subgroup. A subsequent method comparison needs an unused split or task, the same candidate recipe, capable contemporary backbones and relevant efficient/diverse-ensemble controls.

Save literature conclusions and the scope actually inspected. Attribute inherited ingredients. A failure to find an exact match is not proof of novelty. Record unsuccessful directions so they are not repeated. Keep original paper scores and all completed evidence unchanged.

## Current execution

At 00:49 UTC, both 18.77 GPUs were training the original Wiki12 family. Six cells had completed. The two combined cells were at epochs 893 and 965 of 1,100. On the one-GPU allocation, the original molecular P cell was at epoch 60 of 100. The original context worker was intentionally paused at epoch 670 by the bounded scheduling controller. It resumes on P exit or the original two-hour scheduling limit. No active comparative scores were opened.

No new superiority, methodological novelty or paper acceptance has been established.
