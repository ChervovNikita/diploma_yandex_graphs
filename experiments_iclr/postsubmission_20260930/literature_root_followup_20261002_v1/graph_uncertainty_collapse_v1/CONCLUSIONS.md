# Graph ensemble uncertainty: primary reading conclusions

**Do Deep Ensembles Actually Capture Uncertainty in Graph Neural Networks?** Pedro C. Vieira, Pedro Ribeiro and Viacheslav Borovitskiy. arXiv:2605.22593v1, 21 May 2026. Primary HTML: https://arxiv.org/html/2605.22593v1. The visible header verifies the title/version/date. Reading was from saved primary HTML/text; no image pixels, author code, datasets or outputs were inspected.

## Exact reading scope

Read abstract; introduction; background/related work; PEMS motivating example; section 4 dataset/model/metric/setup/results prose and the visible table; sections 5.1/5.2; discussion and conclusion; Appendix A recipe table; Appendix B structural-shift setup/results. References were viewed to resolve cited dataset/model sources, not to claim those papers were read. Appendix C plots and figure pixels were not inspected. No statement here independently reproduces the authors' results.

## Supported takeaways

The authors study standard independently initialized, same-architecture message-passing ensembles. Their classification pool averages **member probabilities**. That differs from our prospective mean-raw-logit primary pool; their Jensen decomposition cannot be transplanted unchanged.

Their seven tasks are PEMS node regression, Cora/Citeseer/Tolokers2 node classification, Artnetviews/Chameleon node regression, and a stratified 5% QM9 HOMO-LUMO graph-regression subset. They report ten runs at fixed data splits and additional five-versus-ten-member comparisons. GCN and GATv2 are the native backbones; the latter has four heads. On the GraphLand tasks, interleaved two-layer MLPs are included. They attribute GCN/GATv2 choices to task-specific contemporary references; a bare minimal GCN port would not reproduce those recipes.

They report relatively small ensemble NLL improvements, occasional worsened calibration, small member-information estimates and the phenomenon they call epistemic collapse. For regression, exchanging the ensemble variance with single/aleatoric-only variance while keeping its mean diagnoses where their NLL gains occur. For classification, they report a true-class probability Jensen gap and a separate entropy/mutual-information decomposition. Structural shift ranks nodes by local clustering coefficient, assigning denser neighborhoods to test; this is a topology-derived shift with fixed role sizes, not an unseen-graph guarantee.

The public primary links https://github.com/PedrV/gnn-uq-inspector. Author-source compatibility and checkpoint/split selection have not been inspected or qualified in this reading.

## Critical limits

1. The scope explicitly excludes graph transformers. Do not treat these results as a universal claim about all graph models or as a measured failure of PolyFormer/Polynormer.
2. A small Jensen gap for the true-class probability does not by itself prove that the full class probability vectors are identical. Incorrect-class probabilities can redistribute while the true-class probability stays fixed. The separate mutual-information evidence is more relevant to full predictive disagreement. Neither metric alone certifies uncertainty quality.
3. Jensen improvement over an ensemble's own members is not an isolated measure of calibration. Its size can reflect error complementarity, member deterioration and confidence changes. Measure NLL, predictive performance and calibration separately.
4. The discussion labels functional convexity as an unproven hypothesis. Similar learned functions do not establish a convex functional optimization landscape, and failed unaligned weight averaging is also compatible with ordinary permutation symmetries. Do not cite this as a theorem that message passing causes collapse.
5. Fixed-split initialization variation and ten repeated fits do not establish uncertainty across splits or graph populations. Historical image-versus-graph likelihood ratios compare different tasks/recipes; they are motivation, not a matched causal test.

## Consequences for our current experiment

The closest justified question is whether graph-informed private-factor initialization creates **useful sustained differences** beyond ordinary warming, stochastic continuation and random tangents. It is not whether more embedding distance is inherently beneficial. The five frozen arms already isolate these alternatives; no protocol or pool is changed after this reading.

The new primary is relevant context for the graph-initialization mechanism and the separate prediction-set proposal. It does not establish novelty for either. Any claim to improved epistemic uncertainty will require task-relevant uncertainty/shift evidence beyond accuracy or route separation.

Tolokers2 and Artnetviews are concrete industrial-task leads for future independent confirmation if the current pilot demonstrates utility. Before adding them, read GraphLand's actual releases, verify target/feature/graph semantics, use its competitive native recipe and prospectively freeze the split/metric. Do not add a favorable task retrospectively or substitute older Tolokers. No new data acquisition or training is authorized by this note alone; existing human authorization still requires the root's exact experimental preparation.

## Discovery memory

Two initial latest-first searches were noisy. Two narrower relevance-ranked searches identified this new paper and rediscovered already-read **Training Diverse Graph Experts for Ensembles** (2510.18370v1). The latter's cached assessments were reused, not reread. **SharpBalance** (2407.12996v1) was seen only in discovery metadata; it is not counted as a primary read or used to claim a theoretical delta. Preserve all four query receipts and access scope.
