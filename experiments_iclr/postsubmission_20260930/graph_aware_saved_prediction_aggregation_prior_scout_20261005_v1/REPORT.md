# Graph conditioned stacking of frozen shared predictions

5 October 2026. Literature assessment and one prospective utility hypothesis.

**Retain one attributed hypothesis:** a small stacker can use graph-local prediction disagreement to improve the fixed pool of an already trained shared ensemble. Its quality is unmeasured. Learned graph gates, stacking, hidden-state attention and graph-aware calibration are established; this proposal asks whether they recover useful information in this particular frozen bank. It creates no launch or gate and requires no new backbone training for the aggregation operation itself.

## The nearest methods

Index69 already contains Link-MoE, stacked link forests, MoE-NP v1, PreGS, HGEN/LHGEL, BatchEnsemble, TabM and LoRA-Ensemble. Link-MoE is especially direct: second-stage, pair-specific softmax weights combine independently trained expert scores using structural heuristics and endpoint features. Its Collab gate uses part of validation for training; that supervision opportunity must be matched rather than inherited silently. PreGS already fuses frozen expert features and logits. [1–3]

Five missing or unresolved primary scopes were read for this assessment:

| Source and exact scope | Established operation and relevant limit |
| --- | --- |
| **GAMLP**, 2108.10097v3, Section 3 | Precompute propagated feature banks, then learn node-specific smoothing, recursive or JK attention and an MLP. Hidden-state or hop fusion is direct ancestry. Full GAMLP includes label propagation and repeated pseudo-label/distillation stages; those are additional operations, not automatically part of a cheap frozen stacker. Its GAMLP description of JKNet/DAGNN is a cited lead; their original papers were not read here. |
| **META-DES**, 1810.01270v1, Section 3 | Learn classifier competence from local accuracy, local posterior behavior, output-profile matches and confidence; select members and majority-vote. Base fitting, meta-training and dynamic-selection reference data have separate roles. Four meta-records per node do not create four independent labeled observations. |
| **Ratio-binned scaling**, 2206.01570v1, Sections II-C2 and IV-A/B | Approximate same-class-neighbor ratio from predictions, fit bin temperatures on validation, preserve each node's class ordering. True neighbor labels occur in the explanatory analysis; they cannot be query features at deployment. |
| **GATS**, 2210.06391v1, Section 5 and Appendix A.3 | Produce graph-conditioned node temperatures from logit distributions, relative confidence, neighbor similarity and distance to training nodes. This is post hoc graph-aware processing of computed outputs. Accuracy preservation requires positive temperatures; the printed softplus-plus-learned-bias formula does not by itself certify an implementation's positivity. The native calibration/selection roles differ from the design below. |
| **MoE-NP**, 2412.00418v3, Section 3 | Random-walk local patterns, a feature-based edge discriminator and degree feed local/global MLP context, then a softmax expert gate. This extends the saved v1 scope, with no additional paper identity. Its printed objective is binary logistic loss despite a multiclass node-classification setting; author-code reduction and exact native reproduction remain unqualified. |

These are bounded method reads, with no result, proof or runtime adoption. The MoE-NP method layout incidentally contains a public benchmark table; its numbers were not used. No target results were accessed. [4–8]

## One concrete modification

For a frozen node ensemble, retain the complete class-logit vector `z_m(v)` for each member. Form probabilities `p_m(v)` and the disagreement residual `r_m(v)=p_m(v)-mean_j p_j(v)`. From the authorized, unchanged graph, compute one cached neighbor mean `P r_m(v)`, preserving self-loop and normalization conventions explicitly. The gate receives each member's own prediction distribution, its residual, its neighbor residual, and degree. No ground-truth homophily, query label or post hoc correctness indicator enters these features.

A small MLP generates softmax member weights `alpha_m(v)`. Its output is the convex probability mixture `sum_m alpha_m(v) p_m(v)`, fitted with ordinary supervised NLL on a reserved fusion-training population. Regularize toward the uniform pool using a fixed prospective rule. That shrinkage controls estimation with few labels; it is not a diversity reward. No masks, new experts, inner updates or backbone gradients are introduced.

The **actual delta from MoE-NP** is the node-pattern input: cached member prediction residuals and their graph-local summaries replace its random-walk/learned-edge-discriminator pattern extractor. The graph supplies context about where a member differs from the pool, rather than a competence oracle. GAMLP and GATS justify this inexpensive information interface; they do not prove it improves accuracy. A lone outlier can preserve useful boundary evidence or be confidently wrong. The hypothesis is that this distinction is predictable on untouched labels, beyond global weighting and a logits-only stacker.

For an unchanged-prior comparison, retain MoE-NP's printed local/global pattern extractor and softmax aggregation operator on the same fixed expert bank and probability endpoint. This bank/endpoint substitution is an explicitly disclosed adaptation, not a reproduction of its published experiment; exact native code and multiclass loss need qualification before any native-reproduction claim. Also retain ordinary logits-only stacking on the same outputs. A win over averaging alone cannot establish that graph-local residuals add value.

This is an attributed composition with a precise utility question, not a cleared novelty claim. The earlier novelty-based router disposition remains historical evidence. The proposal needs neither hard selection nor a claim that its gate is unprecedented. Every member has already run: it adds graph-summary and aggregation work and saves no executed route computation.

## Cheap first comparison and strong controls

Use one complete node graph with a properly reserved fusion population and an existing fully authenticated four-member bank. Keep the bank, checkpoint and prediction population fixed. Fit only small aggregation maps; do not start new backbone fits to create this diagnostic.

Compare mean raw logits, mean probabilities, per-member scalar-temperature averaging, parameter-free mean class-rank fusion, global constant weights, ordinary logits-only stacking, the adapted unchanged MoE-NP gate, and the proposed graph-residual gate. For a link adaptation, rank fusion must use each complete fixed query candidate universe; the node operation is not silently transferable to link-query space. These are necessary cheap controls before adding hidden-state branches.

Give a capable **single stacker** every prediction/context feature and the same nonlinear and multiplication primitives. It can reproduce the gated mixture exactly and can learn a joint class decoder. Such a single already consumes the ensemble's computed evidence; this comparison concerns the aggregation architecture, not elimination of the original ensemble acquisition cost. Retain a competent standalone single-GNN quality anchor and an ordinary independent ensemble where their already acquired banks have the same eligible label and selection roles. Apply the same aggregation training opportunities to the independent bank. Without those comparisons, a result supports useful output processing only, not a special quality benefit from sharing.

Intermediate hidden states can later supply a GAMLP/JK-style cached depth bank if they already exist with authenticated custody. Keep member coordinates separate until aligned; averaging independently learned hidden coordinates is not justified by their equal dimensions. The first test uses aligned class predictions and introduces no hidden-state export or extra GNN run.

## Prevent supervision leakage

Reserve four disjoint roles before backbone training: backbone-fit labels, backbone-selection labels, fusion-fit labels, and final aggregation-assessment labels. Fusion-fit labels train temperatures, global weights and stackers; select their complexity on a separately predetermined subdivision. Final assessment labels choose no model, calibration, gate, graph statistic or stopping epoch. All baselines receive identical fusion supervision and selection opportunity.

Alternatively use correctly masked out-of-fold TRAIN predictions, with label-derived graph inputs and target facts excluded for each fold. For link tasks, exclusion includes target edges and any forbidden incident-label/support information. That alternative generally requires additional backbone fitting and therefore is not the cheap first option. Transductive use of allowed unlabeled graph/features is compatible with the design, but it does not make connected nodes independent observations.

An already selected VALID population can support only a declared retrospective exploration. Splitting it now or cross-fitting only the aggregator cannot erase its use in checkpoint selection. No untouched fusion population was authenticated in this task. If the existing bank lacks one, stop the claimed honest test; do not create a new expensive training campaign merely to make this memo executable.

## Falsification and uncertainty

Stop the graph-residual explanation if it does not improve complete-task quality over the strongest cheap control and the unchanged-prior/logits-only stackers. Better NLL or ECE without the desired accuracy gain supports calibration utility only. If the joint single matches, use the simpler same-evidence architecture; if the independent bank matches, sharing-specific benefit remains unsupported. Graph context cannot repair an identical error component carried by every member.

Report complete paired-block gaps and graph-dependent uncertainty. A small positive selected-development difference is unconfirmed. A future worthwhile-effect threshold and uncertainty rule must be declared before untouched assessment, with no subgroup, gate-depth, mask or seed search after failure. There is no predictive gain or new gate established here.

## Sources

1. [Index69](../literature_memory/index_v69/LITERATURE_INDEX.json), records 0, 5, 8, 79, 97, 101, 114, 115 and 126; reused conclusions only.
2. [Link-MoE](https://arxiv.org/abs/2402.08583v2) and [graph link stacking](https://arxiv.org/abs/1909.07578), saved scopes in index69.
3. [Saved graph-view fusion assessment](../learnable_graph_view_quality_prior_synthesis_20261005_v1/ASSESSMENT.md), [router decision](../moe_direction_root_v1/DECISION.json), and [quality triage](../quality_literature_to_design_triage_20261005_v1/REPORT.md).
4. [GAMLP exact v3](https://arxiv.org/html/2108.10097v3).
5. [META-DES exact v1](https://arxiv.org/html/1810.01270v1), DOI [10.1016/j.patcog.2014.12.003](https://doi.org/10.1016/j.patcog.2014.12.003).
6. [Ratio-binned GNN calibration exact v1](https://arxiv.org/html/2206.01570v1).
7. [GATS exact v1](https://arxiv.org/html/2210.06391v1).
8. [MoE-NP exact v3](https://arxiv.org/html/2412.00418v3).

Exact public retrievals, paragraph/equation locators, reused bindings and read accounting are saved alongside this memo. No target endpoint, model, data, score or training-history payload was opened, and no scientific computation or launch occurred.
