# Next research decision after completed MolBACE evidence

15 October 2026. The completed MolBACE evidence supports improving member quality while measuring whether their differences help the served prediction. It does not support optimizing disagreement alone. Private BN substantially improves the members; neither it nor a simple averaging change surpasses independent four on mean selected VALID AUC. **Defer the dependent run decision until all nine BN by geometry by correction fits reach a terminal state and the complete prespecified contrasts are reviewed.** The two hypotheses below are bounded options for that later decision, not new run recommendations.

## What the completed evidence establishes

All results use seeds 101, 202 and 303, one official scaffold VALID split of 151 graphs, and the original selected checkpoints. TEST remains closed. AUC values below are percentages; pooling gain is pooled minus mean member AUC. [Completed pilot](../pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.md), [route analysis](../analysis/complete_route_diagnostic_v1/REPORT.md), and [BN decomposition](../analysis/member_pool_gap_decomposition_v1/REPORT.md).

| Bank | Mean member AUC | Pooled AUC | Pooling gain | Pooled BCE |
| --- | ---: | ---: | ---: | ---: |
| Native single | 73.32 | 73.32 | 0.00 | 0.779 |
| Independent four | 73.41 | 74.85 | 1.44 | 0.890 |
| Base GNNM | 62.01 | 67.78 | 5.77 | 0.719 |
| Alignment and variance | 64.24 | 71.17 | 6.93 | 0.551 |
| Private input correction | 66.07 | 71.76 | 5.69 | 0.941 |
| Both ingredients | 61.00 | 69.28 | 8.28 | 0.882 |
| Private BN | 72.63 | 73.61 | 0.99 | 1.086 |

**Weak members are an observed limitation.** Base GNNM benefits from averaging in every seed but starts 11.40 points below independent four in mean member AUC. Correction improves member AUC in all three seeds; geometry does so only in seed 101. Private BN improves mean member AUC by 17.55, 6.20 and 8.11 points over base, leaving gaps to independent members of 0.44, 1.26 and 0.64 points. This does not identify shared parameterization as the cause, or require every useful member to beat the native single. [BN member analysis](../analysis/bn_route_diagnostic_v1/REPORT.md).

**Available alternatives need not be usable alternatives.** The combined ingredients have the fewest positive–negative pairs reversed by every route: 123.7 versus 341.3 for independent four. Yet their members and pool are weaker. Majority-correct but pool-wrong counts expose cancellation, but do not construct a feasible gating oracle. The remaining private-BN pool gap of 1.233 points decomposes arithmetically into a 0.781-point mean member gap and a 0.452-point pooling-gain difference. The latter is not an identified causal diversity effect.

**Aggregation and probability quality remain separate questions.** Probability averaging raises base GNNM AUC from 67.778 to 69.011, still 5.885 points below independent four under that same rule. Private BN remains the strongest completed shared bank by mean AUC under either fixed rule, with gaps of 1.233 and 1.429 points. Correction BCE improves from 0.941 to 0.638 under probability averaging while AUC changes from 71.758 to 71.587. These fixed readouts were not independently reselected and do not choose a new serving policy. [All-bank aggregation](../analysis/complete_fixed_aggregation_v1/REPORT.md).

## What normalization does and does not explain

At the selected private-BN weights, own banks average 73.61 AUC, tied banks 59.06, and the mean of all three cyclic mismatches 57.98. Own banks beat tied and average mismatch in every seed, although one seed-101 shift reaches 77.22. This establishes sensitivity to route–bank correspondence at those weights. Separately averaging saved means and variances produces neither the original shared EMA nor a pooled variance. It cannot isolate the original private-BN gain. [Bank assignment](../analysis/bn_bank_assignment_v1/REPORT.md).

The original shared EMA advances four times per logical batch, with later routes receiving greater weight; each private bank advances once. Geometry and input correction can change route distributions, so running-state mixing may moderate their observed effects. Under ordinary TRAIN BN using minibatch statistics, buffer ownership alone leaves the instantaneous mathematical TRAIN output and gradient unchanged when inputs, RNG and parameters match. Evaluation, checkpoint selection and stopping can still differ. The original factorial therefore estimates ingredient effects **conditional on shared BN**, while comparisons between separately selected shared and private fits do not isolate a same-weight buffer effect. No private history can be reconstructed from a shared EMA, and VALID is not a recalibration population. [BN state](../catalogue/BN_STATE.md).

The nine-fit specification adds private-BN geometry, correction and combination for the three fixed seeds, reusing the completed BN-only and shared-BN cells. Review all outcomes, AUC and BCE, epochs and costs before interpreting moderation. Positive moderation with a still-negative private-BN interaction means reduced antagonism, not positive interaction. No partial nine-fit scores enter this decision. [Frozen specification](../pilot_20261015_bn_geometry_correction_v1/SPEC.json).

## Distinct catalogue questions still open

“Open” here means the completed MolBACE evidence does not supply the specified predictive comparison; an implementation or short engineering receipt is not such a comparison. Historical closed recipes retain their outcomes.

| Mechanism | Missing scientific comparison |
| --- | --- |
| L1 training credit | Own risk versus all-block or private half-pool credit under the declared native reductions. |
| L2 and C2 persistent responsibility | Uniform versus fixed positive weights; interpretable TRAIN structure versus its exact mass-matched permutation. |
| C3 distinct context targets | Common chemical-context mixture versus route-specific atom and bond targets, with matched target controls. |
| C6 learned aggregation | Mean versus global versus query-conditional weights under the same complete all-block objective. |
| C7 joint link-pattern supervision | Joint structural targets versus the declared independent-bit or count controls on a compatible link task. |

C1 has completed alignment-plus-variance evidence, but its alignment-only attribution remains open. The variance term penalizes low within-class sample variance; it does not guarantee escape from exact collapse. C4 input correction and C8 private BN each have evidence in this one molecular setting; this does not settle their interactions or cross-task transfer. The current nine fits address their declared BN interactions. [Catalogue](../catalogue/CATALOGUE.md), [interactions](../catalogue/INTERACTIONS.md), [current-source correction](../quality_architecture_literature_20261015_v1/ADDENDUM_CURRENT_SOURCE.md).

Deferred mechanisms remain distinct: actual native shared-step interference before a faithful PCGrad/CAGrad comparison; TRAIN retrieval; private factor distributions; changed propagation operators; paid initialization or learning-response credit; functional-diversity interventions; and complete predictive task definitions for prototypes. The defined class-code flow lead also remains deferred because it introduces a solver and serving contract without current evidence of shared-ensemble benefit. A CE residual is not flow matching. These are not reasons to reopen exact closed recipes or add a Cartesian product.

Retained literature constrains interpretation: TabM permits individually weaker but collectively useful members; Graph Experts warns that oracle domains or structure alone need not yield practical fusion gains; GraphDIVE supplies molecular view/fusion ancestry, not a verified repulsion control here. Negative gradient cosine does not establish harmful Adam interference. Endpoint-weight ancestry provides no posterior or variance guarantee. Current C6 has a complete objective and detached prediction descriptors; its older builder description is superseded. These scoped readings establish no novelty guarantee. [Retained synthesis](../literature/SCOPED_CONCLUSIONS.md), [quality reading](../quality_architecture_literature_20261015_v1/REPORT.md), [C6 correction](../quality_architecture_literature_20261015_v1/ADDENDUM_CURRENT_SOURCE.md), [resampling](../literature/structured_resampling_prior_scan_v1/CONCLUSIONS.md), [flow disposition](../literature/flow_matching_shared_backbone_method_v1/READING_AND_DECISION.md).

## Two representative hypotheses for the later decision

Both options use MolBACE GIN with dropout 0.5, the same three paired seeds, whole VALID selection and native operators, optimizer and task reductions. The currently prepared packages retain shared sequential native BN; keep that identical across each option's arms. Substituting private BN would require a separately frozen interaction definition after the nine-fit review. Retain existing M1 and genuine I4 as context; neither is the mechanism control. Report mean and worst member quality, served AUC/BCE, repaired and introduced pair errors, convergence and actual work. No VALID-fitted gate, coefficient search or outcome-selected seed is part of either option.

### Conditional aggregation can exploit useful alternatives

Compare the source-defined mean, global and conditional aggregation methods, all trained with **the same all-block J = 0.5 F + 0.5 L of the actual served output**. The controls add 0 and 4 parameters; the conditional gate adds 20. Its detached confidence and disagreement descriptors feed a hidden-4 scorer, with a 0.05 route floor and exact uniform-mean initialization. Detachment removes the descriptor gradient path to logits; the direct weighted-logit path remains. Mean trained only with F is context, not the matched primary control. Also retain each selected gated bank's same-state uniform mean. [Exact methods](../conditional_aggregation_v1/METHODS.json).

The hypothesis predicts a positive paired mean AUC difference for conditional versus global and mean J-trained pipelines, and a positive conditional-versus-mean readout difference on the same selected conditional bank. Nonpositive pipeline differences leave conditional utility unsupported; a nonpositive same-bank difference rejects the proposed readout advantage at those states. Improvement only over mean-own-F would not distinguish aggregation from changed credit. AUC improvement with worse BCE supports ranking only. Pair-coverage or entropy changes alone cannot rescue a negative predictive result.

### Persistent graph size responsibility provides useful specialization

Compare the existing uniform GNNM control, within-TRAIN-class atom-count quartile responsibility, and its frozen whole-vector permutation within each class. The declared weights are **w_m(i) = 0.5 + 2 q_m(i)**; tie handling preserves sum_m q_m(i) = 1 and per-class member weight mass N_c exactly. Shared parameters receive ordinary F; private parameters receive weighted W, with unchanged native mixed-batch BCE reduction, no minibatch renormalization, no added parameters and the same raw-logit mean serving. The permutation preserves the weight law and class/member masses while removing the graph-size relationship. An unmatched exponential-IID table is not needed. [Exact methods](../structural_class_responsibility_v1/METHODS.json).

The hypothesis predicts a positive paired mean pooled-AUC difference over the permutation and uniform controls, with member quality and BCE retained. A nonpositive structured-minus-permuted difference rejects the proposed benefit of size structure; equal improvements by both weighted arms support generic reweighting instead. Improvement confined to assigned TRAIN size strata, with no pooled VALID gain, does not establish useful ensemble specialization. This option isolates assignment first; pooled-credit, context-target and adapter combinations remain separate questions.

Three selected-VALID seeds provide descriptive exploratory evidence. Their routes, molecules and graph pairs do not enlarge the number of independent training repetitions. Future confirmation remains separate from this authorized exploratory programme.
