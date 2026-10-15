## Shared-backbone ensemble research — 15 October 2026

We are developing several trainable routes through one native GNN, with small
additional parameter cost. Each compatible setting compares the native single
model, original GNNM and four independently trained native models. The current
residual and responsibility studies have **no confirmed quality advantage over
independent ensembles**; earlier setting-specific findings remain in the history.

### Latest completed comparison

The MolBACE/GIN studies now contain **39 unique completed fits and 33 prediction
banks**, with three paired seeds and the full official TRAIN/VALID split.
Native operators, features, learning rate and weight decay are unchanged.
Every fit uses its first-best validation-AUC checkpoint under the frozen
minimum 100 / maximum 1,000 / patience 50 controller. TEST remains closed.

| Method | Mean VALID AUC ± sample SD (%) | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.32 ± 1.17 | 0.779 |
| Independent four-model ensemble | 74.85 ± 0.50 | 0.890 |
| Original GNNM | 67.78 ± 4.39 | 0.719 |
| GNNM + alignment/variance | 71.17 ± 2.66 | 0.551 |
| GNNM + residual correction | 71.76 ± 3.91 | 0.941 |
| GNNM + both ingredients | 69.28 ± 1.80 | 0.882 |
| GNNM + private BN state | 73.61 ± 1.06 | 1.086 |
| Private BN + alignment/variance | 73.65 ± 1.64 | 1.122 |
| Private BN + correction | 72.19 ± 1.41 | 0.816 |
| Private BN + both | 73.03 ± 1.61 | 1.030 |
| Residual-only routes + private BN | 74.52 ± 2.66 | 1.952 |

Higher AUC means better ranking; lower BCE means better probability loss.
These are selected validation results from one small, repeatedly examined
scaffold split. They do not establish transfer or statistical equivalence.
All seeds, losses, unsuccessful combinations and costs are retained.

The newest method replaces the BatchEnsemble factors with four rank-eight
additive corrections after the native input encoder. Their initial output is
zero, so all routes start with the native model's function. They share the GNN
weights and BN affine parameters, and keep separate BN running statistics.
There are **19,200 extra learned parameters: 1.0183%** of the native GIN.

Removing the factors improved AUC over factors-plus-residuals in all three
seeds (+2.33 points on average), while worsening BCE. The result is still
0.33 points below I4 on average, with much greater seed variation. Individual
routes have mean AUC 74.56%, but averaging contributes −0.04 points. About
95.54% of pool ranking errors are shared by every route. The next objective
is useful specialization while preserving member quality.

New analysis of every completed bank locates the probability-loss problem:
residual-only worsens loss on positive molecules and improves it on negatives
in all three seeds. At a fixed diagnostic threshold of 0.5, it misses
105/95/78 of 130 positives, versus I4's 67/46/60. The five largest individual
loss increases account for only 12–38% of net excess BCE. Ranking recovery
therefore does not imply reliable probability estimates. The observed TRAIN/
VALID prevalence difference is reported without fitting a validation correction.

- [Class-specific errors and confidence analysis](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/completed_probability_failure_v1/INTERPRETATION.md)

- [New results, paired differences, uncertainty and costs](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_residual_only_private_bn_v1/complete_results_v1/INTERPRETATION.md)
- [Earlier complete factorial and error analysis](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_bn_geometry_correction_v1/complete_results_v1/INTERPRETATION.md)
- [Convergence audit of the preceding 36 fits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/completed_molbace_convergence_v1/FINAL_INTERPRETATION.md)

### Current specialization study

The running six-fit study compares uniform route supervision with graph-size
responsibilities and a within-class shuffled-assignment control. Shared
parameters receive the native unweighted loss; only the small private
corrections receive the weighted loss. The same mean-logit serving rule and
native training settings are retained. Six new fits reuse all paid controls. The source was published in commit
`7ad17c23a3e6bd66d3bb5a8900b2cc565dce3c7f`, and training started on the
authorized allocation at 16:21 UTC. At 18:31 UTC, three of six fits had completed and the fourth had reached 126 full epochs. We will interpret the comparison after all six fits finish. No partial
accuracy result is used to change the methods or stopping rules.

ACE (ICCV 2021) already separates shared and specialist gradient credit. That
principle is attributed to prior work. Our empirical question is whether the
association with graph structure improves these shared GNN routes. A learned
readout remains a separate possibility: fixed-bank oracle bounds show limited
room, without demonstrating a trainable gate.

- [Frozen six-fit protocol](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/pilot_20261015_structural_private_risk_v1/PROTOCOL.md)
- [Closest prior work and claim limits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/literature/structural_private_credit_followup_v1/NOTE.md)
- [Why route quality precedes more diversity](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/next_quality_mechanism_20261015_v1/RECOMMENDATION.md)
- [Next setting: PubMed / PolyFormer-Cheb, 24-fit preparation](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/second_setting_protocol_v1/DECISION.md)
- [Probability-quality hypothesis and its limits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/quality_specialization_synthesis_v1/SYNTHESIS.md)

The next source preparation uses the native PubMed/Cheb setting, without BN.
Its fixed VALID split has class counts 72, 1,919 and 1,952. All classes occur,
but the smallest has only 1.83% of the records. We retain the source split and
selection, report per-class results, and disclose prior PubMed development.
The proposed comparison requires 24 new fits because previous PubMed methods
used different native settings. No transfer fits have started. A separate
proper-loss/readout hypothesis addresses probability quality, with no claim
that an NLL gain alone would meet the accuracy-over-I4 goal.

The PubMed implementation is now source-reviewed and has passed all three
full-graph CPU checks: uniform, structured and shuffled responsibilities. The
checks exercised six training updates and nine validation passes, including
fresh checkpoint restore. Actual parameter overhead is **8,192 / 592,435 =
1.38%**. GPU cost measurement and full training remain pending; these checks
provide no accuracy result.

A separate prospective idea directly supervises positive–negative score
orderings while retaining the native classification loss. It includes
independent ensembles trained with the same auxiliary, so a better loss alone
cannot be mistaken for an advantage of sharing. Its source is now implemented, including genuine native controls and a TRAIN-only support inspector. All four CPU checks have now passed, including two training updates and fresh checkpoint restore per variant. The complete proposed 21-fit comparison is frozen and its allocation preflight passed. Its full-epoch CUDA cost probe is prepared; no scientific fit has run.

The default next scientific priority is the prepared **PubMed/Cheb transfer**.
Another task and native backbone without BN provide a stronger scope test than
further decisions on the same small MolBACE VALID split. Pairwise competence
remains a separate hypothesis. Both settings have prior development exposure;
neither constitutes independent confirmation. Actual full outcomes and costs
will determine admission.

- [Next-study priority, controls and falsifiers](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/next_study_priority_20261015_v1/NOTE.md)

- [Current research direction and decisions](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/CURRENT_DIRECTION_20261015.md)
- [Actual pairwise support and CPU qualification](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/PAIR_QUALIFICATION_SUMMARY_25.json)
- [Pairwise implementation and bounded checks](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/molbace_pairwise_competence_v1/README.md)
- [Pairwise prior work and claim limits](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pairwise_competence_prior_followup_v1/NOTE.md)
- [Full-graph implementation checks and costs](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/PUBMED_CPU_SUMMARY_24.json)
- [Ranking-supervision hypothesis and required controls](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/pairwise_competence_synthesis_v1/NOTE.md)

### Broad benchmark preparation

The existing proposal contains **6,460 physical fits across 7 datasets,
8 architecture families and 27 fixed source presets**: 270 source references,
1,080 independent native fits and 5,110 shared fits. M1 reuses I4 body 0.
An additional 270 export tasks give 6,730 tasks and 6,190 prediction rows.
New research pilots are separate from this unlaunched roster.

The 27 contracts, 99 split proofs and 43 evaluation populations are prepared.
The genuine composed-normalization molecular pipeline passed 38 updates,
restore, selection and serving. The repaired NCN pipeline passed two epochs,
34 updates and four whole-VALID calls. The original BUDDY pipeline also passed two epochs, 3,170 updates, four VALID
calls, restore and selected-state export. Its child wall time was 1,066.69
seconds. The separate composed-normalization BUDDY check remains pending.
These are execution checks, not scientific model-quality results.

The selected node-state replay now passes: one restored model reproduced all
665 VALID predictions exactly, with four native forwards and zero training.
The preceding failed attempt exposed legitimate non-tensor checkpoint metadata;
the scorer now preserves and checks that state. Both attempts remain recorded.
NCNC selected serving and the composed BUDDY check are still pending.
Final serving coverage, measured three-GPU placement and a reliable total
budget remain incomplete. One measured MolHIV GIN-VN TRAIN epoch took 582.45
seconds; its 100-epoch TRAIN proxy is 16.18 worker-hours before validation.
It is not a campaign ETA. All 18.77 workers remain disabled. The large run
requires the user's review and explicit GO.

- [Plain-language benchmark design](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/BENCHMARK_OVERVIEW.md)
- [Method coverage and matched controls](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/benchmark_review_map_v1/README.md)
- [Native source settings](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/expanded_study_design_v1/NATIVE_SETTINGS.json)
- [Runner preparation and outstanding work](experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/README.md)

The dated sections below preserve earlier states and proposals.
