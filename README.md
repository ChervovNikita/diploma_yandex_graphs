# GNNM: shared propagation in graph ensembles

## Shared-backbone ensemble research — 15 October 2026

We are developing several trainable routes through one native GNN, with small
additional parameter cost. Each compatible setting compares the native single
model, original GNNM and four independently trained native models. The current
methods have **no confirmed quality advantage over independent ensembles**.

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
authorized allocation at 16:21 UTC. At 17:30 UTC, two of six fits had completed and the third was running. We will interpret the comparison after all six fits finish. No partial
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
cannot be mistaken for an advantage of sharing. Its source is now implemented, including genuine native controls and a TRAIN-only support inspector. All four CPU checks have now passed, including two training updates and fresh checkpoint restore per variant. The proposed 21 scientific fits have not run.

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
## Real-data smoke checks — 14 October 2026

Completed **167 distinct configuration checks** across the redesigned study:
56 node, 96 molecular and 15 link cases. Each uses real source TRAIN data,
two initial optimizer updates, complete VALID evaluation and checkpoint
save/reload. A third update is run and replayed; all four callbacks and all
four independently owned I4 optimizers are charged. TEST is not scored.

The latest seeded attempts report **94 strict passes and
73 failures**. The separate functional diagnostic confirms
146 cases; incomplete
legacy telemetry remains unknown. Floating replay failures stay failed in
the reports, even when initialization, updates and exact restoration succeed.
These checks establish execution evidence, not prediction quality or convergence.

The source audit corrected link bias scaling and limited NCNC output factors
to the top-level task scorer, retaining its native recursive completion.
Corrected link cases were rerun. All earlier outcomes and repair history remain.

A concurrent Physics Mono/private-ensemble-credit + NCNC/I4 probe passed. Its maximum sampled own GPU usage was 64.7 GiB, with at least 13.6 GiB device memory free. This qualifies the tested allocation pairing, not every combination or the 18.77 runtime.

**Full scientific launch remains disabled.** Complete epoch handling,
source/main selectors, independent I4 stopping/checkpoint ownership, selected
TEST serving and the executable three-GPU launcher still require integration.
The old v29 launcher remains retired; no full-campaign GO has been received.

- [Smoke methodology and limits](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/README.md)
- [All configuration checks and retained failures](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/REPORT.md)
- [Remaining full-run implementation](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/controller/FULL_RUN_GAPS.md)

## Revised wrapper benchmark — 13 October 2026

**The earlier 80,850-fit proposal is superseded.** The new proposed study keeps
the reference backbone, features, preprocessing, splits, evaluator and native
optimizer presets unchanged. Its main comparison is a native single model,
four genuinely independent native models, and a four-route GNNM with one shared
native backbone and private factors at explicitly mapped input/output boundaries.
There is no learning-rate or weight-decay search.

| Task | Native architectures | Datasets | Fixed presets |
| --- | --- | --- | ---: |
| Node classification | PolyFormer, separately reported Mono/Cheb source settings | CiteSeer, PubMed, Coauthor CS, Coauthor Physics | 8 |
| Molecular graph classification | GCN, GCN+virtual, GIN, GIN+virtual | MolHIV, MolBACE | 16 |
| Link prediction | NCN, NCNC, BUDDY | OGBL Collab | 3 |

That is **7 datasets, 8 native architectures, 27 fixed presets and 2,480 planned
cold fits**: 270 native source-horizon checks plus 2,210 main comparison fits,
with ten paired trials per preset. The single reuses exactly member zero of the
ordinary ensemble. Original-paper GNNM experiments are not rerun.

The additional GNNM candidates change only training on the same wrapper:
mixed member/ensemble loss; ensemble-loss credit confined to private factors;
fixed private example responsibilities; and a within-route supervised contrastive
loss on eligible node tasks. None is claimed successful or novel from this design.

Published scores and our source-horizon checks are distinct from the main
longer-training comparison. Main caps/minimums/patience are 10,000/2,000/100
for PolyFormer and 1,000/100/50 for molecular and link presets. Native LR,
WD, optimizer groups and constant schedules remain fixed. Curves, stopping
reasons, failures, per-trial scores and paired uncertainty will be retained.

OGB's two published dropout presets are separately reported because the paper
does not identify each winning value. Conflicting Polynormer paper/release rows
and unavailable HGB hidden-test protocols remain explicitly deferred/excluded.

**Design and metadata checks are complete; the scientific runner is not ready.**
The portable boundary core and loss rules have passed static checks. Exact
source-specific categorical/parallel/reused-boundary adapters, data/trainer
integration, checkpoint/evaluator integration and physical resource admission
are still pending. No new fits have started and no GO has been received.
The old public v29 launcher refuses execution. Current preparation accesses
only the authorized one-GPU allocation; three-GPU assignments are prospective.

- [Complete revised design and methods](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/README.md)
- [Exact proposed counts](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/SUMMARY.json)
- [Fixed source settings and published score cells](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/SOURCE_CELLS.csv)
- [Exact fit roster](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/FIT_ROSTER.csv)
- [Native-model preservation contract](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/methods/PRESERVATION_CONTRACT.md)
- [Portable core and pending source ports](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/runtime/README.md)

The dated proposals and research outcomes below are preserved history.

## Superseded massive-study proposal — 13 October 2026

**The finite source and protocol inventory is complete; resource admission and scientific launch approval remain incomplete.** The exact frozen v29 master contains **80,850 model-fit groups in 98 study specifications**, supplied by **51 packets**, across **12 datasets and 21 named architecture recipes**. The earlier count of 16 named architectures was incomplete; source/protocol configurations retain their alias mappings.

The final canonical coverage audit records **273 canonical rows and 361 aliases**: **263 prepared selected recipe rows** (228 scientific variants and 35 references/controls), and **10 explicit inactive, nonbenchmark, or source-only rows**. No canonical source-port row is missing and no row is excluded because of historical outcome. This establishes source/protocol preparation, not an accuracy result or a count of executed methods.

The final source ports include the copied-native relation policies and complete private-hop panel, literal structural/cardinality and conditional-pattern procedures, direct hidden-state head refits, affine head-capacity controls, and learned graph-serving operators. The named nonlinear-head alias remains an explicit unimplemented source disposition. Namespace import successors preserve numerical bodies. Protected76, bounded HPO, per-GPU commands, and checkpoint-retention helpers are bound to the final controller. All engineering failures and their costs remain recorded.

The publication includes reviewable authored source files, compact coverage proofs, and a sealed **13,038,922-byte source archive with 3,060 exact repository-relative source/metadata paths**. The final PLAN and SUMMARY and all 98 study specifications are compressed. The archive manifest records per-file bytes and SHA256 hashes; its restore helper uses exclusive writes and rejects unequal existing files. Datasets, checkpoints, banks, private keys, installed dependency trees, `.venv`, and transport payloads are excluded from publication. Large scientific artifacts remain on the server; installed libraries stay in the environment provenance.

**Resource admission, server77 live admission, and ETA are incomplete. No GO has been received. This is not a claim that the study is ready to run.** Current preparation uses the allocation only; 18.77 has not been accessed. Source scheduling weights are planning bounds and do not establish execution time or concurrent GPU feasibility. Original paper scores and all completed outcomes remain unchanged. Original GNNM remains a comparison on compatible new datasets, and the original five paper datasets are excluded from the proposed massive study.

- [Exact final proposal and prepared wrapper commands](experiments_iclr/postsubmission_20260930/final_massive_audit_20261013_v1/PROPOSAL_V2.md)
- [Exact registered counts](experiments_iclr/postsubmission_20260930/final_massive_audit_20261013_v1/EXACT_COUNTS.json)
- [Dataset family map](experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/DATASET_FAMILY_MAP.json)
- [Architecture family map](experiments_iclr/postsubmission_20260930/final_sync_master_v29_20261013_v1/ARCHITECTURE_FAMILY_MAP.json)
- [Final audit and manifest/bundle/resource bindings](experiments_iclr/postsubmission_20260930/final_massive_audit_20261013_v1/AUDIT_REPORT.md)
- [Final resource report and remaining physical admission limits](experiments_iclr/postsubmission_20260930/final_v29_resource_status_20261013_v1/FINAL_RESOURCE_REPORT.md)
- [Final finite run census](experiments_iclr/postsubmission_20260930/final_census_master_v29_20261013_v1/RUN_CENSUS.md)
- [Closed canonical source audit](experiments_iclr/postsubmission_20260930/final_canonical_coverage_audit_20261013_v1/CLOSED_SOURCE_AUDIT_v3.json)
- [All 273 source dispositions](experiments_iclr/postsubmission_20260930/final_canonical_coverage_audit_20261013_v1/source_port_successor_review_v3/ALL273_REVIEW_TABLE.csv)
- [Exact final v29 audit binding](experiments_iclr/postsubmission_20260930/final_canonical_coverage_audit_20261013_v1/FINAL_MASTER_V29_BINDING.json)
- [Explicit source publication inventory and restoration instructions](experiments_iclr/postsubmission_20260930/publication_inventory_v29_20261013_v1/README.md)
- [Final packet input index](experiments_iclr/postsubmission_20260930/practical_master_inputs_20261013_v29/INPUT_INDEX.json)
- [Preserved predecessor README at expected HEAD](experiments_iclr/postsubmission_20260930/publication_inventory_v29_20261013_v1/ROOT_README_AT_EXPECTED_HEAD.md)

The dated research history below preserves its recorded outcomes and observation times.

## Research status — 12 October 2026

**The larger study is being prepared and remains disabled until the user approves its exact run manifest.** Original GNNM’s five paper datasets will not be rerun. Original GNNM, capable single models and genuine independent ensembles will be tested on compatible new datasets.

| Work | Current state | Meaning |
| --- | --- | --- |
| Physics private GRU memory | Last observed: 14/30 fresh fit groups complete; full cohort still training | Wait for the whole comparison before interpreting results |
| PubMed private landmark memory | Complete: 36 fresh groups, 54 trajectories, 39 CAL endpoints | No improvement over its source; all combined criteria failed |
| Training refactor | Full-budget engine; endpoint-conditioned and bootstrap procedures; original OGB author runner; frozen v8 manifest loads 95,856 job records | Engineering verification; no new predictive results |
| New study data | WikiCS, Amazon Photo, CiteSeer, PubMed, Coauthor CS/Physics, MolHIV, MolBACE and ogbl-collab in v8; zero new study fits/scores | Original GNNM included across all 34 standardized studies; original five excluded |
| Large-study coverage | Disabled v8: 38,309 planned model-fit groups plus 160 separate author trajectories | Remaining method ports, resource costs and final sync must close before approval |

PubMed LM4 and its listed source-equivalent controls selected checkpoint 0 after 100 executed continuation updates. Its mean development accuracy was **90.8308%**, exactly the selected source score in each seed. The mean difference versus equipped I4 was +0.2280 points with one negative seed; calibrated NLL was worse. This preserves a failed procedure and motivates better prospective training diagnostics, without claiming a learned improvement.

The refactor records complete TRAIN/SELECT trajectories, sample-weighted molecular epoch losses, actual optimizer counts and checkpoint/RNG restoration. Two experiment slots per GPU are prepared for the allocation plus 18.77's two GPUs, with fixed assignments and global phase gates. Current preparation uses allocation only; 18.77 synchronization and live environment checks are pending.

The catalog has 273 canonical entries across 73 families, including controls, diagnostics and unfinished implementations. It is **not a count of runnable methods**. Source-only and blocked procedures add no executed experiments.

The [v8 run roster](experiments_iclr/postsubmission_20260930/training_pipeline_v2_20261012/docs/INTEGRATED_RUN_ROSTER_20261012_V8.md) gives the current finite counts, budgets and GPU assignments. Eight new Coauthor memory checks passed; two Physics coupled-loss cases exceeded 32 and 60 GiB child limits. A source-preserving recomputation option passed six synthetic CPU checks and awaits actual-input memory qualification. No accuracy or novelty claim follows from these engineering checks.

- [Current research state](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/RESEARCH_STATE.md)
- [Training pipeline and preparation status](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/training_pipeline_v2_20261012/README.md)
- [Complete PubMed outcome and checkpoint selections](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/pubmed_LM4_complete_interpretation_root_20261012_v1/REPORT.md)
- [Recurrent-memory protocol and source](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/native_meanSAGE_private_GRU_execution_source_20261011_v1/README.md)
- [Complete outcomes and their reference limits](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/complete_quality_gain_reference_confirmation_evidence_map_20261011_v1/REPORT.md)

### Current recurrent-memory architecture

One ordinary source GNN first learns a useful local representation. Its first layer is then frozen. Four routes read that common representation along four fixed, label-free graph traversals. They share the main GRU weights, while small private gate factors and biases can learn different decisions about retaining graph history. A shared residual readout adds the resulting memory to each route's representation; the native GNN decoder predicts separately for each route.

```mermaid
flowchart TD
    X[Node features and graph] --> H[One acquired local GNN representation]
    H --> A[Private gate factors: route 1]
    H --> B[Private gate factors: route 2]
    H --> C[Private gate factors: route 3]
    H --> D[Private gate factors: route 4]
    A --> T[Shared native decoder applied separately to each route]
    B --> T
    C --> T
    D --> T
    T --> P[Four class-probability predictions]
    P --> M[Mean class probabilities]
```

All four memory routes use the same slow GRU weights. Each traverses the complete graph component in both directions and resets its state at component boundaries. The candidate uses **one pretrained source GNN**. Additional source models are acquired only for the independent references.

The fixed comparison includes decoder-only continuation, ordinary and factorized single learners using all four traversals, private native-factor placement, and genuine ordinary/factorized independent ensembles. Each independent learner has its own optimizer, validation selection and four traversals. Member competence compares four private candidate routes with four independent body-average predictions. All runs and calibration endpoints must finish before development outcomes are released together.

Recurrent BatchEnsemble, gated graph models and graph memory have prior literature. [Closest primary comparisons and conditional follow-up controls](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/common_P1_private_memory_primary_comparison_20261011_v1/REPORT.md) describe those overlaps. A new methodological advantage remains unproven. A worthwhile prediction result must survive stronger references, mechanism controls and an unused confirmation pipeline before paper promotion.

### Current landmark-memory architecture

LM4 uses one freshly learned native PolyFormer source. Its raw representation immediately before the class head is frozen. Four groups of fixed, label-free graph landmarks use private learned address offsets to collect remote node evidence. Graph-distance biases condition this exchange. Shared attention maps read that memory back into the common representation, and a shared native class head produces four route predictions whose class probabilities are averaged.

S64 and FS64 equip single learners with all 64 landmarks and ordinary or factorized memory maps. Their I4 references contain four complete, independently selected equipped learners, each using its own fresh source ancestor. Competence compares four private LM4 routes with four independent bodies. All 36 fresh fit groups, 54 optimizer trajectories and 39 calibration endpoints must close before one complete development release. PubMed is an encountered graph; fresh roles do not establish unused confirmation. No quality or novelty advantage is claimed.

## Strongest complete quality evidence

Accuracy is in percent; differences are percentage points. Each row is one complete, matched comparison. Evaluation populations and training procedures differ between rows.

| Setting | Shared procedure | Accuracy | Genuine independent reference | Accuracy | Shared minus independent |
| --- | --- | ---: | --- | ---: | ---: |
| WikiCS, selected development, 3 optimizer seeds | Internal-factor SAGE F | 80.6219 | Internal-factor I4 | 80.0847 | +0.5372 |
| WikiCS, selected development, 3 optimizer seeds | Endpoint-conditioned SAGE E | 81.1971 | Equally equipped I4 | 80.8937 | +0.3034 |
| Penn94, official split 0 heldout nodes, 10 optimizer repeats | Internal-factor SAGE | 78.1535 | Internal-factor I4 | 75.2993 | +2.8542 |
| Physics, heldout nodes under 10 mask/optimizer protocols | Endpoint-conditioned SAGE E | 91.3264 | Equally equipped I4 | 92.7891 | -1.4628 |

**WikiCS:** F's gains are positive in all three seeds, including +0.7205 points over the matched joint untied control. Its descriptive 95% interval against I4 is [-0.2653, 1.3398], and its mean calibrated-NLL allowance fails. E's smaller gain over its own equipped I4 also has an interval including zero. These are encountered development results.

**Penn94:** the +2.8542-point gain is positive in all ten repeats, with paired 95% interval [2.5464, 3.1620]. Its members improve by 2.1602 points. Original boundary GNNM reaches **79.1334%**, and the LINKX task reference reaches **84.6852%**. Thus internal sharing improves its matched independent reference while the full predeclared improvement criterion fails. The ten repeats share one graph and one fixed evaluation population.

**Physics:** E improves unchanged sharing by 0.7861 points but loses to its capable independent reference. The source meanSAGE single reaches **93.0590%**. This complete confirmation closes that E recipe; favorable development numbers do not establish transfer.

Within these completed comparisons, E supplies the highest new shared score in its WikiCS setting, while original boundary GNNM remains the stronger GNNM procedure on Penn94. There is no confirmed universally best extension.

- [WikiCS internal-factor sharing comparison](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/SAGE_matched_joint_reference_77_closed_history_20261011_v1/REPORT.md)
- [WikiCS endpoint comparison](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/native_SAGE_endpoint_value_closed_interpretation_root_20261011_v1/REPORT.md)
- [Penn94 complete results, uncertainty and costs](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/native_SAGE_Penn94_complete_confirmation_summary_root_20261011_v1/REPORT.md)
- [Physics complete confirmation](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/native_SAGE_Physics_complete_confirmation_summary_root_20261011_v1/REPORT.md)

### Completed architectural tests and decisions

The earlier graph-tour scanner finished all 30 comparison groups and 48 training trajectories, including inherited references. Its candidate scored **93.0132%**, compared with **93.1100%** unchanged sharing and **93.2850%** an equipped independent ensemble on the same encountered Physics protocols. Weaker members and losses during averaging outweighed its extra correct alternatives. That exact recipe is closed.

[Complete scanner outcomes and errors](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/graph_walk_complete_family_history_synthesis_20261011_v1/REPORT.md) remain preserved. The current GRU changes state-dependent memory, training stages and private parameter allocation; it is a separate fixed hypothesis. Other completed failures, costs and decisions remain in the ledger. Known ingredients can support a useful method, but their combination alone supplies no novelty or accuracy claim.

## Original benchmark, paper and reproducibility

The original five-dataset benchmark remains central to the paper. Its models, scores and reported selection procedure are preserved. The [complete anonymous V5 source](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/codex/postsubmission-research-20260930/experiments_iclr/postsubmission_20260930/complete_anonymous_quality_draft_20261011_v5/README.md) contains the original tables and the added matched/heldout studies. It is a working revision, uncompiled and unreviewed; pagination and a new acceptance recommendation remain unverified.

Original implementations are `models.py`, `run_base.py` and `run_common.py`. Environment files are `pyproject.toml`, `uv.lock` and `environment.yml`. The documented runtime uses Python 3.11, Torch 2.1.2 with CUDA 11.8, PyG 2.7 and DGL 2.4.

The original README remains available at an immutable commit, preserving all commands and detailed evidence instructions:

- [Repository contents and original usage](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/5fc577ece84f5c894797c9099ce43e8772b283c8/README.md#repository-contents)
- [Original benchmark table commands](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/5fc577ece84f5c894797c9099ce43e8772b283c8/README.md#recalculate-the-original-benchmark-tables)
- [Compact evidence verifiers](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/5fc577ece84f5c894797c9099ce43e8772b283c8/README.md#verify-the-compact-extension-evidence)
- [Fresh experiment setup, preparation and preflight](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/5fc577ece84f5c894797c9099ce43e8772b283c8/README.md#run-new-experiments)
- [Data and interpretation](https://github.com/ChervovNikita/diploma_yandex_graphs/blob/5fc577ece84f5c894797c9099ce43e8772b283c8/README.md#data-and-interpretation)

Research source, fixed protocols and compact evidence are versioned on `codex/postsubmission-research-20260930`. Full datasets, checkpoints and member predictions stay on authorized research servers. Git records source changes, complete scalar results, uncertainty, failures, costs and decisions. Historical original launch-time source/environment/data fingerprints are incomplete; the newer fixed studies bind their inputs and execution evidence explicitly.

Fresh independent Codex reviewers apply the requested paper-review guidance after substantial supported progress and a complete manuscript revision. They receive immutable anonymous manuscript/evidence with no requested verdict. A clear acceptance recommendation and the overall research goal remain unmet.
