# GNNM: shared-backbone graph ensembles

GNNM learns graph-layer weights shared by several member trajectories. Private factors give the members different transformations. The original five-dataset benchmark compares prediction quality with single models and independent ensembles. Its reported scores remain unchanged.

## Paper and implementation

The [complete anonymous V5 source](experiments_iclr/postsubmission_20260930/complete_anonymous_quality_draft_20261011_v5/README.md) keeps the five-dataset benchmark central. It includes the matched Roman Empire analyses, completed native SAGE and packed comparisons, and the fixed-feature head diagnostic. The appendix is shorter, with every numerical table and figure preserved in the paper or explicitly named text supplements. V3 remains intact. The source is uncompiled; pagination is unverified. It is a working revision, not an asserted copy of the submitted manuscript.

Original implementations remain in `models.py`, `run_base.py` and `run_common.py`. Research source, fixed protocols and compact evidence are under `experiments_iclr/postsubmission_20260930/`. The [portable Physics package](experiments_iclr/postsubmission_20260930/portable_conditional_SAGE_source_20261011_v2/README.md) provides prepare/acquire/export/score commands without cluster credentials. It has one actual same-runtime update and strict model/optimizer/RNG restore check. Clean installation, external portability and complete CLI execution remain unverified.

Large datasets, checkpoints and member-logit archives stay on authorized servers. They are not included in these compact Git publications. Explicit inventories and push receipts identify the public files.

## Current prediction evidence

On encountered WikiCS development, endpoint-conditioned shared SAGE (E4) reaches 81.1971% accuracy, versus 80.6282% unchanged sharing, 80.8937% equally equipped independent I4, 81.0517% receiver-only N4 and 80.2174% equipped M1. The three-seed interval against I4 includes zero. I4 retains better NLL and Brier, and receiver conditioning explains most of E4's baseline gain. The smaller packed reference reaches 79.7876% with better raw NLL than E4. These are selected development outcomes, without unused confirmation or a certified new primitive.

- [Complete endpoint comparison](experiments_iclr/postsubmission_20260930/native_SAGE_endpoint_value_closed_interpretation_root_20261011_v1/REPORT.md)
- [Complete packed comparison](experiments_iclr/postsubmission_20260930/native_SAGE_PE4_execution_root_20261011_v1/complete_closed_joined_evidence_v1/complete_joined_raw_analysis_v1/COMPLETE_JOINED_RAW_SUMMARY.json)
- [Current research state](experiments_iclr/postsubmission_20260930/RESEARCH_STATE.md) and [decision history](experiments_iclr/postsubmission_20260930/research_ledger.json)

The [complete error audit](experiments_iclr/postsubmission_20260930/WikiCS_E_frozen_strata_diagnostic_root_20261011_v1/REPORT.md) finds a common false class above truth in every member on 69.38% of E4's remaining error readouts. This constrains unchanged nonnegative scalar pooling, without proving absent class information in the hidden states. The [fixed affine-head comparison](experiments_iclr/postsubmission_20260930/WikiCS_E_fixed_feature_head_capacity_summary_root_20261011_v1/INTERPRETATION.txt) fails its declared utility criteria: 50 repairs and 51 harms. All 40 newly covered original common-rival errors disappear during pooling. That exact recipe is closed.

The [source–message association diagnostic](experiments_iclr/postsubmission_20260930/WikiCS_E_source_context_association_summary_root_20261011_v1/REPORT.txt) completes 36 banks and 108 trajectories with no training. A fixed within-neighborhood cyclic mapping changes which sender context conditions each factual message. Applying it at both layers reduces accuracy by 0.2844 points for E4, 0.2086 for M1 and 0.0822 for I4. E4 has 99 repairs and 144 harms. Its larger served loss includes pooling effects; its mean-member deterioration is similar to I4's. The same conditional dependence appears in the capable references. One restricted mapping does not remove covariance, estimate permutation uncertainty or identify a retrained effect. All methods, classes and adverse results are retained.

## Complete comparisons and active confirmation

The full Penn94 study has closed 70 banks, 100 actual training bodies, 90 calibration endpoints and 90 final scoring rows. Internal shared SAGE scores 78.1535% versus 75.2993% equally equipped genuine I4: +2.8542 accuracy points, positive in all ten paired optimizer blocks, paired 95% interval [2.5464, 3.1620]. Original boundary GNNM scores 79.1334% and LINKX context scores 84.6852%. The complete required four-contrast improvement criterion therefore fails; the secondary placement gate remains closed. Raw NLL is worse than I4, while separate-CAL temperature calibration improves that contrast and passes the fixed practical harm screens. This is an internal-factor SAGE comparison on one graph, not a replication of endpoint E or a state-of-the-art claim.

- [Complete Penn94 results and limits](experiments_iclr/postsubmission_20260930/native_SAGE_Penn94_complete_confirmation_summary_root_20261011_v1/REPORT.md)
- [Complete Penn94 member and pooling analysis](experiments_iclr/postsubmission_20260930/Penn94_complete_shared_training_history_synthesis_20261011_v1/REPORT.md)

The complete native PyG GNN-FiLM WikiCS comparison has closed 9 banks, 18 fits and 45 fixed calibration endpoints. FiLM M1/shared4/genuine I4 reach 77.2153/79.9836/78.4161% raw accuracy. Existing endpoint E reaches 81.1971%, exceeding shared FiLM by 1.2135 points and FiLM I4 by 2.7809 points, positive in all three paired development seeds. Raw/calibrated NLL and Brier also favor E. These fixed-recipe adaptation results are exploratory, not unused confirmation, capacity-matched attribution or a published-score reproduction. Native FiLM includes additive receiver modulation, per-edge activation, self records and skip, so it is richer than the earlier N control.

- [Complete FiLM comparison](experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_complete_summary_root_20261011_v1/REPORT.md)
- [Whole-result compact evidence](experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1/complete_closed_evidence_v1)
- [Cross-graph member/retention synthesis](experiments_iclr/postsubmission_20260930/broad_shared_training_history_after_Penn_FiLM_20261011_v1/REPORT.md)

The complete Physics confirmation has finished all 60 banks, 90 training bodies and 60 fixed calibration endpoints. E scores 91.3264%, versus 90.5403% unchanged sharing, 91.2982% receiver conditioning, 92.2233% equipped M1, 92.7891% genuine I4 and 93.0590% the source-derived meanSAGE single. E improves BASE by 0.7861 points in all ten blocks but loses 1.4628 points to I4 in all ten. Its endpoint increment is only 0.0282 points with mixed signs. The full primary criterion fails, endpoint attribution stays closed, and calibrated-NLL protection fails against I4. Complete vectors, intervals, errors, every class and paid cost scope remain available. These ten paired mask/optimizer protocols use one graph, not ten independent domains.

- [Complete Physics confirmation and failed criteria](experiments_iclr/postsubmission_20260930/native_SAGE_Physics_complete_confirmation_summary_root_20261011_v1/REPORT.md)
- [Complete Physics scalar evidence](experiments_iclr/postsubmission_20260930/native_SAGE_Physics_confirmation_root_20261011_v1/complete_closed_evidence_v1)
- [Updated decisions after Physics](experiments_iclr/postsubmission_20260930/Physics_complete_shared_training_history_decision_synthesis_20261011_v1/REPORT.md)

The attributed Rank-1 BNN acquisition proposal and native v2 source are unexecuted. They test learned versus fixed Gaussian fast-factor noise with capable point/stochastic single and independent references. A separate exact original boundary-GNNM WikiCS comparator is prepared but requires an explicit prospective roster amendment and complete reader before any launch. After Physics, all ten fresh original-GNNM TRAIN/SELECT reference acquisitions completed in 608.85 seconds with no native failure. Their SELECT-only outputs are acquisition diagnostics; no new CAL/TEST result or original-member coverage/proper-loss evidence is claimed. This is encountered-graph exploration and cannot rescue the closed confirmation. Source preparation is not an accuracy or novelty result. No new 18.77 training has started while the paired Mac is offline.

## Research and review practice

Preserve original scores, failures, costs and decisions. Use complete tasks, capable references, paired comparisons, fixed selection rules and uncertainty. Known components retain their citations. Fewer stored weights do not establish lower computation or peak memory. The [fixed-state explanation](experiments_iclr/postsubmission_20260930/native_SAGE_VALUE_fixed_hidden_state_expressivity_note_20261011_v1/NOTE.md) attributes pre-mean source-conditioned distinctions to existing conditional-message work; it does not establish a whole-GNN or ensemble-specific hierarchy.

Fresh independent Codex reviewers apply the requested peer-review guidance after substantial supported progress and a complete paper update. They receive an immutable anonymous paper and evidence, without author history or a requested verdict. No reviewer is called after each small experiment. No acceptance verdict or genuine goal completion has been established.


The [closest BSNN primary update](experiments_iclr/postsubmission_20260930/BSNN_primary_version_publication_update_20261011_v1/REPORT.md) verifies its September 2025 v2. Its shared-network and conditional sampled-geometry ensemble overlap is unchanged. Official publication remains unverified. This is a versioned method revisit, not a new paper or novelty clearance. The ten-bank common-builder attribution consumer is source-ready, conditional on full Physics support and a separate future admission.

### Complete TRAIN gradient diagnostic and paper revision

The complete TRAIN-only diagnostic has 15 banks, 24 restored states and 51 route backwards, zero updates and no new VALID/TEST scores. It finishes in 71.40 seconds. Shared-gradient signs vary by seed and do not support uniform overwhelming opposing gradients; no gradient-surgery fit follows. This selected-state/dropout measurement does not establish training dynamics or causality.

Complete anonymous V5 now integrates Penn94, native FiLM and the whole Physics outcome, with all original scores and dependencies preserved. It is sealed and uncompiled, with 157 static source checks. Every complete failure and bounded positive result remains visible. Original five-dataset scores and dependencies remain central and unchanged. Fresh manuscript review follows substantial complete integration, not each experiment. The scientific goal remains unmet.

### Dedicated scientific research

Three distinct research roles investigate new architecture mechanisms, implement controlled studies, and synthesize completed experiments. Root selects, runs, interprets and publishes representative paired comparisons. The dedicated architecture researcher checks strongest primary-source alternatives and targets shared-route member competence and lost correct predictions. No new primitive or general architecture improvement has yet been established.

The complete sign-aware dropout study closed42fresh banks,69selected optimizer states and42separate-CALendpoints. Its candidate reaches90.9972% versus90.9835%same-input IID,92.5508%capableI4and92.5723%source meanSAGE. All primary/attribution criteria fail. The exact rule is closed. [Complete outcomes, uncertainty, member/pooling errors and costs](experiments_iclr/postsubmission_20260930/Physics_dropout_pilot_complete_development_summary_root_20261011_v1/REPORT_V2.md) are retained. Restricted conditional variance did not establish prediction gains.

The attribute-peer family has completed. Its exact SX4 candidate failed the declared quality/member/peer-order/retention criteria; every result and cost is retained below. The former peer acquisition and watcher are closed.

## Complete peer outcome and next architecture hypothesis

The complete peer family closed30banks,48optimizer trajectories,57physical bodies and30CALendpoints/15,000updates. SX4scores93.0914% versus93.1100%unchangedsharing,93.1374%genuine I4and93.2127%ordinarysource. It fails quality/member-gain/peer-order/retention criteria despite passing competence/risk allowances. The exact rule is closed. [Every outcome, error count, uncertainty and cost](experiments_iclr/postsubmission_20260930/meanSAGE_attribute_peer_feedback_complete_development_summary_20261011_v1/REPORT.md) is retained. The full compact JSON can be reconstructed byte-for-byte from [small publication partitions](experiments_iclr/postsubmission_20260930/peer_complete_summary_publication_partitions_root_20261011_v1/README.md).

Under this source recipe, shared members are only.0958points below genuine I4members. That differs from earlier whole recipes; no specific initialization/optimizer/clipping cause is identified. The SM4control has a modest+.1388raw/member gain versusS4at allthree blocks, but mixed signs versus capableI4/sourceP1. It is not selected as a replacement candidate or unused-confirmation winner. Computers confirmation remains unadmitted for these failed fixed candidates.

The [next single nominated hypothesis](experiments_iclr/postsubmission_20260930/shared_graph_walk_state_architecture_scout_20261011_v1/REPORT.md) changes acquired graph context: persistent randomized tours with bidirectional gated state between competent meanSAGElayers. It compares strong all-four-tour tied singles and genuine equippedI4, with common-tour and memory-reset controls. Janossy/CRaWl/Graph-Mamba and related ancestry remain explicit. The scientist preserves existingRademacher starts; all-ones initialization remains separate and inactive. The implementation passed one independent derivative fixture and one actual full-graph resource/state/RNG qualification. The fixed study is now running on the authorized sole GPU allocation: 18 new banks / 27 fresh trajectories, with 12 complete old reference banks / 21 original trajectories retained unchanged. Strong tied four-tour singles also test an architectural benefit for a single GNN. A shared-ensemble contribution requires additional gains beyond these equally equipped singles and genuine I4. No prediction outcome has been opened, and no quality or methodological-novelty claim is established.

- [Actual scientific launch](experiments_iclr/postsubmission_20260930/graph_walk_acquisition_root_20261011_v1/ACTUAL_LAUNCH.json)
- [Closest graph-scan and recurrent-BatchEnsemble primary comparisons](experiments_iclr/postsubmission_20260930/shared_graph_walk_scan_ensemble_prior_addendum_20261011_v1/REPORT.md)
