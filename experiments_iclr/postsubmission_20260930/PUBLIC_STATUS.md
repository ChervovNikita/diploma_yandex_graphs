# Current GNNM research status

Updated: 2026-10-05 03:00 UTC. Goal active and incomplete. Original manuscript scores remain unchanged. No new method has established superiority over competent singles and independent ensembles with a cleared novelty claim; no new manuscript acceptance exists.

## New completed quality comparison

The complete, prospectively fixed Citeseer-HeaRT development comparison finished all 36 physical fits. All native checkpoint selections and prediction identities were audited before reconstructing the seven families. TEST remains unopened.

| Model | Mean VALID MRR (%) |
| --- | ---: |
| Native single | 26.7811 |
| Ordinary independent four | 27.9204 |
| Shared four without endpoint frames | 28.4115 |
| Shared four with one common frame | 28.0053 |
| Shared four with private frames | 28.1795 |
| Single receiving all four frame features | 26.9768 |
| Independent four with frames | 28.0116 |

Each family has three paired seed blocks on one fixed split, with 227 validation positives and 500 fixed hard negatives per positive. These are validation-selected development scores. Every primary descriptive seed interval includes zero. There is no statistical-significance or graph-generality claim. [Complete results and interpretation](citeseer_frame_complete_root_adoption_20261005_v1/REPORT.md), [unrounded audit](citeseer_frame_complete_analysis_20261005_v2/RESULTS.json).

The private-frame candidate passes the directional comparisons with all four primary controls but fails the frozen requirement to improve over unchanged shared four: -0.2320 percentage points, lower in two of three blocks. It is not promoted to confirmation as a supported accuracy extension. Favorable subgroup descriptions cannot reverse this decision. [Frozen-rule application](citeseer_frame_complete_root_adoption_20261005_v1/ROOT_DECISION.json).

The unchanged shared ensemble's observed +0.4911-point mean versus independent four is encouraging new task evidence for the existing recipe. It does not establish the endpoint-frame extension, a new principle, or held-out confirmation.

The previous five-seed ogbl-collab TEST family remains: single64 66.4426%, independent4 67.6298%, private completion4 67.2909%, pooled completion4 67.0673%, capacity70 66.7642% Hits@50. Private completion still trails independent four. That TEST is consumed and cannot confirm a later design. [Preserved results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Current method development

The separate training hypothesis asks the shared backbone to support useful private learning across endpoint-separated TRAIN queries. It specifies a virtual private update, an ensemble-aware backbone update, and private recomputation at the new backbone; serving uses the committed weights without adaptation. A capable adapted single, the same-rule untied ensemble, detached-update and matched query/masking controls are required. No scientific fit of this new rule has run. [Complete method specification](shared_backbone_quality_next_hypothesis_20261005_v1/REPORT.md).

The one full-shape CPU derivative check found that native sparse dense-input adjoints lack the required second-order graph. A nonzero mixed gradient and passing chain decomposition were insufficient; the explicit sparse identities failed. Directional finite differences were also inconclusive/unstable. The committed update was barred and all model/input state stayed unchanged. This is an implementation limitation, not predictive evidence against the hypothesis. An explicit constant-adjacency recursive-adjoint wrapper is being prepared; it must preserve forward/first-gradient behavior and qualify higher derivatives before any fit. [Actual diagnostic](shared_core_private_learning_native_derivative_execution_20261005_v1/RESULT.json).

BatchEnsemble/TabM, HousE/GoldE, LowFER/NTN, ANIL and BMAML supply direct ingredients. The new MetaReg/MLDG method readings also establish train-only transfer optimization and deployment without adaptation. Those cannot be novelty claims. The graph query construction and served ensemble learning recipe remain unproved differences. [New scoped conclusions](shared_core_train_only_meta_prior_root_20261005_v1/REPORT.md).

Literature memory v63 retains **238 scoped conclusion records, 186 paper groups and two software groups**. These are not full-paper reading totals. All predecessor conclusions/history are preserved, with exact source/scope bindings. [Index adoption notes](literature_memory/index_v63/ROOT_ADOPTION_NOTES.md).

## Running work and comparisons

- Amazon observation59 at02:42UTC: 10/15 fits complete; split2_gnnm_boundary_4_seed43 at1760/2700updates; no failure. Next metadata observation60 when useful. [Receipt](amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0059_RESULT.json).
- The recovered 18.77 route physically verifies both queues and current children. Observationv6 at02:44UTC: conditional Collab5/9, separate seed1 epoch94; DDI6/12, joint seed1. No outcomes opened, jobs restarted or signals sent. Next fresh observationv7 when useful. [Receipt](compact_owned_queue_monitor_20261005_v6/OBSERVATION.json).
- Fixed Citeseer structural/frame descriptions are authorized after both successful completion gates. They retain all families/bins, permit only within-model map distances, and cannot rescue the failed frame contrast. The bounded CPU analysis completed and its compact descriptions were authenticated. They do not support a reflection-specific benefit. [Interpretation](citeseer_frame_mechanism_analysis_execution_20261005_v1/INTERPRETATION.md).
- Native PENCIL CPU imports passed on the actual installed runtime. One full native zero-update TRAIN/VALID resource probe is authorized after the completed NCN queue, fresh singleton-route verification and at least34GiB free GPU memory. It retains native batch/workers/precision and full coverage, with24GiB allocated/28GiB reserved/128GiB own-process RSS/one-hour bounds. The staging-only failure was preserved and the single probe completed normally:31TRAIN batches/7740queries,400VALID batches/102255unique queries restored to113727occurrences,0updates, parameters unchanged. Peak allocated9.395GiB/reserved20.178GiB; own+loader RSS41.615GiB. Co-resident resource evidence does not establish Adam readiness or predictive competence. [Actual complete resource result](pencil_citeseer_zero_update_resource_execution_20261005_v1/RESULTS_SUMMARY.md). [Import evidence](pencil_citeseer_cpu_import_qualification_20261005_v1/RESULTS_SUMMARY.md), [resource plan](pencil_citeseer_bounded_zero_update_resource_plan_20261005_v1/README.md).

## Publication and custody

Latest verified GitHub ref before this adoption is f91e6dc0477be893e52237293e9f712dd4fca00a, branch codex/postsubmission-research-20260930. Reviewed new sources/conclusions and completed result receipts await the next publication. No18.77 Git synchronization is claimed. [Verified push](publication/shared_core_learning_rule_and_pencil_readiness_20261005_v1/PUSH_RECEIPT.json).

Previous full status and ledger are preserved in [the coordination snapshot](coordination_snapshots/20261005_completed_citeseer_and_train_only_prior_v1/PUBLIC_STATUS.md); failed candidates are not renamed as successes. Fresh manuscript reviews use the requested skill with immutable evidence and no requested verdict after method/evidence materially changes. Deliberate work remains inside authorized project repositories. Seven-GPU access is forwarding only. No sudo, PDF compilation, GENLINK, unrelated-data access or host-setting changes.
