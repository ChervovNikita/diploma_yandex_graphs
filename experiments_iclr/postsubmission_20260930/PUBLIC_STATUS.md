# Current GNNM research status

Updated 2026-10-06T20:53:44.019362+00:00. **Goal active and unmet.** No supported new-method superiority or fresh manuscript acceptance has been established. Original manuscript scores are unchanged.

## Initialization study is training

One common native model first trained for 20 cycles using TRAIN only, without validation selection. The fixed study then compares seven conditions across seeds 0, 1 and 2: all-layer random signs, a TabM-inspired first-layer normal initialization, an unchanged warm lift, edge-difference covariance initialization, centred hidden-feature covariance initialization, native single, and four independent models copied from the common warm state. Each condition receives 60 postwarm cycles. Dense shared weights and private factors remain trainable.

The feature covariance control uses graph-derived hidden states; it is not graph-free. The independent control has four separate full encoders and optimizers, independent postwarm streams, and the union of the four inner query banks per member. It does not use four independently trained warm donors. Finite initialization perturbations do not guarantee preserved competence.

The first warm fit completed 3,660 Adam updates with clean exit 0. At 20:50 UTC, the first seed's random-sign condition had reached cycle 56/60 and 10,248 updates; its exact live process identity was verified. Seeds 1 and 2 are being staged. No complete three-seed quality comparison is available. TEST is closed. [Fixed source and plan](citeseer_gnnm_initialization_pilot_preparation_20261006_v1/PLAN.json). [Warm and launch evidence](citeseer_gnnm_initialization_pilot_activation_root_20261006_v2/WARM_PROGRESS01.json).

## Known ranking control on 18.77

The fixed scaled-BPR control compares native single, independent four and shared four on complete HeaRT Citeseer. The original training source and sampler remain unchanged. A corrected independent loss reference passed the actual three-pass gradient, Adam-state and parameter gates for all three architectures, followed by complete TRAIN cost cycles. The original failed analytical-reference gate remains preserved; its float32 primitive discrepancy was diagnosed without changing the training loss or original tolerances.

Full training began at 19:54 UTC. The single completed 60 cycles and 10,980 updates with clean exit 0 at 20:29 UTC. Independent four reached cycle 32 at the 20:50 UTC observation; shared four remains queued. All retain 60 cycles and TEST stays closed. This established ranking loss is a control, not methodological novelty. [Corrected activation](citeseer_known_ranking_control_matched_reference_activation_root_20261006_v2/ROOT_RELEASE.json). [Preserved first failure](citeseer_ranking77_actual_qualification_failure_root_20261006_v1/CONCLUSIONS.md).

## Broader architecture tests

The structural candidate learns nonseparable interactions on edges inside a query's common-neighbour subgraph. Its dense encoder and message matrices are shared; modulation and readouts are private. It adds graph information beyond the current endpoint-product and unary common-neighbour summary. Tied, blind, count-only, same-information single and independent controls are part of the fixed plan. CAR already uses internal common-neighbour edges; OCN/NCN2 and subgraph GNNs provide close ancestry. Numerical qualification and paid complete-cycle cost are pending. No novelty or predictive success is established. [Prototype](shared_NCN_private_structural_interaction_prototype_20261006_v1/REPORT.md).

A separate control retains one shared encoder and gives each member a full native nonlinear NCN head, rather than diagonal factors. It has 2,801,673 parameters. This known architecture tests whether factorized heads limit quality. Source review is complete; numerical checks and training are pending. [Capacity control](citeseer_shared_encoder_full_native_heads_supplement_preparation_20261006_v1/REPORT.md).

## Completed Citeseer comparison and error analysis

All 39 selected fits, covering 13 conditions and three paired seeds, completed. Endpoint live-credit achieved mean selected VALID MRR 0.243122 and lost to its single, independent and random-frame controls in every block. This configuration stops. The ordinary shared joint baseline reached MRR 0.296971, versus 0.280665 for native single and 0.291541 for native independent four; it won MRR in two of three seeds but had lower Hits@10 than independent four. Its mean member MRR was 0.292948, versus 0.271605 for independent members. Shared members are not universally weaker.

A once-only CPU analysis of twelve saved VALID banks found that 74.74% of the shared ensemble's pooled strict positive–negative ranking errors were wrong in every member, versus 45.23% for independent four. Common errors averaged 29.84 versus 15.32 negatives per query. A query-best existing shared member still averaged 32.97 strict errors per query. This limits query-wise member selection and convex averaging of existing logits; it does not establish sharing causality or exclude arbitrary candidate-dependent aggregation. No new forwards or training were paid and no original score was recalculated. [Complete comparison](shared_private_transfer_complete39_D2_owned_operation_20261006_v2/PREDICTIVE_SUMMARY.csv). [Error conclusion](shared_private_transfer_complete39_common_negative_owned_operation_20261006_v1/RESULT_CONCLUSION.md).

## Other completed quality decisions

Amazon G0 failed: 41.9763% accuracy for live correction, versus 46.3046% independent four, 45.3655% single and 45.2021% initial state on 2,449 reserved TRAIN-A nodes. It repaired 224 initial errors and corrupted 303 correct predictions. The graph-free control had identical accuracy. All twelve models and 44 forwards preceded assessment labels; original VALID/TEST remained closed. The published CMCL continuation reached 41.5680%. No quality winner is promoted. Native references strongly minimized their recorded TRAIN likelihood losses; that does not identify the cause of poor assessment accuracy. [Full outcome and limits](amazon_G0_complete_comparison_result_root_disposition_20261006_v1/CONCLUSIONS.md). [TRAIN audit](amazon_G0_actual_TRAIN_curve_configuration_audit_20261006_v1/REPORT.md).

The earlier once-only Collab confirmation found private shared four +0.84828 percentage points over native single, but -0.33888 points against independent four. Its private-versus-pooled interval crossed zero. This remains a local gain over single, not broad superiority. [Heldout audit](ncnc_all25_heldout_fresh_evidence_audit_20261004_v1/REPORT.md).

## Literature, provenance and publication

Saved conclusions are reused before additional primary-source searches. The base index 72 contains 260 scoped conclusion records covering 207 paper identities and two software identities; these are not full-paper reading counts. Later scoped supplements preserve prior credits. Generic shared/private branching, centred covariance initialization and graph modulation are established ideas; a new contribution requires both a distinct mechanism and supporting comparisons.

The latest verified pushed commit is `f91853d6b879cb8d654856e2f9f3941c1bb82214`. New reviewed sources, compact results and actual execution receipts await publication. Checkpoints and full logits remain on authorized servers. Scientific work uses only the one-GPU allocation and 18.77; the seven-GPU route is forwarding only. Failed runs, costs and decisions are preserved. Fresh paper reviewers receive immutable anonymous evidence and no requested verdict.
