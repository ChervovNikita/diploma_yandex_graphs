# Current GNNM research status

Updated: 2026-10-05T08:09:47.941324+00:00. The goal is active and unmet. No new method has established both methodological novelty and better prediction than strong singles and independently trained ensembles. Original manuscript scores remain unchanged; no new acceptance verdict exists.

## The learning rule being tested

Each member learns from TRAIN links excluding nodes involved in the current outer queries. The shared graph backbone is trained according to how well the members predict those outer links after their private learning step. Private updates are then recomputed at the new backbone and committed once. Prediction uses the trained ensemble directly.

This tests transferable private learning rather than embedding repulsion. Endpoint-conditioned sampling and live differentiation are separate controls. Graph context and training histories remain shared, so this is conditional transductive regularization, not independent cross-fitting. ANIL/BMAML, MLDG/MetaReg, SELAR, graph meta-learning and Episodic DG bound the learning principle; the complete graph rule's usefulness and novelty remain unresolved. [Eight-family synthesis](quality_literature_to_design_triage_20261005_v1/REPORT.md). [Mechanism and interpretation limits](shared_ensemble_method_decision_20261005_v1/REPORT.md).

## Current experiment and matched single control

The original ten-cell, three-block pilot is unchanged: sixty complete TRAIN cycles, full VALID every five cycles, eleven misses, first maximum four-decimal MRR. All thirty authenticated fits are required before comparative analysis. TEST remains closed. Historical independent/F4 anchors retain their disclosed support and budget differences. Jointly trained native four is not an independently trained ensemble.

At 2026-10-05T08:06:25.952305+00:00, singleton b0 has 2/10 operationally complete, current b0_E_end_joint at cycle10/episode13. At 2026-10-05T08:06:26.508230+00:00, 18.77 b1 has 1/10, current b1_E_random_live at cycle60/episode19; b2 has 2/10, current b2_U_end_live at cycle7/episode57. All owned queues and children physically match, no failures or score reads. [Singleton observation](shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/observation_20261005T080625Z/OBSERVATION.json). [77 observation](shared_private_transfer_gpu77_block_observation_20261005_v4/OBSERVATION.json).

The single-control audit found that the original capable single relocates dense head bases to private learning. A separate fixed-row0 F1 control preserves F4's parameter roles and uses four explicit inner streams in one averaged-gradient private update. It keeps the original thirty and rich single intact. Both provider-specific implementation checks passed (one attempt each, exit0,1003 numerical comparison records each). Three complete TRAIN-only cost cycles passed: live145.38s, detached132.69s and ordinary joint67.85s. No VALID/TEST values were accessed. These are implementation/cost facts, not accuracy results. Conservative original F4 per-rule bounds are retained for the nine-fit companion, whose launch adapters are being finalized and independently reviewed; its fits are not released. [Source review](shared_private_transfer_row0_companion_source_review_20261005_v1/REPORT.md). [Singleton gate](shared_private_transfer_row0_fp32_execution_root_20261005_v1/REPORT.md). [77 gate](shared_private_transfer_row0_gpu77_fp32_execution_root_20261005_v1/REPORT.md). [Complete costs](shared_private_transfer_row0_complete_cost_execution_root_20261005_v1/REPORT.md).

## Completed quality evidence

Citeseer-HeaRT's complete36-fit development comparison gives mean VALID MRR28.4115% for unchanged shared F4,27.9204% for independently trained four,26.7811% for native single and28.1795% for private frames. Three paired blocks on one split; every primary descriptive interval includes zero. The frame candidate failed its frozen improvement-over-F4 gate and is not promoted. TEST remains unopened. [Audited comparison](citeseer_frame_complete_root_adoption_20261005_v1/REPORT.md).

The five-seed Collab TEST result remains private completion67.2909%, independent four67.6298% and native single66.4426% Hits@50. That TEST is consumed and cannot confirm a later design. [Preserved comparison](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Other running studies and literature

Amazon/Polynormer observation at08:10:36UTC finds12/15 complete, independent member1 atupdate744/2700, exact owned live processes and no failure. PENCIL metadata at08:10:37UTC finds seed0 at epoch259/1040updates, no completed fit/failure; this is metadata rather than a fresh physical identity check. Older77 Collab7/9 and DDI6/12 remain historical observations. Comparative outcomes stay closed until the corresponding complete families are authenticated.

Canonical literature indexv63 retains238 scoped conclusions across186 paper groups plus two software groups; these are not full-paper reading counts. A separately saved TMetaNet scope adds topology-conditioned learning-rate ancestry, with zero full-paper reads and no change to the frozen operator or novelty claim. [Scoped source record](private_learning_credit_recent_prior_20261005_v1/REPORT.md).

Latest verified GitHub head is ac953dfc241adf01e22bf28c0c1105f4f7c1b62a, exact ref verified at 2026-10-05T07:20:02.972864+00:00. New F1 source/gates/costs and this status await compact publication. Raw evidence and checkpoints remain on authorized servers. The seven-GPU allocation is a MacLink relay only. No sudo, host/mount changes, PDF compilation, GENLINK or unrelated-data access. Failed ideas and decisions remain preserved. Paper review requires fresh context, immutable anonymous inputs and no requested verdict.
