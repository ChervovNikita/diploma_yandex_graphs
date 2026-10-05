# Current GNNM research status

Updated: 2026-10-05 07:14 UTC. Goal active and unmet. No new method has established superiority over competent singles and independently trained ensembles. Original manuscript scores remain unchanged; no new acceptance verdict exists.

## Current method and decision

The active hypothesis trains the shared graph backbone through each member's private learning. Private layers learn from TRAIN links excluding endpoints of the current outer queries; the backbone update differentiates through that learning. Private updates are then recomputed and committed at the new backbone. Inference uses the trained ensemble directly. The episodes remain correlated, conditional transductive training: they do not establish independent cross-fitting or unseen-node generalization.

Meta-learning, persistence and recomputation have close prior in ANIL/BMAML, MLDG/MetaReg, SELAR, graph meta-learning and Episodic DG. The unresolved question is whether the endpoint-conditioned private-learning rule addresses a useful graph sharing failure. An endpoint/random × live/detached comparison separates the proposed mechanism from generic meta-regularization and sampler regularization. [Eight-family literature-to-design synthesis](quality_literature_to_design_triage_20261005_v1/REPORT.md). [Bounded nearest-prior stress test](endpoint_private_transfer_novelty_stress_test_20261005_v1/REPORT.md).

The single-model audit found a parameter-role confound: the original capable single adapts its whole nonlinear head privately, while F4 keeps dense head weights shared. A separate row0 single companion is being implemented with F4's shared/private partition and four inner dropout streams. Its proposed nine fits preserve the original thirty-fit plan and the richer single control. Source preparation is not an execution release or a result. [Partition audit and matched-control design](shared_private_transfer_member_count_partition_audit_20261005_v1/REPORT.md).

A separate successor scout found no defensible additional mechanism in private query-conditioned structural corrections: retained BUDDY/LPFormer/OCN already supply the information repair, and additional capacity remains a competing explanation. This direction is not promoted. [Saved decision](graph_ensemble_quality_successor_scout_20261005_v1/REPORT.md).

## Running experiment families

The private-transfer pilot has ten cells in each of three paired seed blocks. It uses sixty complete TRAIN cycles, full VALID every five cycles, eleven misses and the first maximum MRR rounded to four decimals. All thirty authenticated fits are required before comparative analysis; TEST stays closed. Prior independent-ensemble/F4 anchors retain their disclosed training-support and budget differences. The jointly trained native four-model cell is not a separately trained ensemble.

- **Singleton b0**, physical observation 07:14:17 UTC: one of ten fits operationally complete; current jointly trained four-model control at cycle31/episode61. Exact queue and child identities live; no failure, score access, restart or new launch. [Observation](shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/observation_20261005T071417Z/OBSERVATION.json).
- **18.77 b1/b2**: both complete ten-cell blocks froze at06:44:42 UTC and launched once at06:46:16/17 UTC on their assigned GPUs. At07:06:25 UTC both owned queues and children were live, zero of ten completed in each, with no failures or score access. b1 was atcycle33/episode25; b2 atcycle38/episode7. [Launch custody](shared_private_transfer_gpu77_block_launch_execution_root_20261005_v1/REPORT.md). [Observation](shared_private_transfer_gpu77_block_observation_20261005_v2/OBSERVATION.json). [Preserved prefit metadata clarification](shared_private_transfer_gpu77_launch_metadata_clarification_20261005_v1/CLARIFICATION.md).
- **Amazon/Polynormer**, latest available observation 06:21:17 UTC: eleven of fifteen complete; current independent member0 update1437/2700. Owned processes live, no failures. [Observation67](amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_launch_execution_receipts_20261003_v2/MONITOR_0067_RESULT.json).
- **PENCIL Citeseer300**, metadata at06:21:16 UTC: seed0 epoch161/648updates, no completed fit or failure. Three seeds; early feature fusion is a declared adaptation, not exact official reproduction. Scores remain closed.
- **Older 18.77 Collab/DDI**, observation06:30:20 UTC: Collab7/9, joint seed2 epoch90/batch4; DDI6/12, joint seed1 elapsed13960.67s. Owned queues/children live, outcomes closed. [Observationv11](compact_owned_queue_monitor_20261005_v11/OBSERVATION.json).

## Completed quality evidence

Citeseer-HeaRT's complete 36-fit development study gives mean VALID MRR28.4115% for unchanged shared F4,27.9204% for independently trained four,26.7811% for native single and28.1795% for private frames. Three paired blocks on one split; every primary descriptive interval includes zero. Private frames failed the frozen improvement-over-F4 gate and are not promoted. TEST is unopened. [Audited result](citeseer_frame_complete_root_adoption_20261005_v1/REPORT.md).

The five-seed Collab TEST result remains private completion67.2909%, independent four67.6298% and native single66.4426% Hits@50. That TEST is consumed and cannot confirm a later design. [Preserved result](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Literature, publication and boundaries

Canonical indexv63 retains238 scoped conclusions across186 paper groups and two software groups. These are not full-paper reading counts. Separate scoped novelty/source decisions retain their exact boundaries without inflated totals.

Latest verified GitHub head before this update: f280d427202d7e14b907c502ee5d2bfb7a9e15da, exact ref verified06:44:13 UTC. That107-file publication preserves the design synthesis, scoped novelty notes, exact sampling equality, qualifier sources and compact success/failure records. [Push receipt](publication/science_triage_and77_qualification_20261005_v1/PUSH_RECEIPT.json). New launch/control decisions await the current compact publication.

Source, data and research outputs stay in authorized project repositories. The seven-GPU allocation is a MacLink relay only; wrong-allocation evidence remains excluded. Normal incidental caches are permitted. No sudo, host-setting/mount changes, PDF compilation, GENLINK or unrelated-data access. Failed ideas and costs remain in the ledger and Git history. Reviews must use immutable anonymous evidence and fresh independent context with no requested verdict.
