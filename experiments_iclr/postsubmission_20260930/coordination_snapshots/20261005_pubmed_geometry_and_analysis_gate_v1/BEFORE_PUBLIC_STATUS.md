# Current GNNM research status

Updated: 2026-10-05T09:05:59.922775+00:00. The goal is active and unmet. No new method has established both methodological novelty and better prediction than strong single models and independently trained ensembles. Original manuscript scores remain unchanged. No new acceptance verdict exists.

## Access and current training

The user temporarily withdrew access to 18.77 on 5 October. Work now uses only the authorized one-GPU allocation. There is no contact with 18.77 or MacLink, including monitoring, fetching, launch, stopping or cleanup. Its detached jobs remain untouched and unobserved after withdrawal. Earlier observations are historical, not evidence that jobs stopped.

The allocation's original pilot has 3 of 10 fits operationally complete. At 09:16:53 UTC, the new endpoint-conditioned live-learning rule was at cycle 2, episode 17. The separately frozen matched-single block launched once at 08:43:11 UTC. At 08:59:54 UTC, its first ordinary joint control was at cycle 15, episode 4, with no completed fit. Both queue and current child identities physically match their launch records, with no recorded failure. These are training progress facts, not prediction results. [Original pilot observation](shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/observation_20261005T085953Z/OBSERVATION.json). [Matched-single observation](shared_private_transfer_row0_companion_root_release_20261005_v1/observation_20261005T085953Z/OBSERVATION.json).

The original 30-fit study and nine-fit matched-single design remain unchanged. All 30 original fits are required before comparative analysis; the full 39 are required for the added mechanism diagnostics. Only three matched-single fits are released on the allocation. The other six remain unreleased while 18.77 access is withdrawn. TEST remains closed. Neither unfinished nor favorable subsets will replace the complete cohort.

## The learning rule and its controls

Four members share the graph backbone. In each episode, their private parameters learn from TRAIN links excluding the nodes involved in the outer positive and negative queries. The backbone is trained according to how well the members predict those outer links after that private step. Private updates are then recomputed at the new backbone and committed once. Prediction uses the trained ensemble directly.

The controls separate endpoint conditioning from differentiation through learning, and compare strong singles, shared members and untied backbones under disclosed training costs. The new matched single preserves the four-member model's shared/private parameter roles and four inner streams; it averages their gradients into one private update. The original richer single remains intact. The matched control changes several ensemble properties together and is not a pure member-count intervention. [Control caveat](shared_private_transfer_member_count_partition_audit_20261005_v1/REPORT.md). [Actual launch review](shared_private_transfer_row0_companion_root_release_20261005_v1/ROOT_REVIEW.md).

ANIL/BMAML, MLDG/MetaReg, SELAR and graph meta-learning already establish nearby learning principles. The endpoint rule is conditional transductive regularization: graph context and training histories remain shared. It is not independent cross-fitting. No global novelty clearance or ensemble-specific benefit has been established. A rich single's ability to represent an ensemble does not logically reject a useful extension, and similar small-study gains do not identify a unique cause. The active rule remains an unresolved empirical test. [Independent screening audit](method_screening_criteria_independent_audit_20261005_v1/REPORT.md).

## What the analysis will measure

Five fixed diagnostics cover the complete quality contrasts; member competence and complementary errors; sampling exposure and paid training; separate versus averaged Adam responses; and the actual contribution of differentiating through private learning. The last two require separately reviewed, discarded TRAIN probes after complete-family release. A correct meta-gradient may be practically small under Adam. Its size, error diversity or an attractive visualization cannot rescue a failed quality comparison. [Prospective analysis and limits](shared_private_transfer_fixed30_plus9_mechanism_analysis_plan_20261005_v1/REPORT.md).

## Completed quality evidence

The complete 36-fit Citeseer-HeaRT development comparison gives mean VALID MRR of 28.4115% for unchanged shared F4, 27.9204% for independent four, 26.7811% for native single and 28.1795% for private frames. Three paired blocks use one split; every primary descriptive interval includes zero. The frame candidate failed its frozen improvement-over-F4 gate and is not promoted. TEST remains unopened. [Audited comparison](citeseer_frame_complete_root_adoption_20261005_v1/REPORT.md).

The five-seed Collab TEST result remains 67.2909% Hits@50 for private completion, 67.6298% for independent four and 66.4426% for native single. This consumed TEST cannot confirm a later design. [Preserved comparison](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Other studies, literature and publication

At 09:02:08 UTC, Amazon/Polynormer has 12 of 15 fits operationally complete; independent member 1 is at update 1653 of 2700, with no recorded failure. PENCIL's first of three fits completed with exit 0 after 300 epochs. Seed 1 is at epoch index 3 and 16 updates. PENCIL's current observation is metadata, not a fresh physical identity check. Outcomes remain closed until each complete family is authenticated.

The adopted literature index v64 contains 241 scoped conclusions across 189 paper groups and two software groups. These are not full-paper reading counts. It preserves all 238 predecessor records and integrates the previously completed TMetaNet, LPSFed and PFedEG method scopes. Persistent graph sharing, averaging and personalized mixtures are established prior; the inspected scopes supply no supported successor or global novelty clearance. The marginal-update-credit synthesis also found no justified new operator. [Scoped literature memory](literature_memory/index_v64/LITERATURE_INDEX.json). [Prior conclusions](persistent_graph_private_learning_credit_followup_20261005_v1/REPORT.md). [Credit synthesis](private_update_marginal_value_design_synthesis_20261005_v1/REPORT.md).

Latest verified GitHub head: 3ed79659e295e0a5a2c41f08c94733b81acb1401, verified at 2026-10-05T09:08:09.772098+00:00. The allocation-only launch records, index v64 and prospective mechanism plan were published. New propagation-prior conclusions, screening corrections and latest observations await the next compact publication. Raw checkpoints and evidence remain on their authorized servers. No sudo, host or mount changes, PDF compilation, GENLINK or unrelated-data access. Failed ideas and costs remain preserved. Paper review requires fresh context, immutable anonymous inputs and no requested verdict.
