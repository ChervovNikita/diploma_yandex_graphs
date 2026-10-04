# GNNM post-submission research

This branch tests whether shared graph ensembles can produce a supported predictive and methodological improvement. Original five-dataset paper scores remain unchanged.

The strongest current lead is member-specific NCNC graph completion on ogbl-collab: five-seed selected-validation +0.8432 percentage points over pooling, with essentially equal independent-ensemble quality. Audit and heldout confirmation are pending. Whole-pattern versus individual-incidence supervision is training on both authorized 18.77GPUs; the authored Amazon Polynormer comparison runs on the authorized one-GPU allocation. No new method advantage or paper acceptance is established.

- [Current measured results and exact runtime status](PUBLIC_STATUS.md)
- [Current plan and acceptance requirements](RESEARCH_STATE.md)
- [Decisions, failed hypotheses and full history](research_ledger.json)
- [Complete NCNC development results](ncnc_complete_family_saved_VALID_summary_20261004_v1/RESULTS_SUMMARY.md)
- [Complete BUDDY heldout analysis](buddy_paired_analysis_execution_20261004_v1/report/REPORT.md)
- [Canonical scoped literature memory](literature_memory/index_v41/LITERATURE_INDEX.json)
- [New structured-single control proposals](ncnc_structured_single_control_source_proposal_20261004_v1/MEMO.md)
- [Verified Pubmed feature equivalence](pubmed_planetoid_raw_feature_equivalence_execution_root_20261004_v1/RESULTS_SUMMARY.md)
- [Previous README and status history](status_history/20261004_0047_before_current_consolidation/README.md)

All comparison families retain fixed protocols, complete cohorts, actual failed attempts and provenance. Raw data/checkpoints stay on their authorized servers. Source qualification and engineering success do not establish predictive superiority, methodological novelty or acceptance.
