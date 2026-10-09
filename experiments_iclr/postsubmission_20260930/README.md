# GNNM post-submission research

We are developing an ensemble for better graph predictions with shared learned weights and private member paths. Original paper scores remain unchanged. A new methodological advantage and fresh manuscript acceptance remain unestablished.

## Current research

Each member makes a complete prediction using shared large weight matrices and its own compact adjustments. We test whether these adjustments can learn complementary graph information while keeping every member accurate. The models start from fresh initialization; independent ensembles are separately trained comparison models.

| Direction | Task and backbone | Current evaluation |
| --- | --- | --- |
| Different attention patterns | WikiCS, Polynormer | Four query/key operators in singles, shared-four models and genuine independent-four ensembles;30/36 cells completed at the latest observation. |
| Different graph propagation | Tolokers, Neural Sheaf Diffusion |15 initial physical fits completed; three fixed centered-factor-prior fits are running. Full comparison awaits all21 result records. |
| Ensemble feedback into private factors | MolHIV, GINE; WikiCS, Polynormer | Full-data comparisons of where the ensemble prediction objective affects learning. Every failed control remains recorded. |
| Complementary relation information | HGB IMDB, SeHGNN |24 genuine reference fits are running before the fixed18 shared-source fits. Members will be trained to contribute actor, director or keyword information to their committee. |

The IMDB source-learning integration and genuine four-body qualification passed on the full input. These checks establish executable updates and checkpoint restoration. Predictive benefit still requires the complete comparisons. The [quality plan](sehgnn_IMDB_paired_pilot_quality_root_freeze_20261009_v1/README.md) fixes micro-F1, strong references, member competence, uncertainty and error analysis before shared-candidate training. The [recent literature notes](recent_semantic_expert_counterfactual_baseline_scope_20261009_v1/PAPER_CONCLUSIONS.json) preserve the inspected HOPE and CoR methods and their limits.

## Preserved completed directions

The complete Wiki24 comparison identifies insufficient complementary correct predictions. A complete final-head diagnostic supplied no useful gain. The complete Context9 target experiment now also fails its unchanged continuation gate: route-specific targets tie or trail their matched control. Preserve these negative results and avoid reopening them through tuning.

The staged one-hop label posterior P0 is closed negative: all 12 endpoints completed 1100 updates and all selected states reconstructed. C4 averages 81.62685% development accuracy, below the joint single and untied label routes; both primary gates fail. It repairs only 2, 3, and 1 native errors, introduces none, and rescues no node missed by every member. The known C&S control remains stronger. Preserve this recipe without a seed, epoch, mixture, or subset rescue.

The separate query-conditioned value gate15 family completed all15 endpoints and selected-state reconstruction. C4 averages81.63949% development accuracy: it ties the matched joint single, trails untied same-backbone routes, and has no useful label-identity qualification. Both primary gates fail; the fixed recipe is closed negative. The frozen native encoder and same-backbone untied heads do not constitute an ordinary independent GNN ensemble.

Earlier runtime observations and every completed or failed experiment remain in the ledger. Use the current status for live counts. The Q/K comparison is active and its prior ancestry is recorded.

- [Current execution and decisions](RESEARCH_STATE.md)
- [Complete query-value gate decision](query_value_gate_complete_decision_root_20261009_v1/NOTE.md)
- [Complete staged P0 decision](staged_posterior_complete_decision_root_20261009_v1/NOTE.md)
- [Disabled callable gate15 source](query_conditioned_value_gate_full15_callable_source_20261009_v1/README.md)
- [Inactive two-hop design](label_two_hop_shared_kernel_prior_design_20261009_v1/REPORT.md)
- [Concise public status](PUBLIC_STATUS.md)
- [Complete graph-context result](context9_whole_family_collection_execution_root_20261008_v1/compact/REPORT.md)
- [Fresh independent result assessment](context9_closed_result_independent_assessment_20261008_v1.md)
- [Representative experiment requirements](canonical_SupCon_and_attention_integration_root_20261008_v1/REPRESENTATIVE_EXPERIMENT_STANDARD.md)
- [Complete classifier diagnostic](direct12_result_synthesis_root_20261008_v1/REPORT.md)
- [Public complete training CLI](portable_context_steering_public_interface_20261008_v1/README.md)
- [Preserved decisions, results and failures](research_ledger.json)
- [Canonical base literature memory](literature_memory/index_v72/LITERATURE_INDEX.json)
- [Current literature supplement](literature_memory/CURRENT_SUPPLEMENT.json)

Raw datasets, representations and checkpoints stay on authorized servers. Earlier notes and failed directions remain preserved in Git and the ledger.
