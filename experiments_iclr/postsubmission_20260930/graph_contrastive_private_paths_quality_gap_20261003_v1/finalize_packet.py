"""Seal one prospective candidate using saved literature conclusions only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
RESEARCH = PACKET.parent
NOW = datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (PACKET / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def binding(relative):
    path = RESEARCH / relative
    return {"path": relative, "sha256": sha(path), "size": path.stat().st_size}


index_path = RESEARCH / "literature_memory/index_v31/LITERATURE_INDEX.json"
assert sha(index_path) == "109cb9b981db5320d509df9d5b4e1bd44336690a954f2c0621c7ca8673365c0c"
index = json.loads(index_path.read_text())
positions = [42, 45, 48, 52, 117, 118, 120, 127, 128, 129]
selected = []
bindings = {"literature_memory/index_v31/LITERATURE_INDEX.json": binding("literature_memory/index_v31/LITERATURE_INDEX.json")}
for position in positions:
    record = index["paper_records"][position]
    b = binding(record["conclusion_file"])
    assert b["sha256"] == record["conclusion_file_sha256"]
    bindings[b["path"]] = b
    selected.append({"index_position_zero_based": position, "read_kind_this_packet": "saved_conclusion_reuse_no_primary_reread", "record": record})

report_paths = [
    "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/ASSESSMENT.md",
    "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/SOURCES_AND_EVIDENCE.json",
    "member_subspace_messages_v1/factor_graph_sensitivity_v1/ASSESSMENT.md",
    "heterogeneous_label_relevant_diversity_theory_20261003_v1/REPORT.md",
    "heterogeneous_label_relevant_diversity_theory_20261003_v1/PAPER_CONCLUSIONS.json",
    "graph_member_filter_quality_gap_20261003_v1/REPORT.md",
    "shared_message_closest_prior_v1/contrastive_repulsion_prior_v1/INSPECTED_PASSAGES.json",
    "genn_primary_access_resolution_20261003_v1/MANIFEST.json",
    "genn_primary_access_resolution_20261003_v1/REPORT.md",
    "literature_memory/index_v31/READ_ACCOUNTING_CORRECTION.json",
]
for path in report_paths:
    bindings[path] = binding(path)
assert bindings["genn_primary_access_resolution_20261003_v1/MANIFEST.json"]["sha256"] == "8049b1915b0883159e14ec8113b368611ae4ac8f9c8bebfc42f52b26f814b703"
assert bindings["genn_primary_access_resolution_20261003_v1/REPORT.md"]["sha256"] == "d173a529db1dc5870a91076d732757cc5c23827bcccc9721ea6093bd7d225afb"

source_custody = json.loads((RESEARCH / "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/SOURCES_AND_EVIDENCE.json").read_text())
own = []
for record in source_custody["own_primary_records"]:
    if record["name"] in {"adp", "gncl", "dncc", "dverge"}:
        own.append({"name": record["name"], "version": record["version"], "metadata": record["metadata"], "prior_primary_retrieval": record["primary"], "prior_passage_coordinates": [{k: v for k, v in passage.items() if k != "text"} for passage in record["inspected_passages"]], "this_packet_read_kind": "Reused interpretation and custody coordinates; no primary HTML/PDF reopened"})
write("REUSED_REFERENCES.json", {"schema": "saved-contrastive-quality-conclusions-v1", "UTC": NOW, "index": bindings["literature_memory/index_v31/LITERATURE_INDEX.json"], "index_counts_not_full_paper_counts": {"conclusion_records": 130, "normalized_paper_ids": 85, "software_ids": 2}, "selected_index_records": selected, "contrastive_saved_primary_scope_records": own, "external_bindings": list(bindings.values()), "hgen_identity_note": "Retained identity arxiv:2509.09843, DOI alias10.24963/ijcai.2025/685; no new unique read counted", "genn_note": "Six-route access check remains closed and unchanged; exact method overlap unresolved", "new_primary_reads": 0})

citations = [
    {"key": "dice", "label": "DICE, Rame and Cord", "canonical_id": "arXiv:2101.05544v1", "primary_url": "https://arxiv.org/html/2101.05544v1", "prior_scope": "Section2.1 Eq1, conditional feature redundancy and supervised information-bottleneck terms", "scope_binding": "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/ASSESSMENT.md"},
    {"key": "cdlg", "label": "CDLG, Zhang, Fu and Li", "canonical_id": "arXiv:2306.11344v1", "primary_url": "https://arxiv.org/html/2306.11344v1", "prior_scope": "SectionIII-C same-node/channel positive and cross-channel negative pair; saved PDF page4 loss-notation check", "scope_binding": "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/ASSESSMENT.md"},
    {"key": "gncl", "label": "GNCL, Buschjaeger, Pfahler and Morik", "canonical_id": "arXiv:2011.02952v2", "primary_url": "https://arxiv.org/html/2011.02952v2", "prior_scope": "Section4.1 Eq5 exact member/ensemble-loss trade-off; Eq4 local Hessian-weighted diversity", "scope_binding": "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/SOURCES_AND_EVIDENCE.json"},
    {"key": "adp", "label": "Improving Adversarial Robustness via Promoting Ensemble Diversity", "canonical_id": "arXiv:1901.08846v3", "primary_url": "https://arxiv.org/html/1901.08846v3", "prior_scope": "Sections3.1–3.3; retained dimensional audit and pooling limits", "scope_binding": "heterogeneous_label_relevant_diversity_theory_20261003_v1/PAPER_CONCLUSIONS.json"},
    {"key": "forde", "label": "Input-gradient space particle inference for neural network ensembles", "canonical_id": "arXiv:2306.02775v3", "primary_url": "https://arxiv.org/html/2306.02775v3", "prior_scope": "Section3.2 true-label normalized input gradients;3.4 minibatch approximation;3.5 differentiation cost", "scope_binding": "member_subspace_messages_v1/factor_graph_sensitivity_v1/ASSESSMENT.md"},
    {"key": "diverse_graph_experts", "label": "Training Diverse Graph Experts for Ensembles: A Systematic Empirical Study", "canonical_id": "arXiv:2510.18370v1", "primary_url": "https://arxiv.org/html/2510.18370v1", "prior_scope": "Sections2.1–2.3 and AppendixB.2; independent experts and heldout fusion", "scope_binding": "literature_root_followup_20261002_v1/REUSED_CONCLUSIONS.json"},
    {"key": "wood", "label": "A Unified Theory of Diversity in Ensemble Learning", "canonical_id": "jmlr:v24/23-0041", "primary_url": "https://jmlr.org/papers/v24/23-0041.html", "prior_scope": "PDFpages9–14; prior equation pixel verification9,10,14", "scope_binding": "heterogeneous_label_relevant_diversity_theory_20261003_v1/PAPER_CONCLUSIONS.json"},
    {"key": "polyformer", "label": "PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer", "canonical_id": "arxiv:2407.14459;doi:10.1145/3637528.3671849", "primary_url": "https://arxiv.org/html/2407.14459v1", "prior_scope": "Saved polynomial tokens, nodewise order attention, residual/FFN method and pinned source qualification", "scope_binding": "coordinate_source_independent_review_v1/strong_backbones_v1/PAPER_CONCLUSIONS.json"},
    {"key": "tfe", "label": "Unifying Homophily and Heterophily for Spectral Graph Neural Networks via Triple Filter Ensembles", "canonical_id": "DOI:10.52202/079017-2966", "primary_url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/a9db2b121c3517fd559ecbe5038701ee-Paper-Conference.pdf", "prior_scope": "Saved complete relevant method/evaluation text scope, not every-page/full-paper certification", "scope_binding": "graph_distinct_quality_literature_20261003_v1/PAPER_CONCLUSIONS.json"},
]
for citation in citations:
    citation["this_packet_read_status"] = "saved_conclusion_reuse_not_new_primary_scope"
    citation["scope_binding_sha256"] = sha(RESEARCH / citation["scope_binding"])
write("CITATIONS.json", {"schema": "exact-version-citation-reuse-v1", "UTC": NOW, "citations": citations, "new_primary_citation_sources_read": 0, "no_global_coverage_or_first_use_claim": True})

write("READ_SCOPES.json", {"schema": "saved-conclusions-only-candidate-scope-v1", "UTC": NOW, "new_primary_method_scopes": 0, "new_unique_primary_papers_read": 0, "full_paper_certifications": 0, "deliberate_retained_primary_method_revisits": 0, "author_source_reads_or_audits": 0, "network_literature_queries": 0, "read_saved_assessments": report_paths[:6], "metadata_custody_check": {"path": report_paths[6], "scope": "JSON top-level structure only, no new passage selection"}, "incidental_archived_primary_quotation_exposure": {"path": "member_subspace_messages_v1/contrastive_be_diversity_literature_v1/SOURCES_AND_EVIDENCE.json", "source": "ADP1901.08846v3", "blocks_visible": [37, 40, 41, 42], "partial_block_visible": 43, "scope_note": "Archived quotations incidentally displayed in first6000 characters during source-custody inspection; no complete method reread, numerical result or new scope certification"}, "prior_scopes_reused": "CITATIONS.json and REUSED_REFERENCES.json; reading source custody does not certify full-paper reads", "current_data_tensors_outcomes_or_experiments": 0})

candidate = {"schema": "single-attributed-method-extension-v1", "UTC": NOW, "candidate_count": 1, "name": "Class- and neighborhood-conditioned functional response diversity on guarded private intermediate paths", "status": "prospective_unexecuted_extension", "attributed_ingredients": ["DICE conditional label-redundancy reduction", "CDLG graph channel contrast/views", "FoRDE task-relevant response/sensitivity diversity", "GNCL competence/diversity trade-off", "Existing all-layer BE private factors", "Ordinary frozen-backbone adaptation and constrained/backtracked optimization"], "operational_delta": ["Anchored class-probability changes under two fixed TRAIN-label-based neighborhood removal views, rather than arbitrary embedding coordinates", "Within-class/neighborhood response centering and normalized pairwise redundancy", "Only existing private intermediate maps adapt; common weights/stem/classifier/statistics frozen", "Actual native/probe CE and response-energy feasibility checks on proposed private steps", "Native inference graph and fixed raw-logit pool unchanged"], "response": "A_m(v)=Concat[p_native_m(v)-p_same_removal_m(v),p_native_m(v)-p_other_removal_m(v)]", "penalty": "D=mean_groups mean_member_pairs <normalize(vec(A_m-group_mean(A_m))),normalize(vec(A_n-group_mean(A_n)))>^2", "invariance": "Function-preserving hidden coordinate changes, classifier nullspace changes and common logit shifts leave D identical. Positive response scaling alone leaves D identical; reference energy bands prevent zero/inflated responses.", "competence_limit": "TRAIN control-role constraints are exact only on their declared deterministic evaluation sets; validation/test competence is not guaranteed", "plausible_mechanism": "Private nonlinear multiscale paths may retain class-relevant alternative neighborhood reliance while common maps remain competent", "new_primitive_or_global_novelty_certified": False, "published_exact_complete_overlap_excluded": False, "genn_overlap": "Unresolved; sealed access packet unchanged", "scope_limits": ["Second-order response decorrelation is not MI/independence", "Supervised evidence-removal masks are not causal counterfactuals", "No objective guarantees CE/calibration/generalization", "Can forbid legitimate graph invariance or yield infeasible updates", "Modern multiscale backbone may already preserve all useful evidence"]}
write("CANDIDATE.json", candidate)

arms = [
    {"id": "native_only", "objective": "Native mean-member CE only", "role": "Ordinary private continuation"},
    {"id": "augmentation_only", "objective": "Native CE plus mean two-probe CE", "role": "Additional supervision/compute baseline"},
    {"id": "candidate", "objective": "Same CE plus0.1D", "role": "One proposed extension"},
    {"id": "class_only", "objective": "Same CE plus response redundancy with class-only groups", "role": "Neighborhood-conditioning necessity"},
    {"id": "hidden_repulsion", "objective": "Same CE plus normalized class/group hidden cosine repulsion", "role": "Ordinary embedding separation"},
    {"id": "dice", "objective": "Same CE plus source-qualified conditional feature redundancy", "role": "Strong label-conditional redundancy prior; charge adversarial estimator"},
    {"id": "forde", "objective": "Same CE plus source-qualified normalized true-label input-gradient diversity", "role": "Strong functional sensitivity prior; explicitly label any graph-domain adaptation and charge derivatives"},
    {"id": "permuted_masks", "objective": "Candidate with degree-matched label-permuted masks/counts", "role": "Graph-label evidence semantics falsifier"},
]
pilot = {"schema": "one-representative-prospective-pilot-v1", "UTC": NOW, "pilots_proposed": 1, "pilots_executed": 0, "launch_authorized_by_this_packet": False, "development_task": {"dataset": "Complete Amazon-ratings node classification", "version_and_native_split_file_hashes": "Must qualify/freeze before admission; no current data accessed", "split_seed_blocks": [{"official_split_id": 0, "optimizer_seed": 17}, {"official_split_id": 1, "optimizer_seed": 29}, {"official_split_id": 2, "optimizer_seed": 43}], "fit_control_train_roles": "Fixed stratified80/20 within official TRAIN; both roles are supervision, not heldout confirmation", "forbidden_label_uses": "No VALIDATION/TEST labels in mask/group/reference/acceptance construction"}, "backbone": "Source-qualified PolyFormer with existing all-layer BE factors and complete member trajectories; M4", "warm_phase": "Competent native mean-member CE, one shared warm bank per block; source/implementation and selector qualification before protocol freeze", "continuation": {"updates": 200, "trainable": "Existing intermediate attention/FFN private factors only", "frozen": "Common maps, stems, classifier heads, running statistics", "native_serving_pool": "Mean raw logits then softmax", "secondary_pool": "Mean probabilities, separately labeled", "paired_streams": True, "native_validation_selector": "Same validation NLL checkpoint rule for all arms; no outcome-based intervention time"}, "views": {"native_graph": "Unchanged complete native graph", "same_class_probe": "Fixed10% sample of edges between fit-role labeled nodes of equal class removed", "other_class_probe": "Fixed10% sample of edges between fit-role labeled nodes of different class removed", "normalization_tokens": "Exactly native normalization and complete per-view polynomial token computation; costs charged", "semantics": "Supervised evidence removal, no causal or label-invariance theorem"}, "group_rule": {"base": "(target class, same-class fit-labeled-neighbor fraction>=0.5)", "no_labeled_neighbors": "Separate fallback", "fit_minimum": 32, "control_minimum": 16, "fallback_order": "Class-only then global; no native supervised targets removed", "minimum_active_fit_coverage": 0.5}, "guards": {"per_member_native_and_probe_control_ce_margin_nats": 0.01, "supported_cells": "Global and supported class/neighborhood cells", "reference": "Copied warm function on deterministic control roles", "response_norm_floor": 1e-5, "response_energy_reference_band": [0.5, 2.0], "same_frozen_active_set_for_all_arms": True, "proposal_scales": [1.0, 0.5, 0.25, 0.125, 0.0], "optimizer_state": "Accept commits corresponding moments once; zero step discards proposed parameter and optimizer-state updates", "generalization_guarantee": False}, "fixed_continuation_arms": arms, "control_qualification": "DICE/FoRDE use competent source-informed fixed recipes qualified without pilot outcomes. If unavailable or weak stand-ins, no launch. Do not silently repair a published loss or use hidden tuning.", "utility_references": ["Competent PolyFormer single", "Capacity-matched PolyFormer single", "Native TFE-GNN single", "M4 independently trained PolyFormer ensemble with competent packing"], "resource_accounting": {"base_acquisitions_member_fit_equivalents": 33, "warm_be_banks": 3, "continuation_m4_banks": 24, "utility_continuation": "Matched paid budget and label visibility; additional cost counted", "must_charge": ["Warm acquisition", "Three complete graph/token caches", "All member paths", "DICE estimator", "FoRDE derivatives", "Guard and backtracking evaluations", "Rejected proposals"], "timing_or_hardware_measurement": None}, "primary_development_screen": {"pooled_validation_nll_mean_min_improvement_nats": 0.01, "comparators": "Augmentation-only and qualified diversity controls", "paired_sign_rule": "Same improvement sign in all3 blocks", "mean_accuracy_and_macro_f1_max_loss_percentage_points": 0.2, "mean_member_nll_max_increase_nats": 0.01, "worst_member_nll_max_increase_nats": 0.02, "intervals": "Descriptive only"}, "falsifiers": ["Lower response redundancy without useful pooled improvement", "Class-only matches: neighborhood grouping unnecessary", "Permuted masks match: evidence semantics unsupported", "Under10% proposals accepted: guarded adaptation not useful under this protocol", "Utility reference matches more cheaply: no quality/cost advantage", "No post-outcome rescue via new sites/groups/masks/losses/seeds"], "heldout_confirmation": "Only after a passing screen: freeze unchanged mechanism/control set for complete Roman-empire with native splits and fresh predetermined seeds; no present confirmation authorized", "broad_grid": False, "frozen_studies_changed": False, "manuscript_verdict": None}
pilot["continuation"]["dropout_and_grad_mode"] = "Dropout off for every Stage B objective/guard; gradients enabled for objectives and disabled for guard evaluation; response separation cannot be random-mask noise"
write("PILOT_SPEC.json", pilot)
write("PROVENANCE.json", {"schema": "prospective-private-path-quality-custody-v1", "UTC": NOW, "packet": PACKET.name, "bindings": list(bindings.values()), "read_accounting": {"new_primary_method_scopes": 0, "full_paper_certifications": 0, "deliberate_retained_primary_revisits": 0, "incidental_archived_primary_quotation_sources": 1, "author_source_audits": 0, "network_queries": 0}, "candidate_count": 1, "prospective_pilot_count": 1, "pilot_execution": 0, "constraints_observed": {"live_outcomes_data_tensors_experiments": False, "frozen_study_edits": False, "broad_grid": False, "manuscript_verdict": False, "genlink_use": False, "pdf_compilation": False, "canonical_ledger_status_edits": False, "index_append": False, "additional_agents": False}, "genn_packet_unchanged": True, "new_write_scope": str(PACKET), "source_execution": "Custody script only; no scientific/model/dataset imports or execution", "not_certified": ["Actual private-path response complementarity", "Quality/cost benefit", "Feasibility", "Whole-paper reading", "Global novelty or exact complete-prior exclusion"]})

payloads = [{"path": str(path.relative_to(PACKET)), "sha256": sha(path), "size": path.stat().st_size} for path in sorted(PACKET.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"]
write("MANIFEST.json", {"schema": "sha256-payload-manifest-v1", "UTC": NOW, "packet": PACKET.name, "excludes": ["MANIFEST.json (self)"], "payload_file_count": len(payloads), "files": payloads})
print(json.dumps({"payload_files": len(payloads), "manifest_sha256": sha(PACKET / "MANIFEST.json"), "report_sha256": sha(PACKET / "REPORT.md"), "new_primary_method_scopes": 0, "full_paper_certifications": 0, "candidate_count": 1, "prospective_pilots": 1, "pilots_executed": 0}, indent=2))
