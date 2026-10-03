"""Seal the six-route metadata/access check. No source/model execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

PACKET = Path(__file__).resolve().parent
RESEARCH = PACKET.parent
NOW = datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (PACKET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(path):
    p = RESEARCH / path
    return {"path": path, "sha256": sha(p), "size": p.stat().st_size}


receipt = json.loads((PACKET / "ROUTE_RECEIPTS.json").read_text())
assert len(receipt) == len({r["url"] for r in receipt}) == 6
assert all(not r["previously_blocked_sciencedirect_url_retried"] for r in receipt)
assert receipt[-1]["key"] == "r6_ssrn_delivery_4535927" and receipt[-1]["status"] == 403
for request in receipt:
    assert sha(PACKET / request["saved_path"]) == request["sha256"]
    assert (PACKET / request["saved_path"]).stat().st_size == request["size"]

seed = "graph_member_filter_quality_gap_20261003_v1"
paths = [
    f"{seed}/discovery/GENN_CROSSREF.json",
    f"{seed}/discovery/OPENALEX.json",
    f"{seed}/discovery/GENN_ACCESS.json",
    f"{seed}/REPORT.md",
    f"{seed}/primary/hgen.pdf",
    f"{seed}/MANIFEST.json",
    "modern_rank_one_block_objective_priors_20261003_v1/REPORT.md",
    "literature_memory/index_v31/LITERATURE_INDEX.json",
    "literature_memory/index_v31/READ_ACCOUNTING_CORRECTION.json",
]
bindings = [binding(path) for path in paths]
assert next(b["sha256"] for b in bindings if b["path"].endswith("index_v31/LITERATURE_INDEX.json")) == "109cb9b981db5320d509df9d5b4e1bd44336690a954f2c0621c7ca8673365c0c"
assert next(b["sha256"] for b in bindings if b["path"] == f"{seed}/MANIFEST.json") == "267adb071ed0407f7c473bfc9d2dec1d27b93c59bddae093b94d5b3b9fb25d9b"
write("INPUT_BINDINGS.json", {"schema": "saved-metadata-access-seeds-v1", "UTC": NOW, "bindings": bindings, "saved_sources_consulted_before_new_requests": [f"{seed}/discovery/GENN_CROSSREF.json", f"{seed}/discovery/OPENALEX.json", f"{seed}/discovery/GENN_ACCESS.json"], "hgen_secondary_locator": {"source": f"{seed}/primary/hgen.pdf", "previously_read_scope": "Page1 introduction; reused previous filter-gap report Section7 characterization only", "characterization": "GEN integrates ensemble operations through GNN training, rather than only prediction", "primary_genn_method_evidence": False, "new_primary_revisit": False}, "closed_or_live_outcomes_accessed": False})

crossref = json.loads((RESEARCH / f"{seed}/discovery/GENN_CROSSREF.json").read_text())["message"]
write("SAVED_METADATA_SEED.json", {"title": crossref["title"][0], "doi": crossref["DOI"], "authors": [a["given"] + " " + a["family"] for a in crossref["author"]], "published": crossref["published"], "first_author_orcid": crossref["author"][0]["ORCID"], "publisher_full_text_locators_not_attempted_in_this_pass": crossref["link"], "earlier_blocked_request": json.loads((RESEARCH / f"{seed}/discovery/GENN_ACCESS.json").read_text()), "secondary_characterization_is_not_primary_exclusion": True})

ss = json.loads((PACKET / receipt[0]["saved_path"]).read_text())
arxiv = ET.fromstring((PACKET / receipt[1]["saved_path"]).read_text())
github = json.loads((PACKET / receipt[2]["saved_path"]).read_text())
openaire = json.loads((PACKET / receipt[3]["saved_path"]).read_text())
orcid = json.loads((PACKET / receipt[4]["saved_path"]).read_text())
oaf = openaire["response"]["results"]["result"][0]["metadata"]["oaf:entity"]["oaf:result"]
pids = [item["$"] for item in oaf["pid"]]
assert "10.1016/j.inffus.2024.102461" in pids and "10.2139/ssrn.4535927" in pids
assert ss["externalIds"]["DOI"] == "10.1016/j.inffus.2024.102461"
assert not ss["isOpenAccess"] and not ss["openAccessPdf"]["url"]
assert github["total_count"] == 0 and not github["incomplete_results"]
assert arxiv.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") == "0"
assert orcid["researcher-url"] == []
findings = {"schema": "targeted-source-locator-findings-v1", "UTC": NOW, "semantic_scholar": ss, "arxiv_exact_title": {"total_results": 0, "entry_count": 0, "coverage_limit": "Exact queried title only"}, "github_exact_title": {"total_count": 0, "incomplete_results": False, "coverage_limit": "Exact queried phrase only; no differently named/unindexed/private-code exclusion"}, "openaire": {"result_count": int(openaire["response"]["header"]["total"]["$"]), "grouped_pids": pids, "titles": oaf["title"], "best_access_right": oaf["bestaccessright"], "child_instances": oaf["children"]["instance"], "ssrn_candidate": {"doi": "10.2139/ssrn.4535927", "abstract_id": "4535927", "locator": "https://doi.org/10.2139/ssrn.4535927", "access_in_metadata": "UNKNOWN", "primary_bytes_obtained": False, "published_version_equivalence_verified": False, "version_alias_certified": False}}, "orcid_researcher_urls": orcid, "ssrn_delivery": {"status": 403, "primary_bytes_obtained": False, "error_page_method_read": False}, "no_primary_or_code_source_available": True}
write("METADATA_FINDINGS.json", findings)

unknown_fields = ["shared_private_frozen_parameter_partition", "member_specific_graph_operators_and_nonlinearity_placement", "per_member_vs_pooled_supervision_and_reduction_factors", "update_timing_and_pre_update_state_reuse", "served_logit_probability_embedding_or_gated_pool", "selective_gradient_stops_or_replacements", "shared_feature_backbone_with_private_graph_filters"]
comparison = {"schema": "unresolved-operation-comparison-v1", "UTC": NOW, "current_policy": {"shared_block": "Trainable W; gradient of CE(mean raw member logits,y)", "private_block": "Original (1/M)*gradient_phi_m CE(z_m,y)", "timing": "Both from one unchanged pre-update state, one combined optimizer step; no post-shared recomputation", "pool": "Mean raw logits then softmax", "reference": "modern_rank_one_block_objective_priors_20261003_v1/REPORT.md"}, "member_filter_family": {"description": "Common feature maps plus member-specific learned graph/neighborhood operators before nonlinear processing; persistent predictive members", "reference": f"{seed}/REPORT.md", "pilot_disposition_preserved": 0}, "genn_method_facts": {field: {"status": "unresolved", "primary_evidence": None} for field in unknown_fields}, "exact_target_equivalence_established": False, "exact_target_exclusion_established": False, "filter_family_overlap_or_exclusion_established": False, "novelty_or_quality_certified": False, "secondary_training_characterization_limit": "HGEN's description establishes relevance, not GENN's exact operation"}
write("METHOD_COMPARISON.json", comparison)

write("READ_SCOPES.json", {"schema": "access-only-read-accounting-v1", "UTC": NOW, "new_primary_method_scopes": 0, "new_full_paper_reads_or_certifications": 0, "author_source_files_read": 0, "author_source_audits": 0, "retained_primary_method_revisits": 0, "new_targeted_routes": 6, "new_http_requests": 6, "distinct_urls": 6, "route_retries": 0, "previously_blocked_sciencedirect_url_retries": 0, "scopes": [{"route_key": r["key"], "saved_path": r["saved_path"], "read_kind": "metadata_locator_response" if r["status"] == 200 else "access_error_receipt_no_primary_method", "primary_method_read": False} for r in receipt], "saved_reference_scope": "Existing conclusions and citation metadata only; source PDF bound by hash, not reopened as a primary method", "zero_count_policy": "Metadata records, a grouped SSRN locator and a 403 page do not count as primary/code or full-paper reading", "experiments_or_current_outcomes": 0})

record = {"canonical_id": "DOI:10.1016/j.inffus.2024.102461", "verified_registered_title": crossref["title"][0], "registered_authors": [a["given"] + " " + a["family"] for a in crossref["author"]], "read_status": "primary_unavailable_metadata_only_bounded_access_routes_closed", "full_read": False, "primary_method_read": False, "author_source_read": False, "saved_takeaway": "Exact metadata identifies GENN; six targeted routes obtained no primary method or author code. OpenAIRE identifies candidate SSRN4535927, whose delivery returned403. Shared/private maps, graph operators, individual/pooled losses, update timing, serving pool and selective gradients remain unresolved.", "new_locator": "DOI:10.2139/ssrn.4535927 (candidate; manuscript/version identity not certified)", "comparison_disposition": "Retain relevant unresolved closest prior; no exact target equivalence/exclusion or member-filter exclusion from unavailable source", "resume_condition": "Newly supplied accessible primary/code, changed access, or newly authorized named route; no default repeat of six requests or prior blocked ScienceDirect landing", "read_scope_reference": "READ_SCOPES.json", "evidence_paths": ["ROUTE_RECEIPTS.json", "METADATA_FINDINGS.json", "INPUT_BINDINGS.json", "METHOD_COMPARISON.json"], "numeric_results_adopted": False, "novelty_or_quality_claim": False}
write("PAPER_CONCLUSIONS.json", {"schema": "metadata-access-limit-conclusions-v1", "UTC": NOW, "paper_records": [record], "new_primary_method_scopes": 0, "new_full_paper_certifications": 0, "author_source_reads": 0, "metadata_access_limit_records": 1, "primary_method_conclusion_records": 0, "candidate_ssrn_version_aliases_certified": 0})
write("RESUME_POLICY.json", {"schema": "bounded-source-route-stop-v1", "UTC": NOW, "status": "six_route_access_check_closed_primary_unavailable", "routes_exhausted": 6, "do_not_repeat_by_default": [r["url"] for r in receipt] + ["https://www.sciencedirect.com/science/article/abs/pii/S1566253524002392"], "newly_identified_unresolved_locator": "https://doi.org/10.2139/ssrn.4535927", "resume_only_on": ["Accessible primary or author code explicitly supplied", "Documented change in access", "Newly authorized named targeted route"], "source_absence_or_novelty_certificate": False, "new_grid_or_pilot": False})
write("PROVENANCE.json", {"schema": "bounded-source-access-custody-v1", "UTC": NOW, "packet": PACKET.name, "input_bindings": bindings, "route_limit": 6, "route_count": 6, "http_request_count": 6, "blocked_sciencedirect_retry": False, "retrieval_script": "fetch_routes.py", "read_accounting": {"new_primary_method_scopes": 0, "full_paper_certifications": 0, "author_source_reads": 0, "retained_primary_revisits": 0, "metadata_access_routes": 6}, "no_current_or_closed_outcomes_accessed": True, "no_experiments_or_training": True, "no_grid_or_pilot": True, "no_gpu_requests": True, "no_canonical_ledger_status_edits": True, "no_index_append": True, "no_additional_agents": True, "new_write_scope": str(PACKET), "limitations": ["Targeted routes do not certify exhaustive source/code absence", "SSRN grouping does not certify version/text equivalence", "No GENN method equality/exclusion claim"]})

payloads = [{"path": str(p.relative_to(PACKET)), "sha256": sha(p), "size": p.stat().st_size} for p in sorted(PACKET.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"]
write("MANIFEST.json", {"schema": "sha256-payload-manifest-v1", "UTC": NOW, "packet": PACKET.name, "excludes": ["MANIFEST.json (self)"], "payload_file_count": len(payloads), "files": payloads})
print(json.dumps({"payload_files": len(payloads), "manifest_sha256": sha(PACKET / "MANIFEST.json"), "report_sha256": sha(PACKET / "REPORT.md"), "new_primary_method_scopes": 0, "full_paper_certifications": 0, "author_source_reads": 0, "routes": 6}, indent=2))
