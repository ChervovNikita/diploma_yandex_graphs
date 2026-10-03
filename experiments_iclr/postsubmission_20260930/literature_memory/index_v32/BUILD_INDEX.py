"""Append saved, bounded conclusions to immutable v31; no retrieval or modeling."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PRIOR = ROOT / "literature_memory/index_v31/LITERATURE_INDEX.json"
PACKET_NAME = "conditional_graph_response_closest_priors_20261003_v1"
PACKET = ROOT / PACKET_NAME


def digest(p):
    b = p.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


old = json.loads(PRIOR.read_text())
new = copy.deepcopy(old)
sources = json.loads((PACKET / "CONCLUSIONS.json").read_text())["sources"]
scopes = json.loads((PACKET / "READ_SCOPES.json").read_text())
verified = json.loads((PACKET / "VERIFICATION.json").read_text())
assert len(old["paper_records"]) == 130
assert scopes["new_primary_method_scopes"] == 3 and scopes["full_primary_papers_read"] == 0
assert verified["status"] == "PASS"
for file in json.loads((PACKET / "MANIFEST.json").read_text())["files"]:
    assert digest(PACKET / file["path"]) == {k: file[k] for k in ["bytes", "sha256"]}

utc = datetime.now(timezone.utc).isoformat()
old_ids = {g["normalized_identifier"] for g in old["canonical_identifier_normalization"]["groups"]}
old_titles = {str(r["conclusion"].get("verified_title", r["conclusion"].get("title", ""))).casefold() for r in old["paper_records"]}
appended = []
for source in sources:
    raw_id = source["version"]
    normalized = re.sub(r"v\d+$", "", raw_id.casefold())
    assert normalized == source["canonical_id"]
    assert normalized not in old_ids
    assert source["title"].casefold() not in old_titles
    index = len(new["paper_records"])
    new["paper_records"].append({
        "canonical_id": raw_id,
        "conclusion": copy.deepcopy(source),
        "conclusion_file": f"{PACKET_NAME}/CONCLUSIONS.json",
        "conclusion_file_sha256": digest(PACKET / "CONCLUSIONS.json")["sha256"],
        "read_scope_reference": {"path": f"{PACKET_NAME}/READ_SCOPES.json", "sha256": digest(PACKET / "READ_SCOPES.json")["sha256"], "paper_canonical_id": raw_id},
        "integration_read_accounting": {"effective_read_status": "first_scoped_primary_method_read_for_new_indexed_identity", "new_unique_paper_identity": True, "normalized_identifier": normalized, "prior_record_indices": [], "full_paper_read": False, "scoped_method_read_event": True, "integration_primary_method_reopen": False},
    })
    new["canonical_identifier_normalization"]["groups"].append({"normalized_identifier": normalized, "kind": "paper", "raw_canonical_identifiers": [raw_id], "explicit_aliases": [], "record_indices": [index]})
    appended.append({"record_index": index, "raw_canonical_id": raw_id, "normalized_identifier": normalized, "title": source["title"], "previous_matching_identity_records": [], "previous_matching_title_records": [], "first_scoped_primary_method_read": True, "full_paper_read": False})

catalog_files = ["REPORT.md", "CONCLUSIONS.json", "READ_SCOPES.json", "PRIMARY_EXCERPTS.json", "CONTROL_RECOMMENDATIONS.json", "MANIFEST.json", "VERIFICATION.json", "PROVENANCE.json"]
catalog_files += sorted(p.name for p in PACKET.glob("RETRIEVAL*.json"))
for name in catalog_files:
    new["existing_packets"].append({"path": f"{PACKET_NAME}/{name}", "sha256": digest(PACKET / name)["sha256"], "kind": "saved_bounded_source_packet_receipt_or_conclusion", "sealed_packet_manifest_sha256": digest(PACKET / "MANIFEST.json")["sha256"], "scope": "Three new scoped method identities, zero full-paper certifications; exact algebraic identity and unresolved metadata-only lead retained. Integration makes zero public requests/primary reopens."})

catalog_paths = {r["path"] for r in new["existing_packets"]}
conclusion_paths = {r["conclusion_file"] for r in new["paper_records"]}
scope_paths = {r["read_scope_reference"]["path"] for r in new["paper_records"] if isinstance(r.get("read_scope_reference"), dict)}
groups = new["canonical_identifier_normalization"]["groups"]
papers = sum(g["kind"] == "paper" for g in groups)
software = sum(g["kind"] != "paper" for g in groups)
assert (len(new["paper_records"]), papers, software) == (133, 88, 2)
new["schema"] = "literature-memory-index-v32"
new["created_UTC"] = utc
new["predecessor_index"] = "literature_memory/index_v31/LITERATURE_INDEX.json"
new["predecessor_index_sha256"] = digest(PRIOR)["sha256"]
new["latest_adoption"] = {"prepared_UTC": utc, "state": "PREPARED_FOR_ROOT_INTEGRITY_REVIEW", "previous_adoption_reference": "literature_memory/index_v31/ADOPTION_RECEIPT.json", "packet_bindings": [{"packet": PACKET_NAME, "manifest_sha256": digest(PACKET / "MANIFEST.json")["sha256"], "report_sha256": digest(PACKET / "REPORT.md")["sha256"], "payload_files_verified": verified["counts"]["manifest_files"], "state": "SAVED_SCOPED_CONCLUSIONS_BOUND"}], "novelty_quality_or_execution_authorized": False}
new["post_v31_append"] = {"UTC": utc, "packet": PACKET_NAME, "new_scoped_primary_paper_identities": 3, "additional_retained_identity_method_scopes": 0, "scoped_method_events_in_packet": 3, "new_full_paper_reads": 0, "integration_primary_method_reads": 0, "integration_public_requests": 0, "canonical_ledger_status_edited": False}
new["read_accounting"].update({
    "state": "PREPARED_FOR_ROOT_INTEGRITY_REVIEW",
    "catalog_entries": len(new["existing_packets"]),
    "conclusion_records": len(new["paper_records"]),
    "normalized_paper_identifiers": papers,
    "software_documentation_identifiers": software,
    "unique_catalog_document_paths": len(catalog_paths),
    "unique_conclusion_source_documents": len(conclusion_paths),
    "unique_referenced_document_paths": len(catalog_paths | conclusion_paths),
    "unique_scope_reference_document_paths": len(scope_paths),
    "latest_adoption_packets": 1,
    "latest_index_growth": {"records": 3, "new_normalized_paper_identities": 3, "scoped_primary_method_events": 3, "first_scoped_primary_method_reads": 3, "retained_identity_scoped_primary_revisits": 0, "full_primary_reads": 0},
    "latest_packet_first_scoped_method_identity": [r["normalized_identifier"] for r in appended],
    "latest_packet_new_scoped_primary_reads": 3,
    "latest_packet_retained_primary_revisits": 0,
    "latest_packet_full_primary_reads": 0,
    "latest_packet_scoped_primary_method_events": 3,
    "latest_packet_bounded_author_source_scope_events": 0,
    "integration_pass_primary_method_reads": 0,
    "integration_pass_full_primary_reads": 0,
    "integration_pass_new_primary_reads": 0,
    "integration_pass_retained_primary_revisits": 0,
    "integration_pass_metadata_identity_checks": 0,
    "latest_accounting_correction_reference": "literature_memory/index_v32/IDENTITY_AND_SCOPE_ACCOUNTING.json",
    "historical_path_catalog_note": "v31 preserved exactly; v32 appends three identities verified from saved packet/retained records only. Record and identity totals are not full-paper read counts.",
})
assert new["paper_records"][:130] == old["paper_records"]
assert new["existing_packets"][:len(old["existing_packets"])] == old["existing_packets"]
assert groups[:len(old["canonical_identifier_normalization"]["groups"])] == old["canonical_identifier_normalization"]["groups"]
write("LITERATURE_INDEX.json", new)
write("IDENTITY_AND_SCOPE_ACCOUNTING.json", {"schema": "additive-index-identity-accounting-v1", "UTC": utc, "predecessor": {"path": str(PRIOR.relative_to(ROOT)), **digest(PRIOR)}, "source_packet": PACKET_NAME, "appended_records": appended, "older_records_preserved_exactly": 130, "older_catalog_entries_preserved_exactly": len(old["existing_packets"]), "older_normalization_groups_preserved_exactly": len(old["canonical_identifier_normalization"]["groups"]), "paper_identities": 88, "software_identities": 2, "conclusion_records": 133, "first_scoped_primary_method_reads_in_source_packet": 3, "retained_identity_revisits_in_source_packet": 0, "new_full_paper_certifications": 0, "integration_primary_reads_or_retrievals": 0, "metadata_only_ACGA_not_counted_as_read_or_appended_paper": True, "cumulative_full_paper_read_total_certified": False})
(OUT / "README.md").write_text("# Literature memory v32\n\nPrepared for root integrity review. Preserves all 130 v31 conclusion records, all 165 catalog entries and all 87 normalization groups exactly. Appends three saved conditional-graph-response source conclusions: Repulsive Deep Ensembles, CF-GNNExplainer and AD-GCL.\n\nThe index has **133 conclusion records, 88 normalized paper identifiers and 2 software identifiers**. These are not full-paper reading totals. The source packet made three first scoped method reads and no full-paper certifications or author-code reads. Index integration made no new primary reads or public requests. The inaccessible AAAI2025 lead remains metadata-only and is not counted as a read paper.\n\nThe exact degree-two kernel-energy identity is retained separately from published operator ancestry. No duplicate matched-kernel fit is proposed. Earlier index files and the canonical ledger/status remain unchanged; this append provides no quality, novelty, execution or acceptance authorization.\n")
write("APPEND_RECEIPT.json", {"schema": "saved-scoped-index-append-receipt-v1", "UTC": utc, "state": "PREPARED_FOR_ROOT_INTEGRITY_REVIEW", "prior_index": digest(PRIOR), "new_index": digest(OUT / "LITERATURE_INDEX.json"), "source_manifest": digest(PACKET / "MANIFEST.json"), "source_report": digest(PACKET / "REPORT.md"), "older_records_preserved_exactly": 130, "added_records": 3, "new_normalized_paper_identities": 3, "new_full_paper_reads": 0, "integration_new_retrievals": 0, "canonical_ledger_status_edited": False, "older_index_edited": False})
print(json.dumps({"records": 133, "papers": papers, "software": software, "catalog_entries": len(new["existing_packets"]), "unique_catalog_paths": len(catalog_paths), "scope_paths": len(scope_paths), "older_records_preserved": 130}))
