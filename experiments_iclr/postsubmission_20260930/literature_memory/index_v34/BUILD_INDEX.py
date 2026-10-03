"""Immutable append-only metadata integration. No old verifier or primary method execution/read."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PREVIOUS = ROOT / "literature_memory/index_v33"
PACKET_NAME = "joint_completion_decoder_uncertainty_literature_20261003_v1"
PACKET = ROOT / PACKET_NAME
LEGACY = ROOT / "shared_message_closest_prior_v1/graph_uncertainty_resolvent_prior_v1"
NEW_METHOD_IDS = {"arxiv:1903.11960", "arxiv:2404.11032"}
LEGACY_IDS = {"arxiv:1811.11103", "arxiv:1911.04965", "arxiv:2006.04064"}
EXPECTED_IDS = NEW_METHOD_IDS | LEGACY_IDS

def digest(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

def binding(path):
    return {"path": str(path.relative_to(ROOT)), **digest(path)}

def load(path):
    return json.loads(path.read_text())

def write(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError("Refusing to overwrite an immutable integration artifact: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def normalized(value):
    text = str(value).strip().casefold()
    hit = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|html)/)(\d{4}\.\d{4,5})(?:v\d+)?", text)
    return "arxiv:" + hit.group(1) if hit else text

def title_key(value):
    return " ".join(str(value).split()).casefold()

def tree_state(folder):
    return [binding(p) for p in sorted(folder.rglob("*")) if p.is_file()]

assert [p.name for p in OUT.iterdir()] == ["BUILD_INDEX.py"], "Fresh unsealed successor only"
utc = datetime.now(timezone.utc).isoformat()
assert digest(PREVIOUS / "LITERATURE_INDEX.json")["sha256"] == "27f94438c11d783cbe3919f91655ef20b7414c9b1dcf5d493d287a91317f8760"
assert digest(PREVIOUS / "MANIFEST.json")["sha256"] == "28eb6f55b089ceba546301aee5a80fa2c5afac20db83cfdf0df8635aa8ab2b85"
assert digest(PACKET / "MANIFEST.json")["sha256"] == "1675713e58807fbc390a26581a0d1e0014492ae28ad474cba5535f2739ff2084"

# Check saved seals/byte inventories; no old verifier invocation or primary parsing.
before = tree_state(PREVIOUS) + tree_state(PACKET)
for row in load(PREVIOUS / "MANIFEST.json")["files"]:
    assert digest(PREVIOUS / row["path"]) == {"bytes": row["bytes"], "sha256": row["sha256"]}
for row in load(PACKET / "MANIFEST.json")["files"].values():
    assert digest(PACKET / row["path"])["sha256"] == row["sha256"]
inventory_rows = []
for line in (PACKET / "FILE_HASHES.sha256").read_text().splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    assert match
    expected_hash, relative = match.groups()
    assert digest(PACKET / relative)["sha256"] == expected_hash
    inventory_rows.append(relative)

old = load(PREVIOUS / "LITERATURE_INDEX.json")
payload = load(PACKET / "PAPER_CONCLUSIONS.json")
source_scope = load(PACKET / "READ_SCOPES.json")
source_manifest = load(PACKET / "MANIFEST.json")
# Only metadata fields are selected from excerpts; passage text is neither surfaced nor inspected.
excerpt_metadata = [{"index": i, "canonical_id": e["canonical_id"], "key": e["key"],
                     "locator": e["locator"], "kind": e["kind"]}
                    for i, e in enumerate(load(PACKET / "PRIMARY_EXCERPTS.json")["excerpts"])]
legacy_metadata = load(LEGACY / "PRIMARY_METADATA.json")
legacy_retrieval = load(LEGACY / "ARXIV_METHOD_RETRIEVAL_RECEIPTS.json")
legacy_passage_metadata = [
    {k: e.get(k) for k in ("key", "title", "arxiv_version", "url", "source_file", "source_sha256", "block_id", "excerpt_sha256")}
    for e in load(LEGACY / "PRIMARY_PASSAGES.json")["passages"]
]
legacy_inputs = [LEGACY / name for name in ("PRIMARY_METADATA.json", "ARXIV_METHOD_RETRIEVAL_RECEIPTS.json",
                                           "METHOD_METADATA_RECEIPT.json", "PRIMARY_PASSAGES.json")]
legacy_before = [binding(p) for p in legacy_inputs]
assert len(old["paper_records"]) == 136 and len(old["existing_packets"]) == 194
old_groups = old["canonical_identifier_normalization"]["groups"]
old_papers = sum(g["kind"] == "paper" for g in old_groups)
old_software = sum(g["kind"] != "paper" for g in old_groups)
assert (old_papers, old_software) == (90, 2)
sources = payload["paper_records"]
assert len(sources) == 5 and {s["canonical_id"] for s in sources} == EXPECTED_IDS
assert set(payload["new_unique_paper_identities"]) == NEW_METHOD_IDS
assert set(payload["reused_paper_identities"]) == LEGACY_IDS
assert source_scope["counts"] == {"fresh_scoped_primary_method_read_events": 5,
                                "new_unique_paper_identities_after_saved_history_check": 2,
                                "reused_identities_reinspected": 3, "full_paper_reads": 0,
                                "author_code_inspections": 0}
assert source_manifest["accounting"]["full_paper_reads"] == 0

def prior_matches(source):
    cid, title = source["canonical_id"], title_key(source["title"])
    records = []
    for i, record in enumerate(old["paper_records"]):
        conclusion = record.get("conclusion", {})
        identifiers = [record.get("canonical_id", ""), conclusion.get("canonical_id", "")]
        titles = [conclusion.get(k, "") for k in ("title", "verified_title")]
        if any(normalized(v) == cid for v in identifiers) or any(title_key(v) == title for v in titles if v):
            records.append(i)
    groups = []
    for i, group in enumerate(old_groups):
        identifiers = [group["normalized_identifier"]] + group.get("raw_canonical_identifiers", []) + group.get("explicit_aliases", [])
        if any(normalized(v) == cid for v in identifiers):
            groups.append(i)
    return records, groups

lookup_decisions = []
for source in sources:
    records, groups = prior_matches(source)
    assert not records and not groups, "Old group/record would require a separately specified append rule"
    matching_metadata = [r for r in legacy_metadata if normalized(r["id"]) == source["canonical_id"]]
    assert all(title_key(r["title"]) == title_key(source["title"]) for r in matching_metadata)
    old_retrievals = [{k: r[k] for k in ("id", "url", "sha256", "status")} for r in legacy_retrieval
                      if normalized(r["url"]) == source["canonical_id"]]
    old_passages = [r for r in legacy_passage_metadata if normalized(r.get("url", "")) == source["canonical_id"]]
    if source["canonical_id"] in LEGACY_IDS:
        assert matching_metadata and old_passages, "Legacy reconciliation needs explicit saved identity evidence"
    else:
        assert not old_passages, "Prior selected-method evidence would alter first-inspection accounting"
    lookup_decisions.append({"key": source["key"], "canonical_id": source["canonical_id"],
        "verified_title": source["title"], "prior_v33_record_indices": records, "prior_v33_group_indices": groups,
        "prior_title_matches": records, "legacy_metadata_rows": matching_metadata,
        "legacy_retrieval_rows": old_retrievals, "legacy_selected_passage_metadata": old_passages,
        "effective_status": "newly_inspected_scoped_method_identity" if source["canonical_id"] in NEW_METHOD_IDS
                            else "legacy_scoped_method_reinspection_and_normalized_index_omission_reconciliation",
        "earlier_retrieval_does_not_certify_read": bool(old_retrievals and not old_passages)})
write("PRIOR_IDENTITY_LOOKUP.json", {"schema": "exact-saved-identity-title-reconciliation-v1", "UTC": utc,
    "predecessor_index": binding(PREVIOUS / "LITERATURE_INDEX.json"), "legacy_metadata_bindings": legacy_before,
    "decisions": lookup_decisions, "primary_methods_reread": 0, "title_only_alias_merges": 0,
    "note": "LDS was previously retrieved and metadata-known; acquisition alone is not a certified method read. "
            "All five normalized groups are missing in v33; only LDS/CORE are newly inspected scoped-method identities in the current source packet."})

# Add only an integration qualification; preserve the source report/excerpts/conclusions exactly.
write("INTEGRATION_LIMITS.json", {"schema": "source-informed-joint-uncertainty-integration-limits-v1", "UTC": utc,
    "preserved_report": binding(PACKET / "REPORT.md"),
    "source_packet_report_modified": False,
    "no_new_learner_or_quality_gap": True, "experiments_or_execution_authorized": False,
    "root_instruction_scope": "Record member-specific completion-feature compatibility when adopting the preserved D/X/P diagnostic.",
    "crossed_diagnostic_limit": {
        "required_representation": "C_m(q_k), with receiving member m's actual private xlin/features and decoder path",
        "own": "D = mean_m f_m(C_m(q_m))",
        "crossed": "X = mean_m mean_k f_m(C_m(q_k))",
        "pooled": "P = mean_m f_m(C_m(mean_k q_k))",
        "conditions": [
            "Cross only completion weights on the same aligned candidate/support coordinates.",
            "Retain receiving member m's actual private xlin feature transform and any required receiving-member context path.",
            "Verify feature dimensions, candidate identifiers, clamp semantics and decoder input contracts permit every C_m(q_k).",
            "Do not assume a common C(q) merely because an upstream encoder is shared.",
            "If C_m depends nonlinearly on q, X-P reflects the whole C_m/decoder path, not decoder curvature alone.",
            "If compatibility cannot be established, the crossed diagnostic is undefined for that architecture and must not be reported as a qualified control.",
            "Any fixed-bank association effect is a known-mechanism/coadaptation diagnostic; it does not establish calibrated posterior sampling or learner novelty."
        ]},
    "paper_limits": {
        "BGCN": "Graph-conditioned model/topology marginalization is prior; a-MMSBM MAP and MC dropout are approximations in node classification.",
        "LDS": "Nonlinear prediction averaging versus mean topology is explicit prior; shared weights are not a graph-conditioned model posterior.",
        "node_copying": "Printed v1 predictive likelihood uses G_obs while conditional fitting uses sampled graphs; do not silently replace that graph.",
        "adaptive_connection_sampling": "Observed-edge masks cannot add missing edges; inspected q(W,Z) is factorized with point-mass global weights.",
        "CORE": "Completion plus stochastic reduction and nonlinear link prediction is prior; native inference uses expected edge weights; theorem scope is qualified."
    }})
limits_reference = binding(OUT / "INTEGRATION_LIMITS.json")
scope_reference = binding(PACKET / "READ_SCOPES.json")
evidence_reference = binding(PACKET / "PRIMARY_EXCERPTS.json")
report_reference = binding(PACKET / "REPORT.md")
conclusion_reference = binding(PACKET / "PAPER_CONCLUSIONS.json")
manifest_reference = binding(PACKET / "MANIFEST.json")

new = copy.deepcopy(old)
groups = new["canonical_identifier_normalization"]["groups"]
decisions = []
for source in sources:
    cid = source["canonical_id"]
    record_index = len(new["paper_records"])
    selected = [e["index"] for e in excerpt_metadata if e["canonical_id"] == cid]
    assert selected and source["full_read"] is False and source["source_code_inspected"] is False
    new_method = cid in NEW_METHOD_IDS
    accounting = {"normalized_identifier": cid, "new_normalized_index_identity": True,
                  "newly_inspected_paper_identity_in_source_packet": new_method,
                  "legacy_identity_reinspection_and_omission_reconciliation": not new_method,
                  "new_unique_paper_identity_per_source_packet": source["new_unique_paper_identity"],
                  "prior_v33_record_indices": [], "scoped_primary_method_read_event_in_source_packet": True,
                  "full_paper_read": False, "integration_primary_read_or_revisit": False,
                  "effective_read_status": "newly_inspected_scoped_method_identity" if new_method else
                    "legacy_identity_reinspection_and_normalized_index_omission_reconciliation"}
    compact = {"key": source["key"], "canonical_id": cid, "versioned_id": source["versioned_id"],
               "verified_title": source["title"], "authors": source["authors"],
               "saved_takeaway": source["closest_prior_role"],
               "verified_method_scope_summary": source["primary_text_verified"],
               "read_status": accounting["effective_read_status"],
               "full_paper_read": False, "author_code_read": False,
               "qualification_limits": source["limits"], "adoption": source["adoption"],
               "scope_reference": {**scope_reference, "paper_canonical_id": cid},
               "evidence_reference": {**evidence_reference, "excerpt_indices": selected, "text_repeated_in_index": False},
               "report_reference": report_reference, "integration_qualification_reference": limits_reference,
               "numeric_results_adopted": False, "new_learner_gap_or_execution_authorized": False}
    new["paper_records"].append({"canonical_id": cid, "conclusion": compact,
        "conclusion_file": conclusion_reference["path"], "conclusion_file_sha256": conclusion_reference["sha256"],
        "source_record_locator": {"container": "paper_records", "key": source["key"]},
        "read_scope_reference": {**scope_reference, "paper_canonical_id": cid},
        "primary_evidence_reference": {**evidence_reference, "paper_canonical_id": cid, "excerpt_indices": selected},
        "source_report_reference": report_reference, "source_packet_manifest_reference": manifest_reference,
        "integration_read_accounting": accounting})
    groups.append({"normalized_identifier": cid, "kind": "paper",
                   "raw_canonical_identifiers": [cid], "explicit_aliases": [], "record_indices": [record_index]})
    decisions.append({"record_index": record_index, "key": source["key"], "canonical_id": cid,
                      **accounting, "bound_excerpt_records": len(selected)})

catalog = ["REPORT.md", "PAPER_CONCLUSIONS.json", "READ_SCOPES.json", "PRIMARY_EXCERPTS.json",
           "PROVENANCE.json", "DISCOVERY_RECEIPT.json", "FILE_HASHES.sha256", "MANIFEST.json"]
for name in catalog:
    new["existing_packets"].append({"path": f"{PACKET_NAME}/{name}", "sha256": digest(PACKET / name)["sha256"],
        "kind": "saved_joint_uncertainty_scoped_conclusion_or_metadata",
        "sealed_packet_manifest_sha256": manifest_reference["sha256"],
        "scope": "Five scoped method events: two newly inspected identities and three legacy reinspections/reconciled omissions. "
                 "No integration primary method reread, full read, author-code certification or execution authorization."})
assert new["paper_records"][:len(old["paper_records"])] == old["paper_records"]
assert new["existing_packets"][:len(old["existing_packets"])] == old["existing_packets"]
assert groups[:len(old_groups)] == old_groups
assert len({g["normalized_identifier"] for g in groups}) == len(groups)
paper_count = sum(g["kind"] == "paper" for g in groups)
software_count = sum(g["kind"] != "paper" for g in groups)
counts = {"records": len(new["paper_records"]), "catalog": len(new["existing_packets"]),
          "normalized_paper_identities": paper_count, "software_identities": software_count}
assert counts == {"records": 141, "catalog": 202, "normalized_paper_identities": 95, "software_identities": 2}

new["schema"] = "literature-memory-index-v34"
new["created_UTC"] = utc
new["predecessor_index"] = str((PREVIOUS / "LITERATURE_INDEX.json").relative_to(ROOT))
new["predecessor_index_sha256"] = digest(PREVIOUS / "LITERATURE_INDEX.json")["sha256"]
new["latest_adoption"] = {"prepared_UTC": utc, "state": "PREPARED_FOR_ROOT_ADOPTION",
    "previous_append_reference": "literature_memory/index_v33/APPEND_RECEIPT.json",
    "packet_bindings": [{"packet": PACKET_NAME, "manifest_sha256": manifest_reference["sha256"],
                        "report_sha256": report_reference["sha256"], "scope_sha256": scope_reference["sha256"],
                        "quoted_evidence_sha256": evidence_reference["sha256"],
                        "state": "SAVED_SCOPED_CONCLUSIONS_BOUND_WITHOUT_OLD_VERIFIER_RERUN"}],
    "integration_limits_reference": limits_reference, "novelty_quality_or_execution_authorized": False}
new["post_v33_append"] = {"UTC": utc, "packet": PACKET_NAME,
    "added_conclusion_records": len(decisions), "new_normalized_paper_identity_groups": len(decisions),
    "newly_inspected_scoped_method_identities": sorted(NEW_METHOD_IDS),
    "legacy_identity_reinspections_and_omission_reconciliations": sorted(LEGACY_IDS),
    "source_packet_scoped_method_events": 5, "source_packet_full_primary_reads": 0,
    "integration_primary_methods_reread": 0, "integration_primary_retrievals": 0,
    "older_index_or_verification_files_modified": False, "canonical_ledger_status_edited": False,
    "interpretation": "No additional learner novelty/quality gap found. Joint marginalization and nonlinear averaging have priors. "
                      "D/X/P is an established-mechanism diagnostic only with compatible receiving-member C_m(q_k) using actual private xlin features.",
    "predecessor_latest_adoption": copy.deepcopy(old["latest_adoption"])}
new["canonical_identifier_normalization"]["post_v33_identity_reconciliation_append"] = {
    "new_normalized_index_groups": [s["canonical_id"] for s in sources],
    "newly_inspected_method_identities": sorted(NEW_METHOD_IDS),
    "legacy_reinspection_and_omission_reconciliation": sorted(LEGACY_IDS),
    "old_groups_preserved_exactly": len(old_groups), "title_only_alias_merges": 0,
    "normalization_growth_is_not_new_paper_read_count": True}
catalog_paths = {v["path"] for v in new["existing_packets"]}
conclusion_paths = {v["conclusion_file"] for v in new["paper_records"]}
scope_paths = {v["read_scope_reference"]["path"] for v in new["paper_records"]
               if isinstance(v.get("read_scope_reference"), dict)}
new["read_accounting"].update({"state": "PREPARED_FOR_ROOT_ADOPTION", "catalog_entries": counts["catalog"],
    "conclusion_records": counts["records"], "normalized_paper_identifiers": paper_count,
    "software_documentation_identifiers": software_count,
    "unique_catalog_document_paths": len(catalog_paths), "unique_conclusion_source_documents": len(conclusion_paths),
    "unique_referenced_document_paths": len(catalog_paths | conclusion_paths),
    "unique_scope_reference_document_paths": len(scope_paths), "latest_adoption_packets": 1,
    "latest_index_growth": {"records": 5, "new_normalized_paper_identities": 5,
                           "newly_inspected_scoped_method_identities": 2, "legacy_identity_reinspections": 3,
                           "legacy_normalization_omissions_reconciled": 3, "scoped_primary_method_events": 5,
                           "full_primary_reads": 0},
    "latest_packet_first_scoped_method_identity": sorted(NEW_METHOD_IDS),
    "latest_packet_new_scoped_primary_reads": 2, "latest_packet_retained_primary_revisits": 3,
    "latest_packet_full_primary_reads": 0, "latest_packet_scoped_primary_method_events": 5,
    "latest_packet_bounded_author_source_scope_events": 0,
    "integration_pass_primary_method_reads": 0, "integration_pass_full_primary_reads": 0,
    "integration_pass_new_primary_reads": 0, "integration_pass_retained_primary_revisits": 0,
    "integration_pass_metadata_identity_checks": 5,
    "latest_accounting_correction_reference": "literature_memory/index_v34/IDENTITY_AND_SCOPE_ACCOUNTING.json",
    "historical_path_catalog_note": "All136 v33 records,194 catalog entries and92 old normalization groups preserved exactly. "
                                  "Five added normalized groups comprise2 newly inspected scoped-method identities and3 legacy omission reconciliations. "
                                  "Record/identity totals are not full-paper or newly read-paper totals."})
write("LITERATURE_INDEX.json", new)
write("IDENTITY_AND_SCOPE_ACCOUNTING.json", {"schema": "append-only-scoped-method-identity-reconciliation-accounting-v1",
    "UTC": utc, "predecessor": binding(PREVIOUS / "LITERATURE_INDEX.json"), "source_packet": PACKET_NAME,
    "decisions": decisions, "older_conclusion_records_preserved_exactly": 136,
    "older_catalog_entries_preserved_exactly": 194, "older_normalization_groups_preserved_exactly": len(old_groups),
    "added_conclusion_records": 5, "added_normalized_paper_identity_groups": 5,
    "newly_inspected_scoped_method_identity_count": 2, "legacy_identity_reinspection_count": 3,
    "legacy_normalization_omissions_reconciled": 3, "source_packet_scoped_method_events": 5,
    "integration_primary_methods_reread": 0, "integration_primary_retrievals": 0,
    "full_paper_read_increment": 0, "author_code_scope_increment": 0,
    "full_paper_total_certified": False, "normalized_growth_not_new_paper_read_count": True, "effective_totals": counts})
readme = """# Literature memory v34

Prepared for root adoption. Preserves all **136 v33 conclusion records, 194 catalog entries and all 92 prior normalization groups exactly**. Five compact scoped source conclusions are appended with immutable report, scope and excerpt bindings.

**Derived totals: 141 conclusion records, 202 catalog entries, 95 normalized paper identities and two software identities.** The five added normalization groups comprise **two newly inspected scoped-method identities (LDS/CORE)** and **three legacy reinspections/reconciled omissions (BGCN/node copying/adaptive connection sampling)**. This is not five newly read papers. LDS was previously metadata-known/retrieved; retrieval alone is not a certified method read. There are zero full-paper reads and zero primary method rereads/retrievals during integration.

No additional learner novelty or quality gap is asserted. Joint graph/model marginalization, pairing and nonlinear predictive averaging have direct priors. D/X/P is only an established-mechanism diagnostic. Crossing requires **C_m(q_k)** compatible with receiving member m's actual private xlin/features and decoder inputs; a shared upstream encoder does not establish a common C(q). See INTEGRATION_LIMITS.json. The preserved source report is unchanged.

Only compact metadata is added here. Primary HTML, extracted method text and raw discovery payloads are checksum-bound in the source packet and remain untracked for publication. No old verifier was run, no earlier timestamp file was edited, and no canonical ledger or Git push was performed. This index authorizes no learner, experiment or numerical execution.
"""
(OUT / "README.md").write_text(readme)
after = tree_state(PREVIOUS) + tree_state(PACKET)
assert before == after
assert legacy_before == [binding(p) for p in legacy_inputs]
write("PREDECESSOR_AND_PACKET_CUSTODY.json", {"schema": "immutable-index-packet-and-legacy-metadata-custody-v1",
    "all_input_files_unchanged": True, "files": before, "legacy_metadata_files": legacy_before,
    "checked_source_inventory_entries": len(inventory_rows), "old_verifiers_executed": False,
    "earlier_timestamp_files_rewritten": False, "source_report_modified": False,
    "primary_html_and_method_text_hashed_only": True})
write("APPEND_RECEIPT.json", {"schema": "compact-identity-reconciliation-append-receipt-v1", "UTC": utc,
    "state": "PREPARED_FOR_ROOT_ADOPTION", "index": digest(OUT / "LITERATURE_INDEX.json"),
    "predecessor_index": digest(PREVIOUS / "LITERATURE_INDEX.json"),
    "predecessor_manifest": digest(PREVIOUS / "MANIFEST.json"), "source_manifest": digest(PACKET / "MANIFEST.json"),
    "source_report": digest(PACKET / "REPORT.md"), "source_scopes": digest(PACKET / "READ_SCOPES.json"),
    "source_excerpts": digest(PACKET / "PRIMARY_EXCERPTS.json"), "preserved_records": 136,
    "preserved_catalog_entries": 194, "preserved_normalization_groups": len(old_groups),
    "added_scoped_conclusions": 5, "added_normalized_paper_groups": 5,
    "newly_inspected_method_identities": 2, "legacy_identity_reinspections": 3,
    "legacy_normalization_omissions_reconciled": 3, "effective_totals": counts,
    "full_primary_read_increment": 0, "integration_primary_method_reads_or_retrievals": 0,
    "old_verifiers_rerun": False, "old_timestamp_files_changed": False,
    "canonical_ledger_status_changed": False, "new_learner_gap_or_experiments_authorized": False})
verification = {"schema": "fresh-read-only-metadata-integration-verification-v1", "UTC": utc, "status": "PASS",
    "verified_predecessor_manifest_sha256": digest(PREVIOUS / "MANIFEST.json")["sha256"],
    "verified_source_manifest_sha256": digest(PACKET / "MANIFEST.json")["sha256"],
    "older_records_preserved_exactly": 136, "older_catalog_entries_preserved_exactly": 194,
    "all92_old_normalization_groups_preserved_exactly": True, "added_records": 5,
    "new_normalized_groups": 5, "two_newly_inspected_and_three_legacy_reinspected": True,
    "all_new_identifiers_and_titles_checked": True, "bound_excerpt_metadata_records": len(excerpt_metadata),
    "effective_totals": counts, "source_and_predecessor_and_legacy_metadata_unchanged": True,
    "old_verifiers_executed": False, "primary_methods_reread_or_retrieved": 0, "full_paper_reads": 0,
    "canonical_ledger_or_Git_push": False,
    "verification_kind": "This fresh stdlib metadata builder verified hashes, identity metadata and append-only preservation. "
                         "No primary method text was inspected, no project numerical code imported/executed, and no old verifier/timestamp was touched."}
write("VERIFICATION.json", verification)
# Verify the serialized successor, including byte-equivalent old JSON values.
written = load(OUT / "LITERATURE_INDEX.json")
assert written["paper_records"][:136] == old["paper_records"]
assert written["existing_packets"][:194] == old["existing_packets"]
assert written["canonical_identifier_normalization"]["groups"][:len(old_groups)] == old_groups
assert before == tree_state(PREVIOUS) + tree_state(PACKET)
files = [{"path": str(p.relative_to(OUT)), **digest(p)} for p in sorted(OUT.rglob("*"))
         if p.is_file() and p.name != "MANIFEST.json"]
write("MANIFEST.json", {"schema": "immutable-compact-index-payload-manifest-v1", "UTC": utc,
    "files": files, "excludes": ["MANIFEST.json"], "input_custody": "PREDECESSOR_AND_PACKET_CUSTODY.json",
    "immutable_inputs_preserved": True, "root_adoption_pending": True})
print(json.dumps({"status": "PASS", "index_sha256": digest(OUT / "LITERATURE_INDEX.json")["sha256"],
    "manifest_sha256": digest(OUT / "MANIFEST.json")["sha256"], "manifest_payloads": len(files),
    **counts, "added_scoped_conclusions": 5, "newly_inspected_method_identities": 2,
    "legacy_identity_reinspections": 3, "full_reads": 0, "integration_primary_method_reads": 0,
    "old_files_unchanged": True}))

