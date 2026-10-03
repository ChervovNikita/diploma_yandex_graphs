"""Append two sealed literature scouts to v40 using stdlib metadata only."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import copy
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
PHASE = OUT.parents[1]
PRIOR = PHASE / "literature_memory/index_v40"
STAMP = datetime.now(timezone.utc).isoformat()
SOURCES = [
    ("ncnc_coherent_completion_recent_literature_scout_20261003_v1",
     "5e24dfcd95598f3048d0d05901d9208c047a2bab99cc1df4a28ff3c0acbd6904",
     "31b6144e60de68513260f710800e970387fdc980dd9d86149eec3b18c2567070", 64, 7045111),
    ("graph_conditioned_low_rank_structural_specialization_literature_scout_20261003_v1",
     "8ca0a59a017fc2aae6c578142d2e34ba204b6d49365a6d0018c276b2988fe599",
     "41948f92232fc39042497c554c7a1b2467ee72211565e7c30dbc887dfb50a6c9", 53, 3052469),
]
PRIOR_HASH = "9a451fedaaa11d5533f4255c061f49404e667f52d9c7e94629d6541861d4730e"


def descriptor(path, base=PHASE):
    data = path.read_bytes()
    return {"path": str(path.relative_to(base)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify(path, wanted):
    actual = descriptor(path)
    assert actual["sha256"] == wanted["sha256"] and actual["bytes"] == wanted["bytes"], str(path)
    if path.suffix == ".json":
        json.loads(path.read_text())


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def normalize(cid):
    # New sources supply verified arXiv IDs; preserve all prior aliases/groups.
    assert re.fullmatch(r"arxiv:\d{4}\.\d{4,5}v\d+", cid.lower()), cid
    return re.sub(r"v\d+$", "", cid.lower())


assert not any((OUT / name).exists() for name in ["LITERATURE_INDEX.json", "ADOPTION_RECEIPT.json", "COUNT_RECONCILIATION.json", "VERIFICATION.json", "MANIFEST.json", "SEAL.json"])
assert descriptor(PRIOR / "LITERATURE_INDEX.json")["sha256"] == PRIOR_HASH
old = json.loads((PRIOR / "LITERATURE_INDEX.json").read_text())
prior_manifest = json.loads((PRIOR / "MANIFEST.json").read_text())
prior_seal = json.loads((PRIOR / "SEAL.json").read_text())
for item in prior_manifest["files"]:
    verify(PRIOR / item["path"], item)
for field in ("manifest", "index", "integration", "verification"):
    verify(PHASE / prior_seal[field]["path"], prior_seal[field])
new = copy.deepcopy(old)
groups = new["canonical_identifier_normalization"]["groups"]
old_groups = old["canonical_identifier_normalization"]["groups"]
lookup = {g["normalized_identifier"]: g for g in groups}
assert len(lookup) == len(groups) == 121
source_bindings = []
identity_rows = []
new_ids = []
added_records = []
catalog_names = {
    "MANIFEST.json": "sealed_scout_manifest",
    "SEAL.json": "sealed_scout_bindings",
    "PAPER_CONCLUSIONS.json": "structured_scoped_paper_conclusions",
    "CITATIONS.json": "verified_versioned_source_identity_metadata",
    "READ_SCOPES.json": "exact_primary_read_scope_metadata",
    "READ_ACCOUNTING.json": "honest_scoped_read_accounting",
    "NAVIGATION_SCOPES.json": "incidental_navigation_scope_metadata",
    "REPORT.md": "bounded_source_only_scientific_scout",
    "CONCLUSIONS.json": "scout_scientific_decision_and_limits",
    "DISCOVERY_QUERIES.json": "bounded_discovery_query_metadata",
    "DISCOVERY_DISPOSITIONS.json": "bounded_discovery_dispositions",
    "SEARCH_LIMITS.json": "preserved_search_limits",
    "INPUT_BINDINGS.json": "scout_input_custody",
    "VERIFICATION.json": "source_only_packet_verification",
}
for name, manifest_hash, seal_hash, count, size in SOURCES:
    source = PHASE / name
    assert descriptor(source / "MANIFEST.json")["sha256"] == manifest_hash
    assert descriptor(source / "SEAL.json")["sha256"] == seal_hash
    manifest = json.loads((source / "MANIFEST.json").read_text())
    seal = json.loads((source / "SEAL.json").read_text())
    assert manifest["payload_count"] == len(manifest["payload"]) == count
    assert manifest["payload_bytes"] == sum(d["bytes"] for d in manifest["payload"]) == size
    expected_paths = {d["path"] for d in manifest["payload"]}
    assert expected_paths == {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}}
    for item in manifest["payload"]:
        verify(source / item["path"], item)
    for field in ("manifest", "report", "read_scopes", "verification"):
        verify(source / seal[field]["path"], seal[field])
    assert seal["payload_count"] == count and seal["payload_bytes"] == size
    acct = json.loads((source / "READ_ACCOUNTING.json").read_text())
    first_reads = acct.get("new_paper_identities_with_scoped_primary_method_read", acct.get("new_identities_with_scoped_primary_method_read"))
    assert first_reads == 6 and acct["full_papers_certified"] == 0
    assert acct["proof_audits"] == acct["author_implementations_inspected"] == 0
    assert not acct["model_training_or_numerical_prediction_executed"]
    assert not acct["canonical_or_index_mutation"]
    conclusions = json.loads((source / "PAPER_CONCLUSIONS.json").read_text())["papers"]
    cites_doc = json.loads((source / "CITATIONS.json").read_text())
    citations = {c["key"]: c for c in cites_doc.get("papers", cites_doc.get("new_scoped_sources", []))}
    scopes_doc = json.loads((source / "READ_SCOPES.json").read_text())
    scopes = {s["key"]: s for s in scopes_doc.get("papers", scopes_doc.get("sources", []))}
    assert len(conclusions) == len(citations) == len(scopes) == 6
    packet_binding = {"packet": name, "manifest": descriptor(source / "MANIFEST.json"),
                      "seal": descriptor(source / "SEAL.json"),
                      "verified_payload_count": count, "verified_payload_bytes": size,
                      "first_scoped_method_identity_count": 6, "full_paper_certifications": 0,
                      "read_accounting": descriptor(source / "READ_ACCOUNTING.json"),
                      "decision_reference": descriptor(source / "CONCLUSIONS.json")}
    source_bindings.append(packet_binding)
    for conclusion in conclusions:
        key = conclusion["key"]
        citation = citations[key]
        cid = citation.get("canonical_id", "arXiv:" + citation.get("versioned_id", ""))
        normalized = normalize(cid)
        scope = scopes[key]
        assert scope["full_paper_read"] is False and scope["proof_audit"] is False
        assert scope.get("source_code_read_or_reproduced", scope.get("author_source_inspected")) is False
        if "numeric_results_adopted" in scope:
            assert not scope["numeric_results_adopted"]
        source_primary = citation.get("primary_source", citation.get("primary"))
        verify(source / source_primary["path"], source_primary)
        existing = lookup.get(normalized)
        # Both scout packets independently declared identities absent from v40;
        # validate that fact without title-based or inferred alias matching.
        assert existing is None, (cid, "unexpected duplicate; preserve scope before reconciling")
        aliases = citation.get("DOI_from_discovery")
        assert aliases is None, "No new cross-scheme alias has been verified in these packets."
        position = len(new["paper_records"])
        record = {
            "canonical_id": cid,
            "conclusion_file": str((source / "PAPER_CONCLUSIONS.json").relative_to(PHASE)),
            "conclusion_file_sha256": descriptor(source / "PAPER_CONCLUSIONS.json")["sha256"],
            "conclusion": copy.deepcopy(conclusion), "citation_metadata": copy.deepcopy(citation),
            "exact_read_scope": copy.deepcopy(scope),
            "read_scope_reference": dict(descriptor(source / "READ_SCOPES.json"), paper_canonical_id=cid, source_key=key),
            "read_accounting_reference": descriptor(source / "READ_ACCOUNTING.json"),
            "navigation_scope_reference": descriptor(source / "NAVIGATION_SCOPES.json"),
            "source_packet_manifest_reference": packet_binding["manifest"],
            "source_packet_seal_reference": packet_binding["seal"],
            "read_status": "source_packet_new_scoped_primary_method_not_full_paper",
            "primary_method_read_in_source_packet": True,
            "full_paper_read": False, "proof_audit": False, "author_code_read": False,
            "scientific_results_or_runtime_adopted": False,
            "integration_primary_method_read": False,
            "global_novelty_or_absence_certificate": False,
        }
        new["paper_records"].append(record)
        group = {"kind": "paper", "normalized_identifier": normalized,
                 "raw_canonical_identifiers": [cid], "record_indices": [position], "explicit_aliases": []}
        groups.append(group)
        lookup[normalized] = group
        new_ids.append(normalized)
        added_records.append(position)
        identity_rows.append({"packet": name, "source_key": key, "canonical_id": cid,
                              "normalized_identifier": normalized, "record_index": position,
                              "title": citation["title"], "disposition": "NEW_VERIFIED_ARXIV_IDENTITY_SCOPED_METHOD",
                              "title_only_merge": False, "full_paper_certification": False})
    names = dict(catalog_names)
    if (source / "REUSED_CONCLUSIONS.json").exists():
        names["REUSED_CONCLUSIONS.json"] = "cached_conclusions_reused_without_new_primary_reads"
    for filename, kind in names.items():
        new["existing_packets"].append(dict(descriptor(source / filename), kind=kind,
            scope="Sealed literature metadata adoption; exact source scopes/limits retained. No whole-paper, scientific execution or global novelty clearance."))

paper_ids = {g["normalized_identifier"] for g in groups if g["kind"] == "paper"}
software_ids = {g["normalized_identifier"] for g in groups if g["kind"] != "paper"}
catalog_paths = {r["path"] for r in new["existing_packets"]}
conclusion_paths = {r["conclusion_file"] for r in new["paper_records"] if "conclusion_file" in r}
scope_paths = {r["read_scope_reference"]["path"] for r in new["paper_records"] if "read_scope_reference" in r}
paths = {"catalog_entries": len(new["existing_packets"]), "unique_catalog_document_paths": len(catalog_paths),
         "unique_conclusion_source_documents": len(conclusion_paths),
         "unique_referenced_document_paths": len(catalog_paths | conclusion_paths),
         "unique_scope_reference_document_paths": len(scope_paths)}
counts = {
    "schema": "literature_memory_v41_count_reconciliation_v1", "UTC": STAMP,
    "predecessor": {"conclusion_records": len(old["paper_records"]),
                    "normalized_papers": sum(g["kind"] == "paper" for g in old_groups),
                    "software_documentation_identities": sum(g["kind"] != "paper" for g in old_groups),
                    "normalized_groups": len(old_groups), "explicit_alias_entries": sum(len(g["explicit_aliases"]) for g in old_groups),
                    "catalog_entries": len(old["existing_packets"])},
    "added": {"conclusion_records": len(added_records), "normalized_papers": len(new_ids),
              "duplicate_verified_identities": 0, "cached_reuse_records_added": 0,
              "source_packet_scoped_method_identity_events": 12, "full_paper_certifications": 0,
              "author_source_scope_events": 0, "proof_or_results_audits": 0,
              "integration_primary_method_reads": 0, "new_alias_entries": 0,
              "catalog_entries": len(new["existing_packets"]) - len(old["existing_packets"])},
    "successor": {"conclusion_records": len(new["paper_records"]), "normalized_papers": len(paper_ids),
                  "software_documentation_identities": len(software_ids), "normalized_groups": len(groups),
                  "explicit_alias_entries": sum(len(g["explicit_aliases"]) for g in groups), **paths},
    "new_identity_rows": identity_rows,
    "count_definition": "Conclusion records preserve historical repeated notes/version scopes; normalized identifiers deduplicate verified identity. Scoped method events are not full-paper reads. Path union definition remains v40's catalog/conclusion-file union; scope paths counted separately.",
    "historical_full_paper_totals_certified": False,
}
assert len(new_ids) == len(set(new_ids)) == 12
assert len(new["paper_records"]) == 180 and len(paper_ids) == 131 and len(software_ids) == 2
assert len(groups) == 133
lowrank_source = PHASE / SOURCES[1][0]
decision = json.loads((lowrank_source / "CONCLUSIONS.json").read_text())
assert decision["decision"] == "NO_WARRANTED_NEW_EXPERIMENT_AT_PRESENT"
assert decision["hypotheses_or_paired_comparisons_proposed"] == 0
recent_source = PHASE / SOURCES[0][0]
recent_decision = json.loads((recent_source / "CONCLUSIONS.json").read_text())
assert recent_decision["control_count"] == 3
assert recent_decision["unresolved_metadata_leads_retained"] == old["unresolved_primary_metadata_leads"]
integration = {
    "schema": "append_only_literature_metadata_integration_v41", "UTC": STAMP,
    "status": "ADOPTED_CANONICAL_LITERATURE_SUCCESSOR_ONLY_NO_ROOT_STATUS_LEDGER_EDIT",
    "predecessor": descriptor(PRIOR / "LITERATURE_INDEX.json"),
    "predecessor_manifest": descriptor(PRIOR / "MANIFEST.json"), "predecessor_seal": descriptor(PRIOR / "SEAL.json"),
    "source_packets": source_bindings, "count_reconciliation": counts,
    "source_payloads_verified": 117, "source_payload_bytes_verified": 10097580,
    "scientific_decisions": {
        "coherent_completion": {"source": descriptor(recent_source / "CONCLUSIONS.json"),
            "preserved": copy.deepcopy(recent_decision),
            "adoption_scope": "Literature attribution, limits and three controls retained; no new learner/run accepted or released by metadata integration."},
        "structural_context_lowrank": {"source": descriptor(lowrank_source / "CONCLUSIONS.json"),
            "decision": decision["decision"], "proposed_hypotheses_or_comparisons": 0,
            "preserved": copy.deepcopy(decision), "global_novelty_clearance": False,
            "adoption_scope": "No warranted experiment at present in this bounded scout; not global literature absence, novelty clearance or predictive-quality rejection of all possible methods."},
    },
    "unresolved_leads": {"old_metadata_leads_preserved_exactly": len(old["unresolved_primary_metadata_leads"]),
        "duplicate_reaffirmed_metadata_leads_added": 0,
        "new_paper_method_uncertainties_preserved_in_original_conclusions": True,
        "unselected_discovery_leads_retained_as_metadata_only_in_bound_dispositions": True},
    "identity_policy": "Verified arXiv identity normalization only; version suffix removed for grouping and retained in scope. No title-only/cross-scheme merges. Both GraphLoRA identities remain distinct. Cached prior conclusion reuse adds no records or reads.",
    "new_learner_or_pilot_adopted": False, "execution_authorized": False,
    "scientific_execution": False, "outcome_access": False, "publication": False,
    "canonical_literature_successor_created": True,
    "root_status_ledger_or_manuscript_edits": False,
}
new["schema"] = "literature-memory-index-v41"
new["created_UTC"] = STAMP
new["predecessor_index"] = integration["predecessor"]["path"]
new["predecessor_index_sha256"] = PRIOR_HASH
new["predecessor_v40_read_accounting_snapshot"] = copy.deepcopy(old["read_accounting"])
new["predecessor_v40_latest_adoption_snapshot"] = copy.deepcopy(old["latest_adoption"])
new["post_v40_append"] = integration
new["latest_adoption"] = integration
new["read_accounting"].update(
    state=integration["status"], conclusion_records=180, normalized_paper_identifiers=131,
    software_documentation_identifiers=2, **paths, latest_adoption_packets=2,
    latest_index_growth=integration, latest_packet_first_scoped_method_identity=new_ids,
    latest_packet_new_scoped_primary_reads=12, latest_packet_scoped_primary_method_events=12,
    latest_packet_full_primary_reads=0, latest_packet_retained_primary_revisits=0,
    latest_packet_retained_abstract_only_scope_upgrades=0, latest_packet_bounded_author_source_scope_events=0,
    integration_pass_new_primary_reads=0, integration_pass_primary_method_reads=0,
    integration_pass_full_primary_reads=0, integration_pass_retained_primary_revisits=0,
    integration_pass_metadata_identity_checks=12, cumulative_scoped_or_full_read_totals_certified=False,
    full_paper_read_total_certified=False,
    historical_path_catalog_note="All168 v40 records,121 normalized groups,11 explicit alias entries and all prior scopes/unresolved leads preserved. Two sealed scouts add12 distinct scoped method identities, zero whole-paper certifications. Structural low-rank no-experiment decision is bounded, not global novelty clearance.",
)
changed = {"schema", "created_UTC", "predecessor_index", "predecessor_index_sha256", "latest_adoption", "read_accounting", "paper_records", "existing_packets", "canonical_identifier_normalization"}
unchanged_keys = [k for k in old if k not in changed]
checks = {
    "old_records_prefix_exact": new["paper_records"][:len(old["paper_records"])] == old["paper_records"],
    "old_groups_prefix_exact": groups[:len(old_groups)] == old_groups,
    "old_catalog_prefix_exact": new["existing_packets"][:len(old["existing_packets"])] == old["existing_packets"],
    "old_unresolved_leads_exact": new["unresolved_primary_metadata_leads"] == old["unresolved_primary_metadata_leads"],
    "old_locator_only_exclusions_exact": new["locator_only_exclusions"] == old["locator_only_exclusions"],
    "normalization_non_group_fields_exact": {k:v for k,v in new["canonical_identifier_normalization"].items() if k != "groups"} == {k:v for k,v in old["canonical_identifier_normalization"].items() if k != "groups"},
    "unchanged_prior_top_level_keys_exact": all(new[k] == old[k] for k in unchanged_keys),
    "predecessor_accounting_snapshot_exact": new["predecessor_v40_read_accounting_snapshot"] == old["read_accounting"],
    "predecessor_adoption_snapshot_exact": new["predecessor_v40_latest_adoption_snapshot"] == old["latest_adoption"],
    "normalized_groups_unique": len(groups) == len({g["normalized_identifier"] for g in groups}),
    "all_new_source_conclusions_and_scopes_preserved_exactly": True,
    "all_new_records_nonfull_noncode_nonproof_nonresults": all(not new["paper_records"][i]["full_paper_read"] and not new["paper_records"][i]["author_code_read"] and not new["paper_records"][i]["proof_audit"] and not new["paper_records"][i]["scientific_results_or_runtime_adopted"] for i in added_records),
}
assert all(checks.values())
write("LITERATURE_INDEX.json", new)
write("COUNT_RECONCILIATION.json", counts)
write("INTEGRATION.json", integration)
receipt = {"schema": "canonical_literature_v41_compact_adoption_receipt_v1", "UTC": STAMP,
    "status": integration["status"], "index": descriptor(OUT / "LITERATURE_INDEX.json"),
    "predecessor": integration["predecessor"], "source_packets": source_bindings,
    "counts": {"prior_records":168,"added_records":12,"records":180,"prior_normalized_papers":119,"added_normalized_papers":12,"normalized_papers":131,"software_identities":2,"preserved_alias_entries":11,"new_full_paper_certifications":0,"integration_primary_reads":0,**paths},
    "count_reconciliation": descriptor(OUT / "COUNT_RECONCILIATION.json"),
    "integration": descriptor(OUT / "INTEGRATION.json"),
    "source_payloads_verified":117,"source_payload_bytes_verified":10097580,
    "structural_lowrank_decision":"NO_WARRANTED_NEW_EXPERIMENT_AT_PRESENT",
    "coherent_completion_controls_preserved":3,
    "global_novelty_clearance":False,"scientific_execution":False,"outcome_access":False,
    "new_learner_or_pilot_adopted":False,"execution_authorized":False,"publication":False,
    "root_status_ledger_or_manuscript_edits":False,
    "predecessor_and_source_packets_unchanged":True,
}
write("ADOPTION_RECEIPT.json", receipt)
source = (OUT / "BUILD_INDEX.py").read_text()
ast.parse(source, filename=str(OUT / "BUILD_INDEX.py"))
compile(source, str(OUT / "BUILD_INDEX.py"), "exec")
verification = {"schema":"literature_v41_stdlib_metadata_verification_v1","UTC":STAMP,
    "status":"PASS_METADATA_CUSTODY_IDENTITY_COUNTS_AND_EXACT_PRESERVATION",
    "preservation_checks":checks,"unchanged_prior_top_level_keys":unchanged_keys,
    "source_payloads_verified":117,"source_payload_bytes_verified":10097580,
    "source_primary_semantics_reread":False,"integration_primary_reads":0,
    "new_record_indices":added_records,"new_normalized_ids":new_ids,
    "builder_AST_compile_without_scientific_imports":True,
    "predecessor_SHA256_unchanged":descriptor(PRIOR / "LITERATURE_INDEX.json")["sha256"],
    "scientific_execution":False,"outcome_access":False,"publication":False,
    "canonical_creation_scope":"New index_v41 only; no overwrite or root status/ledger/manuscript edit.",
}
write("VERIFICATION.json", verification)
files = ["BUILD_INDEX.py","LITERATURE_INDEX.json","COUNT_RECONCILIATION.json","INTEGRATION.json","ADOPTION_RECEIPT.json","VERIFICATION.json"]
for name in files:
    if name.endswith(".json"):
        json.loads((OUT/name).read_text())
write("MANIFEST.json", {"schema":"sealed_canonical_literature_index_v41_manifest_v1","UTC":STAMP,
    "status":"SEALED_CANONICAL_LITERATURE_SUCCESSOR_ONLY",
    "payload":[descriptor(OUT/name,OUT) for name in files],"payload_count":len(files),
    "payload_bytes":sum((OUT/name).stat().st_size for name in files),
    "source_inputs":[integration["predecessor"],integration["predecessor_manifest"],integration["predecessor_seal"]]+[d for s in source_bindings for d in (s["manifest"],s["seal"])],
    "scientific_execution":False,"outcome_access":False,"publication":False,
    "canonical_creation_scope":"index_v41 only; root status/ledger untouched."})
write("SEAL.json", {"schema":"sealed_canonical_literature_index_v41_seal_v1","UTC":STAMP,
    "status":"SEALED_CANONICAL_LITERATURE_SUCCESSOR_ONLY",
    "manifest":descriptor(OUT/"MANIFEST.json"),"index":descriptor(OUT/"LITERATURE_INDEX.json"),
    "adoption_receipt":descriptor(OUT/"ADOPTION_RECEIPT.json"),"count_reconciliation":descriptor(OUT/"COUNT_RECONCILIATION.json"),
    "verification":descriptor(OUT/"VERIFICATION.json"),
    "records":180,"normalized_papers":131,"software_identities":2,
    "new_scoped_method_identities":12,"new_full_paper_certifications":0,
    "structural_lowrank_decision":"NO_WARRANTED_NEW_EXPERIMENT_AT_PRESENT",
    "global_novelty_clearance":False,"scientific_execution":False,"outcome_access":False,
    "publication":False,"root_status_ledger_or_manuscript_edits":False})
assert descriptor(PRIOR / "LITERATURE_INDEX.json")["sha256"] == PRIOR_HASH
for path in OUT.iterdir():
    if path.is_file():
        path.chmod(0o444)
OUT.chmod(0o555)
print(json.dumps({"index":descriptor(OUT/"LITERATURE_INDEX.json"),"receipt":descriptor(OUT/"ADOPTION_RECEIPT.json"),"counts":descriptor(OUT/"COUNT_RECONCILIATION.json"),"manifest":descriptor(OUT/"MANIFEST.json"),"seal":descriptor(OUT/"SEAL.json"),"totals":counts["successor"]},indent=2))
