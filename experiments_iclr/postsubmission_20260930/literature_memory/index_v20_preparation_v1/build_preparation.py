"""Prepare an unadopted index with conclusion JSON and byte hashes only.

Do not parse or display retained/new paper bodies. Writes are confined to this
preparation folder; adoption and active-index changes belong to the parent.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

P = Path(__file__).resolve().parent
BASE = P.parents[1]
OLD_PATH = "literature_memory/index_v19/LITERATURE_INDEX.json"
OLD_SHA = "831d760fb3bbe12fb88203a51b251398fb50eb3263938bf117c604ac4a84d109"
SPECS = [
    {
        "packet": "graph_error_specialization_gap_search_v1",
        "manifest_sha256": "5ebb479bbce602bc28c2ff8f5c921c3bdf879e96dbb8c6b32f5b42b9b57c6588",
        "conclusion_sha256": "9ca590c1e408cf9f539cd52a6252e61ded3e6fa6bc57f00e78388c55ca26a682",
        "records": 2,
        "proposed_catalog": [
            ("PAPER_CONCLUSIONS.json", "structured_paper_conclusions", "Two scoped primary method conclusions; no full read or retained-primary reread."),
            ("REPORT.md", "existing_packet_report", "SEA/sigmaN-Ens nearest-prior gap analysis; no new learner, superiority or execution admission."),
            ("READ_SCOPES.json", "bounded_primary_read_scopes", "Exact versioned scoped reads, limits and no full-paper certification."),
            ("REFRESH_ANALYSIS.md", "source_bound_mathematical_analysis", "Divergent route geometry, joint projection and finite pooled-quality safeguards; no new primary identifiers from this analysis."),
            ("CANDIDATE_SPEC.json", "prospective_pulse_specification", "One conditional fixed-time graph-error pulse, fresh prelude and full additional cost; not implemented or admitted."),
        ],
    },
    {
        "packet": "graph_specific_error_gap_search_v1",
        "manifest_sha256": "7859406d0604b51d0c3dca76774ea2f0c87cc3d3dc7e969db30f35a625fc540a",
        "conclusion_sha256": "ccb9d79d5014a20283909cd7c3b4d76387352e0af27c1260a10cdae1ec076a34",
        "records": 4,
        "proposed_catalog": [
            ("PAPER_CONCLUSIONS.json", "structured_paper_conclusions", "Four new scoped method conclusions; publication/version/access limits preserved."),
            ("REPORT.md", "existing_packet_report", "Graph-specific equivalence risks and bounded mechanism gap; no global absence, novelty, superiority or admission claim."),
            ("READ_SCOPES.json", "bounded_primary_read_scopes", "Exact method pages/blocks, published/preprint scope distinction and locator limits."),
            ("PILOT_KILL_CRITERIA.json", "prospective_pilot_falsifiers", "Topology, finite geometry, acquisition, query/cache boundary, practical quality and whole-cost kill criteria; no new grid or persistent admission."),
        ],
    },
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(name, value):
    (P / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def bound(rel, expected=None):
    data = (BASE / rel).read_bytes()
    digest = sha(data)
    if expected is not None:
        assert digest == expected, f"Immutable binding mismatch: {rel}"
    return {"path": rel, "bytes": len(data), "sha256": digest}


def normalize(raw):
    text = raw.strip()
    arxiv = re.search(r"(?i)arxiv:\s*([^;\s]+)", text)
    if arxiv:
        return "arxiv:" + re.sub(r"v\d+$", "", arxiv.group(1).lower())
    doi = re.search(r"(?i)doi:\s*([^;\s]+)", text)
    if doi:
        return "doi:" + doi.group(1).lower().removeprefix("https://doi.org/")
    return text


inputs = [bound(OLD_PATH, OLD_SHA)]
old = json.loads((BASE / OLD_PATH).read_text())
draft = copy.deepcopy(old)
old_records = old["paper_records"]
groups = copy.deepcopy(old["canonical_identifier_normalization"]["groups"])
for group in groups:
    for index in group["record_indices"]:
        assert normalize(old_records[index]["canonical_id"]) == group["normalized_identifier"]

prior_refs = {}
for row in old["existing_packets"]:
    if row["path"] in prior_refs:
        assert prior_refs[row["path"]] == row["sha256"]
    prior_refs[row["path"]] = row["sha256"]
for row in old_records:
    if row["conclusion_file"] in prior_refs:
        assert prior_refs[row["conclusion_file"]] == row["conclusion_file_sha256"]
    prior_refs[row["conclusion_file"]] = row["conclusion_file_sha256"]
prior_ref_bindings = [bound(rel, digest) for rel, digest in sorted(prior_refs.items())]
assert len(prior_refs) == 100

additions = []
pending = []
payload_bindings = []
catalog_added = []
for spec in SPECS:
    packet = spec["packet"]
    manifest_binding = bound(f"{packet}/MANIFEST.json", spec["manifest_sha256"])
    inputs.append(manifest_binding)
    manifest = json.loads((BASE / packet / "MANIFEST.json").read_text())
    verified_payload = []
    for row in manifest["payload"]:
        b = bound(f"{packet}/{row['path']}", row["sha256"])
        assert b["bytes"] == row["bytes"]
        verified_payload.append(b)
    payload_bindings.extend(verified_payload)
    by_name = {row["path"].removeprefix(packet + "/"): row for row in verified_payload}
    assert by_name["PAPER_CONCLUSIONS.json"]["sha256"] == spec["conclusion_sha256"]
    conclusions = json.loads((BASE / packet / "PAPER_CONCLUSIONS.json").read_text())
    records = conclusions["paper_records"]
    assert len(records) == spec["records"]
    read_accounting = conclusions["read_accounting"]
    assert read_accounting["new_scoped_primary_method_reads"] == spec["records"]
    assert read_accounting["new_full_primary_reads"] == 0
    assert read_accounting["retained_primary_rereads"] == 0
    for row in records:
        assert row["full_read"] is False
        additions.append({
            "canonical_id": row["canonical_id"],
            "conclusion_file": f"{packet}/PAPER_CONCLUSIONS.json",
            "conclusion_file_sha256": spec["conclusion_sha256"],
            "conclusion": copy.deepcopy(row),
            "read_scope_reference": {
                "path": f"{packet}/READ_SCOPES.json",
                "sha256": by_name["READ_SCOPES.json"]["sha256"],
                "paper_canonical_id": row["canonical_id"],
            },
        })
    for name, kind, scope in spec["proposed_catalog"]:
        item = {
            "path": f"{packet}/{name}", "sha256": by_name[name]["sha256"],
            "kind": kind, "sealed_packet_manifest_sha256": spec["manifest_sha256"], "scope": scope,
        }
        if name == "PAPER_CONCLUSIONS.json":
            item["records"] = spec["records"]
        catalog_added.append(item)
    seal_binding = None
    if (BASE / packet / "SEAL.json").is_file():
        seal_binding = bound(f"{packet}/SEAL.json")
        seal = json.loads((BASE / packet / "SEAL.json").read_text())
        assert seal["manifest_sha256"] == spec["manifest_sha256"]
        assert seal["payload_count"] == len(verified_payload)
        inputs.append(seal_binding)
    pending.append({
        "packet": packet, "manifest_sha256": spec["manifest_sha256"],
        "seal_sha256": seal_binding["sha256"] if seal_binding else None,
        "payload_files_verified": len(verified_payload),
        "report_sha256": by_name["REPORT.md"]["sha256"],
        "conclusion_file_sha256": spec["conclusion_sha256"],
        "scope_file_sha256": by_name["READ_SCOPES.json"]["sha256"],
        "proposed_paper_conclusions": len(records), "earlier_scoped_method_reads": len(records),
        "earlier_full_primary_reads": 0, "earlier_retained_primary_rereads": 0,
        "state": "VERIFIED_PENDING_PARENT_REVIEW_AND_ADOPTION",
        "verification": "Manifest and every payload byte hashed; paper bodies not parsed/read in preparation.",
    })

assert len(additions) == 6 and len(catalog_added) == 9
draft["paper_records"] = old_records + additions
draft["existing_packets"] = old["existing_packets"] + catalog_added
assert draft["paper_records"][:len(old_records)] == old_records
assert draft["existing_packets"][:len(old["existing_packets"])] == old["existing_packets"]

# Preserve prior groups exactly; add by normalized identifier and explicit alias.
lookup = {}
for i, group in enumerate(groups):
    for ident in [group["normalized_identifier"]] + group["explicit_aliases"]:
        lookup[normalize(ident)] = i
new_group_ids = []
for offset, wrapper in enumerate(additions, len(old_records)):
    raw = wrapper["canonical_id"]
    normalized = normalize(raw)
    record = wrapper["conclusion"]
    aliases = []
    explicit_doi = record.get("canonical_doi") or record.get("doi")
    if explicit_doi:
        aliases.append("doi:" + explicit_doi.lower().removeprefix("https://doi.org/"))
    matching = {lookup[x] for x in [normalized] + aliases if x in lookup}
    assert len(matching) <= 1, f"Cross-group identity conflict: {raw}"
    if matching:
        i = matching.pop()
        group = groups[i]
        group["record_indices"].append(offset)
        if raw not in group["raw_canonical_identifiers"]:
            group["raw_canonical_identifiers"].append(raw)
        group["explicit_aliases"] = list(dict.fromkeys(group["explicit_aliases"] + aliases))
    else:
        i = len(groups)
        group = {"normalized_identifier": normalized, "kind": "paper", "raw_canonical_identifiers": [raw], "explicit_aliases": aliases, "record_indices": [offset]}
        groups.append(group)
        new_group_ids.append(normalized)
    for ident in [normalized] + aliases:
        lookup[ident] = i
assert len(new_group_ids) == 6
assert groups[:len(old["canonical_identifier_normalization"]["groups"])] == old["canonical_identifier_normalization"]["groups"]
draft["canonical_identifier_normalization"] = {
    "rules": copy.deepcopy(old["canonical_identifier_normalization"]["rules"]),
    "groups": groups,
    "pending_alias_note": "New explicit DOI alias retained for PNAS link stacking; canonical NAACL DOI remains its own identity. GRAND read source is exact NeurIPS2020 PDF, not latest-arXiv version certification. No title-only identity merging.",
}

catalog_paths = {row["path"] for row in draft["existing_packets"]}
conclusion_paths = {row["conclusion_file"] for row in draft["paper_records"]}
union_paths = catalog_paths | conclusion_paths
scope_refs = {}
for row in draft["paper_records"]:
    ref = row.get("read_scope_reference")
    if ref:
        if ref["path"] in scope_refs:
            assert scope_refs[ref["path"]] == ref["sha256"]
        scope_refs[ref["path"]] = ref["sha256"]
scope_bindings = [bound(rel, digest) for rel, digest in sorted(scope_refs.items())]
counts = {
    "conclusion_records": len(draft["paper_records"]),
    "normalized_paper_identifiers": sum(g["kind"] == "paper" for g in groups),
    "software_documentation_identifiers": sum(g["kind"] == "software_documentation" for g in groups),
    "unique_conclusion_source_documents": len(conclusion_paths),
    "catalog_entries": len(draft["existing_packets"]),
    "unique_catalog_document_paths": len(catalog_paths),
    "unique_referenced_document_paths": len(union_paths),
    "reference_count_definition": "Same as v19: union of catalog paths and conclusion_file paths; scope-only bindings audited separately.",
    "unique_scope_reference_document_paths": len(scope_refs),
    "cataloged_source_math_conclusions_not_paper_records": old["read_accounting"]["cataloged_source_math_conclusions_not_paper_records"],
    "pending_paper_conclusions": len(additions), "pending_new_normalized_paper_identifiers": len(new_group_ids),
    "pending_packet_earlier_scoped_primary_method_reads": 6,
    "preparation_new_primary_reads": 0, "preparation_full_primary_reads": 0, "preparation_retained_primary_rereads": 0,
    "full_paper_read_total_certified": False, "cumulative_scoped_or_full_read_totals_certified": False,
    "state": "PROPOSED_UNADOPTED_COUNTS_ONLY",
}
assert counts["conclusion_records"] == 98
assert counts["normalized_paper_identifiers"] == 54
assert counts["software_documentation_identifiers"] == 2
assert counts["unique_conclusion_source_documents"] == 28
assert counts["catalog_entries"] == 112 and counts["unique_catalog_document_paths"] == 107 and counts["unique_referenced_document_paths"] == 109
draft["created_UTC"] = datetime.now(timezone.utc).isoformat()
draft["predecessor_index_sha256"] = OLD_SHA
draft["predecessor_index"] = {"path": OLD_PATH, "sha256": OLD_SHA}
draft["preparation_status"] = "UNADOPTED_DRAFT_PARENT_REVIEW_REQUIRED"
draft["pending_integration"] = {
    "packet_bindings": pending,
    "earlier_index_latest_adoption_preserved": copy.deepcopy(old["latest_adoption"]),
    "earlier_index_read_accounting_preserved": copy.deepcopy(old["read_accounting"]),
    "earlier_predecessor_metadata_preserved": {"predecessor_index": old["predecessor_index"], "predecessor_index_sha256": old["predecessor_index_sha256"]},
    "execution_or_index_adoption_authorized_by_preparation": False,
    "decision_limits": "Preserve conditional one-pulse/finite geometry/full acquisition/cost boundaries; no novelty, superiority, persistent schedule or scientific launch inference.",
}
draft["read_accounting"] = {
    **counts,
    "scope": old["read_accounting"]["scope"],
    "latest_adoption_packets": 0, "pending_adoption_packets": 2,
    "integration_pass_new_primary_reads": 0, "integration_pass_full_primary_reads": 0, "integration_pass_retained_primary_revisits": 0,
    "source_math_review_new_primary_identifiers": 0, "source_math_review_new_primary_reads": 0,
}
dump("DRAFT_LITERATURE_INDEX.json", draft)
dump("ADDITION_RECORDS.json", {"schema": "gnnm-literature-index-preparation-additions-v1", "proposed_records": additions, "normalized_new_identifiers": new_group_ids, "state": "UNADOPTED"})
dump("COUNTS.json", counts)
dump("PREPARATION_DIFF.json", {
    "prior_index": {"path": OLD_PATH, "sha256": OLD_SHA},
    "old_records_preserved_exactly": len(old_records), "old_catalog_entries_preserved_exactly": len(old["existing_packets"]),
    "old_normalization_groups_preserved_exactly": len(old["canonical_identifier_normalization"]["groups"]),
    "new_record_indices_0based": list(range(len(old_records), len(draft["paper_records"]))),
    "new_groups": [g for g in groups if g["normalized_identifier"] in new_group_ids],
    "catalog_additions": catalog_added,
    "new_conclusions_exact_source_objects_preserved": True,
    "math_conclusion_count_unchanged": 10,
    "active_index_status_ledger_changed": False,
    "additional_source_availability_note_cataloged": False,
    "parent_action": "Review immutable bindings/scopes/counts, then separately adopt a versioned active index and publication metadata if authorized.",
})
dump("INPUT_BINDINGS.json", {
    "schema": "gnnm-literature-index-preparation-input-bindings-v1",
    "index_and_pending_packet_bindings": inputs,
    "prior_union_reference_bindings": prior_ref_bindings,
    "pending_packet_payload_bindings": payload_bindings,
    "scope_reference_bindings": scope_bindings,
    "hash_only_primary_bytes": "Every sealed payload byte verified without parsing or displaying paper bodies. No primary text reread counted.",
})

for row in inputs + prior_ref_bindings + payload_bindings + scope_bindings:
    current = bound(row["path"], row["sha256"])
    assert current["bytes"] == row["bytes"]
roundtrip = json.loads((P / "DRAFT_LITERATURE_INDEX.json").read_text())
assert roundtrip["paper_records"][:len(old_records)] == old_records
for source, spec in zip([json.loads((BASE / s["packet"] / "PAPER_CONCLUSIONS.json").read_text()) for s in SPECS], SPECS):
    received = [r["conclusion"] for r in additions if r["conclusion_file"] == spec["packet"] + "/PAPER_CONCLUSIONS.json"]
    assert received == source["paper_records"]
dump("VERIFICATION.json", {
    "schema": "gnnm-literature-index-preparation-verification-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
    "immutable_v19_hash_verified": OLD_SHA,
    "prior_union_references_verified": len(prior_ref_bindings),
    "pending_packet_payloads_verified": len(payload_bindings),
    "scope_reference_documents_verified": len(scope_bindings),
    "old_records_catalog_groups_preserved": True,
    "six_new_conclusion_objects_match_exact_source_JSON": True,
    "normalized_ID_dedup_and_explicit_alias_checks": "PASS",
    "distinct_papers": 54, "distinct_software": 2, "conclusion_records": 98,
    "JSON_roundtrip": "PASS", "active_index_status_ledger_mutations": 0,
    "preparation_new_primary_reads": 0, "preparation_full_primary_reads": 0, "preparation_retained_primary_rereads": 0,
    "paper_body_parsing_or_scientific_execution": False,
    "adopted": False, "scientific_execution_authorized": False,
})

payload = []
for file in sorted(P.iterdir()):
    if file.is_file() and file.name not in {"MANIFEST.json", "SEAL.json"}:
        data = file.read_bytes()
        payload.append({"path": file.name, "bytes": len(data), "sha256": sha(data)})
dump("MANIFEST.json", {"schema": "gnnm-literature-index-preparation-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "unadopted_preparation_only": True, "scientific_execution_authorized": False, "payload": payload})
manifest_bytes = (P / "MANIFEST.json").read_bytes()
for row in payload:
    data = (P / row["path"]).read_bytes()
    assert sha(data) == row["sha256"] and len(data) == row["bytes"]
dump("SEAL.json", {"schema": "gnnm-literature-index-preparation-seal-v1", "manifest_sha256": sha(manifest_bytes), "manifest_bytes": len(manifest_bytes), "payload_count": len(payload), "payload_verified": True, "active_index_status_ledger_changed": False, "adopted": False, "scientific_execution_authorized": False})
print(json.dumps({"preparation": str(P), "manifest_sha256": sha(manifest_bytes), "draft_index_sha256": sha((P / "DRAFT_LITERATURE_INDEX.json").read_bytes()), "payloads": len(payload), "proposed_counts": counts}))
