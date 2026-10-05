"""Append two completed saved method scopes; local JSON/byte custody only."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / "literature_memory/index_v65"
SCOUT = BASE / "graph_endpoint_private_meta_precedent_scout_20261005_v1"
PREV_INDEX_PIN = "4b8cd1ed8a6b1dd1b8be705ad35a662d4f346af7de760053bd0ab6e68d5c97e7"
PREV_MANIFEST_PIN = "32eba89a433200978534f14666b90a219a67aa9fa276a2d921dab66e283fcc37"
SCOUT_MANIFEST_PIN = "e5d0a1d00497f65d127de522f716cd9835fd71a220b20e23d0437219f328a1a4"
SCOUT_LEDGER_PIN = "25965ebbf35e41ce8b33aadefc4ef0a594935aaac988af7c3fa5d92a45251da8"
MAX_BYTES = 2_000_000


def ref(path):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE) and not path.is_symlink()
    raw = path.read_bytes()
    return {"path": str(path.relative_to(BASE)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def save(name, data):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def normalized(raw):
    return {re.sub(r"v\d+$", "", part.strip().lower()) for part in raw.split(";")}


def metrics(index):
    catalog = {r["path"] for r in index["existing_packets"]}
    conclusions = {r["conclusion_file"] for r in index["paper_records"]
                   if isinstance(r.get("conclusion_file"), str)}
    scopes = {r["read_scope_reference"]["path"] for r in index["paper_records"]
              if "read_scope_reference" in r}
    legacy = {r["read_scope_file_reference"]["path"] for r in index["paper_records"]
              if "read_scope_file_reference" in r}
    groups = index["canonical_identifier_normalization"]["groups"]
    return {"conclusion_records": len(index["paper_records"]),
            "normalized_paper_identifiers": sum(g["kind"] == "paper" for g in groups),
            "software_documentation_identifiers": sum(g["kind"] != "paper" for g in groups),
            "catalog_entries": len(index["existing_packets"]),
            "unique_catalog_document_paths": len(catalog),
            "unique_conclusion_source_documents": len(conclusions),
            "unique_referenced_document_paths": len(catalog | conclusions),
            "unique_scope_reference_document_paths": len(scopes),
            "unique_scope_reference_document_paths_including_legacy_field_alias": len(scopes | legacy)}


def verify_manifest(folder, pin, seal_count_required):
    manifest = ref(folder / "MANIFEST.json")
    assert manifest["sha256"] == pin
    rows = json.loads((folder / "MANIFEST.json").read_text())["files"]
    verified = [manifest]
    for row in rows:
        path = folder / row["path"]
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        got = ref(path)
        assert (got["bytes"], got["sha256"]) == (row["bytes"], row["sha256"]), row["path"]
        verified.append(got)
    seal = json.loads((folder / "SEAL.json").read_text())
    assert seal["manifest_sha256"] == pin
    if seal_count_required:
        assert seal["payload_files"] == len(rows)
    else:
        for seal_key, name in [("report_sha256", "REPORT.md"),
                               ("conclusions_sha256", "CONCLUSIONS.json"),
                               ("source_ledger_sha256", "SOURCE_LEDGER.json"),
                               ("primary_scopes_sha256", "PRIMARY_SCOPES.json"),
                               ("comparison_sha256", "COMPARISON.md"),
                               ("verification_sha256", "VERIFICATION.json")]:
            assert ref(folder / name)["sha256"] == seal[seal_key]
    return verified + [ref(folder / "SEAL.json")]


def main():
    assert {p.name for p in HERE.iterdir()} == {"BUILD_INDEX.py"}
    now = datetime.now(timezone.utc).isoformat()
    predecessor = verify_manifest(PREV, PREV_MANIFEST_PIN, True)
    scout_files = verify_manifest(SCOUT, SCOUT_MANIFEST_PIN, False)
    assert ref(PREV / "LITERATURE_INDEX.json")["sha256"] == PREV_INDEX_PIN
    assert ref(SCOUT / "SOURCE_LEDGER.json")["sha256"] == SCOUT_LEDGER_PIN
    previous = json.loads((PREV / "LITERATURE_INDEX.json").read_text())
    before = metrics(previous)
    assert before == {k: previous["read_accounting"][k] for k in before}

    # Hash inert referenced files; do not import source or reread primary text semantically.
    input_checks = []
    for row in json.loads((SCOUT / "CANDIDATE_BINDINGS.json").read_text())["files"]:
        got = ref(BASE / row["path"])
        assert (got["bytes"], got["sha256"]) == (row["bytes"], row["sha256"])
        input_checks.append({**got, "binding_role": "candidate bytes only"})
    reused_checks = []
    for row in json.loads((SCOUT / "REUSED_SCOPE_BINDINGS.json").read_text())["retained"]:
        got = ref(BASE / row["saved_source_file"])
        assert got["sha256"] == row["saved_source_file_sha256"]
        assert row["new_primary_credit"] == 0
        reused_checks.append({**got, "new_primary_credit": 0})
    source = json.loads((SCOUT / "SOURCE_LEDGER.json").read_text())
    candidates = source["new_sources"]
    assert [r["canonical_id"] for r in candidates] == ["arxiv:2107.08765v2", "arxiv:2208.05716v2"]
    assert source["read_accounting"]["new_bounded_primary_method_scopes"] == 2
    assert source["read_accounting"]["new_full_paper_reads"] == 0
    assert source["read_accounting"]["retained_primary_rereads"] == 0
    excerpts = json.loads((SCOUT / "PRIMARY_SCOPES.json").read_text())
    receipts = json.loads((SCOUT / "PRIMARY_RETRIEVAL.json").read_text())
    selected_container_count = 0
    selected_paragraph_count = 0
    for pos, row in enumerate(candidates):
        assert receipts["requests"][pos] == row["retrieval"]
        assert receipts["retrieved_utc"] == row["retrieval_batch_utc"]
        sections = excerpts[row["key"]]
        assert len(sections) == len(row["scope"])
        for section, scope in zip(sections, row["scope"]):
            assert {k: v for k, v in section.items() if k != "text"} == scope
            assert hashlib.sha256(section["text"].encode()).hexdigest() == scope["selected_text_sha256"]
            selected_container_count += 1
            selected_paragraph_count += len(scope["paragraph_ids"])
        assert not row["full_paper_read"] and not row["results_adopted"] and not row["author_code_read"]
    assert selected_container_count == 10 and selected_paragraph_count == 39
    source_conclusions = json.loads((SCOUT / "CONCLUSIONS.json").read_text())
    assert not source_conclusions["global_novelty_clearance"]
    assert not source_conclusions["exact_complete_combination_found_in_inspected_scopes"]

    save("SOURCE_BINDINGS.json", {"UTC": now, "predecessor_files": predecessor,
        "files": scout_files, "source_manifest_checks": [{"packet": SCOUT.name,
        "manifest_sha256": SCOUT_MANIFEST_PIN, "verified_payload_files": len(scout_files)-2}],
        "candidate_input_hash_checks": input_checks, "reused_scope_hash_checks": reused_checks,
        "selected_scope_text_hashes_checked": selected_container_count,
        "integrity_checks_only": True, "integration_semantic_primary_reads": 0,
        "integration_primary_retrievals": 0, "previously_completed_scoped_primary_method_reads": 2,
        "source_completed_full_paper_reads": 0, "H_GRAM_Meta_iKG_new_read_credit": 0,
        "deleted_full_HTML_hashes_are_saved_receipt_claims_not_recomputed_raw_hashes": True})

    index = copy.deepcopy(previous)
    current_metadata = ["schema", "created_UTC", "latest_adoption", "read_accounting",
                        "predecessor_index", "predecessor_index_sha256"]
    index["integration_v66_predecessor_v65_snapshot"] = {
        **{k: copy.deepcopy(previous[k]) for k in current_metadata},
        "index_reference": ref(PREV / "LITERATURE_INDEX.json"),
        "manifest_reference": ref(PREV / "MANIFEST.json"), "seal_reference": ref(PREV / "SEAL.json")}
    index.update(schema="literature-memory-index-v66", created_UTC=now,
        predecessor_index="literature_memory/index_v65/LITERATURE_INDEX.json",
        predecessor_index_sha256=PREV_INDEX_PIN)
    groups = index["canonical_identifier_normalization"]["groups"]
    added = []
    keys = set()
    for pos, row in enumerate(candidates):
        canonical = row["canonical_id"]
        identity = next(iter(normalized(canonical)))
        assert not any(identity in normalized(item) for g in groups for item in
            [g["normalized_identifier"]] + g.get("raw_canonical_identifiers", []) + g.get("explicit_aliases", []))
        assert not any(identity in normalized(r.get("canonical_id", "")) for r in index["paper_records"])
        payload = {"normalized_identifier": identity, "versioned_canonical_id": canonical.lower(),
                   "exact_read_scope": row["scope"]}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert (identity, key) not in keys
        keys.add((identity, key))
        position = len(index["paper_records"])
        retrieval = row["retrieval"]
        index["paper_records"].append({"canonical_id": canonical, "normalized_identifier": identity,
            "conclusion": copy.deepcopy(row),
            "conclusion_file": str((SCOUT / "SOURCE_LEDGER.json").relative_to(BASE)),
            "conclusion_file_sha256": SCOUT_LEDGER_PIN, "conclusion_source_selector": f"new_sources/{pos}",
            "read_scope_reference": {**ref(SCOUT / "SOURCE_LEDGER.json"), "selector": f"new_sources/{pos}/scope"},
            "exact_read_scope": copy.deepcopy(row["scope"]),
            "read_status": "Previously completed bounded primary method scope; not a new integration read or full-paper read",
            "source_packet": SCOUT.name, "source_packet_binding_reference": ref(HERE / "SOURCE_BINDINGS.json"),
            "source_extracted_text_reference": {**ref(SCOUT / "PRIMARY_SCOPES.json"), "selector": row["key"]},
            "source_retrieval_reference": {**ref(SCOUT / "PRIMARY_RETRIEVAL.json"), "selector": f"requests/{pos}"},
            "source_primary_retrieval_metadata": {"url": retrieval["url"], "sha256": retrieval["sha256"],
                "bytes": retrieval["bytes"], "raw_HTML_retained": False,
                "hash_status": "Saved original retrieval receipt; deleted raw HTML not rehashed in integration"},
            "scope_deduplication_key_sha256": key,
            "scope_deduplication_key_schema": "normalized identity + version + exact saved scope",
            "scoped_method_read": True, "full_paper_read": False, "author_source_read": False,
            "numeric_result_transfer": False, "predictive_adoption": False,
            "global_novelty_clearance": False, "execution_authorized": False,
            "integration_pass_new_primary_reads": 0, "integration_pass_primary_reread": False})
        groups.append({"normalized_identifier": identity, "kind": "paper",
            "raw_canonical_identifiers": [canonical], "explicit_aliases": [], "record_indices": [position]})
        added.append({"canonical_id": identity, "exact_canonical_id": canonical, "record_index": position,
            "new_identity": True, "full_paper_read": False, "source_packet": SCOUT.name,
            "scope_deduplication_key_sha256": key})

    for name in ["SOURCE_LEDGER.json", "CONCLUSIONS.json", "PRIMARY_SCOPES.json", "PRIMARY_RETRIEVAL.json",
                 "REPORT.md", "COMPARISON.md", "DEDUPLICATION.json", "REUSED_SCOPE_BINDINGS.json",
                 "CANDIDATE_BINDINGS.json", "MANIFEST.json", "SEAL.json", "VERIFICATION.json"]:
        row = {**ref(SCOUT / name), "kind": "saved_graph_endpoint_private_update_scoped_prior",
            "source_bindings_reference": ref(HERE / "SOURCE_BINDINGS.json"),
            "scope": "Two previously completed bounded methods only; all reused/discovery/metadata material earns zero new reading credit. No novelty, numeric, predictive or execution adoption."}
        assert not any(r["path"] == row["path"] for r in index["existing_packets"])
        index["existing_packets"].append(row)

    assert index["paper_records"][:len(previous["paper_records"])] == previous["paper_records"]
    assert index["existing_packets"][:len(previous["existing_packets"])] == previous["existing_packets"]
    assert groups[:len(previous["canonical_identifier_normalization"]["groups"])] == previous["canonical_identifier_normalization"]["groups"]
    assert {k: v for k, v in index["canonical_identifier_normalization"].items() if k != "groups"} == {
        k: v for k, v in previous["canonical_identifier_normalization"].items() if k != "groups"}
    for k in previous:
        if k not in current_metadata + ["paper_records", "existing_packets", "canonical_identifier_normalization"]:
            assert index[k] == previous[k], k
    totals = metrics(index)
    assert totals["conclusion_records"] == before["conclusion_records"] + 2
    assert totals["normalized_paper_identifiers"] == before["normalized_paper_identifiers"] + 2
    assert totals["software_documentation_identifiers"] == before["software_documentation_identifiers"]
    account = copy.deepcopy(previous["read_accounting"])
    account.update(totals, state="PROSPECTIVE_SAVED_GRAPH_METHOD_SCOPE_ADOPTION_PENDING_ROOT_REVIEW",
        historical_path_catalog_note="All 243 v65 records, group/catalog prefixes, source events, decision linkages and historical/summary fields retained. Only AUX-TS and TMAG completed scopes appended; zero integration primary/full reads. Changed current metadata retained exactly in the v65 snapshot.",
        latest_packet_new_scoped_primary_reads=2, latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=2, latest_packet_new_paper_identity_groups=2,
        integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0, integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0, integration_pass_retrievals=0)
    index["read_accounting"] = account
    index["latest_adoption"] = {"UTC": now, "status": "PROSPECTIVE_PENDING_ROOT_REVIEW",
        "predecessor": ref(PREV / "LITERATURE_INDEX.json"), "previous_records_preserved": True,
        "added_records": added, "source_completed_new_scoped_method_reads": 2, "source_full_paper_reads": 0,
        "integration_primary_reads": 0, "integration_primary_rereads": 0, "integration_retrievals": 0,
        "H_GRAM_Meta_iKG_new_read_credit": 0, "source_binding_reference": ref(HERE / "SOURCE_BINDINGS.json"),
        "novelty_or_numeric_or_predictive_or_execution_adoption": False}
    index["graph_endpoint_private_update_prior_limits_v66"] = {
        "source_conclusions_reference": ref(SCOUT / "CONCLUSIONS.json"),
        "AUX_TS": copy.deepcopy(candidates[0]["findings"]),
        "TMAG": copy.deepcopy(candidates[1]["findings"]),
        "TMAG_unresolved_details": copy.deepcopy(candidates[1]["unresolved_details"]),
        "source_conclusions_preserved_without_rewrite": True,
        "allowed_statement": source_conclusions["allowed_statement"],
        "interpretation_limits": copy.deepcopy(source_conclusions["interpretation_limits"]),
        "scope_limits": "Two bounded saved method scopes, zero full-paper reads; reused H-GRAM/Meta-iKG and all metadata leads earn zero new credit. No global absence/novelty clearance or utility/execution verdict."}

    encoded = (json.dumps(index, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    assert json.loads(encoded) == index and len(encoded) <= MAX_BYTES
    save("SIZE_LIMIT_CHECK.json", {"UTC": now, "index_bytes": len(encoded),
        "decimal_2MB_limit": MAX_BYTES, "within_limit": True,
        "serialization": "Full schema-compatible compact UTF-8 JSON, sorted object keys",
        "measured_before_index_write": True, "JSON_roundtrip_equal": True,
        "history_discarded": False, "new_raw_texts_embedded": False,
        "predecessor_content_including_legacy_embedded_fields_preserved": True, "publisher_modified": False})
    with (HERE / "LITERATURE_INDEX.json").open("xb") as stream:
        stream.write(encoded)
    save("DELTA.json", {"UTC": now, "prospective": True,
        "predecessor_index": ref(PREV / "LITERATURE_INDEX.json"), "successor_index": ref(HERE / "LITERATURE_INDEX.json"),
        "before": before, "after": totals, "metric_deltas": {k: totals[k]-before[k] for k in totals},
        "added_records": added, "previously_completed_scoped_reads_adopted": 2,
        "integration_primary_reads": 0, "integration_primary_rereads": 0, "integration_retrievals": 0,
        "new_full_paper_reads": 0, "H_GRAM_Meta_iKG_new_read_credit": 0,
        "scope_counts": {"selected_HTML_containers": selected_container_count,
        "paragraph_containers": selected_paragraph_count, "counts_are_not_full_paper_reads": True},
        "source_claims_rewritten": False, "canonical_status_or_ledger_modified": False})
    save("VERIFICATION.json", {"UTC": now, "status": "PASS",
        "predecessor_index_sha256": PREV_INDEX_PIN, "predecessor_manifest_sha256": PREV_MANIFEST_PIN,
        "predecessor_records_preserved": len(previous["paper_records"]),
        "predecessor_group_order_and_values_preserved": True, "predecessor_catalog_prefix_preserved": True,
        "all_other_predecessor_fields_preserved": True, "changed_current_metadata_snapshotted_exactly": current_metadata,
        "history_source_events_and_decision_linkages_preserved": True, "source_manifest_and_seal_verified": True,
        "candidate_input_bindings_verified": len(input_checks), "reused_scope_bindings_verified": len(reused_checks),
        "selected_text_hashes_and_exact_scope_metadata_verified": True,
        "retrieval_receipt_fields_verified_without_retrieval": True, "deleted_full_HTML_not_independently_rehashed": True,
        "two_distinct_normalized_identity_groups_and_scoped_keys": True,
        "duplicate_adoption_detectable_keys": sorted([list(k) for k in keys]), "added_records": added,
        "recomputed_metrics": totals, "source_bindings_reference": ref(HERE / "SOURCE_BINDINGS.json"),
        "index_bytes": len(encoded), "within_decimal_2MB": True, "full_JSON_roundtrip_equal": True,
        "new_raw_texts_not_embedded": True, "prior_uncertified_cumulative_read_flags_preserved": True,
        "integration_primary_reads": 0, "integration_primary_rereads": 0, "integration_retrievals": 0,
        "new_full_paper_reads": 0, "H_GRAM_Meta_iKG_new_read_credit": 0,
        "no_numeric_predictive_novelty_execution_adoption": True, "canonical_status_or_ledger_modified": False})
    with (HERE / "ROOT_ADOPTION_NOTES.md").open("x", encoding="utf-8") as stream:
        stream.write(
            f"# Prospective literature index v66\n\n"
            f"Prepared for root review. The compact successor contains {totals['conclusion_records']} conclusion records, "
            f"{totals['normalized_paper_identifiers']} normalized paper groups and {totals['software_documentation_identifiers']} software groups. "
            "All 243 v65 records, prior group/catalog prefixes, source events, decision linkages and existing summary/history fields are preserved. "
            "The six changed current metadata fields are retained exactly in `integration_v66_predecessor_v65_snapshot`.\n\n"
            "Only AUX-TS (`arxiv:2107.08765v2`) and TMAG (`arxiv:2208.05716v2`) are appended, at record indices 243 and 244. "
            "Their original saved source rows, exact headings/paragraph/equation/algorithm locators, selected-text hashes, receipt hashes and reading limitations are retained. "
            "No unverified DOI alias or title-based merge is introduced. H-GRAM/Meta-iKG are reused memory with zero new reading credit; all metadata leads stay unresolved.\n\n"
            "The source scout completed two bounded primary method scopes, zero full-paper reads and zero retained primary rereads. "
            "This integration performs zero searches, retrievals, semantic primary rereads, new primary reads, full-paper reads or scientific/numerical work. "
            "Paper-group counts remain distinct from reading totals; the uncertified cumulative-reading flags are preserved.\n\n"
            "AUX-TS strengthens virtual/meta/recomputed graph-update ancestry, but commits on the full supervised batch including the meta fold. "
            "TMAG specifies interaction-set separation and meta-test adaptation, not the exact positive-and-negative endpoint exclusion or persistent served private ensemble. "
            "The new scoped summary preserves these qualifications; no global absence, novelty, utility, acceptance or execution verdict is adopted.\n\n"
            f"The complete JSON is {len(encoded):,} bytes, below the decimal 2,000,000-byte cap, with a parsed roundtrip equality check. "
            "Both predecessor and source manifests/seals, seven candidate byte bindings, all reused-scope bindings, ten selected-text hashes and the two saved retrieval receipts were checked. "
            "Deleted full HTML remains represented by original receipt hashes and was not independently retrieved or rehashed. "
            "SOURCE_BINDINGS, DELTA, VERIFICATION and SIZE_LIMIT_CHECK document the exact append. Canonical status/ledger and publisher are unchanged.\n")
    # Recheck all inputs after serialization, without displaying primary text.
    for row in predecessor + scout_files + input_checks + reused_checks:
        got = ref(BASE / row["path"])
        assert (got["bytes"], got["sha256"]) == (row["bytes"], row["sha256"])
    rows = [{"path": p.name, "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(HERE.iterdir()) if p.is_file()]
    save("MANIFEST.json", {"UTC": now, "prospective": True, "files": rows})
    save("SEAL.json", {"manifest_sha256": ref(HERE / "MANIFEST.json")["sha256"], "payload_files": len(rows)})
    print(json.dumps({**totals, "index_bytes": len(encoded),
        "index_sha256": ref(HERE / "LITERATURE_INDEX.json")["sha256"],
        "manifest_sha256": ref(HERE / "MANIFEST.json")["sha256"],
        "seal_sha256": ref(HERE / "SEAL.json")["sha256"], "status": "PROSPECTIVE_READY_FOR_ROOT_REVIEW"}))


if __name__ == "__main__":
    main()
