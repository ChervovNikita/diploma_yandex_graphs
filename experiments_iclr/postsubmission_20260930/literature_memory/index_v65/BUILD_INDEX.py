"""Integrate two completed scopes into the full v65 JSON; custody work only.

No searches, primary retrievals, semantic primary rereads or scientific execution.
Preserves predecessor values, snapshots changed current metadata, and measures
compact UTF-8 serialization before writing the index under the decimal 2MB cap.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / "literature_memory/index_v64"
SCOUT = BASE / "graph_shared_backbone_information_failure_scout_20261005_v1"
AUDIT = BASE / "method_screening_criteria_independent_audit_20261005_v1/REPORT.md"
PREV_INDEX_PIN = "943fb0f1a4a33ec78e06c26690d64caad7fd5a0269068fba2134d42dc8b432dd"
PREV_MANIFEST_PIN = "aafe20a2ab3ccc26416aa9d63f1b0146f7384bf36709640649f6bcf69395fe8a"
SCOUT_MANIFEST_PIN = "5db010311755a26deacb37b8f49c6aba6144e835a8a05fca28545c3ddd21d53e"
SCOUT_CONCLUSIONS_PIN = "b9fac1a4b50d998d23a37baa013f34483c8d48387ce0eecb91930ebf86fa131c"
AUDIT_PIN = "f1a6b8edf206f7a8ed3716f793d345aad2455859b6a4ff863c842e36db4a2205"
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


def verify_manifest(folder, pin):
    manifest = ref(folder / "MANIFEST.json")
    assert manifest["sha256"] == pin
    data = json.loads((folder / "MANIFEST.json").read_text())
    rows = data if isinstance(data, list) else data["files"]
    verified = [manifest]
    for row in rows:
        path = folder / row["path"]
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        got = ref(path)
        assert (got["bytes"], got["sha256"]) == (row["bytes"], row["sha256"]), row["path"]
        verified.append(got)
    seal = json.loads((folder / "SEAL.json").read_text())
    assert seal["manifest_sha256"] == pin and seal["payload_files"] == len(rows)
    return verified + [ref(folder / "SEAL.json")]


def main():
    assert {p.name for p in HERE.iterdir()} == {"BUILD_INDEX.py"}
    now = datetime.now(timezone.utc).isoformat()
    predecessor = verify_manifest(PREV, PREV_MANIFEST_PIN)
    scout_files = verify_manifest(SCOUT, SCOUT_MANIFEST_PIN)
    assert ref(PREV / "LITERATURE_INDEX.json")["sha256"] == PREV_INDEX_PIN
    assert ref(SCOUT / "PAPER_CONCLUSIONS.json")["sha256"] == SCOUT_CONCLUSIONS_PIN
    assert ref(AUDIT)["sha256"] == AUDIT_PIN
    previous = json.loads((PREV / "LITERATURE_INDEX.json").read_text())
    old_metrics = metrics(previous)
    assert old_metrics == {k: previous["read_accounting"][k] for k in old_metrics}

    source_inputs = []
    for row in json.loads((SCOUT / "INPUT_BINDINGS.json").read_text())["files"]:
        got = ref(BASE / row["path"])
        assert (got["bytes"], got["sha256"]) == (row["bytes"], row["sha256"])
        source_inputs.append({**got, "unchanged": True})
    spec = ref(BASE / "shared_private_transfer_paired_pilot_preparation_20261005_v2/STUDY_SPEC.json")
    scout_verification = json.loads((SCOUT / "VERIFICATION.json").read_text())
    assert spec["sha256"] == scout_verification["frozen_study_spec_sha256"]

    source = json.loads((SCOUT / "PAPER_CONCLUSIONS.json").read_text())
    candidates = source["records"]
    assert len(candidates) == 2
    assert [r["canonical_id"] for r in candidates] == ["arxiv:2608.09031v1", "arxiv:2310.07430v1"]
    assert source["reading_credit"]["new_bounded_method_scopes"] == 2
    assert source["reading_credit"]["new_full_paper_reads"] == 0
    # Mechanical excerpt-integrity checks; primary text is never displayed or read semantically.
    excerpts = json.loads((SCOUT / "SCOPE_EXCERPTS.json").read_text())
    receipts = json.loads((SCOUT / "PRIMARY_RETRIEVAL.json").read_text())
    assert len(excerpts) == len(receipts) == len(candidates)
    section_count = container_count = 0
    for pos, row in enumerate(candidates):
        assert receipts[pos]["arxiv_id"] == row["retrieval"]["arxiv_id"]
        assert excerpts[pos]["key"] == receipts[pos]["key"]
        for field in ["url", "retrieved_utc", "status", "final_url", "sha256", "bytes"]:
            assert receipts[pos][field] == row["retrieval"][field]
        sections = excerpts[pos]["selected_complete_subsections"]
        assert len(sections) == len(row["exact_read_scope"])
        for section, scope in zip(sections, row["exact_read_scope"]):
            for field in ["title", "html_id", "paragraph_ids", "equation_ids", "figure_ids"]:
                assert section[field] == scope[field]
            assert hashlib.sha256(section["text"].encode()).hexdigest() == scope["selected_text_sha256"]
            section_count += 1
            container_count += len(scope["paragraph_ids"])
    assert section_count == source["reading_credit"]["complete_subsections"]
    assert container_count == source["reading_credit"]["selected_paragraph_theorem_or_list_item_containers"]

    correction = {"audit_reference": ref(AUDIT),
        "interpretation": "The fixed-complete-input obstruction applies to downstream-only heads. An upstream representation-changing extension may have finite-sample, regularization, estimation or optimization value even if a competent single represents its predictor and an untied learner admits its rule. No source establishes that benefit here; the unsupported successor remains unpromoted. Strong quality comparisons and frozen gates remain unchanged.",
        "prior_source_claims_rewritten": False,
        "no_single_representational_impossibility_requirement": True,
        "no_non_detection_as_equivalence": True}
    save("SOURCE_BINDINGS.json", {"UTC": now, "predecessor_files": predecessor,
        "files": scout_files, "source_manifest_checks": [{"packet": SCOUT.name,
        "manifest_sha256": SCOUT_MANIFEST_PIN, "verified_payload_files": len(scout_files)-2}],
        "source_input_hash_checks": source_inputs, "frozen_spec_integrity_reference": spec,
        "screening_correction": correction, "integrity_checks_only": True,
        "integration_semantic_primary_reads": 0, "integration_primary_retrievals": 0,
        "previously_completed_scoped_primary_method_reads": len(candidates),
        "source_completed_full_paper_reads": 0, "new_raw_source_text_embedded": False,
        "deleted_full_HTML_hashes_are_saved_receipt_claims_not_recomputed_raw_hashes": True})

    index = copy.deepcopy(previous)
    current_metadata = ["schema", "created_UTC", "latest_adoption", "read_accounting",
                        "predecessor_index", "predecessor_index_sha256"]
    index["integration_v65_predecessor_v64_snapshot"] = {
        **{k: copy.deepcopy(previous[k]) for k in current_metadata},
        "index_reference": ref(PREV / "LITERATURE_INDEX.json"),
        "manifest_reference": ref(PREV / "MANIFEST.json"), "seal_reference": ref(PREV / "SEAL.json")}
    index.update(schema="literature-memory-index-v65", created_UTC=now,
        predecessor_index="literature_memory/index_v64/LITERATURE_INDEX.json",
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
                   "exact_read_scope": row["exact_read_scope"]}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert (identity, key) not in keys
        keys.add((identity, key))
        position = len(index["paper_records"])
        scope_ref = {**ref(SCOUT / "PAPER_CONCLUSIONS.json"), "selector": f"records/{pos}/exact_read_scope"}
        retrieval = row["retrieval"]
        index["paper_records"].append({"canonical_id": canonical, "normalized_identifier": identity,
            "conclusion": copy.deepcopy(row),
            "conclusion_file": str((SCOUT / "PAPER_CONCLUSIONS.json").relative_to(BASE)),
            "conclusion_file_sha256": SCOUT_CONCLUSIONS_PIN, "conclusion_source_selector": f"records/{pos}",
            "read_scope_reference": scope_ref, "exact_read_scope": copy.deepcopy(row["exact_read_scope"]),
            "read_status": "Previously completed bounded primary method scope; not a new integration read or full-paper read",
            "source_packet": SCOUT.name, "source_packet_binding_reference": ref(HERE / "SOURCE_BINDINGS.json"),
            "source_extracted_text_reference": {**ref(SCOUT / "SCOPE_EXCERPTS.json"), "selector": str(pos)},
            "source_retrieval_reference": {**ref(SCOUT / "PRIMARY_RETRIEVAL.json"), "selector": str(pos)},
            "source_primary_retrieval_metadata": {"url": retrieval["url"], "sha256": retrieval["sha256"],
                "bytes": retrieval["bytes"], "raw_HTML_retained": False,
                "hash_status": "Saved original retrieval receipt; deleted raw HTML not rehashed in integration"},
            "scope_deduplication_key_sha256": key,
            "scope_deduplication_key_schema": "normalized identity + version + exact saved scope",
            "source_scope_deduplication_key_sha256": row["scope_deduplication_key_sha256"],
            "integration_adoption_interpretation": copy.deepcopy(correction),
            "scoped_method_read": True, "full_paper_read": False, "author_source_read": False,
            "numeric_result_transfer": False, "predictive_adoption": False,
            "global_novelty_clearance": False, "execution_authorized": False,
            "integration_pass_new_primary_reads": 0, "integration_pass_primary_reread": False})
        groups.append({"normalized_identifier": identity, "kind": "paper",
            "raw_canonical_identifiers": [canonical], "explicit_aliases": [], "record_indices": [position]})
        added.append({"canonical_id": identity, "exact_canonical_id": canonical, "record_index": position,
            "new_identity": True, "full_paper_read": False, "source_packet": SCOUT.name,
            "scope_deduplication_key_sha256": key})

    for name in ["PAPER_CONCLUSIONS.json", "SCOPE_EXCERPTS.json", "PRIMARY_RETRIEVAL.json",
                 "REPORT.md", "MANIFEST.json", "INPUT_BINDINGS.json", "VERIFICATION.json"]:
        row = {**ref(SCOUT / name), "kind": "saved_graph_propagation_information_failure_scoped_prior",
            "source_bindings_reference": ref(HERE / "SOURCE_BINDINGS.json"),
            "scope": "Previously completed bounded methods; metadata/discovery custody is not reading credit. Zero integration primary/full reads or predictive, novelty or execution adoption."}
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
    assert totals["conclusion_records"] == old_metrics["conclusion_records"] + len(candidates)
    assert totals["normalized_paper_identifiers"] == old_metrics["normalized_paper_identifiers"] + len(candidates)
    assert totals["software_documentation_identifiers"] == old_metrics["software_documentation_identifiers"]
    account = copy.deepcopy(previous["read_accounting"])
    account.update(totals, state="PROSPECTIVE_SAVED_GRAPH_METHOD_SCOPE_ADOPTION_PENDING_ROOT_REVIEW",
        historical_path_catalog_note="All v64 records, group/catalog prefixes, source events, decision linkages and historical fields retained. Two completed scout scopes appended; zero integration primary/full reads. Changed current metadata is retained in the v64 snapshot.",
        latest_packet_new_scoped_primary_reads=len(candidates), latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=len(candidates),
        latest_packet_new_paper_identity_groups=len(candidates), integration_pass_new_primary_reads=0,
        integration_pass_full_primary_reads=0, integration_pass_primary_method_reads=0,
        integration_pass_author_source_semantic_reads=0, integration_pass_experimental_score_artifact_reads=0,
        integration_pass_retrievals=0)
    index["read_accounting"] = account
    index["latest_adoption"] = {"UTC": now, "status": "PROSPECTIVE_PENDING_ROOT_REVIEW",
        "predecessor": ref(PREV / "LITERATURE_INDEX.json"), "previous_records_preserved": True,
        "added_records": added, "source_completed_new_scoped_method_reads": len(candidates),
        "source_method_scopes_already_root_adopted": True, "source_full_paper_reads": 0,
        "integration_primary_reads": 0, "integration_primary_rereads": 0, "integration_retrievals": 0,
        "source_binding_reference": ref(HERE / "SOURCE_BINDINGS.json"),
        "screening_correction_reference": ref(AUDIT),
        "novelty_or_numeric_or_predictive_or_execution_adoption": False}
    index["graph_propagation_information_failure_prior_limits_v65"] = {
        "HOPPER": "Learnable graph/hop-conditioned extraction and finite-memory structural recurrence are prior single-model operations. Linearity is conditional on coefficients; proofs, stability recipe and results are unqualified in the saved scope.",
        "NBA_GNN": "Directed-edge states exclude reverse-edge messages, with a degree-one exception and incoming/outgoing readout. Proposition1 is a statement-only scope; no proof, implementation or result is adopted.",
        "fixed_input_boundary": "Only a downstream intervention with identical complete common inputs is blocked by an exact information collision. Changing the shared encoder or adding context changes that premise.",
        "adoption_interpretation": correction["interpretation"],
        "source_conclusions_preserved_without_rewrite": True,
        "scope_limits": "Two completed bounded method scopes, not full papers. No global novelty/absence, quality, current-NCN failure or supported successor claim. Frozen controls/gates unchanged."}

    # Measure the complete compatible object before opening the index output.
    encoded = (json.dumps(index, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    assert json.loads(encoded) == index
    assert len(encoded) <= MAX_BYTES, (len(encoded), MAX_BYTES)
    save("SIZE_LIMIT_CHECK.json", {"UTC": now, "index_bytes": len(encoded),
        "decimal_2MB_limit": MAX_BYTES, "within_limit": True,
        "serialization": "Full schema-compatible JSON, compact separators, UTF-8, sorted object keys",
        "measured_before_index_write": True, "JSON_roundtrip_equal": True,
        "history_discarded": False, "new_raw_texts_embedded": False,
        "predecessor_content_including_legacy_embedded_fields_preserved": True, "publisher_modified": False})
    with (HERE / "LITERATURE_INDEX.json").open("xb") as stream:
        stream.write(encoded)
    save("DELTA.json", {"UTC": now, "prospective": True,
        "predecessor_index": ref(PREV / "LITERATURE_INDEX.json"),
        "successor_index": ref(HERE / "LITERATURE_INDEX.json"), "before": old_metrics, "after": totals,
        "metric_deltas": {k: totals[k]-old_metrics[k] for k in totals}, "added_records": added,
        "previously_completed_scoped_reads_adopted": len(candidates),
        "integration_primary_reads": 0, "integration_primary_rereads": 0,
        "integration_retrievals": 0, "new_full_paper_reads": 0,
        "scope_counts": {"complete_subsections": section_count,
                         "saved_paragraph_theorem_or_list_item_containers": container_count,
                         "counts_are_not_full_paper_reads": True},
        "source_claims_rewritten": False, "canonical_status_or_ledger_modified": False})
    save("VERIFICATION.json", {"UTC": now, "status": "PASS",
        "predecessor_index_sha256": PREV_INDEX_PIN, "predecessor_manifest_sha256": PREV_MANIFEST_PIN,
        "predecessor_records_preserved": len(previous["paper_records"]),
        "predecessor_group_order_and_values_preserved": True, "predecessor_catalog_prefix_preserved": True,
        "all_other_predecessor_fields_preserved": True, "changed_current_metadata_snapshotted_exactly": current_metadata,
        "history_source_events_and_decision_linkages_preserved": True,
        "source_manifest_and_seal_verified": True, "source_input_bindings_verified": len(source_inputs),
        "selected_text_hashes_and_exact_scope_metadata_verified": True,
        "retrieval_receipt_fields_verified_without_retrieval": True,
        "deleted_full_HTML_not_independently_rehashed": True,
        "two_distinct_normalized_identity_groups_and_scoped_keys": True,
        "duplicate_adoption_detectable_keys": sorted([list(k) for k in keys]),
        "added_records": added, "recomputed_metrics": totals,
        "source_bindings_reference": ref(HERE / "SOURCE_BINDINGS.json"),
        "index_bytes": len(encoded), "within_decimal_2MB": True, "full_JSON_roundtrip_equal": True,
        "new_raw_texts_not_embedded": True, "prior_uncertified_cumulative_read_flags_preserved": True,
        "integration_primary_reads": 0, "integration_primary_rereads": 0,
        "integration_retrievals": 0, "new_full_paper_reads": 0,
        "no_numeric_predictive_novelty_execution_adoption": True,
        "screening_correction_applied_to_interpretation_not_original_claims": True,
        "canonical_status_or_ledger_modified": False})
    (HERE / "ROOT_ADOPTION_NOTES.md").write_text(
        f"# Prospective literature index v65\n\n"
        f"Recommend root adoption after reviewing this sealed integration packet. It contains {totals['conclusion_records']} conclusion records, {totals['normalized_paper_identifiers']} normalized paper groups and {totals['software_documentation_identifiers']} software groups. All {old_metrics['conclusion_records']} v64 records and every prior group/catalog value, source event, decision linkage and historical field are preserved. Changed current metadata is retained exactly in the v64 snapshot.\n\n"
        "Only the two already-adopted saved scopes are appended: HOPPER (arxiv:2608.09031v1) and NBA-GNN (arxiv:2310.07430v1). The source recorded five complete subsections and 21 paragraph/theorem/list-item containers, zero full-paper reads, no proof audit or author-code/result read. Integration adds zero searches, retrievals, semantic primary rereads, primary reads or full-paper reads. Paper-group counts are not full-paper reading counts; uncertified cumulative flags remain unchanged.\n\n"
        "Original scoped source conclusions are retained verbatim. The adopted screening correction is recorded separately: a fixed-complete-input obstruction excludes a downstream-only remedy under its assumptions; an upstream representation change may still yield finite-sample or optimization benefit. An available capable single or untied implementation is not a logical rejection. The unsupported successor remains unpromoted; no quality, novelty, current-NCN failure or inference claim is adopted. Frozen study and gates remain unchanged.\n\n"
        f"The full schema-compatible UTF-8 JSON was measured before writing: {len(encoded):,} bytes, below the 2,000,000-byte publication cap. Compact whitespace changes no parsed value. Saved excerpt hashes and versioned receipt fields were mechanically checked; deleted full HTML is represented by receipt hashes, not re-certified raw custody. SOURCE_BINDINGS, DELTA and VERIFICATION document the exact append. Canonical ledger/state and publisher were not changed.\n",
        encoding="utf-8")
    rows = [{"path": p.name, "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(HERE.iterdir()) if p.is_file()]
    save("MANIFEST.json", {"UTC": now, "prospective": True, "files": rows})
    save("SEAL.json", {"manifest_sha256": ref(HERE / "MANIFEST.json")["sha256"], "payload_files": len(rows)})
    # Recheck all unchanged source bytes after constructing the packet.
    for row in predecessor + scout_files + source_inputs + [ref(AUDIT), spec]:
        got = ref(BASE / row["path"])
        assert (got["bytes"], got["sha256"]) == (row["bytes"], row["sha256"])
    print(json.dumps({**totals, "index_bytes": len(encoded), "index_sha256": ref(HERE / "LITERATURE_INDEX.json")["sha256"],
        "manifest_sha256": ref(HERE / "MANIFEST.json")["sha256"],
        "seal_sha256": ref(HERE / "SEAL.json")["sha256"], "status": "PROSPECTIVE_READY_FOR_ROOT_REVIEW"}))


if __name__ == "__main__":
    main()
