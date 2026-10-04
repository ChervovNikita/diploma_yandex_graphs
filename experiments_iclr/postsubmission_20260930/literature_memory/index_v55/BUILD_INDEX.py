"""Add one completed method scope and one separately counted incidental locator.

Reads only saved notes, scope metadata and hashes; no primary semantic reread.
Every previous record/catalog/history field is preserved or snapshotted.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / "literature_memory/index_v54"
PACKET = P / "graph_covariance_initialization_closest_prior_20261004_v1"
EXPECTED_MANIFEST = "2d8e407ca9cd43640e6666f6e80927ade8436c1f42433b485d57b3eb80c4536b"
NEW_ID = "arxiv:2602.15747"
LOCATOR_ID = "arxiv:2011.09468"


def ref(path):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sealed(folder, expected=None):
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    seal = json.loads((folder / "SEAL.json").read_text())
    sha = ref(folder / "MANIFEST.json")["sha256"]
    assert sha == seal["manifest_sha256"]
    if expected is not None:
        assert sha == expected
    for item in manifest["files"]:
        observed = ref(folder / item["path"])
        assert observed["bytes"] == item["bytes"]
        assert observed["sha256"] == item["sha256"]
    return manifest


def normalize(identifier):
    return re.sub(r"v\d+$", "", identifier.strip().lower())


def canonical_group(groups, identifier):
    found = []
    for index, group in enumerate(groups):
        aliases = ([group["normalized_identifier"]] +
                   group.get("raw_canonical_identifiers", []) +
                   group.get("explicit_aliases", []))
        if identifier in {normalize(raw) for raw in aliases}:
            found.append(index)
    assert len(found) <= 1
    return found[0] if found else None


def metrics(index):
    catalog = {row["path"] for row in index["existing_packets"]}
    conclusions = {row["conclusion_file"] for row in index["paper_records"]
                   if isinstance(row.get("conclusion_file"), str)}
    scopes = {row["read_scope_reference"]["path"] for row in index["paper_records"]
              if "read_scope_reference" in row}
    aliases = {row["read_scope_file_reference"]["path"] for row in index["paper_records"]
               if "read_scope_file_reference" in row}
    groups = index["canonical_identifier_normalization"]["groups"]
    return dict(conclusion_records=len(index["paper_records"]),
                normalized_paper_identifiers=sum(g["kind"] == "paper" for g in groups),
                software_documentation_identifiers=sum(g["kind"] != "paper" for g in groups),
                catalog_entries=len(index["existing_packets"]),
                unique_catalog_document_paths=len(catalog),
                unique_conclusion_source_documents=len(conclusions),
                unique_referenced_document_paths=len(catalog | conclusions),
                unique_scope_reference_document_paths=len(scopes),
                unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes | aliases))


def dedup(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def main():
    utc = datetime.now(timezone.utc).isoformat()
    sealed(PREV)
    packet_manifest = sealed(PACKET, EXPECTED_MANIFEST)
    previous = json.loads((PREV / "LITERATURE_INDEX.json").read_text())
    conclusions = json.loads((PACKET / "PAPER_CONCLUSIONS.json").read_text())
    scopes = json.loads((PACKET / "READ_SCOPES.json").read_text())
    bindings = json.loads((PACKET / "SOURCE_BINDINGS.json").read_text())
    assert metrics(previous)["conclusion_records"] == 217
    assert metrics(previous)["normalized_paper_identifiers"] == 165
    assert len(conclusions["methods"]) == 1
    paper = conclusions["methods"][0]
    assert normalize(paper["canonical_id"]) == NEW_ID
    assert scopes["accounting"]["genuinely_new_scoped_primary_method_papers"] == 1
    assert scopes["accounting"]["previously_unexamined_retained_abstract_introduction_papers"] == 1
    assert scopes["accounting"]["new_full_paper_reads"] == 0
    for binding in bindings["external_read_only_bindings"]:
        assert ref(P / binding["path"]) == binding

    old_groups = previous["canonical_identifier_normalization"]["groups"]
    assert canonical_group(old_groups, NEW_ID) is None
    assert canonical_group(old_groups, LOCATOR_ID) is None
    # Retain only exact completed/explicitly exposed scopes, not the truncated locator list.
    method_scope = scopes["new_method_scope"]
    exact_indices = sorted(set(method_scope["abstract_and_introduction_inspected_indices"] +
                               method_scope["complete_relevant_method_and_covariance_passage_indices"] +
                               method_scope["other_exact_selected_primary_exposure_indices"]))
    primary_ref = method_scope["primary_html"]
    blocks_ref = ref(PACKET / "primary/2602_15747v1_blocks.json")
    method_payload = dict(canonical_id=NEW_ID, requested_versioned_id="2602.15747v1",
                          primary_sha256=primary_ref["sha256"],
                          blocks_sha256=blocks_ref["sha256"],
                          exact_blocks_zero_based=exact_indices)
    method_key = dedup(method_payload)
    assert not any(r.get("scope_deduplication_key_sha256") == method_key
                   for r in previous["paper_records"])
    assert not any(normalize(r["canonical_id"]) == NEW_ID for r in previous["paper_records"])

    incidental = scopes["retained_first_exposure"]
    assert normalize(incidental["canonical_id"]) == LOCATOR_ID
    unread_path = P / "graph_pooled_curvature_intervention_20261003_v1/READ_SCOPES.json"
    unread_record = json.loads(unread_path.read_text())["retrieved_but_method_unread"]
    assert normalize(unread_record["canonical_id"]) == LOCATOR_ID
    assert unread_record["method_blocks_read"] == []
    assert unread_record["supplies_substantive_claim"] is False
    raw_incidental = ref(P / "graph_pooled_curvature_intervention_20261003_v1/primary/2011.09468v1_html.html")
    locator_payload = dict(canonical_id=LOCATOR_ID, requested_versioned_id="2011.09468v1",
                           primary_sha256=raw_incidental["sha256"],
                           blocks_sha256=incidental["binding"]["sha256"],
                           exact_blocks_zero_based=[3, 5],
                           scope_kind="retained_incidental_abstract_and_intro_only")
    locator_key = dedup(locator_payload)
    old_locators = previous.get("locator_only_exclusions", [])
    assert not any(normalize(r["canonical_id"]) == LOCATOR_ID for r in previous["paper_records"])
    assert not any(normalize(r["canonical_id"]) == LOCATOR_ID for r in old_locators)
    assert not any(r.get("scope_deduplication_key_sha256") == locator_key for r in old_locators)

    index = copy.deepcopy(previous)
    replaced = ("schema", "created_UTC", "latest_adoption", "read_accounting",
                "predecessor_index", "predecessor_index_sha256")
    index["integration_v55_predecessor_v54_snapshot"] = {
        **{key: copy.deepcopy(previous.get(key)) for key in replaced},
        "index_reference": ref(PREV / "LITERATURE_INDEX.json"),
        "manifest_reference": ref(PREV / "MANIFEST.json"),
        "seal_reference": ref(PREV / "SEAL.json"),
        "actual_recomputed_metrics": metrics(previous),
    }
    index.update(schema="literature-memory-index-v55", created_UTC=utc,
                 predecessor_index=str((PREV / "LITERATURE_INDEX.json").relative_to(P)),
                 predecessor_index_sha256=ref(PREV / "LITERATURE_INDEX.json")["sha256"])
    record_index = len(index["paper_records"])
    record = dict(canonical_id=NEW_ID, normalized_identifier=NEW_ID,
                  requested_versioned_id="2602.15747v1", conclusion=copy.deepcopy(paper),
                  conclusion_file=str((PACKET / "PAPER_CONCLUSIONS.json").relative_to(P)),
                  conclusion_file_sha256=ref(PACKET / "PAPER_CONCLUSIONS.json")["sha256"],
                  read_scope_reference={**ref(PACKET / "READ_SCOPES.json"), "selector": "new_method_scope"},
                  exact_read_scope=copy.deepcopy(method_scope), compact_primary_scope=method_payload,
                  primary_payload_reference=primary_ref, primary_blocks_reference=blocks_ref,
                  source_packet=PACKET.name, source_packet_manifest_reference=ref(PACKET / "MANIFEST.json"),
                  source_report_reference=ref(PACKET / "REPORT.md"),
                  scope_deduplication_key_sha256=method_key, full_paper_read=False,
                  proof_audit=False, numeric_result_transfer=False, predictive_adoption=False,
                  global_novelty_clearance=False, execution_authorized=False,
                  integration_pass_primary_reread=False, integration_pass_new_primary_reads=0,
                  integration_read_status="Previously completed sealed scope adopted; integration adds zero reads",
                  genuinely_new_scoped_paper_identity_in_source_packet=True)
    index["paper_records"].append(record)
    index["canonical_identifier_normalization"]["groups"].append(
        dict(normalized_identifier=NEW_ID, kind="paper", raw_canonical_identifiers=[NEW_ID, "arXiv:2602.15747v1"],
             record_indices=[record_index], explicit_aliases=[]))
    locator = dict(canonical_id=LOCATOR_ID, requested_versioned_id="2011.09468v1",
                   verified_title=unread_record["title"], status="Incidental retained abstract and first intro anecdote only",
                   source_reference=ref(PACKET / "READ_SCOPES.json"),
                   exact_read_scope=copy.deepcopy(incidental), compact_primary_scope=locator_payload,
                   scope_deduplication_key_sha256=locator_key,
                   prior_retrieved_unread_scope_reference=ref(unread_path),
                   prior_retrieved_unread_scope_record=copy.deepcopy(unread_record),
                   primary_payload_reference=raw_incidental,
                   not_GNCL=True, correct_GNCL_identifier="arxiv:2011.02952",
                   paper_record_added=False, scoped_primary_method_read=False, full_paper_read=False,
                   supplies_substantive_claim=False, scope_is_not_exact_operation_exclusion=True,
                   new_primary_retrieval=False, integration_primary_reads=0,
                   reason="No method/results/proof scope; preserved for honest exposure accounting without a method-read or canonical-paper-group increment.")
    index["locator_only_exclusions"].append(locator)

    catalog_added = []
    for name in ("REPORT.md", "PAPER_CONCLUSIONS.json", "READ_SCOPES.json", "PRIMARY_PASSAGES.json",
                 "REUSED_CONCLUSIONS.json", "SOURCE_BINDINGS.json", "VERIFICATION.json", "MANIFEST.json", "SEAL.json"):
        entry = {**ref(PACKET / name), "kind": "stored_scoped_graph_covariance_closest_prior",
                 "scope": "One completed method scope plus one separately counted incidental retained abstract/intro; zero integration reads or gain/novelty adoption"}
        assert not any(row["path"] == entry["path"] for row in index["existing_packets"])
        index["existing_packets"].append(entry)
        catalog_added.append(entry["path"])
    added = [dict(canonical_id=NEW_ID, record_index=record_index, new_identity=True,
                  scope_deduplication_key_sha256=method_key, integration_new_primary_reads=0,
                  full_paper_read=False)]
    index["latest_adoption"] = dict(UTC=utc, source=ref(PACKET / "MANIFEST.json"), added_records=added,
                                    added_locator_only_scopes=[dict(canonical_id=LOCATOR_ID, scope_deduplication_key_sha256=locator_key)],
                                    predecessor=ref(PREV / "LITERATURE_INDEX.json"), previous_records_preserved=True,
                                    source_completed_new_scoped_method_reads=1,
                                    source_incidental_retained_abstract_intro_exposures=1,
                                    source_full_paper_reads=0, integration_primary_reads=0,
                                    novelty_or_numeric_or_predictive_or_execution_adoption=False)
    index["post_v55_append"] = copy.deepcopy(index["latest_adoption"])
    index["canonical_identifier_normalization"]["post_v55_scope_append"] = dict(
        UTC=utc, new_paper_identities=[NEW_ID], separate_locator_identity=LOCATOR_ID,
        locator_not_added_as_paper_group=True, added_record_indices=[record_index],
        all_old_groups_preserved_exactly=True, new_title_or_cross_scheme_alias_inferences=0,
        integration_primary_reads=0,
        deduplication_audit_reference=str((HERE / "DEDUPLICATION_AUDIT.json").relative_to(P)))
    index["graph_covariance_initialization_limits_v55"] = dict(
        report_reference=ref(PACKET / "REPORT.md"), scope_reference=ref(PACKET / "READ_SCOPES.json"),
        candidate_protocol_reference=ref(P / "graph_curvature_selector_source_preparation_20261004_v3/PROTOCOL.json"),
        established_ancestry="Shared-backbone end-to-end shallow ensembles; exactly centered head samples; GGN/Laplace covariance initialization; orientation versus total variance; retained graph filters/VJP/bilevel/ensemble-loss ancestry",
        regression_UQ_result_not_classification_accuracy_evidence=True,
        mean_logit_preservation_or_useful_covariance_orientation_not_new_principles=True,
        complete_current_protocol_equivalence_established=False,
        narrow_candidate="Graph-error VJP candidate span at matched finite Jensen gap, chosen via one discarded live full-model coupled-Adam own-CE trial; native continuation",
        random_permuted_fixed_graph_common_warm_and_competent_ensemble_controls_needed=True,
        GGN_Laplace_and_gradient_SVD_controls_proposed_only=True,
        MAP_Laplace_interpretation_at_short_nonstationary_warm_unjustified=True,
        all13_new_numerical_constants_unfrozen=True,
        numerical_qualification_unestablished=True, predictive_gain_unestablished=True,
        methodological_novelty_unestablished=True, no_global_absence_certificate=True,
        Gradient_Starvation_not_GNCL=True, incidental_abstract_method_not_read=True,
        no_execution_or_manuscript_or_publication_adoption=True)

    account = copy.deepcopy(previous["read_accounting"])
    for key in list(account):
        if key.startswith("latest_packet_"):
            old = account[key]
            account[key] = 0 if isinstance(old, int) else [] if isinstance(old, list) else None
    account.update(metrics(index))
    account.update(latest_accounting_correction_reference="literature_memory/index_v55/VERIFICATION.json",
                   latest_index_growth=1, latest_adoption_packets=1,
                   latest_packet_genuinely_new_scoped_paper_identities=1,
                   latest_packet_new_paper_identity_groups=1,
                   latest_packet_new_scoped_primary_reads=1,
                   latest_packet_scoped_primary_method_events=1,
                   latest_packet_previously_completed_scoped_read_adoptions=1,
                   latest_packet_first_scoped_method_identity=[NEW_ID],
                   latest_packet_retained_incidental_abstract_intro_scopes=1,
                   latest_packet_separate_locator_only_scope_adoptions=1,
                   latest_packet_locator_not_counted_as_method_paper_id=LOCATOR_ID,
                   latest_packet_full_primary_reads=0,
                   latest_packet_cached_index_conclusion_records_reused=27,
                   latest_packet_source_scope_accounting_note="One new scoped method identity plus one retained incidental abstract/intro exposure; no full read, no new source/code or scientific execution. Integration reads saved metadata/notes and hashes only.",
                   integration_pass_metadata_identity_checks=2,
                   integration_pass_primary_method_reads=0, integration_pass_new_primary_reads=0,
                   integration_pass_full_primary_reads=0, integration_pass_retained_primary_revisits=0,
                   integration_pass_author_source_semantic_reads=0, integration_pass_project_source_semantic_reads=0,
                   integration_pass_incidental_primary_exposures=0,
                   full_paper_read_total_certified=False, cumulative_scoped_or_full_read_totals_certified=False,
                   historical_path_catalog_note="All217v54 records/catalog/history preserved. One completed method scope appended:218conclusion records,166paper identities+2software. Separate Gradient Starvation abstract/intro locator is not a method paper or full read. Integration adds0reads.",
                   state="ROOT_DELEGATED_ADDITIVE_COVARIANCE_CLOSEST_PRIOR_MEMORY")
    index["read_accounting"] = account

    assert index["paper_records"][:217] == previous["paper_records"]
    assert index["existing_packets"][:len(previous["existing_packets"])] == previous["existing_packets"]
    assert index["canonical_identifier_normalization"]["groups"][:len(old_groups)] == old_groups
    assert index["locator_only_exclusions"][:len(old_locators)] == old_locators
    for key, value in previous["canonical_identifier_normalization"].items():
        if key != "groups":
            assert index["canonical_identifier_normalization"][key] == value
    allowed = set(replaced) | {"paper_records", "existing_packets", "canonical_identifier_normalization", "locator_only_exclusions"}
    for key, value in previous.items():
        if key not in allowed:
            assert index[key] == value
    groups = index["canonical_identifier_normalization"]["groups"]
    assert len({g["normalized_identifier"] for g in groups}) == len(groups)
    current = metrics(index)
    assert current["conclusion_records"] == 218
    assert current["normalized_paper_identifiers"] == 166
    assert current["software_documentation_identifiers"] == 2
    save("DEDUPLICATION_AUDIT.json", dict(
        policy="Exact canonical identity/version/raw bytes/extracted block bytes/explicit inspected scope. Reused ancestry is not appended again; abstract locator remains separate.",
        method=dict(payload=method_payload, scope_deduplication_key_sha256=method_key,
                    prior_identity_or_scope_matches=[], exact_new_scope_already_present=False,
                    new_method_record_index=record_index),
        incidental_locator=dict(payload=locator_payload, scope_deduplication_key_sha256=locator_key,
                                prior_index_identity_or_exact_scope_matches=[], exact_new_scope_already_present=False,
                                prior_same_version_bytes_retrieved_but_method_unread=True,
                                prior_source_scope_reference=ref(unread_path),
                                duplicate_method_read_or_new_retrieval_count_added=0,
                                canonical_paper_group_added=False),
        cached_ancestry_records_reused_without_append=27))
    save("LITERATURE_INDEX.json", index)
    save("VERIFICATION.json", dict(
        UTC=utc, predecessor_index_reference=ref(PREV / "LITERATURE_INDEX.json"),
        predecessor_manifest_and_seal_verified=True, source_manifest_reference=ref(PACKET / "MANIFEST.json"),
        source_manifest_entries_verified=len(packet_manifest["files"]),
        source_external_bindings_verified=len(bindings["external_read_only_bindings"]),
        previous_records_unchanged=217, previous_catalog_prefix_unchanged=True,
        previous_groups_exactly_preserved=True, previous_history_preserved=True,
        previous_locator_prefix_preserved=True, added_records=1,
        new_paper_identity_groups=1, separately_adopted_retained_abstract_intro_locators=1,
        new_record_indices=[record_index], duplicate_canonical_groups=0,
        duplicate_exact_new_scopes=0, metrics=current,
        source_completed_new_method_scope_papers=1, source_completed_incidental_retained_abstract_intro_scopes=1,
        integration_primary_reads=0, full_paper_read_claims=0,
        metric_or_novelty_or_method_or_execution_adoption=False,
        canonical_ledger_status_manuscript_or_publication_modified=False,
        catalog_added_paths=catalog_added))
    rows = [dict(path=f.name, bytes=f.stat().st_size, sha256=ref(f)["sha256"])
            for f in sorted(HERE.iterdir()) if f.is_file()]
    save("MANIFEST.json", dict(schema="literature-memory-manifest-v55", created_UTC=utc, files=rows))
    save("SEAL.json", dict(schema="literature-memory-seal-v55", manifest_sha256=ref(HERE / "MANIFEST.json")["sha256"], immutable=True))
    sealed(HERE)
    print(json.dumps(dict(metrics=current, index_sha256=ref(HERE / "LITERATURE_INDEX.json")["sha256"],
                          manifest_sha256=ref(HERE / "MANIFEST.json")["sha256"], seal_sha256=ref(HERE / "SEAL.json")["sha256"])))


if __name__ == "__main__":
    main()
