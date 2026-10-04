"""Integrate two completed calibration scopes; no primary method rereading."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / "literature_memory/index_v53"
PACKET = P / "amazon_accuracy_calibration_scout_20261004_v1"
EXPECTED_MANIFEST = "40f61db2b94cdd5c59475a89fd1d51b606afcf1c9792786b0879e4b30b0303ac"
NEW_ID = "arxiv:2609.33764"
RETAINED_ID = "arxiv:2409.05755"


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sealed(folder, expected=None):
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    seal = json.loads((folder / "SEAL.json").read_text())
    actual = ref(folder / "MANIFEST.json")["sha256"]
    assert actual == seal["manifest_sha256"]
    if expected is not None:
        assert actual == expected
    for item in manifest["files"]:
        observed = ref(folder / item["path"])
        assert observed["bytes"] == item["bytes"] and observed["sha256"] == item["sha256"]
    return manifest


def normalize(identifier):
    return re.sub(r"v\d+$", "", identifier.strip().lower())


def canonical_group(groups, identifier):
    found=[]
    for i, group in enumerate(groups):
        aliases=[group["normalized_identifier"]]+group.get("raw_canonical_identifiers",[])+group.get("explicit_aliases",[])
        if identifier in {normalize(raw) for raw in aliases}:
            found.append(i)
    assert len(found) <= 1
    return found[0] if found else None


def metrics(index):
    catalog={row["path"] for row in index["existing_packets"]}
    conclusions={row["conclusion_file"] for row in index["paper_records"] if isinstance(row.get("conclusion_file"),str)}
    scopes={row["read_scope_reference"]["path"] for row in index["paper_records"] if "read_scope_reference" in row}
    aliases={row["read_scope_file_reference"]["path"] for row in index["paper_records"] if "read_scope_file_reference" in row}
    groups=index["canonical_identifier_normalization"]["groups"]
    return dict(conclusion_records=len(index["paper_records"]),normalized_paper_identifiers=sum(g["kind"]=="paper" for g in groups),software_documentation_identifiers=sum(g["kind"]!="paper" for g in groups),catalog_entries=len(index["existing_packets"]),unique_catalog_document_paths=len(catalog),unique_conclusion_source_documents=len(conclusions),unique_referenced_document_paths=len(catalog|conclusions),unique_scope_reference_document_paths=len(scopes),unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes|aliases))


def exact_ranges(value):
    assert isinstance(value,list) and all(isinstance(r,list) and len(r)==2 and all(isinstance(x,int) for x in r) for r in value)
    assert all(1<=a<=b for a,b in value)
    return sorted([list(r) for r in value])


def main():
    utc=datetime.now(timezone.utc).isoformat()
    prior_manifest=sealed(PREV)
    scout_manifest=sealed(PACKET,EXPECTED_MANIFEST)
    previous=json.loads((PREV/"LITERATURE_INDEX.json").read_text())
    source_conclusions=json.loads((PACKET/"PAPER_CONCLUSIONS.json").read_text())["paper_records"]
    source_account=json.loads((PACKET/"READ_ACCOUNTING.json").read_text())
    bindings=json.loads((PACKET/"SOURCE_BINDINGS.json").read_text())
    assert source_account["new_paper_scoped_reads"]==source_account["retained_paper_incremental_scopes"]==1
    assert source_account["full_paper_reads"]==0 and not source_account["project_data_or_scores_or_checkpoints_opened"] and not source_account["scientific_execution"]
    assert {row["canonical_id"] for row in source_conclusions}=={NEW_ID,RETAINED_ID}
    assert metrics(previous)["conclusion_records"]==215
    assert metrics(previous)["normalized_paper_identifiers"]==164
    assert metrics(previous)["software_documentation_identifiers"]==2
    old_groups=previous["canonical_identifier_normalization"]["groups"]
    retained_group_index=canonical_group(old_groups,RETAINED_ID)
    assert retained_group_index is not None and canonical_group(old_groups,NEW_ID) is None
    for binding in bindings:
        observed=ref(P/binding["path"])
        assert observed["bytes"]==binding["bytes"] and observed["sha256"]==binding["sha256"]
        # Only source-byte hashing checks saved excerpts, not a semantic reread.
        if "excerpt_bindings" in binding:
            lines=(P/binding["path"]).read_bytes().splitlines(keepends=True)
            for excerpt in binding["excerpt_bindings"]:
                assert hashlib.sha256(b"".join(lines[excerpt["first_line"]-1:excerpt["last_line"]])).hexdigest()==excerpt["sha256"]

    index=copy.deepcopy(previous)
    replaced=("schema","created_UTC","latest_adoption","read_accounting","predecessor_index","predecessor_index_sha256")
    index["integration_v54_predecessor_v53_snapshot"]={**{key:copy.deepcopy(previous.get(key)) for key in replaced},"index_reference":ref(PREV/"LITERATURE_INDEX.json"),"manifest_reference":ref(PREV/"MANIFEST.json"),"seal_reference":ref(PREV/"SEAL.json"),"actual_recomputed_metrics":metrics(previous),"retained_group_before_extension":copy.deepcopy(old_groups[retained_group_index])}
    index.update(schema="literature-memory-index-v54",created_UTC=utc,predecessor_index=str((PREV/"LITERATURE_INDEX.json").relative_to(P)),predecessor_index_sha256=ref(PREV/"LITERATURE_INDEX.json")["sha256"])
    audit=[]
    added=[]
    for paper in source_conclusions:
        canonical=normalize(paper["canonical_id"])
        versioned=paper.get("versioned_id",canonical.removeprefix("arxiv:")+paper["version"] if "version" in paper else None)
        assert normalize("arxiv:"+versioned)==canonical and paper["full_paper_read"] is False
        source_path=P/paper["source_file"]
        assert ref(source_path)["sha256"]==paper["source_sha256"]
        scope=exact_ranges(paper["exact_read_scope"])
        binding=next(b for b in bindings if b["path"]==paper["source_file"])
        assert exact_ranges(binding["scope"])==scope
        payload=dict(canonical_id=canonical,versioned_id=versioned,source_sha256=paper["source_sha256"],exact_line_ranges=scope)
        dedup=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        prior_matches=[]
        same_byte_prior_scopes=[]
        for i,record in enumerate(previous["paper_records"]):
            if normalize(record["canonical_id"])!=canonical:
                continue
            old=record.get("conclusion",{})
            exact=old.get("exact_read_scope",{})
            prior_matches.append(dict(record_index=i,scope_deduplication_key_sha256=record.get("scope_deduplication_key_sha256")))
            assert record.get("scope_deduplication_key_sha256")!=dedup
            if isinstance(exact,dict) and exact.get("primary_text_sha256")==paper["source_sha256"] and isinstance(exact.get("primary_text_lines"),list):
                prior_ranges=exact_ranges(exact["primary_text_lines"])
                assert prior_ranges!=scope
                same_byte_prior_scopes.append(dict(record_index=i,exact_line_ranges=prior_ranges))
        positions=set()
        for row in same_byte_prior_scopes:
            for a,b in row["exact_line_ranges"]:positions.update(range(a,b+1))
        current_lines={line for a,b in scope for line in range(a,b+1)}
        audit.append(dict(canonical_id=canonical,versioned_id=versioned,source_sha256=paper["source_sha256"],scope_deduplication_key_sha256=dedup,prior_identity_record_matches=prior_matches,prior_scopes_bound_to_same_source_bytes=same_byte_prior_scopes,exact_new_scope_already_present=False,overlapping_saved_line_count=len(current_lines&positions),new_relative_to_same_byte_bound_prior_scope_line_count=len(current_lines-positions),overlap_is_retained_read_not_new_paper_identity=True))
        record_index=len(index["paper_records"])
        record=dict(canonical_id=canonical,normalized_identifier=canonical,requested_versioned_id=versioned,conclusion=copy.deepcopy(paper),conclusion_file=str((PACKET/"PAPER_CONCLUSIONS.json").relative_to(P)),conclusion_file_sha256=ref(PACKET/"PAPER_CONCLUSIONS.json")["sha256"],read_scope_reference={**ref(PACKET/"SOURCE_BINDINGS.json"),"selector_source_path":paper["source_file"]},exact_read_scope=copy.deepcopy(scope),source_packet=PACKET.name,source_packet_manifest_reference=ref(PACKET/"MANIFEST.json"),source_report_reference=ref(PACKET/"REPORT.md"),primary_payload_reference=ref(source_path),scope_deduplication_key_sha256=dedup,full_paper_read=False,proof_audit=False,numeric_result_transfer=False,predictive_adoption=False,global_novelty_clearance=False,execution_authorized=False,integration_pass_primary_reread=False,integration_pass_new_primary_reads=0,integration_read_status="Completed saved scope adopted; integration adds no primary read",genuinely_new_scoped_paper_identity_in_source_packet=paper["new_identity"])
        index["paper_records"].append(record)
        group_index=canonical_group(index["canonical_identifier_normalization"]["groups"],canonical)
        if paper["new_identity"]:
            assert group_index is None and canonical==NEW_ID
            index["canonical_identifier_normalization"]["groups"].append(dict(normalized_identifier=canonical,kind="paper",raw_canonical_identifiers=[canonical,"arxiv:"+versioned],record_indices=[record_index],explicit_aliases=[]))
        else:
            assert group_index==retained_group_index and canonical==RETAINED_ID
            index["canonical_identifier_normalization"]["groups"][group_index]["record_indices"].append(record_index)
        added.append(dict(canonical_id=canonical,record_index=record_index,new_identity=paper["new_identity"],source_scope_kind=paper["read_kind"],scope_deduplication_key_sha256=dedup,integration_new_primary_reads=0,full_paper_read=False))

    catalog_added=[]
    for name in ("REPORT.md","PAPER_CONCLUSIONS.json","READ_ACCOUNTING.json","SOURCE_BINDINGS.json","CALIBRATION_PROPOSAL.json","VERIFICATION.json","MANIFEST.json","SEAL.json"):
        entry={**ref(PACKET/name),"kind":"stored_scoped_amazon_accuracy_calibration","scope":"One completed new-paper scope plus one retained incremental table scope; zero integration reads or scientific adoption"}
        assert not any(row["path"]==entry["path"] for row in index["existing_packets"])
        index["existing_packets"].append(entry);catalog_added.append(entry["path"])
    index["canonical_identifier_normalization"]["post_v54_scope_append"]=dict(UTC=utc,new_paper_identities=[NEW_ID],retained_identity_scope_extensions=[RETAINED_ID],added_record_indices=[215,216],retained_group_record_index_prefix_preserved=True,other_old_groups_preserved_exactly=len(old_groups)-1,new_title_or_cross_scheme_alias_inferences=0,integration_primary_reads=0,deduplication_audit_reference=str((HERE/"DEDUPLICATION_AUDIT.json").relative_to(P)))
    index["latest_adoption"]=dict(UTC=utc,source=ref(PACKET/"MANIFEST.json"),added_records=added,predecessor=ref(PREV/"LITERATURE_INDEX.json"),previous_records_preserved=True,source_completed_new_paper_scopes=1,source_completed_retained_incremental_scopes=1,source_full_paper_reads=0,integration_primary_reads=0,novelty_or_numeric_or_predictive_or_execution_adoption=False)
    index["post_v54_append"]=copy.deepcopy(index["latest_adoption"])
    index["amazon_accuracy_calibration_limits_v54"]=dict(report_reference=ref(PACKET/"REPORT.md"),source_bindings_reference=ref(PACKET/"SOURCE_BINDINGS.json"),proposal_reference=ref(PACKET/"CALIBRATION_PROPOSAL.json"),baseline_context_manifest_reference=ref(P/"amazon_ratings_baseline_context_20261003_v1/MANIFEST.json"),baseline_context_report_reference=ref(P/"amazon_ratings_baseline_context_20261003_v1/REPORT.md"),ordinary_reference_source_manifest_reference=ref(P/"accuracy_first_graph_view_reference_source_preparation_20261004_v2/MANIFEST.json"),native_GNNM_source_manifest_reference=ref(P/"accuracy_first_native_gnnm_reference_source_preparation_20261004_v2/MANIFEST.json"),Polynormer_paper_release_differences_preserved="Release uses two heads and200+2500updates; paper wording differs. No recipe blend.",published_TEST_metrics_not_development_gate=True,earlier_FIT_only_results_not_full_TRAIN_controls=True,optional_SAGE_calibration_not_adopted_or_queued=True,seven_additional_SAGE_fits_only_proposed=True,DGL_runtime_or_operator_equivalence_unqualified=True,SAGE_preprocessing_has_no_added_self_loops=True,Polynormer_preprocessing_adds_native_self_loops=True,requirements_404_preserved=True,new_feature_paper_reconstructed_graph_and_mask_equivalence_unverified=True,new_feature_paper_uses_Macro_F1_selector_and_richer_pretraining_inputs=True,new_feature_repositories_and_data_not_inspected=True,metadata_only_candidate_methods_not_primary_reads=True,no_global_SOTA_or_novelty_or_predictive_gain_clearance=True,scientific_comparison_freeze_or_execution_adopted=False)

    account=copy.deepcopy(previous["read_accounting"])
    # Preserve previous current metadata in the predecessor snapshot; reset this adoption's fields.
    for key in list(account):
        if key.startswith("latest_packet_"):
            account[key]=0 if isinstance(account[key],int) else [] if isinstance(account[key],list) else None
    account.update(metrics(index))
    account.update(latest_accounting_correction_reference="literature_memory/index_v54/VERIFICATION.json",latest_index_growth=2,latest_adoption_packets=1,latest_packet_genuinely_new_scoped_paper_identities=1,latest_packet_new_paper_identity_groups=1,latest_packet_new_scoped_primary_reads=1,latest_packet_previously_completed_scoped_read_adoptions=2,latest_packet_retained_primary_incremental_scopes=1,latest_packet_retained_primary_revisits=1,latest_packet_qualified_scope_extensions=1,latest_packet_first_scoped_method_identity=[NEW_ID],latest_packet_full_primary_reads=0,latest_packet_cached_index_conclusion_records_reused=None,latest_packet_retrieval_failure_records=1,latest_packet_author_source_retrieved_only_files=0,latest_packet_author_code_semantic_file_scopes=None,latest_packet_saved_author_source_scope_count=source_account["new_author_source_scopes_retained_identity"],latest_packet_source_scope_accounting_note="Scout records two newly retrieved module/utils scopes plus retained source reinspection; integration certifies no aggregate reinspection total",integration_pass_metadata_identity_checks=2,integration_pass_primary_method_reads=0,integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,integration_pass_retained_primary_revisits=0,integration_pass_author_source_semantic_reads=0,integration_pass_project_source_semantic_reads=0,full_paper_read_total_certified=False,cumulative_scoped_or_full_read_totals_certified=False,historical_path_catalog_note="All215v53 records/history preserved. Two completed scopes appended: one new paper and one retained incremental table scope.217scoped conclusion records,165paper identities+2software are not full-paper-read totals; integration adds0reads.",state="ROOT_DELEGATED_ADDITIVE_CALIBRATION_SCOPE_MEMORY")
    index["read_accounting"]=account

    assert index["paper_records"][:215]==previous["paper_records"]
    assert index["existing_packets"][:len(previous["existing_packets"])]==previous["existing_packets"]
    for i,old in enumerate(old_groups):
        current=index["canonical_identifier_normalization"]["groups"][i]
        if i==retained_group_index:
            assert current["record_indices"]==old["record_indices"]+[215]
            assert {k:v for k,v in current.items() if k!="record_indices"}=={k:v for k,v in old.items() if k!="record_indices"}
        else:assert current==old
    for key,value in previous["canonical_identifier_normalization"].items():
        if key!="groups":assert index["canonical_identifier_normalization"][key]==value
    allowed=set(replaced)|{"paper_records","existing_packets","canonical_identifier_normalization"}
    for key,value in previous.items():
        if key not in allowed:assert index[key]==value
    norms=[g["normalized_identifier"] for g in index["canonical_identifier_normalization"]["groups"]]
    assert len(norms)==len(set(norms))
    current=metrics(index)
    assert current["conclusion_records"]==217 and current["normalized_paper_identifiers"]==165 and current["software_documentation_identifiers"]==2
    save("DEDUPLICATION_AUDIT.json",dict(policy="Exact canonical/version/source-byte/range identity; no title-only merges. Retained overlap preserved without new-paper count.",scopes=audit))
    save("LITERATURE_INDEX.json",index)
    save("VERIFICATION.json",dict(UTC=utc,predecessor_index_reference=ref(PREV/"LITERATURE_INDEX.json"),predecessor_manifest_and_seal_verified=True,source_manifest_reference=ref(PACKET/"MANIFEST.json"),source_manifest_entries_verified=len(scout_manifest["files"]),source_bindings_verified=len(bindings),previous_records_unchanged=215,previous_catalog_prefix_unchanged=True,previous_history_preserved=True,previous_groups_exactly_preserved_except_append_only_retained_record_index=True,retained_group_before_extension_saved=True,new_paper_identity_groups=1,retained_paper_incremental_table_scopes=1,added_records=2,duplicate_canonical_groups=0,duplicate_exact_new_scopes=0,new_record_indices=[215,216],metrics=current,source_completed_new_paper_scopes=1,source_completed_retained_incremental_scopes=1,integration_primary_reads=0,full_paper_read_claims=0,metric_or_novelty_or_method_or_execution_adoption=False,canonical_ledger_status_or_publication_modified=False,catalog_added_paths=catalog_added))
    rows=[dict(path=f.name,bytes=f.stat().st_size,sha256=ref(f)["sha256"]) for f in sorted(HERE.iterdir()) if f.is_file()]
    save("MANIFEST.json",dict(schema="literature-memory-manifest-v54",created_UTC=utc,files=rows))
    save("SEAL.json",dict(schema="literature-memory-seal-v54",manifest_sha256=ref(HERE/"MANIFEST.json")["sha256"],immutable=True))
    sealed(HERE)
    print(json.dumps(dict(metrics=current,index_sha256=ref(HERE/"LITERATURE_INDEX.json")["sha256"],manifest_sha256=ref(HERE/"MANIFEST.json")["sha256"],seal_sha256=ref(HERE/"SEAL.json")["sha256"])))


if __name__=="__main__":
    main()
