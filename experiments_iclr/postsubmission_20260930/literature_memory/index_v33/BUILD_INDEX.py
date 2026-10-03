"""Append compact saved LP conclusions; never run or mutate earlier verifiers."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PREVIOUS = ROOT / "literature_memory/index_v32"
PACKET_NAME = "ncnc_private_completion_decoder_closest_priors_20261003_v1"
PACKET = ROOT / PACKET_NAME


def digest(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write(name, obj):
    path = OUT / name
    if path.exists():
        raise RuntimeError("Refusing to overwrite a prepared index artifact: " + str(path))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


# Read only saved metadata. Primary payloads are checksummed, not parsed or read as methods.
utc = datetime.now(timezone.utc).isoformat()
before = [{"path": str(p.relative_to(ROOT)), **digest(p)} for folder in [PREVIOUS, PACKET] for p in sorted(folder.rglob("*")) if p.is_file()]
assert digest(PREVIOUS / "LITERATURE_INDEX.json")["sha256"] == "200e645f46e4dc10a42d52a98f06ffb4699d9cd7b958bb886e749b806627e625"
assert digest(PREVIOUS / "MANIFEST.json")["sha256"] == "241d56e9f661dc8c6a659ca6c8da16ceba767eb3dcb28516ceb135b70d4dcf08"
assert digest(PACKET / "MANIFEST.json")["sha256"] == "ee2b6f4e27ac1a1736f69df8799c1e3cc3cb42cb5b8010c0c29503f21b1ac7bf"
assert digest(PACKET / "VERIFICATION.json")["sha256"] == "4848341eb49f5c5b603f7d66de411ecbda8d26350c2b0b90294c2899b9dd569d"
for folder in [PREVIOUS, PACKET]:
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        assert digest(folder / item["path"]) == {k:item[k] for k in ["bytes","sha256"]}
assert json.loads((PACKET / "VERIFICATION.json").read_text())["status"] == "PASS"

old = json.loads((PREVIOUS / "LITERATURE_INDEX.json").read_text())
new = copy.deepcopy(old)
payload = json.loads((PACKET / "PAPER_CONCLUSIONS.json").read_text())
scope = json.loads((PACKET / "READ_SCOPES.json").read_text())
quoted = json.loads((PACKET / "PRIMARY_EXCERPTS.json").read_text())
assert len(old["paper_records"]) == 133
assert payload["first_scoped_primary_method_identity_count"] == 2
assert payload["retained_targeted_method_revisit_count"] == 1
assert payload["full_paper_read_count"] == 0
assert scope["public_discovery_calls"] == 6 and scope["total_scoped_method_read_events"] == 3
groups = new["canonical_identifier_normalization"]["groups"]
old_groups = old["canonical_identifier_normalization"]["groups"]
group_lookup = {g["normalized_identifier"]:g for g in groups}
title_lookup = {str(r["conclusion"].get("verified_title",r["conclusion"].get("title",""))).casefold() for r in old["paper_records"]}
excerpt_binding = {"path": f"{PACKET_NAME}/PRIMARY_EXCERPTS.json", **digest(PACKET / "PRIMARY_EXCERPTS.json")}
scope_binding = {"path": f"{PACKET_NAME}/READ_SCOPES.json", **digest(PACKET / "READ_SCOPES.json")}
decisions = []


def append_record(source, retained=False):
    normalized = source["canonical_id"].casefold()
    raw_id = source.get("versioned_id", source["canonical_id"])
    key = "pencil" if retained else source["key"]
    previous_indices = list(group_lookup[normalized]["record_indices"]) if retained else []
    if not retained:
        assert normalized not in group_lookup
        assert source["verified_title"].casefold() not in title_lookup
    else:
        assert normalized == "arxiv:2602.01553" and normalized in group_lookup
        assert not set(source["new_selected_ids"]) & set(source["previous_ids_not_repeated"])
    selected_excerpt_indices = [i for i,e in enumerate(quoted["excerpts"]) if e["key"] == key]
    assert len(selected_excerpt_indices) == {"iecnc":3,"egae":9,"pencil":28}[key]
    compact = {k:copy.deepcopy(source[k]) for k in ["key","verified_title","saved_takeaway","read_status","full_paper_read","author_code_read"]}
    compact["canonical_id"] = raw_id
    compact["version"] = source.get("version",source.get("versioned_id"))
    if retained:
        compact["qualification_limits"] = ["Observed token-derived propagation adjacency is not an NCNC missing-edge completion bank", "LRP-PENCIL versus SEAL theorem does not establish NCNC/candidate superiority at fixed cost; LRP absent from experiments", "No author-code, empirical superiority or full-paper certification"]
    else:
        for k in ["operational_overlap","remaining_delta","qualification_limits"]:
            compact[k] = copy.deepcopy(source[k])
    compact["evidence_reference"] = {**excerpt_binding, "source_key":key, "excerpt_indices":selected_excerpt_indices, "text_repeated_in_index":False}
    compact["scope_reference"] = scope_binding
    compact["numeric_results_adopted"] = False
    idx = len(new["paper_records"])
    accounting = {"normalized_identifier":normalized,"new_unique_paper_identity":not retained,"prior_record_indices":previous_indices,"scoped_primary_method_read_event_in_source_packet":True,"full_paper_read":False,"integration_primary_read_or_revisit":False,"effective_read_status":"retained_identity_targeted_primary_scope_extension" if retained else "first_scoped_primary_method_read_for_new_indexed_identity"}
    new["paper_records"].append({"canonical_id":raw_id,"conclusion":compact,"conclusion_file":f"{PACKET_NAME}/PAPER_CONCLUSIONS.json","conclusion_file_sha256":digest(PACKET / "PAPER_CONCLUSIONS.json")["sha256"],"source_record_locator":{"container":"retained_scope_extension" if retained else "new_records","key":source["key"]},"read_scope_reference":{**scope_binding,"paper_canonical_id":raw_id},"primary_evidence_reference":{**excerpt_binding,"source_key":key,"excerpt_indices":selected_excerpt_indices},"integration_read_accounting":accounting})
    if retained:
        g = group_lookup[normalized]
        if raw_id not in g["raw_canonical_identifiers"]:
            g["raw_canonical_identifiers"].append(raw_id)
        g["record_indices"].append(idx)
    else:
        g = {"normalized_identifier":normalized,"kind":"paper","raw_canonical_identifiers":[raw_id],"explicit_aliases":[],"record_indices":[idx]}
        groups.append(g)
        group_lookup[normalized] = g
    decisions.append({"record_index":idx,"key":key,"canonical_id":raw_id,**accounting,"quoted_passages":len(selected_excerpt_indices)})


for source in payload["new_records"]:
    append_record(source)
append_record(payload["retained_scope_extension"], retained=True)
catalog = ["REPORT.md","PAPER_CONCLUSIONS.json","READ_SCOPES.json","PRIMARY_EXCERPTS.json","REUSED_CONCLUSIONS.json","DISCOVERY_DISPOSITIONS.json","DISCOVERY_RECEIPTS.json","PROVENANCE.json","PREVIOUS_SCOPE_LOOKUP.json","INPUT_BINDINGS.json","MANIFEST.json","VERIFICATION.json"]
catalog += sorted(p.name for p in PACKET.glob("PRIMARY_RETRIEVAL*.json"))
for name in catalog:
    new["existing_packets"].append({"path":f"{PACKET_NAME}/{name}","sha256":digest(PACKET/name)["sha256"],"kind":"saved_LP_source_conclusion_or_receipt","sealed_packet_manifest_sha256":digest(PACKET/"MANIFEST.json")["sha256"],"scope":"Two first scoped method identities and one PENCIL retained-identity scope extension; zero full-paper/author-code certifications. Integration makes no primary reread or retrieval."})

assert new["paper_records"][:133] == old["paper_records"]
assert new["existing_packets"][:len(old["existing_packets"])] == old["existing_packets"]
for old_group,new_group in zip(old_groups,groups[:len(old_groups)]):
    if old_group["normalized_identifier"] != "arxiv:2602.01553":
        assert old_group == new_group
    else:
        expected = copy.deepcopy(old_group)
        expected["record_indices"].append(135)
        assert expected == new_group
assert len(groups) == len(old_groups)+2
assert len({g["normalized_identifier"] for g in groups}) == len(groups)
paper_count = sum(g["kind"] == "paper" for g in groups)
software_count = sum(g["kind"] != "paper" for g in groups)
assert (len(new["paper_records"]),paper_count,software_count) == (136,90,2)

catalog_paths = {v["path"] for v in new["existing_packets"]}
conclusion_paths = {v["conclusion_file"] for v in new["paper_records"]}
scope_paths = {v["read_scope_reference"]["path"] for v in new["paper_records"] if isinstance(v.get("read_scope_reference"),dict)}
new["schema"] = "literature-memory-index-v33"
new["created_UTC"] = utc
new["predecessor_index"] = "literature_memory/index_v32/LITERATURE_INDEX.json"
new["predecessor_index_sha256"] = digest(PREVIOUS/"LITERATURE_INDEX.json")["sha256"]
new["latest_adoption"] = {"prepared_UTC":utc,"state":"PREPARED_FOR_ROOT_ADOPTION","previous_append_reference":"literature_memory/index_v32/APPEND_RECEIPT.json","packet_bindings":[{"packet":PACKET_NAME,"manifest_sha256":digest(PACKET/"MANIFEST.json")["sha256"],"report_sha256":digest(PACKET/"REPORT.md")["sha256"],"quoted_evidence_sha256":excerpt_binding["sha256"],"saved_verification_sha256":digest(PACKET/"VERIFICATION.json")["sha256"],"state":"SAVED_SCOPED_CONCLUSIONS_BOUND_WITHOUT_OLD_VERIFIER_RERUN"}],"novelty_quality_or_execution_authorized":False}
new["post_v32_append"] = {"UTC":utc,"packet":PACKET_NAME,"added_conclusion_records":3,"new_unique_paper_identities":2,"retained_PENCIL_scope_extensions":1,"source_packet_scoped_method_events":3,"source_packet_full_primary_reads":0,"integration_primary_reads_or_retrievals":0,"older_index_or_verification_files_modified":False,"canonical_ledger_status_edited":False,"interpretation":"Native independent NCNC already preserves completion/decoder association. Remaining hypothesis is the own-versus-post-clamp-mean weight association under specified shared/factorized paths; no new completion primitive or present quality/cost advantage."}
new["read_accounting"].update({"state":"PREPARED_FOR_ROOT_ADOPTION","catalog_entries":len(new["existing_packets"]),"conclusion_records":136,"normalized_paper_identifiers":90,"software_documentation_identifiers":2,"unique_catalog_document_paths":len(catalog_paths),"unique_conclusion_source_documents":len(conclusion_paths),"unique_referenced_document_paths":len(catalog_paths|conclusion_paths),"unique_scope_reference_document_paths":len(scope_paths),"latest_adoption_packets":1,"latest_index_growth":{"records":3,"new_normalized_paper_identities":2,"scoped_primary_method_events":3,"first_scoped_primary_method_reads":2,"retained_identity_scoped_primary_revisits":1,"full_primary_reads":0},"latest_packet_first_scoped_method_identity":[v["canonical_id"] for v in payload["new_records"]],"latest_packet_new_scoped_primary_reads":2,"latest_packet_retained_primary_revisits":1,"latest_packet_full_primary_reads":0,"latest_packet_scoped_primary_method_events":3,"latest_packet_bounded_author_source_scope_events":0,"integration_pass_primary_method_reads":0,"integration_pass_full_primary_reads":0,"integration_pass_new_primary_reads":0,"integration_pass_retained_primary_revisits":0,"integration_pass_metadata_identity_checks":0,"latest_accounting_correction_reference":"literature_memory/index_v33/IDENTITY_AND_SCOPE_ACCOUNTING.json","historical_path_catalog_note":"All133 v32 conclusions preserved exactly. Three compact LP scope records add two unique DOI identities and a PENCIL revisit. Record/identity totals are not full-paper read totals."})
new["canonical_identifier_normalization"]["post_v32_scope_append"] = {"new_paper_identities":[v["canonical_id"] for v in payload["new_records"]],"retained_identity":"arxiv:2602.01553","retained_group_change":"append record_index135 only; previous fields, IDs and indices unchanged","explicit_alias_merges":0}

write("LITERATURE_INDEX.json",new)
write("IDENTITY_AND_SCOPE_ACCOUNTING.json",{"schema":"compact-scoped-index-append-accounting-v1","UTC":utc,"predecessor":{ "path":"literature_memory/index_v32/LITERATURE_INDEX.json",**digest(PREVIOUS/"LITERATURE_INDEX.json")},"source_packet":PACKET_NAME,"decisions":decisions,"older_conclusion_records_preserved_exactly":133,"older_catalog_entries_preserved_exactly":len(old["existing_packets"]),"older_normalization_groups_preserved_exactly_except_PENCIL_record_index_append":len(old_groups),"added_conclusion_records":3,"added_unique_paper_identities":2,"retained_PENCIL_scope_revisits":1,"scoped_primary_method_events_in_source_packet":3,"full_paper_read_increment":0,"author_code_scope_increment":0,"integration_primary_rereads_or_retrievals":0,"full_paper_total_certified":False,"metadata_only_inaccessible_framework_not_counted":True})
(OUT/"README.md").write_text("# Literature memory v33\n\nPrepared for root adoption. Preserves all **133 v32 conclusion records** and 179 catalog entries exactly. Appends three compact LP scope records, with quoted passages and reading limits referenced by immutable hashes. Earlier v32 files, sealed LP files and their verification timestamps remain unchanged.\n\n**Effective totals: 136 conclusion records, 90 normalized paper identities and 2 software identities.** The increment is **3 scoped conclusion records, 2 new paper identities (IECNC/E-GAE), and 1 PENCIL retained-identity revisit**. No full-paper or author-code certification is added. Integration reread no primary methods, made no public request and reran no old verifier. These totals are not full-paper reading counts.\n\nIECNC's printed entropy/softmax/common-neighbor semantics remain qualification limits. PENCIL's observed-token adjacency reconstruction and LRP/SEAL theorem limits remain distinct from NCNC completion. E-GAE supplies pre-decoder latent fusion ancestry. Native independent NCNC already preserves its own completion/decoder association; the remaining source-qualified hypothesis concerns preserving it under the specified shared/factorized implementation versus a post-clamp pooled-weight twin. No current quality/cost advantage, global novelty or manuscript verdict is implied.\n")
after = [{"path":v["path"],**digest(ROOT/v["path"])} for v in before]
assert before == after
write("PREDECESSOR_AND_PACKET_CUSTODY.json",{"schema":"immutable-index-and-packet-custody-v1","all_files_unchanged":True,"files":before,"earlier_verifiers_executed":False,"earlier_timestamp_files_rewritten":False})
write("APPEND_RECEIPT.json",{"schema":"compact-scoped-index-append-receipt-v1","UTC":utc,"state":"PREPARED_FOR_ROOT_ADOPTION","index":digest(OUT/"LITERATURE_INDEX.json"),"predecessor_index":digest(PREVIOUS/"LITERATURE_INDEX.json"),"predecessor_manifest":digest(PREVIOUS/"MANIFEST.json"),"LP_manifest":digest(PACKET/"MANIFEST.json"),"LP_report":digest(PACKET/"REPORT.md"),"LP_quoted_primary_evidence":digest(PACKET/"PRIMARY_EXCERPTS.json"),"older_conclusion_records_preserved_exactly":133,"added_scope_records":3,"added_unique_papers":2,"PENCIL_revisits":1,"records":136,"paper_identities":90,"software_identities":2,"primary_reads_or_retrievals_in_integration":0,"old_verifiers_rerun":False,"old_index_packet_or_timestamp_files_changed":False,"canonical_ledger_status_changed":False})
write("VERIFICATION.json",{"schema":"read-only-scoped-index-integration-check-v1","UTC":utc,"status":"PASS","older_records_preserved":133,"older_catalog_entries_preserved":179,"older_normalization_groups":90,"only_old_group_extension":"arxiv:2602.01553 record_indices+135","new_records":3,"new_unique_papers":2,"retained_scope_revisits":1,"records":136,"papers":90,"software":2,"sealed_packet_excerpt_count":40,"all_v32_and_LP_files_unchanged":True,"old_verifiers_run":False,"integration_primary_rereads":0,"verification_kind":"Fresh local metadata/hash/identity consistency check; earlier verification receipts only checksum/read, no old code execution or timestamp overwrite."})
files = [{"path":str(f.relative_to(OUT)),**digest(f)} for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='MANIFEST.json']
write("MANIFEST.json",{"schema":"immutable-compact-index-payload-manifest-v1","files":files,"excludes":["MANIFEST.json"],"input_custody":"PREDECESSOR_AND_PACKET_CUSTODY.json","immutable_inputs_preserved":True})
print(json.dumps({"index_sha256":digest(OUT/"LITERATURE_INDEX.json")["sha256"],"manifest_sha256":digest(OUT/"MANIFEST.json")["sha256"],"manifest_files":len(files),"records":136,"papers":90,"software":2,"added_scope_records":3,"added_unique_papers":2,"PENCIL_revisits":1,"old_files_unchanged":True}))
