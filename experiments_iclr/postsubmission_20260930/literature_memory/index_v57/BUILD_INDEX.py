"""Adopt three saved method scopes; preserve v56 and separate identity reconciliation.

Only saved conclusions, scope metadata and local hashes are inspected. No primary
semantic read, retrieval, experimental artifact or scientific execution is added.
"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / "literature_memory/index_v56"
DIVE = P / "graph_subgraph_disagreement_dive_primary_scope_20261005_v1"
DIRECTED = P / "private_cavity_propagation_method_synthesis_20261005_fresh_v1"
SCOUT = P / "graph_conditioned_prediction_disagreement_primary_scout_20261005_v1"
PINS = {
    DIVE.name: "e0e00542e41e89c1fddc3d7f057cb15a0223206ced181b783b1d6e9e6f7aaf8a",
    DIRECTED.name: "c8b8a06e9c21e15b5ed1a4cc2e22f51fae3abb9ff75ec82fc95487060457da8c",
}


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_manifest(folder, expected=None, require_seal=False):
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    pin = ref(folder / "MANIFEST.json")["sha256"]
    assert expected is None or pin == expected
    seal_path = folder / "SEAL.json"
    if seal_path.exists():
        assert json.loads(seal_path.read_text())["manifest_sha256"] == pin
    else:
        assert not require_seal and expected is not None
    for item in manifest["files"]:
        path = folder / item["path"]
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        observed = ref(path)
        assert (observed["bytes"], observed["sha256"]) == (item["bytes"], item["sha256"])
    return dict(packet=folder.name, manifest_reference=ref(folder / "MANIFEST.json"),
                separate_seal_present=seal_path.exists(),
                seal_reference=ref(seal_path) if seal_path.exists() else None,
                all_manifest_files_verified=True,
                payloads_read_only=all(not (f.stat().st_mode & 0o222)
                                       for f in folder.rglob("*") if f.is_file()))


def tokens(raw):
    return {re.sub(r"v\d+$", "", part.strip().lower())
            for part in raw.split(";") if part.strip()}


def group_match(groups, identifier):
    found = []
    for group in groups:
        aliases = ([group["normalized_identifier"]] +
                   group.get("raw_canonical_identifiers", []) +
                   group.get("explicit_aliases", []))
        if any(identifier in tokens(raw) for raw in aliases):
            found.append(group)
    assert len(found) <= 1
    return found[0] if found else None


def metrics(index):
    catalog = {r["path"] for r in index["existing_packets"]}
    conclusions = {r["conclusion_file"] for r in index["paper_records"]
                   if isinstance(r.get("conclusion_file"), str)}
    scopes = {r["read_scope_reference"]["path"] for r in index["paper_records"]
              if "read_scope_reference" in r}
    legacy = {r["read_scope_file_reference"]["path"] for r in index["paper_records"]
              if "read_scope_file_reference" in r}
    groups = index["canonical_identifier_normalization"]["groups"]
    return dict(conclusion_records=len(index["paper_records"]),
                normalized_paper_identifiers=sum(g["kind"] == "paper" for g in groups),
                software_documentation_identifiers=sum(g["kind"] != "paper" for g in groups),
                catalog_entries=len(index["existing_packets"]),
                unique_catalog_document_paths=len(catalog),
                unique_conclusion_source_documents=len(conclusions),
                unique_referenced_document_paths=len(catalog | conclusions),
                unique_scope_reference_document_paths=len(scopes),
                unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes | legacy))


def main():
    utc = datetime.now(timezone.utc).isoformat()
    verify_manifest(PREV, require_seal=True)
    bindings = [verify_manifest(folder, PINS[folder.name]) for folder in (DIVE, DIRECTED)]
    previous = json.loads((PREV / "LITERATURE_INDEX.json").read_text())
    assert metrics(previous)["conclusion_records"] == 220
    assert metrics(previous)["normalized_paper_identifiers"] == 168
    dive_paper = json.loads((DIVE / "PAPER_CONCLUSIONS.json").read_text())
    dive_scope = json.loads((DIVE / "READ_SCOPES.json").read_text())
    directed_scopes = json.loads((DIRECTED / "READ_SCOPES.json").read_text())
    directed_limits = json.loads((DIRECTED / "CONCLUSION.json").read_text())
    retrievals = json.loads((DIRECTED / "RETRIEVAL.json").read_text())
    abstract = json.loads((SCOUT / "READ_SCOPES.json").read_text())["scopes"][2]
    assert tokens(abstract["canonical_id"]) == {"arxiv:2408.04400"}
    assert abstract["method_blocks_read"] == []
    assert abstract["type"] == "metadata abstract only; retrieved primary HTML unread"
    assert ref(P / abstract["primary"]["path"]) == abstract["primary"]
    assert dive_scope["primary"] == abstract["primary"]
    assert dive_scope["new_method_scopes"] == 1 and dive_scope["full_paper_certifications"] == 0
    assert dive_scope["primary_results_tables_or_analyses_read"] is False
    assert dive_paper["full_read"] is False
    assert directed_scopes["accounting"]["new_primary_method_scopes"] == 2
    assert directed_scopes["accounting"]["full_paper_certifications"] == 0
    assert directed_limits["new_principle_established"] is False
    assert directed_limits["predictive_gain_established"] is False
    assert directed_limits["first_cohort"]["launch_authorized"] is False

    # These paper conclusions are derived only from the saved synthesis at the
    # stated line locators. No primary passage is reopened to create this index.
    descriptions = [
        ("arxiv:1904.01561", 0, 46,
         "Directed bond states exclude the reverse message, retain an initial-edge-state skip, aggregate back to atom states and train end to end; Section 3.6 averages independently initialized and trained D-MPNNs.",
         "Directed-bond reasoning, reverse exclusion and directed-model ensembling are established. Molecular inputs/readout do not certify the proposed citation-node task or restricted shared/private return factors."),
        ("arxiv:1705.08415", 1, 48,
         "Learned graph operators, nonbacktracking line-graph states and node/edge incidence maps; Appendix A.3 relates nonbacktracking inference to I,D,A/Bethe-Hessian operators.",
         "Learned node/edge operator-bank ancestry is established. The exact retrieved v4 title is retained; this scope establishes no candidate predictive utility or first-use clearance for learned scalar backtracking coefficients."),
    ]
    specs = [dict(identifier="arxiv:2408.04400", folder=DIVE, paper=dive_paper,
                  scope=dive_scope, selector=None, conclusion_file="PAPER_CONCLUSIONS.json",
                  conclusion_selector=None, retained_abstract_identity=True)]
    for identifier, selection, line, operation, boundary in descriptions:
        scope = directed_scopes["new_primary_method_scopes"][selection]
        assert tokens(scope["canonical_id"]) == {identifier}
        paper = dict(canonical_id=scope["canonical_id"],
                     verified_title=scope["title_of_exact_retrieved_version"],
                     read_status="Previously completed scoped primary method; not full paper",
                     operation=operation, prior_boundary=boundary,
                     full_paper_read=False, author_source_read=False,
                     numeric_results_adopted=False, predictive_gain_established=False,
                     new_principle_established=False, global_novelty_clearance=False,
                     implementation_executed=False)
        specs.append(dict(identifier=identifier, folder=DIRECTED, paper=paper, scope=scope,
                          selector=f"new_primary_method_scopes/{selection}",
                          conclusion_file="SYNTHESIS.txt", conclusion_selector=dict(line_start=line, line_end=line),
                          retained_abstract_identity=False))

    index = copy.deepcopy(previous)
    replaced = ("schema", "created_UTC", "latest_adoption", "read_accounting",
                "predecessor_index", "predecessor_index_sha256")
    index["integration_v57_predecessor_v56_snapshot"] = {
        **{key: copy.deepcopy(previous.get(key)) for key in replaced},
        "index_reference": ref(PREV / "LITERATURE_INDEX.json"),
        "manifest_reference": ref(PREV / "MANIFEST.json"),
        "seal_reference": ref(PREV / "SEAL.json"),
        "failure_history_reference": ref(PREV / "INITIAL_BUILD_FAILURE.json"),
    }
    index.update(schema="literature-memory-index-v57", created_UTC=utc,
                 predecessor_index=str((PREV / "LITERATURE_INDEX.json").relative_to(P)),
                 predecessor_index_sha256=ref(PREV / "LITERATURE_INDEX.json")["sha256"])
    groups = index["canonical_identifier_normalization"]["groups"]
    added = []
    for spec in specs:
        folder, identifier = spec["folder"], spec["identifier"]
        assert tokens(spec["paper"]["canonical_id"]) == {identifier}
        scope_ref = ref(folder / "READ_SCOPES.json")
        if spec["selector"] is not None:
            scope_ref["selector"] = spec["selector"]
        conclusion_ref = ref(folder / spec["conclusion_file"])
        payload = dict(canonical_id=identifier, conclusion=conclusion_ref,
                       conclusion_selector=spec["conclusion_selector"],
                       exact_scope=spec["scope"], scope_reference=scope_ref)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
        assert not any(r.get("scope_deduplication_key_sha256") == key
                       for r in index["paper_records"])
        group = group_match(groups, identifier)
        old_indices = copy.deepcopy(group["record_indices"]) if group else []
        group_added = group is None
        new_identity = group_added and not spec["retained_abstract_identity"]
        position = len(index["paper_records"])
        record = dict(canonical_id=spec["paper"]["canonical_id"], normalized_identifier=identifier,
                      conclusion=copy.deepcopy(spec["paper"]),
                      conclusion_file=conclusion_ref["path"], conclusion_file_sha256=conclusion_ref["sha256"],
                      read_scope_reference=scope_ref, exact_read_scope=copy.deepcopy(spec["scope"]),
                      scope_deduplication_key_sha256=key, source_packet=folder.name,
                      source_packet_manifest_reference=ref(folder / "MANIFEST.json"),
                      full_paper_read=False, scoped_method_read=True, author_source_read=False,
                      numeric_result_transfer=False, predictive_adoption=False,
                      global_novelty_clearance=False, execution_authorized=False,
                      integration_pass_primary_reread=False, integration_pass_new_primary_reads=0,
                      genuinely_new_paper_identity=new_identity,
                      normalized_identity_group_added=group_added,
                      retained_abstract_identity_scope_upgrade=spec["retained_abstract_identity"])
        if spec["conclusion_selector"] is not None:
            record["conclusion_source_selector"] = spec["conclusion_selector"]
            primary = next(r for r in retrievals if tokens("arxiv:" + r["url"].rsplit("/", 1)[1]) == {identifier})
            primary_ref = ref(P / spec["scope"]["html"])
            assert (primary_ref["bytes"], primary_ref["sha256"]) == (primary["bytes"], primary["sha256"])
            record["primary_payload_reference"] = primary_ref
            record["source_synthesis_limits_reference"] = ref(DIRECTED / "CONCLUSION.json")
        else:
            record["primary_payload_reference"] = copy.deepcopy(dive_scope["primary"])
            record["prior_abstract_scope_reference"] = {**ref(SCOUT / "READ_SCOPES.json"), "selector": "scopes/2"}
            record["prior_abstract_scope"] = copy.deepcopy(abstract)
        index["paper_records"].append(record)
        if group is None:
            group = dict(normalized_identifier=identifier, kind="paper",
                         raw_canonical_identifiers=[], explicit_aliases=[], record_indices=[])
            groups.append(group)
        if spec["paper"]["canonical_id"] not in group["raw_canonical_identifiers"]:
            group["raw_canonical_identifiers"].append(spec["paper"]["canonical_id"])
        group["record_indices"].append(position)
        added.append(dict(canonical_id=identifier, record_index=position,
                          predecessor_record_indices=old_indices, new_identity=new_identity,
                          normalized_identity_group_added=group_added,
                          retained_abstract_identity_scope_upgrade=spec["retained_abstract_identity"],
                          full_paper_read=False, scope_deduplication_key_sha256=key))

    catalog_added = []
    for folder, names in [(DIVE, ["PAPER_CONCLUSIONS.json", "READ_SCOPES.json", "REPORT.md", "CITATIONS.json", "MANIFEST.json"]),
                          (DIRECTED, ["SYNTHESIS.txt", "READ_SCOPES.json", "CONCLUSION.json", "RETRIEVAL.json", "MANIFEST.json"])]:
        for name in names:
            row = {**ref(folder / name), "kind": "saved_scoped_method_memory",
                   "source_manifest_sha256": PINS[folder.name],
                   "scope": "Previously completed bounded method conclusions; zero integration reads or result/novelty/execution adoption"}
            assert not any(r["path"] == row["path"] for r in index["existing_packets"])
            index["existing_packets"].append(row)
            catalog_added.append(row["path"])

    groups.sort(key=lambda g: g["normalized_identifier"])
    assert index["paper_records"][:220] == previous["paper_records"]
    assert index["existing_packets"][:len(previous["existing_packets"])] == previous["existing_packets"]
    for old in previous["canonical_identifier_normalization"]["groups"]:
        assert group_match(groups, old["normalized_identifier"].lower()) == old
    for key in previous:
        if key not in replaced and key not in {"paper_records", "existing_packets", "canonical_identifier_normalization"}:
            assert index[key] == previous[key]
    observed = metrics(index)
    assert observed["conclusion_records"] == 223
    assert observed["normalized_paper_identifiers"] == 171
    assert observed["software_documentation_identifiers"] == 2
    assert sum(a["new_identity"] for a in added) == 2
    assert sum(a["retained_abstract_identity_scope_upgrade"] for a in added) == 1

    # Replace stale latest-pass counters; their exact values remain in the one
    # predecessor snapshot above. Identity groups and newly encountered papers
    # are distinct because v56 left DIVE's earlier abstract identity ungrouped.
    account = {k: copy.deepcopy(v) for k, v in previous["read_accounting"].items()
               if not k.startswith(("latest_", "integration_pass_"))}
    account.update(observed, state="ROOT_COMPLETED_SCOPED_METHOD_MEMORY_ADOPTION",
                   historical_path_catalog_note="All 220 v56 records, groups, catalog entries, aliases, exclusions and failure history preserved. Three saved scopes appended; DIVE is a retained abstract identity upgraded to method scope. Two newly encountered paper identities and one previously ungrouped identity reconciled; zero integration/full reads.",
                   latest_adoption_packets=2, latest_index_growth=3,
                   latest_packet_new_scoped_primary_reads=3, latest_packet_full_primary_reads=0,
                   latest_packet_previously_completed_scoped_read_adoptions=3,
                   latest_packet_genuinely_new_scoped_paper_identities=2,
                   latest_packet_new_paper_identity_groups=3,
                   latest_packet_previously_read_identity_omissions_reconciled=1,
                   latest_packet_retained_abstract_only_scope_upgrades=1,
                   latest_packet_new_author_source_reads=0,
                   latest_packet_background_published_numeric_field_incidental_exposures=1,
                   latest_accounting_correction_reference=str((HERE / "VERIFICATION.json").relative_to(P)),
                   integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0,
                   integration_pass_primary_method_reads=0, integration_pass_retained_primary_revisits=0,
                   integration_pass_author_source_semantic_reads=0, integration_pass_project_source_semantic_reads=0,
                   integration_pass_incidental_primary_exposures=0, integration_pass_metadata_identity_checks=3,
                   integration_pass_experimental_score_artifact_reads=0,
                   integration_pass_saved_literature_numeric_field_incidental_exposures=1,
                   integration_pass_saved_literature_numeric_field_exposure_note="An overly broad saved-index identity query incidentally displayed the already indexed Tokenphormer published-result field, repeated in retained records. No primary result section or experimental/frozen-cohort outcome artifact was read; no numerical claim is adopted.")
    index["read_accounting"] = account
    index["latest_adoption"] = dict(UTC=utc, predecessor=ref(PREV / "LITERATURE_INDEX.json"),
                                    previous_records_preserved=True, added_records=added,
                                    source_completed_new_scoped_method_reads=3, source_full_paper_reads=0,
                                    newly_encountered_paper_identities=2, retained_abstract_identity_method_upgrades=1,
                                    normalized_paper_group_growth=3, previously_ungrouped_identity_reconciliations=1,
                                    integration_primary_reads=0, packet_bindings=bindings,
                                    novelty_or_numeric_or_predictive_or_execution_adoption=False)
    index["graph_directed_view_diversity_limits_v57"] = dict(
        DIVE="Continuing learned graph-view diversity is established. Native private encoders and validation-selected single serving do not establish shared-body uniform-pool complementarity. Printed sampler/backward/Jaccard ambiguities remain unresolved.",
        D_MPNN="Directed bond propagation, reverse exclusion, initial-state skip and independently trained directed-model averaging are established; scoped molecular ancestry only.",
        nonbacktracking="Learned nonbacktracking line-graph/operator-bank propagation and node/edge incidence maps are established. Exact retrieved v4 title retained; no full proof/source/runtime audit.",
        prospective_private_returns="B/J edge-operator bank with restricted parameter tying. One untested empirical sharing-bias hypothesis; no distinct propagation principle, predictive gain, novelty clearance or scientific launch adopted.")
    verification = dict(UTC=utc, predecessor_reference=ref(PREV / "LITERATURE_INDEX.json"),
                        predecessor_records_preserved=220, predecessor_catalog_preserved=True,
                        predecessor_groups_preserved_exactly=True, earlier_history_fields_preserved=True,
                        predecessor_failure_history_reference=ref(PREV / "INITIAL_BUILD_FAILURE.json"),
                        exact_new_source_hashes_verified=True, packet_bindings=bindings,
                        added_records=added, catalog_added=catalog_added, recomputed_metrics=observed,
                        source_completed_scoped_method_reads=3, newly_encountered_paper_identities=2,
                        retained_abstract_identity_upgrades=1, normalized_group_reconciliations=1,
                        scope_flags_verified=True, integration_primary_reads=0, new_full_paper_reads=0,
                        integration_experimental_artifact_reads=0,
                        integration_saved_literature_numeric_field_incidental_exposures=1,
                        no_numeric_predictive_novelty_or_execution_adoption=True,
                        DIVE_identity_note="Earlier scout READ_SCOPES.json scopes/2 identifies DIVE abstract exposure, but v56 has no DIVE paper record/group. Add one precise method record and reconcile one group; do not count DIVE as a newly encountered identity.")
    save("LITERATURE_INDEX.json", index)
    save("VERIFICATION.json", verification)
    (HERE / "ROOT_ADOPTION_NOTES.md").write_text(
        "# Literature index v57\n\n"
        "223 conclusion records cover 171 normalized paper groups and two software groups. "
        "All 220 v56 records, prior groups, catalog entries and failure history are preserved.\n\n"
        "Three completed method scopes are adopted: DIVE, D-MPNN (1904.01561v4), and "
        "nonbacktracking line-graph GNN (1705.08415v4, exact v4 title retained). DIVE upgrades "
        "an earlier abstract identity; its missing v56 group is reconciled. Thus group growth is "
        "three, while only two identities are newly encountered. No full-paper reads or "
        "integration primary reads are added.\n\n"
        "DIVE supplies continuing graph-view diversity ancestry with single-member serving. "
        "Directed messages and learned node/edge operator banks are established. Private "
        "return factors remain one untested sharing-bias hypothesis equivalent to a restricted "
        "B/J bank. No numeric, predictive, novelty or execution claim is adopted.\n")
    rows = [dict(path=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(HERE.iterdir()) if f.is_file()]
    save("MANIFEST.json", dict(UTC=utc, files=rows))
    save("SEAL.json", dict(manifest_sha256=ref(HERE / "MANIFEST.json")["sha256"], payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():
            f.chmod(0o444)
    print(json.dumps(dict(**observed, newly_encountered_paper_identities=2,
                          retained_abstract_identity_upgrades=1,
                          integration_primary_reads=0, new_full_paper_reads=0)))


if __name__ == "__main__":
    main()
