"""Build the source-only v40 successor from exact sealed metadata inputs.

Stdlib metadata transformation only. This does not import scientific code,
read fitted outputs, edit canonical state, or authorize scientific execution.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import re

PHASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = PHASE / "literature_memory/index_v39"
SCOUT = PHASE / "graph_structure_conditioned_specialization_scout_20261003_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor(path):
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path), "bytes": path.stat().st_size}


def verify(path, expected):
    assert sha(path) == expected["sha256"], (str(path), "sha256")
    assert path.stat().st_size == expected["bytes"], (str(path), "bytes")
    if path.suffix == ".json":
        json.loads(path.read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


assert sha(PRIOR / "LITERATURE_INDEX.json") == "07fffb6ea830c261b61cc34b37f7be25d958ae85f50e6cb375e3d45ed4bf3505"
assert (PRIOR / "LITERATURE_INDEX.json").stat().st_size == 713917
assert sha(SCOUT / "MANIFEST.json") == "aaa5a102ccb5689546162abaa5520fa41a6fd3f8ffeec0750c1b69479220ab67"
assert (SCOUT / "MANIFEST.json").stat().st_size == 8544
assert sha(SCOUT / "SEAL.json") == "b4ac8d2de7ce192c7a7d80d5788c3576c08466ad411c8903b2d8b718310beaac"
assert (SCOUT / "SEAL.json").stat().st_size == 1078
assert not any((OUT / n).exists() for n in ["LITERATURE_INDEX.json", "INTEGRATION.json", "VERIFICATION.json", "MANIFEST.json", "SEAL.json"])

prior_manifest = json.loads((PRIOR / "MANIFEST.json").read_text())
for item in prior_manifest["files"]:
    verify(PRIOR / item["path"], item)
scout_manifest = json.loads((SCOUT / "MANIFEST.json").read_text())
scout_seal = json.loads((SCOUT / "SEAL.json").read_text())
for item in scout_manifest["files"]:
    verify(SCOUT / item["path"], item)
assert len(scout_manifest["files"]) == scout_manifest["payload_count"] == 46
assert sum(x["bytes"] for x in scout_manifest["files"]) == scout_manifest["payload_bytes"] == 3497085
assert {x["path"] for x in scout_manifest["files"]} == {
    str(p.relative_to(SCOUT)) for p in SCOUT.rglob("*")
    if p.is_file() and p.name not in {"MANIFEST.json", "SEAL.json"}
}
for key in ["manifest", "report", "paper_conclusions", "prospective_test"]:
    verify(PHASE / scout_seal[key]["path"], scout_seal[key])
assert scout_manifest["source_inputs"][0] == descriptor(PRIOR / "LITERATURE_INDEX.json")
assert not scout_manifest["execution_authorized"]
assert not scout_manifest["new_method_or_pilot_adopted"]
assert not scout_seal["execution_authorized"]

old = json.loads((PRIOR / "LITERATURE_INDEX.json").read_text())
new = copy.deepcopy(old)
papers = json.loads((SCOUT / "PAPER_CONCLUSIONS.json").read_text())["papers"]
scopes = json.loads((SCOUT / "READ_SCOPES.json").read_text())
accounting = json.loads((SCOUT / "READ_ACCOUNTING.json").read_text())
assert len(papers) == scopes["new_scoped_primary_methods"] == accounting["new_scoped_primary_methods"] == 5
assert scopes["full_paper_certifications"] == accounting["new_full_paper_certifications"] == 0
assert accounting["scientific_launches"] == 0
assert not accounting["author_source_read_or_imported"]
assert not accounting["proofs_or_reported_results_audited"]

groups = new["canonical_identifier_normalization"]["groups"]
old_ids = {g["normalized_identifier"] for g in groups}
new_ids = []
conclusion_ref = descriptor(SCOUT / "PAPER_CONCLUSIONS.json")
scope_ref = descriptor(SCOUT / "READ_SCOPES.json")
manifest_ref = descriptor(SCOUT / "MANIFEST.json")
seal_ref = descriptor(SCOUT / "SEAL.json")
for paper in papers:
    canonical = re.sub(r"v\d+$", "", paper["canonical_id"].lower())
    assert canonical not in old_ids and canonical not in new_ids
    assert not paper["full_paper_read"] and not paper["author_code_read"]
    assert not paper["numeric_results_reproduced"] and not paper["global_novelty_or_absence_certificate"]
    assert paper["exact_read_scope"]["primary_method_read"]
    assert not paper["exact_read_scope"]["full_paper_read"]
    assert not paper["exact_read_scope"]["proofs_certified"]
    assert not paper["exact_read_scope"]["reported_results_adopted"]
    position = len(new["paper_records"])
    new["paper_records"].append({
        "canonical_id": paper["canonical_id"],
        "conclusion_file": conclusion_ref["path"],
        "conclusion_file_sha256": conclusion_ref["sha256"],
        "conclusion": copy.deepcopy(paper),
        "exact_read_scope": copy.deepcopy(paper["exact_read_scope"]),
        "read_scope_reference": dict(scope_ref, paper_canonical_id=paper["canonical_id"]),
        "source_packet_manifest_reference": manifest_ref,
        "source_packet_seal_reference": seal_ref,
        "integration_primary_method_read": False,
    })
    groups.append({
        "kind": "paper", "normalized_identifier": canonical,
        "raw_canonical_identifiers": [paper["canonical_id"]],
        "record_indices": [position], "explicit_aliases": [],
    })
    new_ids.append(canonical)

catalog = [
    ("MANIFEST.json", "sealed_scout_manifest", "46 source payloads; scoped reads only; no scientific adoption."),
    ("SEAL.json", "sealed_scout_bindings", "Exact report, conclusions and prospective-test bindings; execution unauthorized."),
    ("PAPER_CONCLUSIONS.json", "structured_paper_conclusions", "Five scoped methods; four recent graph sources and one older specialization ancestor. No whole-paper/code/proof/result certification."),
    ("READ_SCOPES.json", "exact_primary_read_scope_metadata", "Versioned HTML, paragraph ranges and equation ordinals. Retrieval is not a read."),
    ("REPORT.md", "bounded_structure_conditioned_specialization_scout", "Null promotion; capable-single and training-assignment ancestry/control requirements."),
    ("READ_ACCOUNTING.json", "bounded_scout_read_accounting", "Five scoped primary methods; zero full-paper certifications, author-source reads, scientific launches or canonical edits."),
    ("REJECTIONS_AND_OPEN_QUESTION.json", "preserved_scout_dispositions", "Rejections and one explicitly unadopted conditional utility question; no novelty clearance."),
    ("PROSPECTIVE_TEST.json", "unadopted_prospective_candidate_specification", "Five-arm Collab comparison with random/loss assignment and capable-single controls; no source or execution release."),
    ("DISCOVERY_DISPOSITIONS.json", "bounded_discovery_dispositions", "Discovery leads are metadata; unresolved scopes remain unresolved and do not prove absence."),
]
for name, kind, scope in catalog:
    new["existing_packets"].append(dict(descriptor(SCOUT / name), kind=kind, scope=scope))

now = datetime.now(timezone.utc).isoformat()
cats = {r["path"] for r in new["existing_packets"]}
cons = {r["conclusion_file"] for r in new["paper_records"] if "conclusion_file" in r}
scope_paths = {r["read_scope_reference"]["path"] for r in new["paper_records"] if "read_scope_reference" in r}
paper_ids = {g["normalized_identifier"] for g in groups if g["kind"] == "paper"}
software_ids = {g["normalized_identifier"] for g in groups if g["kind"] != "paper"}
assert len(groups) == len({g["normalized_identifier"] for g in groups})
integration = {
    "schema": "bounded_literature_successor_integration_v40", "UTC": now,
    "status": "PREPARED_SUCCESSOR_NO_CANONICAL_STATUS_OR_LEDGER_EDIT",
    "predecessor": descriptor(PRIOR / "LITERATURE_INDEX.json"),
    "predecessor_manifest": descriptor(PRIOR / "MANIFEST.json"),
    "source_manifest": manifest_ref, "source_seal": seal_ref,
    "source_payload_files_verified": 46, "source_payload_bytes_verified": 3497085,
    "preserved_predecessor_conclusion_records": len(old["paper_records"]),
    "added_conclusion_records": 5, "successor_conclusion_records": len(new["paper_records"]),
    "new_normalized_paper_identities": new_ids,
    "normalized_paper_identifiers": len(paper_ids), "software_documentation_identifiers": len(software_ids),
    "existing_groups_preserved_exactly": len(old["canonical_identifier_normalization"]["groups"]),
    "identifier_normalization_rules_preserved_exactly": True,
    "title_only_alias_merges": 0, "duplicate_new_identities": 0,
    "source_packet_first_scoped_primary_method_reads": 5,
    "source_packet_recent_graph_method_scopes": 4,
    "source_packet_older_specialization_ancestor_scopes": 1,
    "source_packet_full_paper_certifications": 0, "source_packet_author_code_reads": 0,
    "source_packet_proof_or_results_audits": 0, "integration_primary_method_reads": 0,
    "integration_retained_primary_revisits": 0, "metadata_identity_checks": 5,
    "operational_decision": "NO_NEW_LEARNER_PROMOTED; ONE_UNADOPTED_CONDITIONAL_UTILITY_QUESTION_SPECIFIED",
    "prior_attribution": {
        "GPM": "Random-walk semantic/anonymous structural tokens, learned pattern weighting, richer-context single and multiscale training/inference evidence.",
        "TAMI": "Time rescaling and persistent target-pair interaction history inside one temporal predictor.",
        "Context_Pooling": "TRAIN relation-conditioned query support and original/context aggregation inside a capable single; independence assumptions remain explicit.",
        "CHAT": "Concentrated heterogeneous walks, connection encodings, contrastive/observation supervision and context averaging of one learner. No normalized joint Bernoulli likelihood established by this scope.",
        "sMCL": "Minimum-member-loss training assignment and emergent specialization. Oracle coverage differs from uniform served quality.",
    },
    "capable_comparator": "Harmonized GPM single, or admitted BUDDY/LPFormer structural single; GPM's published Collab80/5/15 split table is incompatible with official-role comparisons.",
    "unadopted_question": "TRAIN-only structural-cell conditioning of private cotangents, native uniform mean-member shared gradients, fixed half global competence anchor, one native optimizer transition and uniform serving. Generally a nonconservative block gradient field with no pooled-descent guarantee.",
    "prospective_test_reference": descriptor(SCOUT / "PROSPECTIVE_TEST.json"),
    "prospective_controls": ["uniform_shared4", "balanced_random_private_allocation", "explicit_sMCL_derived_loss_assignment_adaptation", "capable_structural_single"],
    "prospective_test_scope": "Full ogbl-Collab; three paired seeds;15 run groups/51 member-or-single trajectories; complete official VALID Hits50; TEST locked. Source harmonization, reductions, tiny cells, ties, parameter ownership and measured costs remain unqualified. No launch release.",
    "rejections_reference": descriptor(SCOUT / "REJECTIONS_AND_OPEN_QUESTION.json"),
    "discovery_dispositions_reference": descriptor(SCOUT / "DISCOVERY_DISPOSITIONS.json"),
    "unresolved_prior_leads_preserved": True,
    "new_method_or_pilot_adopted": False, "execution_authorized": False,
    "new_scientific_execution_releases": 0, "scientific_launches": 0,
    "frozen_J_F_and_running_sources_changed": False, "canonical_edits": False,
    "methodological_novelty_established": False,
    "global_prior_absence_or_predictive_advantage_claimed": False,
    "verification": {
        "old_paper_records_prefix_equal": new["paper_records"][:len(old["paper_records"])] == old["paper_records"],
        "old_normalization_groups_prefix_equal": groups[:len(old["canonical_identifier_normalization"]["groups"])] == old["canonical_identifier_normalization"]["groups"],
        "old_catalog_prefix_equal": new["existing_packets"][:len(old["existing_packets"])] == old["existing_packets"],
        "old_locator_metadata_equal": new["locator_only_exclusions"] == old["locator_only_exclusions"],
        "old_unresolved_metadata_equal": new["unresolved_primary_metadata_leads"] == old["unresolved_primary_metadata_leads"],
        "predecessor_files_manifest_verified": True,
        "scout_manifest_seal_and_all_payload_hashes_verified": True,
        "all_five_new_scopes_declared_nonfull_noncode_nonproof_nonresults": True,
    },
    "recomputed_path_accounting": {
        "catalog_entries": len(new["existing_packets"]), "unique_catalog_document_paths": len(cats),
        "unique_conclusion_source_documents": len(cons), "unique_referenced_document_paths": len(cats | cons),
        "unique_scope_reference_document_paths": len(scope_paths),
    },
}
assert all(integration["verification"].values())
new["schema"] = "literature-memory-index-v40"
new["created_UTC"] = now
new["predecessor_index"] = integration["predecessor"]["path"]
new["predecessor_index_sha256"] = integration["predecessor"]["sha256"]
new["predecessor_v39_read_accounting_snapshot"] = copy.deepcopy(old["read_accounting"])
new["post_v39_append"] = integration
new["latest_adoption"] = integration
new["read_accounting"].update(
    state=integration["status"], conclusion_records=len(new["paper_records"]),
    normalized_paper_identifiers=len(paper_ids), software_documentation_identifiers=len(software_ids),
    **integration["recomputed_path_accounting"], latest_adoption_packets=1,
    latest_index_growth=integration, latest_packet_first_scoped_method_identity=new_ids,
    latest_packet_new_scoped_primary_reads=5, latest_packet_scoped_primary_method_events=5,
    latest_packet_full_primary_reads=0, latest_packet_retained_primary_revisits=0,
    latest_packet_retained_abstract_only_scope_upgrades=0, latest_packet_bounded_author_source_scope_events=0,
    integration_pass_new_primary_reads=0, integration_pass_primary_method_reads=0,
    integration_pass_full_primary_reads=0, integration_pass_retained_primary_revisits=0,
    integration_pass_metadata_identity_checks=5,
    cumulative_scoped_or_full_read_totals_certified=False, full_paper_read_total_certified=False,
    historical_path_catalog_note="All163 v39 conclusions and116 normalized groups preserved exactly. Five new scoped identities appended. Four recent graph scopes plus one older specialization ancestor are not whole-paper certifications. Prior unresolved leads, failures, aliases and groupings remain intact; no new learner promoted.",
)
changed_existing = {"schema", "created_UTC", "predecessor_index", "predecessor_index_sha256", "latest_adoption", "read_accounting", "paper_records", "existing_packets", "canonical_identifier_normalization"}
unchanged_keys = [k for k in old if k not in changed_existing]
assert all(new[k] == old[k] for k in unchanged_keys)
assert {k:v for k,v in new["canonical_identifier_normalization"].items() if k != "groups"} == {k:v for k,v in old["canonical_identifier_normalization"].items() if k != "groups"}
write("LITERATURE_INDEX.json", new)
write("INTEGRATION.json", integration)
verification = {
    "schema": "literature-memory-v40-source-only-verification-v1", "UTC": now,
    "predecessor": integration["predecessor"], "predecessor_manifest": integration["predecessor_manifest"],
    "scout_manifest": manifest_ref, "scout_seal": seal_ref,
    "source_payload_files_verified": 46, "source_payload_bytes_verified": 3497085,
    "preservation_checks": integration["verification"],
    "unchanged_existing_top_level_keys": unchanged_keys,
    "normalization_non_group_fields_exactly_preserved": True,
    "normalized_identifier_groups_unique": True, "old_record_indices_and_aliases_exactly_preserved": True,
    "new_record_indices": list(range(len(old["paper_records"]), len(new["paper_records"]))),
    "new_canonical_identities": new_ids, "recomputed_path_accounting": integration["recomputed_path_accounting"],
    "counts_are_scoped_reads_not_whole_paper_certifications": True,
    "metadata_integration_primary_reads": 0, "scientific_execution": False,
    "canonical_status_ledger_or_manuscript_edits": False, "execution_authorized": False,
    "validation": "Stdlib source metadata JSON/hash/size and exact equality checks only.",
}
write("VERIFICATION.json", verification)
manifest_files = ["BUILD_INDEX.py", "LITERATURE_INDEX.json", "INTEGRATION.json", "VERIFICATION.json"]
for n in manifest_files:
    if n.endswith(".json"):
        json.loads((OUT/n).read_text())
write("MANIFEST.json", {
    "schema": "append_only_literature_metadata_manifest_v40", "UTC": now,
    "status": "SEALED_PREPARED_SUCCESSOR_NO_CANONICAL_STATUS_OR_LEDGER_EDIT",
    "files": [{"path": n, "sha256": sha(OUT/n), "bytes": (OUT/n).stat().st_size} for n in manifest_files],
    "payload_count": len(manifest_files), "source_inputs": [integration["predecessor"], integration["predecessor_manifest"], manifest_ref, seal_ref],
    "execution_authorized": False, "scientific_launches": 0, "canonical_edits": False,
})
write("SEAL.json", {
    "schema": "literature-memory-v40-source-only-seal-v1", "UTC": now,
    "manifest": descriptor(OUT / "MANIFEST.json"),
    "index": descriptor(OUT / "LITERATURE_INDEX.json"),
    "integration": descriptor(OUT / "INTEGRATION.json"),
    "verification": descriptor(OUT / "VERIFICATION.json"),
    "payload_hashes_sizes_JSON_and_preservation_verified": True,
    "execution_authorized": False, "canonical_edits": False,
})
for n in manifest_files:
    item = next(x for x in json.loads((OUT/"MANIFEST.json").read_text())["files"] if x["path"] == n)
    verify(OUT/n, item)
for p in OUT.iterdir():
    if p.is_file():
        os.chmod(p, 0o444)
os.chmod(OUT, 0o555)
print(json.dumps({
    "manifest": descriptor(OUT / "MANIFEST.json"), "seal": descriptor(OUT / "SEAL.json"),
    "index": descriptor(OUT / "LITERATURE_INDEX.json"), "integration": descriptor(OUT / "INTEGRATION.json"),
    "conclusion_records": len(new["paper_records"]), "normalized_papers": len(paper_ids),
    "software_identifiers": len(software_ids), "preserved_old_groups": len(old["canonical_identifier_normalization"]["groups"]),
    "path_accounting": integration["recomputed_path_accounting"],
}, indent=2))
