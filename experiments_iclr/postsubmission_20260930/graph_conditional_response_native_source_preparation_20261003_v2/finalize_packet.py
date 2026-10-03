"""Stdlib-only successor custody binder/sealer; never imports numerical source."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESEARCH = PROJECT / "postsubmission_research_20260930"
PREVIOUS = RESEARCH / "graph_conditional_response_qualification_preparation_20261003_v1"
PREVIOUS_MANIFEST_SHA256 = "a25174ad0c4ae8258dde9f2a3f28333f3a0c4c7618f1aad0bcd699c0685b65a9"
MODERN = RESEARCH / "continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3"
AUTHOR = RESEARCH / "coordinate_source_independent_review_v1/strong_backbones_v1"
PORTS = RESEARCH / "efficient_graph_ensemble_ports_preparation_v1"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(name, content):
    (ROOT / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def bind_inputs():
    if (ROOT / "MANIFEST.json").exists():
        raise RuntimeError("A sealed successor is immutable; use a new version")
    if digest(PREVIOUS / "MANIFEST.json") != PREVIOUS_MANIFEST_SHA256:
        raise RuntimeError("Immutable previous preparation manifest changed")
    inputs = {}

    def add(path, role, expected=None, read_scope="custody_hash_only"):
        path = path.resolve()
        relative = path.relative_to(PROJECT).as_posix()
        observed = digest(path)
        if expected is not None and observed != expected:
            raise RuntimeError(f"Source/protected binding changed: {relative}")
        if relative in inputs:
            if inputs[relative]["sha256"] != observed:
                raise RuntimeError("Conflicting external binding")
            inputs[relative]["roles"].append(role)
            return
        inputs[relative] = {"path": relative, "sha256": observed, "size": path.stat().st_size,
                            "roles": [role], "read_scope": read_scope}

    add(PREVIOUS / "MANIFEST.json", "immutable_previous_preparation_manifest", PREVIOUS_MANIFEST_SHA256)
    previous = json.loads((PREVIOUS / "MANIFEST.json").read_text())
    for row in previous["files"]:
        add(PREVIOUS / row["path"], "immutable_previous_preparation_payload", row["sha256"])
    inherited = json.loads((PREVIOUS / "PROVENANCE.json").read_text())["external_bindings"]
    if len(inherited) != 19:
        raise RuntimeError("Expected nineteen unchanged inherited bindings")
    for row in inherited:
        add(PROJECT / row["path"], "inherited_previous_binding:" + row["role"], row["sha256"])

    scope = "exact_source_body_or_interface_scope_in_READ_SCOPES_not_imported"
    add(PORTS / "native_polyformer.py", "retained_native_class_copy",
        "b1bf001258d160f8e98d5042a775ae475b6ea5e515e2af901287f8fea0382730", scope)
    add(PORTS / "native_polyformer_outer.py", "retained_native_outer_copy",
        "b4bf6669ebb869bfe0130bdb4d0be2132b31af71817b58219f1a96566f38df96", scope)
    add(MODERN / "prototype/native_polyformer_preprocess.py", "retained_native_preprocessing_copy",
        "36d558c1f29b3fd9ad9084e61efc7ec22aae2de49c3b3d76fde63b52542fcb0f", scope)
    for path in (PORTS / "ports.py", PORTS / "PROVENANCE.json",
                 RESEARCH / "closest_controls_v1/all_layer_sage_port/all_layer_sage_adapter.py"):
        add(path, "existing_attributed_local_interface_source", read_scope=scope)
    for name, expected in {
        "AUTHOR_SOURCE_BINDINGS.json": "85d5753264d77c8df7b6846b494c35204310919fbf00e67df733eca597aaf321",
        "IMPLEMENTATION_BINDINGS.json": "edbd0d9fb3490ea204c0e7347e32854a52e3f84696488ccacdd2852ab483805f",
        "RUNTIME_DEPENDENCIES.json": "664f801e93993e17f8ec159117c5c130ef1a37891f29d5fe59265e55be37a3eb",
    }.items():
        add(MODERN / name, "retained_author_or_runtime_metadata", expected, "metadata_only")
    author_rows = json.loads((MODERN / "AUTHOR_SOURCE_BINDINGS.json").read_text())["bindings"]
    polyformer_rows = [r for r in author_rows if r["author_repository"] == "air029/PolyFormer"]
    if len(polyformer_rows) != 6 or any(r["commit"] != "d390f39e88d0eaac80318fdc7704bd3bf3cf8b13" for r in polyformer_rows):
        raise RuntimeError("Pinned author binding scope changed")
    for row in polyformer_rows:
        add(MODERN / row["local_path"], "retained_commit_pinned_author_copy", row["sha256"], scope)
        add(AUTHOR / row["path"], "commit_pinned_author_body_comparison", row["sha256"], scope)
    saved_be = RESEARCH / "efficient_paths"
    add(saved_be / "SOURCES_MANIFEST.json", "saved_Edward2_snapshot_metadata", read_scope="one_source_row_metadata_only")
    add(saved_be / "sources/batchensemble_source.py", "saved_Edward2_dense_bias_convention",
        "c1d402ec51bfa649e478a220d02050eda902f71d6e93b746a64a9cb79983b2cb", scope)

    receipt = json.loads((ROOT / "PYG_CODE_FETCH_RECEIPT.json").read_text())
    for row in receipt["requests"]:
        path = (ROOT / row["path"]).resolve()
        path.relative_to(ROOT / "pyg_source_pins")
        if row["commit"] != "76ff9c2ce18c8cebf52122b57e2aeadce9793d10" or digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise RuntimeError("New pinned PyG code custody mismatch")
    write_json("PROVENANCE.json", {
        "schema": "native-source-successor-preparation-provenance-v2",
        "UTC": datetime.now(timezone.utc).isoformat(), "packet": ROOT.name,
        "previous_preparation": PREVIOUS.relative_to(PROJECT).as_posix(),
        "previous_manifest_sha256": PREVIOUS_MANIFEST_SHA256,
        "previous_payload_files_bound": previous["payload_file_count"],
        "previous_external_bindings_carried_forward": len(inherited),
        "external_bindings": list(inputs.values()),
        "new_primary_code_retrieval_files": len(receipt["requests"]),
        "new_full_paper_read_certifications": 0,
        "primary_code_read_scopes": "READ_SCOPES.json; AST body equality is structural verification, not a paper read",
        "existing_all_layer_PolyFormer_port_verified": False,
        "new_composition": "attributed complete sequential native R/S member-path source prepared before warm acquisition",
        "numerical_imports_or_execution_in_this_preparation": False,
        "native_check_families": "three prepared; separate root authorization required; unexecuted",
        "old_generic_CPU_qualification": "parent reported separately; not native/source/runtime/control admission",
        "roles_masks_feasibility_or_sampling_executed": False,
        "data_labels_checkpoints_or_live_outcomes_accessed": False,
        "training_fit_or_resource_measurements": None,
        "old_packets_or_canonical_ledgers_modified": False,
        "pilot_frozen_or_launched": False,
        "efficacy_or_novelty_claim": False,
        "unresolved_scientific_or_qualification_gates": [
            "competent explicitly adapted Amazon warm recipe",
            "mask unit/rounding/seed/neighbor and TRAIN role rounding prospective adoption",
            "exact per-node Alternative A oracle/feasibility or explicit distinct Alternative B null",
            "cross-arm active/reference and class-only energy guard semantics",
            "strong source-informed DICE and FoRDE implementations/state custody",
            "separately authorized actual native CPU checks and source/runtime precision qualification",
            "complete-data resource step before any later resource request or pilot admission",
        ],
    })


def seal():
    if (ROOT / "MANIFEST.json").exists():
        raise RuntimeError("A sealed successor is immutable; use a new version")
    receipt = json.loads((ROOT / "SOURCE_VERIFICATION.json").read_text())
    if receipt["status"] != "source_checks_passed" or receipt["numerical_native_checks_executed"] != 0:
        raise RuntimeError("Permitted source-only verification receipt required")
    observed = {p.relative_to(ROOT).as_posix(): (digest(p), p.stat().st_size) for p in ROOT.rglob("*")
                if p.is_file() and p.name not in ("MANIFEST.json", "SOURCE_VERIFICATION.json")}
    verified = {r["path"]: (r["sha256"], r["size"]) for r in receipt["verified_payload_bindings"]}
    if observed != verified:
        raise RuntimeError("Source receipt is stale; record source verification again before sealing")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    if any("__pycache__" in p.parts or p.suffix == ".pyc" for p in files):
        raise RuntimeError("No runtime artifacts may enter a source-only packet")
    write_json("MANIFEST.json", {
        "schema": "sha256-native-source-preparation-manifest-v2",
        "UTC": datetime.now(timezone.utc).isoformat(), "packet": ROOT.name,
        "excludes": ["MANIFEST.json (self)"], "payload_file_count": len(files),
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "size": p.stat().st_size}
                  for p in files],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--bind-inputs-only", action="store_true")
    action.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    if args.bind_inputs_only:
        bind_inputs()
    else:
        seal()
    print(json.dumps({"status": "inputs_bound" if args.bind_inputs_only else "sealed",
                      "numerical_execution": False, "pilot_launched": False}))
