"""Independent supervisor successor source/AST/metadata checks only."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
PHASE = REVIEW.parent
V1 = PHASE / "graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v1"
V2 = PHASE / "graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v2"
DRIVER = PHASE / "graph_ncNC_structural_pattern_pilot_preparation_20261003_v3"
DRIVER_REVIEW = PHASE / "graph_ncNC_structural_pattern_v3_empty_tensor_hash_independent_source_review_20261003_v1"
ALLOWED = {".py", ".json", ".md", ".sh", ".patch", ".txt", ".log", ".html"}


def descriptor(path):
    assert path.suffix in ALLOWED and path.is_file() and not path.is_symlink(), path
    raw = path.read_bytes()
    raw.decode("utf-8")
    return {"path": str(path.relative_to(PHASE)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def bound(path, row):
    result = descriptor(path)
    assert result["bytes"] == row.get("bytes", row.get("size")), path
    assert result["sha256"] == row["sha256"], path
    return result


def packet(root, expected_manifest, expected_seal):
    manifest = json.loads((root/"MANIFEST.json").read_text())
    seal = json.loads((root/"SEAL.json").read_text())
    mdesc, sdesc = descriptor(root/"MANIFEST.json"), descriptor(root/"SEAL.json")
    assert mdesc["sha256"] == expected_manifest == seal["manifest_sha256"]
    assert sdesc["sha256"] == expected_seal
    names, rows, python, json_paths = set(), [], [], []
    for row in manifest["files"]:
        assert row["path"] not in names
        names.add(row["path"])
        path = (root/row["path"]).resolve()
        assert path.is_relative_to(root.resolve())
        rows.append(bound(path, row))
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(), filename=str(path))
            compile(tree, str(path), "exec")
            python.append({"path": row["path"], "top_level_imports": [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]})
        if path.suffix == ".json":
            json.loads(path.read_text())
            json_paths.append(row["path"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert actual == names | {"MANIFEST.json", "SEAL.json"}
    return {"manifest": mdesc, "seal": sdesc, "payloads": rows, "payload_count": len(rows),
            "exact_inventory": True, "Python_AST_compile_without_execution": python,
            "JSON_payloads_parsed": json_paths}


def main():
    evidence = {"schema": "ncnc-supervisor-v2-independent-source-evidence-v1", "UTC": datetime.now(timezone.utc).isoformat()}
    evidence["V1"] = packet(V1, "3cfc893897fa86c64aeb36962b3ecbf2653fb5be9358c7e017a32c5392f0bf48", "05f693a84ae82a720432094bb4d73706da270eb1198a6e198b27ab00d1020506")
    evidence["V2"] = packet(V2, "dd853261be34c61b472813d5f452d7c0fe1ecd78088da35b22cf5c4f7b4cba21", "177050181607fc81dac2f244d373903e649003ba581f101163ef3d285cf463d2")
    evidence["driver"] = packet(DRIVER, "fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89", "7473c5483ea8fb1fc70a8c524620b38077a6f4e5d1e664bbe8103c5fe782bf2c")
    driver_review = json.loads((DRIVER_REVIEW/"REVIEW.json").read_text())
    assert driver_review["status"] == "passed" and driver_review["source"]["manifest"] == evidence["driver"]["manifest"]
    assert descriptor(DRIVER_REVIEW/"REVIEW.json")["sha256"] == "062a7cea6701a0f1b56822dfcabe8313b4c50d977e767a14bd897bc7bce44315"
    evidence["driver_review"] = descriptor(DRIVER_REVIEW/"REVIEW.json")
    evidence["driver_review_seal"] = descriptor(DRIVER_REVIEW/"SEAL.json")
    old = {r["path"]:r for r in json.loads((V1/"MANIFEST.json").read_text())["files"]}
    new = {r["path"]:r for r in json.loads((V2/"MANIFEST.json").read_text())["files"]}
    assert old.keys() <= new.keys()
    changed = sorted(p for p in old if old[p]["sha256"] != new[p]["sha256"])
    evidence["diff_inventory"] = {"changed_original_payloads": changed, "added_payloads": sorted(new.keys()-old.keys()), "deleted_payloads": []}
    diffs = []
    for p in changed:
        diffs.append("".join(difflib.unified_diff((V1/p).read_text().splitlines(True), (V2/p).read_text().splitlines(True), fromfile="V1/"+p, tofile="V2/"+p)))
    (REVIEW/"V1_TO_V2_DIFF.patch").write_text("\n".join(diffs))
    adaptations = [("graph_ncNC_structural_pattern_pilot_preparation_20261003_v3", "graph_ncNC_structural_pattern_pilot_preparation_20261003_v2"),
                   ("graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2", "graph_ncNC_structural_pattern_numerical_execution_root_20261003_v1"),
                   ("Only independently repaired V3 may be released", "Only independently repaired V2 may be released")]
    evidence["reverse_adaptations"] = []
    for p in ("supervise_numerical.py", "numerical_child.py"):
        text = (V2/p).read_text()
        counts = []
        for after, before in adaptations:
            counts.append({"after": after, "before": before, "occurrences": text.count(after)})
            text = text.replace(after, before)
        assert text == (V1/p).read_text()
        evidence["reverse_adaptations"].append({"path": p, "adaptations": counts, "restores_exact_V1_bytes": True})
    assert (V1/"CAPS.json").read_bytes() == (V2/"CAPS.json").read_bytes()
    caps = json.loads((V2/"CAPS.json").read_text())["caps"]
    assert caps == {"wall_seconds": 1800, "host_RSS_bytes": 17179869184, "cuda_peak_allocated_bytes": 8589934592, "cuda_peak_reserved_bytes": 8589934592}
    template = json.loads((V2/"ROOT_RELEASE_NUMERICAL_TEMPLATE.json").read_text())
    assert template["example_only_NOT_AUTHORIZATION"] is True and template["root_authorization_reference"] is None
    assert template["authorized_stages"] == template["authorized_invocations"] == []
    assert template["driver_manifest_sha256"] == evidence["driver"]["manifest"]["sha256"]
    assert template["numerical_caps"] == caps
    assert template["supervision_manifest_sha256"] == "REPLACE_WITH_SEALED_SUPERVISION_MANIFEST_SHA256"
    repo = "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git"
    remote_phase = repo + "/experiments_iclr/postsubmission_20260930"
    execution = remote_phase + "/graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2"
    assert template["example_invocations_NOT_AUTHORIZED"] == [{"stage": "numerical", "unit": "pair", "base_seed": 0, "output_directory": execution + "/numerical/run01"}]
    assert template["numerical_driver_path"] == remote_phase + "/graph_ncNC_structural_pattern_pilot_preparation_20261003_v3/pattern_run.py"
    custody = template["predecessor_failure_custody"]
    assert descriptor(PHASE/custody["path"])["sha256"] == custody["sha256"]
    assert custody["exact_V2_identity_retained"] and custody["root_cross_version_cost_carry_required"]
    rebind = json.loads((V2/"V2_REBIND.json").read_text())
    for key in ("predecessor_support_manifest", "predecessor_support_seal", "predecessor_passed_review", "pilot_V3_manifest", "pilot_V3_seal", "pilot_V3_repair_custody"):
        bound(PHASE/rebind[key]["path"], rebind[key])
    assert descriptor(V2/"PRESERVED_V1_SUPERVISOR_REVIEW.json")["sha256"] == rebind["predecessor_passed_review"]["sha256"]
    previous_review = json.loads((PHASE/rebind["predecessor_passed_review"]["path"]).read_text())
    assert previous_review["status"] == "passed"
    evidence["predecessor_review"] = rebind["predecessor_passed_review"]
    evidence["retained_predecessor_limits"] = previous_review["limits"]
    evidence["candidate_caps"] = caps
    evidence["prospective_root_template"] = template
    evidence["rebind_custody"] = rebind
    evidence["physical_and_runtime_constants"] = {"repository": repo, "exact_route": "shmelev@192.168.18.77", "fresh_execution_root": execution,
        "Python": "/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12", "Python_SHA": "14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2",
        "allowed_GPU_UUIDs": ["GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998", "GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced"], "poll_seconds": 0.25,
        "normal_termination_accounting": "os.wait4", "ordinary_child_session": "start_new_session=True", "all_constants_other_than_explicit_paths_unchanged_by_exact_reverse_bytes": True}
    evidence["read_accounting"] = {"stdlib_source_AST_JSON_text_only": True, "AST_compile_without_execution": True,
        "supervisor_child_project_numerical_execution_or_imports": False, "arrays_checkpoints_runtime_binaries_read_or_hashed": False,
        "SSH_remote_or_allocation_access": False, "input_or_canonical_edits": False}
    (REVIEW/"EVIDENCE.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({"status": "PASS_SOURCE_METADATA_CHECKS", "supervisor_payloads": len(new), "supervisor_Python_ASTs": 2,
                      "reverse_adaptations_exact": True, "caps_identical": True, "driver_review": evidence["driver_review"]}))


if __name__ == "__main__":
    main()
