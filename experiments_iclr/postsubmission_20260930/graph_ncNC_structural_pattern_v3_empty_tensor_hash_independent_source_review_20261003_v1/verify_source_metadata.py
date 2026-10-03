"""Independent stdlib source/AST/metadata review; never import project code."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
PHASE = REVIEW.parent
V2 = PHASE / "graph_ncNC_structural_pattern_pilot_preparation_20261003_v2"
V3 = PHASE / "graph_ncNC_structural_pattern_pilot_preparation_20261003_v3"
FAILURE = PHASE / "graph_ncNC_structural_pattern_numerical_execution_root_20261003_v1"
ALLOWED = {".py", ".json", ".md", ".sh", ".patch", ".txt", ".log", ".html"}


def descriptor(path, relative_to=PHASE):
    assert path.suffix in ALLOWED or path.name == ".gitignore", path
    assert path.is_file() and not path.is_symlink(), path
    raw = path.read_bytes()
    raw.decode("utf-8")
    return {"path": str(path.relative_to(relative_to)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def bound(path, row):
    result = descriptor(path)
    assert result["bytes"] == row.get("bytes", row.get("size")), path
    assert result["sha256"] == row["sha256"], path
    return result


def packet(root, expected_manifest=None, exact_inventory=True):
    mdesc = descriptor(root / "MANIFEST.json")
    if expected_manifest:
        assert mdesc["sha256"] == expected_manifest, root
    manifest = json.loads((root / "MANIFEST.json").read_text())
    files = []
    names = set()
    python = []
    json_paths = []
    for row in manifest["files"]:
        assert row["path"] not in names, row
        names.add(row["path"])
        path = (root / row["path"]).resolve()
        assert path.is_relative_to(root.resolve()), path
        files.append(bound(path, row))
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(), filename=str(path))
            compile(tree, str(path), "exec")
            python.append(row["path"])
        if path.suffix == ".json":
            json.loads(path.read_text())
            json_paths.append(row["path"])
    seal_desc = None
    if (root / "SEAL.json").is_file():
        seal = json.loads((root / "SEAL.json").read_text())
        assert seal["manifest_sha256"] == mdesc["sha256"], root
        seal_desc = descriptor(root / "SEAL.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    expected = names | {"MANIFEST.json"} | ({"SEAL.json"} if seal_desc else set())
    if exact_inventory:
        assert actual == expected, (root, sorted(actual - expected), sorted(expected - actual))
    return {"manifest": mdesc, "seal": seal_desc, "payloads": files,
            "payload_count": len(files), "payload_bytes": sum(r["bytes"] for r in files),
            "Python_AST_compiled_without_execution": python, "JSON_payloads_parsed": json_paths,
            "exact_inventory": actual == expected}


def main():
    evidence = {"schema": "ncnc-v3-independent-source-evidence-v1",
                "UTC": datetime.now(timezone.utc).isoformat()}
    evidence["V2"] = packet(V2, "fd962780a1cf563720bb4addada6bdeb8598132d77474bb6351b159b53177931")
    evidence["V3"] = packet(V3, "fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89")
    assert evidence["V2"]["seal"]["sha256"] == "0a04a7f6713281811a48f26228f52fd06e36e203c9617f07d28bae7880f751db"
    assert evidence["V3"]["seal"]["sha256"] == "7473c5483ea8fb1fc70a8c524620b38077a6f4e5d1e664bbe8103c5fe782bf2c"
    old = {r["path"]: r for r in json.loads((V2 / "MANIFEST.json").read_text())["files"]}
    new = {r["path"]: r for r in json.loads((V3 / "MANIFEST.json").read_text())["files"]}
    assert old.keys() <= new.keys()
    changed = [p for p in old if old[p]["sha256"] != new[p]["sha256"]]
    changed_python = sorted(p for p in changed if p.endswith(".py"))
    assert changed_python == ["pilot_data.py", "source_check.py"]
    assert old["PILOT_PLAN.json"] == new["PILOT_PLAN.json"]
    evidence["diff_inventory"] = {"changed_original_payloads": changed,
                                  "added_payloads": sorted(new.keys() - old.keys()),
                                  "deleted_payloads": [], "changed_original_Python": changed_python,
                                  "unchanged_Python": sorted(p for p in old if p.endswith(".py") and p not in changed),
                                  "scientific_plan_unchanged": True}
    all_diffs = []
    for p in changed:
        diff = "".join(difflib.unified_diff((V2/p).read_text().splitlines(True),
                                         (V3/p).read_text().splitlines(True),
                                         fromfile="V2/"+p, tofile="V3/"+p))
        all_diffs.append(diff)
    (REVIEW / "V2_TO_V3_DIFF.patch").write_text("\n".join(all_diffs))
    original = (V2 / "pilot_data.py").read_text()
    repaired = (V3 / "pilot_data.py").read_text()
    old_line = '    digest.update(memoryview(array).cast("B"))\n'
    new_line = '    if array.size:\n        digest.update(memoryview(array).cast("B"))\n'
    assert original.count(old_line) == 1
    assert original.replace(old_line, new_line) == repaired
    old_tree = ast.parse(original)
    new_tree = ast.parse(repaired)
    old_fn = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "tensor_sha")
    new_fn = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "tensor_sha")
    assert len(old_fn.body) == len(new_fn.body) == 4
    guard = new_fn.body[2]
    assert isinstance(guard, ast.If) and not guard.orelse
    assert ast.dump(guard.test) == ast.dump(ast.Attribute(value=ast.Name(id="array", ctx=ast.Load()), attr="size", ctx=ast.Load()))
    assert len(guard.body) == 1 and ast.dump(guard.body[0]) == ast.dump(old_fn.body[2])
    assert all(ast.dump(old_fn.body[i]) == ast.dump(new_fn.body[i]) for i in (0, 1, 3))
    evidence["tensor_sha_AST"] = {"old_function": ast.dump(old_fn), "new_function": ast.dump(new_fn),
                                  "exact_guard_only_change": True, "shape_dtype_header_unchanged": True,
                                  "nonempty_bytes_expression_unchanged": True,
                                  "empty_shape_or_strides_cast_skipped": True,
                                  "runtime_tensor_or_array_execution": False}
    dependencies = json.loads((V3 / "DEPENDENCIES.json").read_text())
    evidence["dependencies"] = [packet(PHASE/r["packet"], r["manifest_sha256"], exact_inventory=False)
                                 for r in dependencies["sealed_packets"]]
    evidence["authorities"] = []
    for row in dependencies["authority_files"]:
        result = descriptor(PHASE/row["path"])
        assert result["sha256"] == row["sha256"]
        json.loads((PHASE/row["path"]).read_text())
        evidence["authorities"].append(result)
    evidence["copied_helpers"] = []
    for row in json.loads((V3 / "COPIED_SOURCE_BINDINGS.json").read_text())["files"]:
        src, dest = PHASE/row["source"], V3/row["destination"]
        source_desc, destination_desc = descriptor(src), descriptor(dest)
        assert source_desc["sha256"] == row["sha256"]
        expected = src.read_text()
        if row["adapted"]:
            if row["destination"] == "pilot_state.py":
                expected = expected.replace('"eventual_test_graph": "native_TRAIN_plus_VALID",', '"test_stage_supported": False,')
            elif row["destination"] == "pilot_data.py":
                expected = expected.replace(old_line, new_line)
                assert destination_desc["sha256"] == row["adapted_sha256"]
            else:
                raise AssertionError(row)
        assert expected == dest.read_text()
        evidence["copied_helpers"].append({"binding": row, "original": source_desc,
                                           "destination": destination_desc, "exact_recorded_adaptation": True})
    fetch = json.loads((FAILURE/"FETCH_RUN01.json").read_text())
    evidence["failure_custody"] = {"fetch": descriptor(FAILURE/"FETCH_RUN01.json"), "files": []}
    assert descriptor(V3/"PRESERVED_V2_NUMERICAL_RUN01/FETCH_RUN01.json")["sha256"] == evidence["failure_custody"]["fetch"]["sha256"]
    for row in fetch["files"]:
        source = bound(FAILURE/"remote_receipts_run01"/row["path"], row)
        copy = bound(V3/"PRESERVED_V2_NUMERICAL_RUN01"/row["path"], row)
        evidence["failure_custody"]["files"].append({"original": source, "copy": copy, "identical": True})
    terminal = json.loads((FAILURE/"remote_receipts_run01/supervision/run01/SUPERVISOR_TERMINAL.json").read_text())
    attempts = json.loads((FAILURE/"remote_receipts_run01/numerical/run01/ATTEMPTS.json").read_text())
    failed = json.loads((FAILURE/"remote_receipts_run01/numerical/run01/FAILED.json").read_text())
    repair = json.loads((V3/"V3_REPAIR.json").read_text())
    assert fetch["terminal_status"] == "FAILED" and fetch["physical_processes"] == []
    assert terminal["status"] == "FAILED" and terminal["child_exit_code"] == 1
    assert terminal["child_physical_identity"]["PID"] == repair["original_failed_child_PID"] == 3161488
    assert terminal["supervisor_inclusive_wall_seconds"] == repair["original_supervisor_wall_seconds"]
    assert terminal["child_wait_wall_seconds"] == repair["original_child_wait_wall_seconds"]
    assert terminal["kernel_child_peak_RSS_bytes"] == repair["original_kernel_peak_RSS_bytes"]
    assert terminal["cap_violation"] is None and terminal["qualification_receipt"] is None
    attempt = attempts["attempts"][0]
    assert attempt["status"] == "FAILED" and attempt["inclusive_wall_seconds"] == repair["original_driver_attempt_wall_seconds"]
    assert attempt["failure"]["exception_type"] == "TypeError"
    assert attempts["identity"] == failed["identity"] == repair["original_failure_identity_preserved"]
    assert attempts["identity"]["driver_manifest_sha256"] == evidence["V2"]["manifest"]["sha256"]
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        assert terminal["CUDA_peak_observation"][key] == repair["original_CUDA_peaks"][key]
    for pin_name in ("predecessor_manifest", "predecessor_seal", "predecessor_passed_source_review", "preserved_fetch_receipt"):
        bound(PHASE/repair[pin_name]["path"], repair[pin_name])
    predecessor_review = PHASE/repair["predecessor_passed_source_review"]["path"]
    assert descriptor(V3/"PRESERVED_V2_REVIEW.json")["sha256"] == descriptor(predecessor_review)["sha256"]
    evidence["failure_observation"] = {"terminal": terminal, "attempts": attempts, "failed": failed,
                                       "source_failure_stack": (FAILURE/"remote_receipts_run01/supervision/run01/CHILD.stderr.log").read_text(),
                                       "not_a_numerical_PASS": True,
                                       "overlapping_wall_scopes_not_additive": True,
                                       "V2_costs_root_carried_separately_from_current_identity_closure": True}
    evidence["read_accounting"] = {"stdlib_only": True, "AST_compile_without_execution": True,
                                    "project_or_numerical_imports_or_execution": False,
                                    "arrays_checkpoints_runtime_binaries_read_or_hashed": False,
                                    "SSH_or_remote_access": False, "canonical_or_input_edits": False}
    (REVIEW/"EVIDENCE.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({"status": "PASS_SOURCE_METADATA_CHECKS", "V3_payloads": len(new),
                      "V3_python": len(evidence["V3"]["Python_AST_compiled_without_execution"]),
                      "dependency_payloads": sum(r["payload_count"] for r in evidence["dependencies"]),
                      "preserved_failure_files": len(fetch["files"])+1, "changed_Python": changed_python}))


if __name__ == "__main__":
    main()
