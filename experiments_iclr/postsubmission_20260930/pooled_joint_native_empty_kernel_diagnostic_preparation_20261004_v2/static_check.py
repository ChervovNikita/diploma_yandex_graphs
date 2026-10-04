#!/usr/bin/env python3
"""Stdlib static checks; no imported project/runtime/Torch or numerical work."""
import ast
import difflib
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace


def check_pin(path, row):
    assert path.stat().st_size == row.get("bytes", row.get("size")), str(path)
    assert sha256(path.read_bytes()).hexdigest() == row["sha256"], str(path)


def scalar_metadata_function(path):
    # Exercise only extracted scalar metadata, with no runtime/model imports.
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "metadata")
    scalar_tree = ast.Module(body=[function],type_ignores=[])
    namespace = {"math":math,"torch":SimpleNamespace(is_tensor=lambda value:False)}
    exec(compile(scalar_tree,str(path)+":scalar_metadata_only","exec"),namespace)
    return namespace["metadata"]


def literal_delta_and_scalar_check(here, prior):
    old = (prior / "operator_trace.py").read_text()
    new = (here / "operator_trace.py").read_text()
    expected = old.replace("import json\n","import json\nimport math\n",1).replace(
        "    if value is None or isinstance(value, (str, int, float, bool)):\n",
        "    # JSON tags describe metadata only; original operation arguments are untouched.\n"
        "    if isinstance(value, float) and not math.isfinite(value):\n"
        "        return {\"type\": \"python_float\", \"value\": str(value)}\n"
        "    if value is None or isinstance(value, (str, int, float, bool)):\n",1)
    assert new == expected, "Runtime code delta is larger/different than the metadata-only branch"
    expected_diff = "".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                               fromfile="v1/operator_trace.py",tofile="v2/operator_trace.py"))
    assert (here / "OPERATOR_TRACE.diff").read_text() == expected_diff
    delta = json.loads((here / "CODE_DELTA.json").read_text())
    check_pin(prior / "operator_trace.py",delta["operator_trace_v1"])
    check_pin(here / "operator_trace.py",delta["operator_trace_v2"])
    for name in ("diagnose.py","LIMITS.json"):
        assert (here / name).read_bytes() == (prior / name).read_bytes(), name
    before = scalar_metadata_function(prior / "operator_trace.py")
    after = scalar_metadata_function(here / "operator_trace.py")
    finite = [None,True,False,1,-1,"finite",0.0,-0.0,1.25,1e308,5e-324]
    for value in finite:
        assert after(value) is value
    assert json.dumps(after(finite),allow_nan=False) == json.dumps(before(finite),allow_nan=False)
    tags = []
    for value,text in ((float("inf"),"inf"),(float("-inf"),"-inf"),(float("nan"),"nan")):
        original = value
        tag = after(value)
        assert tag == {"type":"python_float","value":text}
        assert value is original and not math.isfinite(value)
        json.dumps(tag,allow_nan=False)
        tags.append(tag)
    try:
        json.dumps(before(float("inf")),allow_nan=False)
    except ValueError as exc:
        assert "Out of range float values" in str(exc)
    else:
        raise AssertionError("V1 scalar-JSON failure was not reproduced")
    arguments = [float("inf"),{"nested":(float("-inf"),float("nan"),-0.0),"finite":1.25}]
    identities = (id(arguments),id(arguments[1]),id(arguments[1]["nested"]))
    encoded = json.dumps(after(arguments),allow_nan=False)
    assert identities == (id(arguments),id(arguments[1]),id(arguments[1]["nested"]))
    assert math.isinf(arguments[0]) and math.isinf(arguments[1]["nested"][0])
    assert math.isnan(arguments[1]["nested"][1]) and arguments[1]["nested"][2].hex() == "-0x0.0p+0"
    return {"literal_runtime_delta_verified":True,"driver_and_limits_byte_identical":True,
            "nonfinite_metadata_tags":tags,"finite_metadata_and_signed_zero_unchanged":True,
            "nested_argument_containers_unmodified":True,"nested_strict_JSON_roundtrip":json.loads(encoded),
            "v1_scalar_JSON_error_reproduced_with_stdlib_only":True,
            "original_overload_call_and_exception_paths_literal_unchanged":True,
            "extracted_scalar_metadata_only":True,"Tensor_sparse_native_or_kernel_tested":False}


def run():
    here = Path(__file__).resolve().parent
    research = here.parent
    parsed = []
    for path in sorted(here.glob("*.py")):
        source = path.read_text()
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        parsed.append(path.name)
    for path in sorted(here.glob("*.json")):
        json.loads(path.read_text())
    bindings = json.loads((here / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["files"]:
        path = (research / row["relative_path"]).resolve()
        assert path.is_relative_to(research)
        check_pin(path, row)
    original = json.loads((research / bindings["native_bindings_relative_path"]).read_text())
    for row in original["files"]:
        path = (research / row["relative_path"]).resolve()
        assert path.is_relative_to(research)
        check_pin(path, row)
    payloads = 0
    manifests = bindings["manifests"] + original["manifests"]
    for row in manifests:
        root = (research / row["packet"]).resolve()
        assert root.is_relative_to(research)
        manifest = root / "MANIFEST.json"
        assert sha256(manifest.read_bytes()).hexdigest() == row["sha256"]
        for entry in json.loads(manifest.read_text())["files"]:
            path = (root / entry["path"]).resolve()
            assert path.is_relative_to(root)
            check_pin(path, entry)
            payloads += 1
    plan = json.loads((here / "PLAN.json").read_text())
    release = json.loads((here / "ROOT_RELEASE.example.json").read_text())
    limits = json.loads((here / "LIMITS.json").read_text())
    diagnose = ast.parse((here / "diagnose.py").read_text())
    case_assignment = next(n for n in diagnose.body if isinstance(n, ast.Assign)
                           and any(isinstance(t,ast.Name) and t.id == "CASES" for t in n.targets))
    cases = ast.literal_eval(case_assignment.value)
    assert list(cases) == [row["case"] for row in plan["cases"]]
    controls = json.loads((here / "ROOT_RELEASE_controls.example.json").read_text())
    assert [row["case"] for row in release["authorized_invocations"]] == ["exact"]
    assert list(cases) == [row["case"] for row in release["authorized_invocations"]+controls["authorized_invocations"]]
    assert release["limits"] == controls["limits"] == limits
    assert release["execution_enabled"] is False and controls["execution_enabled"] is False
    assert all("diagnostic_execution_root_20261004_v2/" in row["output_directory"]
               for row in release["authorized_invocations"]+controls["authorized_invocations"])
    failure = json.loads((research / next(row["relative_path"] for row in bindings["files"]
                                         if row["key"] == "failed_receipt")).read_text())
    assert release["cuda_visible_devices"] == failure["runtime_identity"]["profile"]["CUDA_VISIBLE_DEVICES"]
    assert failure["status"] == "FAIL" and failure["data_files_opened"] == [] and failure["fits"] == 0
    observer_failure = json.loads((research / next(row["relative_path"] for row in bindings["files"]
                                                 if row["key"] == "observer_v1_receipt")).read_text())
    assert observer_failure["status"] == "ERROR_OBSERVED_NO_QUALIFICATION"
    assert observer_failure["exception_type"] == "ValueError"
    assert observer_failure["exception"] == "Out of range float values are not JSON compliant: inf"
    assert observer_failure["native_qualification_pass"] is False and observer_failure["fits"] == 0
    assert observer_failure["identity"]["diagnostic_manifest_sha256"] == bindings["diagnostic_v1_manifest_sha256"]
    metadata_check = literal_delta_and_scalar_check(here,research / bindings["diagnostic_v1_packet"])
    return {"status":"PASS_STATIC_METADATA_AND_CUSTODY_ONLY","AST_and_compile_files":parsed,
            "diagnostic_source_and_failure_pins_verified":len(bindings["files"]),
            "original_source_pins_verified":len(original["files"]),
            "dependency_manifests_verified":len(manifests),
            "dependency_manifest_payload_entries_verified":payloads,
            "disabled_release_cases_and_bounds_checked":True,
            "disabled_exact_first_release_and_distinct_controls":True,"metadata_check":metadata_check,
            "v1_observer_failure_preserved_without_kernel_attribution":True,
            "project_or_Torch_imported":False,"native_or_GPU_executed":False,
            "dataset_arrays_opened":False,"old_checkpoints_or_science_outcomes_opened":False,
            "kernel_cause_determined":False,"native_qualification_pass":False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
