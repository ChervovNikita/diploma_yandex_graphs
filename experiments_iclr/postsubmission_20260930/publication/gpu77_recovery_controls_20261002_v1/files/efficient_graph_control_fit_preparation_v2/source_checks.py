"""Source/stdlib checks only. Never imports scientific sources or fits a model."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "efficient_graph_ensemble_ports_preparation_v1"
PREVIOUS = ROOT.parent / "efficient_graph_control_fit_preparation_v1"
PREVIOUS_MANIFEST_SHA256 = "7672fcb1601dada27bb6f015df86eca05eb946563b23e9448dc97a6ad4b1ce39"
REUSED = ("ports.py", "qualifications.py", "test_ports.py", "native_polyformer.py",
    "native_polyformer_outer.py", "native_polynormer.py", "sources/gat_conv_2_7_pinned.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject(call):
    try:
        call()
    except (ValueError, KeyError, TypeError):
        return True
    raise AssertionError("Invalid synthetic protocol metadata was accepted")


def previous_packet_checks(protocol):
    """Seal preservation and AST equality outside the two authorized repairs."""
    manifest_path = PREVIOUS / "MANIFEST.json"
    assert sha(manifest_path) == PREVIOUS_MANIFEST_SHA256
    previous_manifest = json.loads(manifest_path.read_text())
    for entry in previous_manifest["files"]:
        path = PREVIOUS / entry["path"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
    previous_protocol = runpy.run_path(str(PREVIOUS / "control_protocol.py"))
    for backend in ("vmap", "sequential"):
        assert protocol["fixed_plan"](packed_backend=backend) == previous_protocol["fixed_plan"](packed_backend=backend)
    for filename in ("control_protocol.py", "fit_controls.py"):
        trees = [ast.parse((folder / filename).read_text()) for folder in (PREVIOUS, ROOT)]
        for tree in trees:
            for node in tree.body:
                if filename == "control_protocol.py" and isinstance(node, ast.FunctionDef) and node.name == "validate_admission":
                    node.body = [ast.Pass()]
                if filename == "fit_controls.py" and isinstance(node, ast.ClassDef) and node.name == "_Recorder":
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and method.name == "charge":
                            method.body = [ast.Pass()]
        assert ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False)
    return dict(previous_manifest_sha256=PREVIOUS_MANIFEST_SHA256,
        previous_files_verified=len(previous_manifest["files"]), previous_packet_unchanged=True,
        both_backend_21_cell_plans_identical=True,
        protocol_ast_except_validate_admission_identical=True,
        fitter_ast_except_charge_identical=True)


def recorder_charge_checks():
    """Actual fitter charge method, mocked synchronization; no scientific runtime."""
    scientific_roots = {"torch", "torch_geometric", "numpy", "scipy", "pandas", "dgl", "sklearn"}
    assert not any(name.split(".")[0] in scientific_roots for name in sys.modules)
    original_import, original_path = builtins.__import__, list(sys.path)

    def stdlib_import(name, *args, **kwargs):
        if name.split(".")[0] in scientific_roots:
            raise AssertionError(f"Scientific import forbidden in source checks: {name}")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = stdlib_import
        sys.path.insert(0, str(ROOT))
        namespace = runpy.run_path(str(ROOT / "fit_controls.py"))
        assert Path(namespace["_protocol_module"].__file__).resolve() == ROOT / "control_protocol.py"
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path
    assert not any(name.split(".")[0] in scientific_roots for name in sys.modules)
    recorder_type = namespace["_Recorder"]
    cases = [("success", None, None),
        ("body_only", ValueError("mock body failure"), None),
        ("synchronization_only", None, RuntimeError("mock synchronization failure")),
        ("body_and_synchronization", ValueError("mock body failure"), RuntimeError("mock synchronization failure")),
        ("base_exception_and_synchronization", KeyboardInterrupt("mock body interruption"), RuntimeError("mock synchronization failure"))]
    rows = []
    for name, body_error, synchronization_error in cases:
        recorder = object.__new__(recorder_type)  # No __init__, output writes or device access.
        recorder.intervals, recorder.active_stage = [], "mock_stage"
        calls, bodies = [], []

        def synchronize():
            calls.append(len(calls) + 1)
            if len(calls) == 2 and synchronization_error is not None:
                raise synchronization_error

        recorder.synchronize = synchronize
        caught = None
        try:
            with recorder.charge("mock_interval"):
                bodies.append(True)
                if body_error is not None:
                    raise body_error
        except BaseException as error:
            caught = error
        assert calls == [1, 2] and bodies == [True] and len(recorder.intervals) == 1
        expected = synchronization_error if synchronization_error is not None else body_error
        assert caught is expected
        if synchronization_error is not None and body_error is not None:
            assert caught.__cause__ is body_error and caught.__context__ is body_error
        interval = recorder.intervals[0]
        assert interval["body_error"] == (f"{type(body_error).__name__}: {body_error}" if body_error is not None else None)
        assert interval["synchronization_error"] == (f"{type(synchronization_error).__name__}: {synchronization_error}" if synchronization_error is not None else None)
        assert interval["category"] == "mock_interval" and interval["stage"] == "mock_stage"
        assert interval["wall_seconds"] >= 0 and interval["cpu_seconds"] >= 0
        rows.append(dict(case=name, raised_expected_exception_identity=True,
            body_error_recorded=interval["body_error"], synchronization_error_recorded=interval["synchronization_error"],
            body_retained_as_explicit_cause=synchronization_error is not None and body_error is not None))
    assert not any(name.split(".")[0] in scientific_roots for name in sys.modules)
    return dict(actual_method="fit_controls._Recorder.charge", mock_synchronization=True,
        scientific_imports=False, bounded_cases=len(rows), cases=rows)


def checks():
    parsed = {}
    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")  # compile only; no imported body execution
        parsed[str(path.relative_to(ROOT))] = dict(sha256=sha(path), ast="passed", compiled="passed")
    reused = []
    for relative in REUSED:
        assert (ROOT / relative).read_bytes() == (SOURCE / relative).read_bytes()
        reused.append(dict(path=relative, sha256=sha(ROOT / relative), unchanged_bytes=True))
    # Execute only this standard-library metadata module. Scientific imports stay untouched.
    protocol = runpy.run_path(str(ROOT / "control_protocol.py"))
    spec_fn, plan_fn = protocol["specification"], protocol["fixed_plan"]
    plan = plan_fn(packed_backend="vmap")
    assert len(plan) == 21
    assert len({protocol["digest"](spec) for spec in plan}) == 21
    assert {spec["seed"] for spec in plan} == {17, 29, 43}
    assert {spec["backbone"] for spec in plan} == {"polyformer_mono", "polynormer_r"}
    assert all(spec["source_split"] == [17, 29, 43].index(spec["seed"]) for spec in plan)
    assert len(plan_fn(packed_backend="sequential")) == 21
    protocol_rejections = [reject(lambda: spec_fn("polynormer_r", "cached_token_mimo_m4", 17)),
        reject(lambda: spec_fn("polyformer_mono", "fixed_cap_packed_m4", 17)),
        reject(lambda: spec_fn("polyformer_mono", "cfg0_single", 19)),
        reject(lambda: spec_fn("polyformer_mono", "cfg0_single", 17, backend="vmap"))]
    # Extract just the two integer algebra functions from ports; no port imports.
    port_tree = ast.parse((ROOT / "ports.py").read_text())
    algebra = ast.Module(body=[node for node in port_tree.body if isinstance(node, ast.FunctionDef)
        and node.name in ("native_parameter_count", "mimo_parameter_count")], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(algebra), "<source-count-algebra>", "exec"), namespace)
    native, mimo = namespace["native_parameter_count"], namespace["mimo_parameter_count"]
    counts = dict(polyformer_native=native("polyformer_mono", 256),
        polynormer_native=native("polynormer_r", 512),
        polyformer_native_m4=4 * native("polyformer_mono", 256),
        polynormer_native_m4=4 * native("polynormer_r", 512),
        polyformer_packed=4 * native("polyformer_mono", 116),
        polyformer_packed_next=4 * native("polyformer_mono", 120),
        polynormer_packed=4 * native("polynormer_r", 248),
        polynormer_packed_next=4 * native("polynormer_r", 256),
        polyformer_mimo=mimo(208), polyformer_mimo_next=mimo(212))
    assert counts["polyformer_packed"] == 4177828 <= 4430644 < counts["polyformer_packed_next"]
    assert counts["polynormer_packed"] == 7707904 <= 7773732 < counts["polynormer_packed_next"]
    assert counts["polyformer_mimo"] == 4308428 <= 4430644 < counts["polyformer_mimo_next"]
    # Synthetic root metadata exercises admission without scientific imports or data.
    spec = plan[0]
    full = dict(schema="efficient-graph-control-root-admission-v1", scientific_decision="frozen_admitted",
        mode="full", decision_id="SYNTHETIC-NOT-ADMISSION", attempt_id="SYNTHETIC-NOT-RUN",
        protocol_sha256=protocol["PROTOCOL_SHA256"], spec_sha256=protocol["digest"](spec),
        packet_manifest_sha256="1" * 64, prepared_graph_sha256="2" * 64,
        train_pack_sha256="3" * 64, validation_pack_sha256="4" * 64,
        prior_graph_init_closure=dict(closed=True, receipt_sha256="7" * 64),
        runtime_binding=dict(torch_version="SYNTHETIC", pyg_version="SYNTHETIC", device="cpu",
            environment_sha256="6" * 64))
    full["root_qualification"] = dict(passed=True, spec_sha256=protocol["digest"](spec),
        runtime_binding=full["runtime_binding"], receipt_sha256="5" * 64,
        packet_manifest_sha256=full["packet_manifest_sha256"],
        prepared_graph_sha256=full["prepared_graph_sha256"],
        train_pack_sha256=full["train_pack_sha256"],
        validation_pack_sha256=full["validation_pack_sha256"],
        passed_checks=protocol["qualification_checks"](spec))
    validate = lambda request: protocol["validate_admission"](spec, request, mode="full", manifest_sha256="1" * 64)
    assert validate(full) == full
    admission_rejections = []
    for key, value in (("scientific_decision", "pending"), ("protocol_sha256", "0" * 64),
                       ("mode", "qualify"), ("spec_sha256", "0" * 64),
                       ("packet_manifest_sha256", "0" * 64), ("train_pack_sha256", "bad")):
        mutated = copy.deepcopy(full)
        mutated[key] = value
        admission_rejections.append(dict(field=key, rejected=reject(lambda r=mutated: validate(r))))
    for key, value in (("passed", False), ("passed_checks", []), ("receipt_sha256", "bad"),
                       ("runtime_binding", dict(torch_version="OTHER", pyg_version="SYNTHETIC", device="cpu",
                           environment_sha256="6" * 64))):
        mutated = copy.deepcopy(full)
        mutated["root_qualification"][key] = value
        admission_rejections.append(dict(field="root_qualification." + key,
            rejected=reject(lambda r=mutated: validate(r))))
    unclosed = copy.deepcopy(full)
    unclosed["prior_graph_init_closure"]["closed"] = False
    admission_rejections.append(dict(field="prior_graph_init_closure.closed",
        rejected=reject(lambda: validate(unclosed))))
    qualification_binding_rejections = []
    for key in ("packet_manifest_sha256", "prepared_graph_sha256", "train_pack_sha256", "validation_pack_sha256"):
        wrong_identity = copy.deepcopy(full)
        wrong_identity["root_qualification"][key] = "8" * 64
        qualification_binding_rejections.append(dict(field="root_qualification." + key,
            case="well_formed_wrong_identity", otherwise_genuine_looking_metadata=True,
            rejected=reject(lambda r=wrong_identity: validate(r))))
        missing_identity = copy.deepcopy(full)
        del missing_identity["root_qualification"][key]
        qualification_binding_rejections.append(dict(field="root_qualification." + key,
            case="missing_identity", rejected=reject(lambda r=missing_identity: validate(r))))
    fitter = (ROOT / "fit_controls.py").read_text()
    fit_tree = ast.parse(fitter)
    allowed_imports = {"__future__", "contextlib", "copy", "hashlib", "importlib", "inspect", "json", "math",
        "os", "pathlib", "random", "resource", "sys", "time", "types", "control_protocol"}
    for node in fit_tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split('.')[0] in allowed_imports for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] in allowed_imports
    fit_node = next(node for node in fit_tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_control")
    assert [arg.arg for arg in fit_node.args.args] == ["spec", "graph", "train", "validation"]
    assert fit_node.args.vararg is None and fit_node.args.kwarg is None
    assert fitter.index("admission = validate_admission(") < fitter.index("torch, pyg, ports = _runtime()")
    assert 'improved = value < best_nll' in fitter
    assert 'for epoch in range(cap + 1)' in fitter
    assert 'optimizer.load_state_dict(selected["adam_state"])' in fitter
    assert 'model.load_state_dict(selected["model_state"], strict=True)' in fitter
    assert 'epoch - best_epoch >= context["patience"]' in fitter
    assert 'reducer="common_logit"' in fitter and 'reducer="native_probability"' not in fitter
    assert 'ports.independent_tuple_positions(len(train.nodes)' in fitter
    assert 'ports.mimo_loss(model.forward_tuples(tuples), tuples)' in fitter
    assert 'check_mimo_correspondence(' in fitter
    previous_checks = previous_packet_checks(protocol)
    charge_checks = recorder_charge_checks()
    return dict(schema="efficient-graph-control-source-checks-v2", status="passed_source_and_stdlib_mock_checks_only",
        ast_and_compile=parsed, unchanged_reused_sources=reused, integer_source_counts=counts,
        fixed_plan_cells=21, protocol_sha256=protocol["PROTOCOL_SHA256"],
        protocol_invalid_cases_rejected=len(protocol_rejections), admission_synthetic_cases_rejected=admission_rejections,
        qualification_binding_cases_rejected=qualification_binding_rejections,
        previous_packet_and_semantics=previous_checks, actual_recorder_charge_mock_checks=charge_checks,
        structural_checks=dict(delayed_scientific_import=True, exact_train_validation_API=True,
            strict_selection=True, epoch_zero_eligibility=True, native_patience=True,
            model_and_adam_restore=True, common_logit_only_selection=True,
            compact_mimo_training_and_correspondence=True),
        not_run=["scientific_imports", "model_construction", "models", "datasets", "labels", "checkpoints",
            "numerical_torch_tests", "GPU", "SSH", "runtime_fits"],
        limitation="AST/string checks and integer algebra do not establish numerical/runtime correctness")


if __name__ == "__main__":
    result = checks()
    (ROOT / "SOURCE_CHECKS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], python_sources=len(result["ast_and_compile"]),
        unchanged_sources=len(result["unchanged_reused_sources"]), fixed_plan_cells=result["fixed_plan_cells"])))
