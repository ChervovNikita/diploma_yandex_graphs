"""Source/stdlib checks only. Never imports scientific sources or fits a model."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "efficient_graph_ensemble_ports_preparation_v1"
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
    return dict(schema="efficient-graph-control-source-checks-v1", status="passed_source_only",
        ast_and_compile=parsed, unchanged_reused_sources=reused, integer_source_counts=counts,
        fixed_plan_cells=21, protocol_sha256=protocol["PROTOCOL_SHA256"],
        protocol_invalid_cases_rejected=len(protocol_rejections), admission_synthetic_cases_rejected=admission_rejections,
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
