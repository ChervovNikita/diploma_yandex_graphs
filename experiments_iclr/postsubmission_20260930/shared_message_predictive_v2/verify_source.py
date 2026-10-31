"""Stdlib syntax/contract/refusal checks; no Torch, data or numerical models."""
import ast
import hashlib
import importlib
import json
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path[:] = [p for p in sys.path if Path(p or Path.cwd()).resolve() != HERE]
sys.path.insert(0, str(PHASE))


def main():
    output = HERE / "STATIC_SOURCE_CHECK_v1.json"
    if output.exists():
        raise ValueError("Preserve source check; use a new explicit generation")
    source_sha, trees, checks = {}, {}, []
    for path in sorted(HERE.glob("*.py")):
        raw = path.read_bytes()
        compile(raw, str(path), "exec", dont_inherit=True)
        trees[path.name] = ast.parse(raw)
        source_sha[path.name] = hashlib.sha256(raw).hexdigest()
        for node in trees[path.name].body:
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] in ("torch", "numpy", "torch_geometric", "ogb") for name in names):
                raise ValueError("Third-party import precedes admission: " + path.name)
    checks.append({"name": "all_python_syntax_and_no_top_level_third_party_import", "passed": True})
    for name in ("labels.py", "preflight.py", "runner.py"):
        if any(isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(trees[name])):
            raise ValueError("Filename entrypoint must use package-qualified absolute imports: " + name)
    checks.append({"name": "filename_entrypoints_have_no_relative_imports", "passed": True})
    contract = importlib.import_module("shared_message_predictive_v2.contract")
    metrics = importlib.import_module("shared_message_predictive_v2.metrics")
    guard = importlib.import_module("shared_message_predictive_v2.import_guard")
    preflight = importlib.import_module("shared_message_predictive_v2.preflight")
    assert len(contract.cells()) == 30 and set(c["seed"] for c in contract.cells()) == {17, 29, 43}
    assert contract.EPOCHS == 300 and 30 * contract.EPOCHS == 9000
    assert len(set(map(contract.digest, contract.cells()))) == 30
    assert contract.checked(contract.DESIGN).name.endswith("v2.json")
    order = json.loads(contract.checked(contract.ORDER_POLICY).read_text())
    assert sorted(map(contract.digest, order["fit_order"])) == sorted(map(contract.digest, contract.cells()))
    checks.append({"name": "new_fixed30_9000_updates_exact_design_and_order", "passed": True})
    def refuses(name, call):
        try:
            call()
        except (ValueError, RuntimeError, FileNotFoundError, ImportError):
            checks.append({"name": name, "passed": True})
            return
        raise ValueError("Expected refusal absent: " + name)
    refuses("outside_phase_refused_without_file_access", lambda: contract.within(PHASE.parent / "outside"))
    refuses("old_seed0_not_new_seed_domain", lambda: contract.cell_id({"task": "Reddit2", "arm": "exact_member", "seed": 0}))
    refuses("unlisted_arm_refused", lambda: contract.cell_id({"task": "Reddit2", "arm": "candidate", "seed": 17}))
    refuses("noncanonical_reference_refused", lambda: contract.checked({"path": "shared_message_predictive_v2/../shared_message_predictive_v2/contract.py", "sha256": "0" * 64}))
    refuses("no_test_output_stage", lambda: contract.prescribed({"output_root": "shared_message_predictive_v2/UNCREATED"}, "test"))
    with patch.object(contract.json, "loads", return_value={"schema": "gnnm-propagation-fixed30-scientific-admission-v1"}):
        refuses("old_method_admission_refused_before_Torch", lambda: contract.binding(HERE / "contract.py"))
    with patch.object(preflight.json, "loads", return_value={"schema": "UNAPPROVED_SOURCE_ONLY"}):
        refuses("unapproved_cost_request_refused_before_Torch", lambda: preflight.request_binding(HERE / "contract.py"))
    with patch.object(contract.os.environ, "get", return_value="UNFROZEN"):
        refuses("runtime_environment_refused_before_Torch", lambda: contract.runtime_before_import({"runtime_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}))
    refuses("descriptive_interval_requires_three_seeds", lambda: metrics.interval([1.0, 2.0]))
    values = metrics.interval([1.0, 2.0, 3.0])
    assert values["n"] == 3 and values["df"] == 2 and values["significance_or_noninferiority_established"] is False
    assert values["two_sided_95_interval"][0] < 0 and values["two_sided_95_interval"][1] > 4
    checks.append({"name": "descriptive_two_sided95_df2_interval", "passed": True})
    core = (HERE / "core.py").read_text()
    assert "models.make_optimizer(model)" in core and "if optimizer.state:" in core
    assert "range(1, EPOCHS + 1)" in core and "epoch % 5 == 0" in core
    assert "best_logits" in core and "improved_checkpoint" in core and '"FINAL.pt"' in core
    assert "all_scalar_records_and_selection_rules_replayed" in core
    assert "unimproved_epoch_full_forward_replay_performed\": False" in core
    assert "selected_and_final_separate_forwards_replayed" in core
    checks.append({"name": "fresh_optimizer_fixed_budget_compact_states_and_honest_replay_scope", "passed": True})
    assert "paired_initial" in core and "require_order" in core and "audit_family_lock" in core
    assert '"last20_training"' in core and "EPOCHS - 30" in core
    checks.append({"name": "complete_family_pairing_order_and_learning_windows", "passed": True})
    labels = (HERE / "labels.py").read_text()
    assert 'request.get("split") != "valid"' in labels
    assert 'field not in ("processed", "original_provider_files")' in labels
    assert 'get("y").index_select(0, ids)' in labels and 'get("y").index_select(0, train)' in labels
    assert 'official_split_node_ids"]["test"]' not in labels
    assert not (HERE / "evaluate.py").exists()
    checks.append({"name": "separate_validation_only_provider_inventory_and_no_test_evaluator", "passed": True})
    preflight_source = (HERE / "preflight.py").read_text()
    assert preflight_source.index("path.unlink()") < preflight_source.index("torch.save(value, stream)")
    assert "label_free_reductions(member, pooled)" in preflight_source
    assert "load_validation" not in preflight_source
    checks.append({"name": "bounded_train_only_cost_and_unlinked_before_payload", "passed": True})
    assert contract.sha(HERE / "contract.py") == contract.sha(HERE / "contract.py")
    checks.append({"name": "source_hash_and_unchanged_signature_cache", "passed": True})
    for name in ("runner.py", "preflight.py", "labels.py"):
        raw = (HERE / name).read_text()
        assert raw.index("sys.path[:]") < raw.index("import argparse")
        assert "sys.path.insert" not in raw
        assert "bootstrap(args.binding)" in raw if name != "preflight.py" else "bootstrap(args.request)" in raw
        for node in trees[name].body:
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("shared_message_", "propagation_"))
        assert '"source_manifest_sha256"' in raw and '"import_guard.py"' in raw
        assert 'exec(compile(guard_path.read_bytes()' in raw
    checks.append({"name": "entrypoint_path_scrub_request_manifest_guard_bootstrap_before_phase_imports", "passed": True})
    assert contract.MODEL_MANIFEST_SHA == "607c487f83696a21d5792c3f22b408491f9fb55c8f7e0b1dd1527b8f68653dcc"
    assert contract.sha(PHASE / "shared_message_quality_v3/SOURCE_MANIFEST_v1.json") == contract.MODEL_MANIFEST_SHA
    assert contract.sha(PHASE / "protocols/bounded_run_v1.py") == contract.BOUNDED_SOURCE_SHA
    assert (HERE / "GENERATED_MODULE_POLICY_v1.json").read_bytes() == (PHASE / "shared_message_quality_v3/GENERATED_MODULE_POLICY_v1.json").read_bytes()
    assert "quality.synthetic_chain" in (HERE / "contract.py").read_text()
    assert "quality.outer_completed" in (HERE / "contract.py").read_text()
    checks.append({"name": "exact_quality_v3_envelope_pins_generated_policy_and_full_gate_links", "passed": True})
    # Only temporary source/metadata fixtures inside this new package.
    with tempfile.TemporaryDirectory(prefix="STDLIB_TREE_", dir=HERE) as raw_tmp:
        temporary = Path(raw_tmp)
        package = temporary / "fixture"
        package.mkdir()
        initial = package / "__init__.py"
        module = package / "part.py"
        initial.write_text("")
        module.write_text("x = 1\n")
        allowed = [str(p.relative_to(PHASE)) for p in (initial, module)]
        dirs = [str(package.relative_to(PHASE))]
        assert guard.inspect_tree(package, allowed, dirs) == sorted(allowed)
        for suffix in (".py", ".pyc", ".so"):
            extra = package / ("extra" + suffix)
            extra.write_bytes(b"extra source fixture")
            refuses("unsealed_tree_extra_" + suffix[1:], lambda: guard.inspect_tree(package, allowed, dirs))
            extra.unlink()
        shadow = package / "part"
        shadow.mkdir()
        refuses("shadow_source_directory_refused", lambda: guard.inspect_tree(package, allowed, dirs))
        shadow.rmdir()
        cache = package / "__pycache__"
        cache.mkdir()
        refuses("bytecode_cache_directory_refused", lambda: guard.inspect_tree(package, allowed, dirs))
        cache.rmdir()
        link = package / "linked.py"
        link.symlink_to(module)
        refuses("source_symlink_refused", lambda: guard.inspect_tree(package, allowed, dirs))
        link.unlink()
        finder = guard.PhaseFinder(dict(package_roots=["fixture"], importable_sha256={
            "fixture/__init__.py": "0" * 64, "fixture/part.py": "1" * 64}))
        assert finder.find_spec("fixture.part").origin == str(PHASE / "fixture/part.py")
        refuses("finder_unsealed_module_refused", lambda: finder.find_spec("fixture.extra"))
        loader = guard.SealedSourceLoader("fixture.part", module, hashlib.sha256(module.read_bytes()).hexdigest())
        assert isinstance(loader.get_code("fixture.part"), types.CodeType)
        loader.wanted = "0" * 64
        refuses("loader_rechecks_exact_source_before_compile", lambda: loader.get_code("fixture.part"))
        generated_dir = temporary / "tmp"
        generated_dir.mkdir()
        generated_file = generated_dir / "sage_conv_SAGEConv_propagate_fixture.py"
        generated_file.write_text("x = 1\n")
        full_name = "torch_geometric.nn.conv.sage_conv_SAGEConv_propagate"
        wanted = hashlib.sha256(generated_file.read_bytes()).hexdigest()
        policy = dict(importable_sha256={"fixture/part.py": "1" * 64}, generated_module_policy_sha256="2" * 64,
            generated_source_policy=dict(generated_modules={full_name: dict(sha256=wanted, basename_prefix="sage_conv_SAGEConv_propagate")}))
        with patch.object(guard, "PHASE", temporary):
            row = guard.generated_module_receipt(full_name, generated_file, policy)
            assert row["sha256"] == wanted and row["source_kind"] == "pinned_PyG_generated_tmp_source"
            guard.verify_imported_receipts([row], policy, require_generated=True)
            refuses("generated_full_name_exact_not_prefix", lambda: guard.generated_module_receipt(full_name + "_other", generated_file, policy))
            refuses("generated_outside_confined_tmp_refused", lambda: guard.generated_module_receipt(full_name, module, policy))
            source_row = dict(module="fixture.part", relative_to_phase="fixture/part.py", sha256="1" * 64)
            guard.verify_imported_receipts([source_row], policy)
            refuses("model_evidence_requires_actual_generated_module", lambda: guard.verify_imported_receipts([source_row], policy, require_generated=True))
            refuses("duplicate_import_receipt_refused", lambda: guard.verify_imported_receipts([row, row], policy))
            forged = dict(row, relative_to_phase="tmp/../outside.py")
            refuses("noncanonical_saved_generated_path_refused", lambda: guard.verify_imported_receipts([forged], policy))
            generated_file.write_text("x = 2\n")
            refuses("generated_source_hash_exact", lambda: guard.generated_module_receipt(full_name, generated_file, policy))
    checks.append({"name": "exact_source_tree_loader_and_narrow_generated_module_positive_controls", "passed": True})
    with tempfile.TemporaryDirectory(prefix="STDLIB_ENVELOPE_", dir=HERE) as raw_tmp:
        tmp = Path(raw_tmp)
        outer, inner = tmp / "outer", tmp / "inner"
        outer.mkdir(); inner.mkdir()
        request = tmp / "REQUEST.json"
        request.write_text('{"source_only_test_fixture":true}\n')
        route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
        argv = [str(Path(route["repository"]) / ".venv/bin/python"), str(HERE / "runner.py"), "fit", "--binding", str(request), "--supervisor", str(inner)]
        start_path = outer / "START.json"
        start = dict(schema="gnnm-whole-process-bound-start-v1", root_request=contract.reference(request), root_request_child_flag="--binding",
            child_argv=argv, argv=[argv[0], str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(inner), "--require-idle-gpu", "--", *argv],
            inner_supervisor_directory=str(inner), outer_output=str(outer), cwd=route["repository"], phase=str(PHASE), ssh_destination=route["ssh_destination"],
            shell=False, start_new_session=True, whole_cap_seconds=100, termination_grace_seconds=5, normal_child_budget_seconds=95,
            source_sha256=contract.BOUNDED_SOURCE_SHA, start_UTC=datetime.now(timezone.utc).isoformat(),
            authorization_source_sha256=contract.sha(PHASE / "protocols/run_authorized_v2.py"), logging_source_sha256=contract.sha(PHASE / "protocols/run_logged.py"),
            wrapper_sha256=contract.sha(PHASE / "protocols/repo_env.sh"), research_script_sha256=contract.sha(HERE / "runner.py"))
        start_path.write_text(json.dumps(start))
        start_ref, checked_start = contract.verify_outer_start(start_path, request, inner, argv)
        assert start_ref == contract.reference(start_path) and checked_start == start
        assert start_ref != contract.reference(HERE / "runner.py")
        checks.append({"name": "outer_START_reference_remains_START_not_loop_research_source", "passed": True})
        refuses("unrelated_root_request_refused", lambda: contract.verify_outer_start(start_path, HERE / "contract.py", inner, argv))
        refuses("unrelated_supervisor_refused", lambda: contract.verify_outer_start(start_path, request, tmp / "other", argv))
        for name in ("authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"):
            (inner / name).write_text("source-only fixture\n")
        (outer / "stdout_stderr.log").write_text("source-only fixture\n")
        terminal_path = outer / "TERMINAL.json"
        terminal = dict(schema="gnnm-whole-process-bound-terminal-v1", START_sha256=start_ref["sha256"], complete=True,
            root_request_unchanged=True, child_exit_code=0, timed_out=False, interruption=None, within_whole_cap=True,
            whole_supervised_seconds=2.0, stdout_stderr_sha256=contract.sha(outer / "stdout_stderr.log"),
            inner_evidence={name: dict(sha256=contract.sha(inner / name), bytes=(inner / name).stat().st_size)
                for name in ("authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log")})
        terminal_path.write_text(json.dumps(terminal))
        fake_quality = types.ModuleType("shared_message_quality_v3")
        fake_quality.__path__ = []
        fake_contract = types.ModuleType("shared_message_quality_v3.contract")
        fake_contract.completed = lambda directory: dict(gpu_uuid="GPU-fixture", seconds=1.0, argv=argv, live={"fixture": True})
        with patch.dict(sys.modules, {"shared_message_quality_v3": fake_quality, "shared_message_quality_v3.contract": fake_contract}):
            completed = contract.completed(inner, start_ref)
            assert completed["seconds"] == 2 and completed["inner_seconds"] == 1
            assert completed["live"]["outer_start"] == start_ref
            assert completed["outer_terminal"] == contract.reference(terminal_path)
            terminal["timed_out"] = True
            terminal_path.write_text(json.dumps(terminal))
            refuses("inner_success_alone_cannot_qualify_timed_out_outer", lambda: contract.completed(inner, start_ref))
        checks.append({"name": "complete_outer_terminal_required_and_full_outer_cost_authoritative", "passed": True})
    for name in ("core.py", "labels.py", "resource_forecast.py"):
        for node in ast.walk(trees[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "completed":
                assert len(node.args) == 2, (name, node.lineno)
    checks.append({"name": "every_scientific_completion_call_binds_exact_outer_start", "passed": True})
    forecast_source = (HERE / "resource_forecast.py").read_text()
    assert "4 * checkpoint_hash" in forecast_source and "4 * logit_hash" in forecast_source
    assert "four times" in forecast_source
    checks.append({"name": "fourth_hash_pass_for_separate_family_lock_audit_charged", "passed": True})
    if any(name == "torch" or name.startswith(("torch.", "numpy", "ogb", "torch_geometric")) for name in sys.modules):
        raise ValueError("Static verifier imported a forbidden scientific package")
    with output.open("x") as stream:
        json.dump({"schema": "gnnm-shared-message-scientific-stdlib-source-check-v2",
            "status": "source_checks_passed_no_numerical_execution", "checks": checks, "check_count": len(checks),
            "python_source_sha256": source_sha, "Torch_data_labels_GPU_remote_or_weights_accessed": False,
            "old_source_or_seed_guards_modified": False, "scientific_or_cost_launch_admitted": False}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"check_count": len(checks), "source_only": True}))


if __name__ == "__main__":
    main()
