"""Local syntax/count/source closure checks only; never imports Torch/data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import copy

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def main():
    checks, hashes = [], {}
    for path in sorted(HERE.glob("*.py")):
        raw = path.read_bytes()
        compile(raw, str(path), "exec", dont_inherit=True)
        ast.parse(raw)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
    checks.append(dict(check="all_Python_AST_and_syntax_without_execution", passed=True))
    package_name = HERE.name
    package_spec = importlib.util.spec_from_file_location(package_name, HERE / "__init__.py", submodule_search_locations=[str(HERE)])
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    def load(name):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, HERE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    counts, contract, guard = load("counts"), load("contract"), load("import_guard")
    expected = {"Reddit2": (86384, 82985, 66, 86897, 513), "ogbn-products": (52680, 51247, 65, 52697, 17)}
    plans = []
    for task, values in expected.items():
        row = counts.plan(task)
        match, params = row["matched_single"], row["parameter_counts"]
        assert (params["shared_mean"], params["single_D64"], match["width"], match["single_parameters"], match["signed_mismatch"]) == values
        target = match["target_parameters"]
        chosen = match["width"]
        assert all((abs(counts.ordinary_count(counts.TASKS[task]["features"], counts.TASKS[task]["classes"], d) - target), d)
                   >= (match["absolute_mismatch"], chosen) for d in range(1, counts.MAX_SINGLE_WIDTH + 1))
        checks.append(dict(check="exact_formula_global_integer_width_minimum_" + task, passed=True))
        plans.append(row)
    assert len(counts.variants()) == 12
    assert len({counts.variant_id(v) for v in counts.variants()}) == 12
    checks.append(dict(check="all_five_arms_twelve_execution_variants_unique", passed=True))
    for name in ("contract.py", "counts.py", "data.py", "qualify.py", "synthetic_checks.py"):
        tree = ast.parse((HERE / name).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in ("torch", "numpy", "ogb", "torch_geometric") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not node.module or node.module.split(".")[0] not in ("torch", "numpy", "ogb", "torch_geometric")
    checks.append(dict(check="entrypoint_guards_before_any_top_level_third_party_import", passed=True))
    try:
        contract.within(PHASE.parent / "outside_phase")
    except ValueError:
        checks.append(dict(check="outside_phase_path_refused_without_payload_access", passed=True))
    else:
        raise AssertionError("Confinement refusal missing")
    models = (HERE / "models.py").read_text()
    qualifier = (HERE / "qualify.py").read_text()
    assert "training_colors(" not in models + qualifier
    assert "mean = mean.detach()" in models and "common[:, None, :].expand_as(normalized)" in models
    assert "class ParameterMatchedBASE(PlainSAGE)" in models
    assert "loss.backward()" in models and "optimizer.step()" in models
    assert "return_loss=False" in models
    assert "torch.save" not in qualifier and "accuracy" not in qualifier
    checks.append(dict(check="whole_detach_new_width_guard_real_backward_and_no_qualification_weight_or_score_save", passed=True))
    assert 'r["median_schedule_weighted_seconds"]' in qualifier and ' / 5' in qualifier
    assert 'median_combined_seconds"], r["id"]' not in qualifier
    checks.append(dict(check="prospective_300_60_train_plus_validation_over_five_selection", passed=True))
    # Small source/evidence fixtures only; no data/tensor/numerical execution.
    with tempfile.TemporaryDirectory(prefix="stdlib_import_fixture_", dir=HERE) as temp:
        root = Path(temp)
        source = root / "member.py"
        source.write_text("VALUE = 7\n")
        allowed = [str(source.relative_to(PHASE))]
        assert guard.inspect_tree(root, allowed, []) == allowed
        for name, kind in (("extra.py", "file"), ("extra.pyc", "file"), ("extra.so", "file"),
                           ("member", "directory"), ("redirect", "symlink")):
            path = root / name
            if kind == "directory":
                path.mkdir()
            elif kind == "symlink":
                path.symlink_to(source)
            else:
                path.write_text("source-only fixture")
            try:
                guard.inspect_tree(root, allowed, [])
            except ValueError:
                pass
            else:
                raise AssertionError("Import-tree mutation was admitted: " + name)
            path.rmdir() if kind == "directory" else path.unlink()
        loader = guard.SealedSourceLoader("source_only_fixture", source, guard.source_sha(source))
        loader.get_code("source_only_fixture")
        source.write_text("VALUE = 8\n")
        try:
            loader.get_code("source_only_fixture")
        except ImportError:
            pass
        else:
            raise AssertionError("Changed loader source was admitted")
    checks.append(dict(check="extra_Python_bytecode_extension_shadow_directory_symlink_and_changed_loader_refused", passed=True))
    with tempfile.TemporaryDirectory(prefix="stdlib_chain_fixture_", dir=HERE) as temp:
        root = Path(temp); output = root / "output"; output.mkdir(); supervisor = root / "inner"; supervisor.mkdir()
        synthetic_path = root / "synthetic.json"; synthetic_path.write_text("{}\n")
        terminal_path = root / "TERMINAL.json"; terminal_path.write_text("{}\n")
        def ref(path):
            return dict(path=str(path.relative_to(PHASE)), sha256=contract.sha(path))
        policy = dict(device="cuda:0", TF32=False, AMP=False)
        tolerance = dict(atol=1e-5, rtol=1e-5)
        synthetic = dict(output_relative_to_phase=str(output.relative_to(PHASE)), gpu_uuid="fixture-GPU",
                         runtime_policy=policy, equivalence_tolerance=tolerance, memory_cap_bytes=64)
        value = dict(synthetic_request=ref(synthetic_path), synthetic_gate={},
                     synthetic_supervisor_directory=str(supervisor.relative_to(PHASE)), synthetic_outer_terminal=ref(terminal_path),
                     gpu_uuid="fixture-GPU", runtime_policy=policy, equivalence_tolerance=tolerance, memory_cap_bytes=64)
        live = dict(command_sha256="fixture-command", environment_sha256="fixture-env", outer_start=dict(path="fixture-start", sha256="fixture-sha"))
        gate = dict(schema="gnnm-shared-message-synthetic-checks-v2", status="passed", source_manifest_sha256="fixture-manifest",
                    request_sha256="fixture-request", live=live, device="cuda:0", gpu=dict(uuid="fixture-GPU"),
                    runtime_policy=policy, observed_runtime_policy=policy, equivalence_tolerance=tolerance, check_count=1, checks=[dict(passed=True)],
                    memory_cap_bytes=64, peak_memory=dict(peak_allocated_bytes=12, peak_reserved_bytes=16),
                    runtime_import_tree={"fixture": []}, imported_phase_modules=[dict(module="fixture", relative_to_phase="fixture.py", sha256="fixture-source")])
        start = dict(live=live, operation="synthetic", request_sha256="fixture-request", source_manifest_sha256="fixture-manifest")
        saved = (contract.request, contract.completed, contract.expected_cli, contract.outer_completed, guard.verify_packet)
        contract.request = lambda path, operation: (copy.deepcopy(synthetic), "fixture-request")
        contract.completed = lambda directory: dict(argv=["fixture-cli"], gpu_uuid="fixture-GPU", live={k: v for k, v in live.items() if k != "outer_start"})
        contract.expected_cli = lambda *args: ["fixture-cli"]
        contract.outer_completed = lambda *args: live["outer_start"]
        guard.verify_packet = lambda wanted: (dict(importable_sha256={"fixture.py": "fixture-source"}), {"fixture": []})
        def save_gate(current):
            (output / "START.json").write_text(json.dumps(start))
            (output / "RESULT.json").write_text(json.dumps(current))
            manifest = dict(schema="gnnm-shared-message-engineering-artifacts-v2", files={p.name: dict(bytes=p.stat().st_size, sha256=contract.sha(p))
                for p in (output / "START.json", output / "RESULT.json")})
            (output / "artifact_manifest.json").write_text(json.dumps(manifest))
            value["synthetic_gate"] = ref(output / "RESULT.json")
        try:
            save_gate(gate)
            assert contract.synthetic_chain(value, "fixture-manifest")["request_sha256"] == "fixture-request"
            for key, changed in (("request_sha256", "other-request"), ("runtime_policy", {}), ("equivalence_tolerance", {}),
                                 ("gpu", dict(uuid="wrong-GPU")), ("live", {}), ("peak_memory", dict(peak_reserved_bytes=65))):
                mutated = copy.deepcopy(gate); mutated[key] = changed; save_gate(mutated)
                try:
                    contract.synthetic_chain(value, "fixture-manifest")
                except ValueError:
                    pass
                else:
                    raise AssertionError("Synthetic evidence mix-and-match admitted: " + key)
            save_gate(gate)
            (output / "artifact_manifest.json").write_text("{}")
            try:
                contract.synthetic_chain(value, "fixture-manifest")
            except ValueError:
                pass
            else:
                raise AssertionError("Broken artifact manifest admitted")
        finally:
            contract.request, contract.completed, contract.expected_cli, contract.outer_completed, guard.verify_packet = saved
    checks.append(dict(check="synthetic_exact_request_runtime_tolerance_GPU_live_memory_and_artifact_chain_mutations_refused", passed=True))
    with tempfile.TemporaryDirectory(prefix="stdlib_outer_fixture_", dir=HERE) as temp:
        root = Path(temp); inner = root / "inner"; inner.mkdir(); outer = root / "outer"; outer.mkdir()
        request_path = root / "request.json"; request_path.write_text("{}\n")
        admitted = dict(wall_cap_seconds=20)
        route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
        argv = contract.expected_cli(request_path, "synthetic", inner)
        start = dict(schema="gnnm-whole-process-bound-start-v1", root_request=dict(path=str(request_path.relative_to(PHASE)), sha256=contract.sha(request_path)),
            root_request_child_flag="--request", child_argv=argv,
            argv=[argv[0], str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(inner), "--require-idle-gpu", "--", *argv],
            inner_supervisor_directory=str(inner), outer_output=str(outer), cwd=route["repository"], phase=str(PHASE), ssh_destination=contract.DESTINATION,
            shell=False, start_new_session=True, whole_cap_seconds=20, termination_grace_seconds=5, normal_child_budget_seconds=15,
            start_UTC="2026-09-30T00:00:00+00:00")
        for key, relative in (("source_sha256", "protocols/bounded_run_v1.py"), ("authorization_source_sha256", "protocols/run_authorized_v2.py"),
            ("logging_source_sha256", "protocols/run_logged.py"), ("wrapper_sha256", "protocols/repo_env.sh"), ("research_script_sha256", HERE.name + "/qualify.py")):
            start[key] = contract.sha(PHASE / relative)
        start_path = outer / "START.json"; start_path.write_text(json.dumps(start))
        (outer / "stdout_stderr.log").write_text("source-only fixture\n")
        for name in ("authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"):
            (inner / name).write_text("source-only metadata fixture\n")
        terminal = dict(schema="gnnm-whole-process-bound-terminal-v1", START_sha256=contract.sha(start_path),
            complete=True, root_request_unchanged=True, within_whole_cap=True, child_exit_code=0, timed_out=False,
            interruption=None, whole_supervised_seconds=8,
            stdout_stderr_sha256=contract.sha(outer / "stdout_stderr.log"),
            inner_evidence={p.name: dict(sha256=contract.sha(p), bytes=p.stat().st_size) for p in inner.iterdir()})
        terminal_path = outer / "TERMINAL.json"
        def save_terminal(row):
            terminal_path.write_text(json.dumps(row))
            return dict(path=str(terminal_path.relative_to(PHASE)), sha256=contract.sha(terminal_path))
        assert contract.outer_completed(save_terminal(terminal), request_path, admitted, inner, argv)["sha256"] == contract.sha(start_path)
        for key, changed in (("timed_out", True), ("whole_supervised_seconds", 21), ("root_request_unchanged", False), ("inner_evidence", {})):
            mutated = copy.deepcopy(terminal); mutated[key] = changed
            try:
                contract.outer_completed(save_terminal(mutated), request_path, admitted, inner, argv)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid outer terminal admitted: " + key)
        changed_start = copy.deepcopy(start); changed_start["whole_cap_seconds"] = 21; start_path.write_text(json.dumps(changed_start))
        try:
            contract.outer_start(start_path, request_path, admitted, inner, argv)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched live outer cap admitted")
    checks.append(dict(check="outer_exact_live_START_and_completed_timeout_cap_request_inner_hash_chain_refusals", passed=True))
    for packet, wanted in (("propagation_cost_engineering_v1", contract.ENGINEERING_PIN),
                           ("propagation_cost_impl_v1", contract.MECHANICS_PIN)):
        path = PHASE / packet / "SOURCE_MANIFEST_local_run01.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == wanted
        manifest = json.loads(path.read_text())
        for row in manifest["files"]:
            raw = (path.parent / row["relative_path"]).read_bytes()
            assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
        checks.append(dict(check="upstream_source_seal_unchanged_" + packet, passed=True))
    assert not any(name == "torch" or name.startswith(("torch.", "numpy", "ogb", "torch_geometric")) for name in sys.modules)
    output = HERE / "STDLIB_SOURCE_CHECK_v1.json"
    with output.open("x") as stream:
        json.dump(dict(schema="gnnm-shared-message-stdlib-source-check-v1", status="passed_stdlib_source_only",
            checks=checks, check_count=len(checks), python_source_sha256=hashes,
            Torch_GPU_provider_data_or_numerical_runtime_executed=False), stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (HERE / "PARAMETER_PLAN_v1.json").open("x") as stream:
        json.dump(dict(schema="gnnm-shared-message-exact-count-plan-v1", tasks=plans,
            counts_are_stdlib_source_formulas=True, runtime_counts_and_resources_unverified=True), stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(status="passed_stdlib_source_only", checks=len(checks))))


if __name__ == "__main__":
    main()
