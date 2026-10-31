"""Narrow stdlib audit of fetched completed cost metadata, never helper execution."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
FETCH = "protocols/FETCH_AUTHORIZED_20260930_run51.json"
TEN = "shared_message_predictive_v2/COMPLETED_TEN_COST_PREFLIGHT_ROOT_AUDIT_authorized1gpu_run01.json"
EVIDENCE = {}
CHECKS = []


def read(relative, json_value=True):
    path = PHASE / relative
    if Path(relative).is_absolute() or str(path.relative_to(PHASE)) != relative or ".." in Path(relative).parts:
        raise ValueError("Noncanonical phase path")
    current = PHASE
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink evidence")
    if path.suffix not in (".json", ".py", ".md", ".sh", ".log", ".jinja") or "/provider/" in relative:
        raise ValueError("Only explicit source and fetched metadata allowed")
    raw = path.read_bytes()
    EVIDENCE[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return json.loads(raw) if json_value else raw.decode()


def ref(relative):
    if relative not in EVIDENCE:
        read(relative, json_value=False)
    return dict(path=relative, sha256=EVIDENCE[relative]["sha256"])


def checked(reference, json_value=True):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Exact path/hash reference required")
    value = read(reference["path"], json_value)
    if ref(reference["path"]) != reference:
        raise ValueError("Changed evidence: " + reference["path"])
    return value


def check(name, value):
    if not value:
        raise AssertionError(name)
    CHECKS.append(dict(name=name, passed=True))


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def audit_chain(label, packet, output_name, supervisor_name, helper_name, request_name):
    output = packet + "/" + output_name
    supervisor = "protocols/logged_runs/" + supervisor_name
    outer = "protocols/bounded_runs/" + supervisor_name
    request_path = packet + "/" + request_name
    request = read(request_path)
    helper = read(packet + "/" + helper_name, False)
    check(label + " exact request/helper hash", request["helper_sha256"] == ref(packet + "/" + helper_name)["sha256"])
    seal = checked(request["source_manifest"])
    for row in seal["files"]:
        path = packet + "/" + row["relative_path"]
        read(path, False)
        check(label + " sealed file " + row["relative_path"], EVIDENCE[path] == dict(sha256=row["sha256"], bytes=row["bytes"]))
    for path, wanted in seal.get("dependency_sha256", {}).items():
        read(path, False)
        check(label + " dependency pin " + path, EVIDENCE[path]["sha256"] == wanted)
    start, terminal = read(outer + "/START.json"), read(outer + "/TERMINAL.json")
    artifact_start, result = read(output + "/START.json"), read(output + "/RESULT.json")
    completion, authorization = read(supervisor + "/completion.json"), read(supervisor + "/authorization.json")
    command, environment = read(supervisor + "/command.json"), read(supervisor + "/environment.json")
    route = read("protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json")
    remote_phase = route["repository"] + "/experiments_iclr/postsubmission_20260930"
    argv = [route["repository"] + "/.venv/bin/python", remote_phase + "/" + packet + "/" + helper_name,
            "--request", remote_phase + "/" + request_path, "--supervisor", remote_phase + "/" + supervisor]
    envelope = [argv[0], remote_phase + "/protocols/run_authorized_v2.py", "--output", remote_phase + "/" + supervisor,
                "--require-idle-gpu", "--", *argv]
    check(label + " exact source-bound argv and route", start["child_argv"] == argv and command["argv"] == argv
        and start["argv"] == envelope and start["root_request"] == ref(request_path)
        and start["root_request_child_flag"] == "--request" and start["inner_supervisor_directory"] == remote_phase + "/" + supervisor
        and start["outer_output"] == remote_phase + "/" + outer and start["cwd"] == route["repository"]
        and command["cwd"] == route["repository"] and start["phase"] == remote_phase
        and start["ssh_destination"] == route["ssh_destination"] and start["shell"] is False and start["start_new_session"] is True)
    source_links = {"source_sha256": "protocols/bounded_run_v1.py", "authorization_source_sha256": "protocols/run_authorized_v2.py",
                   "logging_source_sha256": "protocols/run_logged.py", "wrapper_sha256": "protocols/repo_env.sh",
                   "research_script_sha256": packet + "/" + helper_name}
    check(label + " envelope and wrapper source pins", all(start[key] == ref(path)["sha256"] for key, path in source_links.items())
        and start["source_sha256"] == request["bounded_source_sha256"])
    check(label + " helper result/START/request/live linkage", artifact_start["request"] == ref(request_path)
        and result["request"] == ref(request_path) and result["live"] == artifact_start["live"]
        and result["live"]["outer_start"] == ref(outer + "/START.json")
        and result["live"]["supervisor_directory"] == supervisor
        and result["live"]["started"] == start["start_UTC"])
    check(label + " environment source and repository confinement", environment["repository"] == route["repository"]
        and environment["phase"] == remote_phase and environment["CUDA_VISIBLE_DEVICES"] == "0"
        and environment["script_sha256"] == request["helper_sha256"] and environment["python_bytecode_disabled"] is True
        and environment["wrapper_sha256"] == start["wrapper_sha256"] and environment["supervisor_sha256"] == start["logging_source_sha256"]
        and all(value == remote_phase or value.startswith(remote_phase + "/") for value in environment["cache_and_temp"].values()))
    check(label + " inner completion links exact bytes", completion["exit_code"] == 0
        and completion["environment_sha256"] == ref(supervisor + "/environment.json")["sha256"]
        and completion["command_sha256"] == ref(supervisor + "/command.json")["sha256"]
        and completion["log_sha256"] == ref(supervisor + "/stdout_stderr.log")["sha256"])
    check(label + " allocation authorization links same GPU and completion", authorization["child_exit_code"] == 0
        and authorization["inner_completion_sha256"] == ref(supervisor + "/completion.json")["sha256"]
        and authorization["ssh_destination"] == route["ssh_destination"] and authorization["gpu_uuid"] == request["gpu_uuid"]
        and authorization["visible_gpu_count"] == 1 and authorization["idle_required"] is True
        and authorization["allocation_guard_sha256"] == start["authorization_source_sha256"]
        and authorization["route_record_sha256"] == ref("protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json")["sha256"])
    names = {"authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"}
    check(label + " exact outer inner-evidence byte/hash chain", set(terminal["inner_evidence"]) == names
        and all(terminal["inner_evidence"][name] == EVIDENCE[supervisor + "/" + name] for name in names)
        and terminal["START_sha256"] == ref(outer + "/START.json")["sha256"]
        and terminal["stdout_stderr_sha256"] == ref(outer + "/stdout_stderr.log")["sha256"])
    check(label + " successful bounded outer completion", terminal["complete"] is True and terminal["child_exit_code"] == 0
        and terminal["timed_out"] is False and terminal["interruption"] is None and terminal["root_request_unchanged"] is True
        and terminal["within_whole_cap"] is True and finite(terminal["whole_supervised_seconds"], True)
        and finite(completion["seconds"], True) and completion["seconds"] <= terminal["whole_supervised_seconds"] <= start["whole_cap_seconds"]
        and start["whole_cap_seconds"] == request["wall_cap_seconds"]
        and start["normal_child_budget_seconds"] == start["whole_cap_seconds"] - start["termination_grace_seconds"])
    times = [start["start_UTC"], authorization["start_UTC"], environment["capture_UTC"], completion["completion_UTC"],
             authorization["completion_UTC"], terminal["terminal_UTC"]]
    dates = [datetime.fromisoformat(value) for value in times]
    check(label + " ordered timezone-aware receipt timestamps", all(date.tzinfo is not None for date in dates) and dates == sorted(dates))
    check(label + " wall clock delta agrees with outer elapsed", math.isclose((dates[-1]-dates[0]).total_seconds(), terminal["whole_supervised_seconds"], abs_tol=.001))
    return dict(request=request, result=result, output=output, source=helper, inner_seconds=completion["seconds"],
                whole_seconds=terminal["whole_supervised_seconds"], refs=dict(request=ref(request_path), result=ref(output + "/RESULT.json"),
                    source_manifest=request["source_manifest"], helper=ref(packet + "/" + helper_name),
                    outer_start=ref(outer + "/START.json"), outer_terminal=ref(outer + "/TERMINAL.json"), supervisor_directory=supervisor))


def run():
    fetch = read(FETCH)
    for path, expected in fetch["files"].items():
        read(path, False)
        check("Fetch byte/hash " + path, EVIDENCE[path] == expected)
    fixed = audit_chain("fixed", "shared_message_cost_admission_v2", "fixed_overhead_authorized1gpu_run01",
        "shared_message_fixed_overhead_v2_authorized1gpu_run01", "fixed_overhead.py", "REQUEST_FIXED_OVERHEAD_ROOT_authorized1gpu_run01.json")
    metric = audit_chain("metric", "shared_message_metric_cost_v1", "artificial_CPU_authorized1gpu_run01",
        "shared_message_metric_cost_v1_authorized1gpu_run01", "metric_cost.py", "REQUEST_METRIC_ROOT_authorized1gpu_run01.json")
    fr, mr = fixed["result"], metric["result"]
    check("Fixed measurement remains metadata-only", fr["status"] == "completed_metadata_only"
        and fr["tensors_labels_models_checkpoints_or_scientific_scores_accessed"] is False
        and fr["quality_novelty_feasibility_or_scientific_admission_established"] is False)
    check("Fixed source/quality pins and work specification exact", fr["work_counts"] == fixed["request"]["work_counts"]
        and fr["surrogate_sizes"] == fixed["request"]["surrogate_sizes"]
        and fr["input_source_bindings"] == fixed["request"]["scientific_source_bindings"]
        and fr["input_engineering_result"] == fixed["request"]["completed_quality_result"])
    for name in ("scientific_source_manifest", "scientific_source_bindings", "completed_quality_result", "completed_quality_root_audit",
                 "root_prior_accounting", "root_independent_delta_audit", "root_ten_cost_audit"):
        checked(fixed["request"][name])
    fixed_manifest = read(fixed["output"] + "/ARTIFACT_MANIFEST.json")
    expected_paths = {fixed["output"] + "/" + name for name in ("START.json", "RESULT.json")}
    expected_paths.update(fixed["output"] + f"/surrogate_fit{cell:02d}/epoch{epoch:03d}.json" for cell in range(30) for epoch in range(1, 301))
    expected_paths.update(fixed["output"] + f"/surrogate_fit{cell:02d}/FIT_END.json" for cell in range(30))
    expected_paths.update(fixed["output"] + f"/surrogate_manifest{index:02d}.json" for index in range(63))
    expected_paths.update(fixed["output"] + "/surrogate_supervisor/" + name for name in
        ("authorization.json", "completion.json", "environment.json", "command.json", "outer_START.json", "outer_TERMINAL.json", "inner.log", "outer.log"))
    check("Fixed manifest declares exact9103expected metadata artifacts", len(fixed_manifest["files"]) == 9103
        and {row["path"] for row in fixed_manifest["files"]} == expected_paths
        and all(set(row) == {"path", "sha256"} and re.fullmatch("[0-9a-f]{64}", row["sha256"]) for row in fixed_manifest["files"]))
    check("Fixed manifest binds fetched START and RESULT", all(row == ref(row["path"]) for row in fixed_manifest["files"] if row["path"] in {fixed["output"] + "/START.json", fixed["output"] + "/RESULT.json"}))
    check("Reported fixed logical bytes before report fit helper cap", 0 < fr["logical_regular_file_bytes_before_report"] < fixed["request"]["disk_cap_bytes"])
    timing = fr["measured_seconds"]
    samples = timing["16_exact_source_closure_full_hash_pass_samples_seconds"]
    check("Sixteen finite source-hash samples and unchanged2048bound", len(samples) == 16 and all(finite(value, True) for value in samples)
        and timing["2048_source_closure_passes_worst_sample_bound_seconds"] == 2048 * max(samples))
    check("All fixed elapsed terms finite", all(finite(value) for key, value in timing.items() if not isinstance(value, list)))
    timed_actual = sum(value for key, value in timing.items() if not isinstance(value, list) and key != "2048_source_closure_passes_worst_sample_bound_seconds") + sum(samples)
    check("Actual timed fixed work fits inner/outer elapsed", timed_actual <= fixed["inner_seconds"] <= fixed["whole_seconds"])
    check("Fixed host RSS positive engineering metadata", type(fr["peak_host_RSS_bytes"]) is int and fr["peak_host_RSS_bytes"] > 0)
    metric_manifest = read(metric["output"] + "/ARTIFACT_MANIFEST.json")
    check("Metric manifest declares exactly START and RESULT", metric_manifest["files"] == [ref(metric["output"] + "/" + name) for name in ("START.json", "RESULT.json")])
    check("Metric calibration remains artificial CPU only", mr["status"] == "completed_artificial_CPU_calibration"
        and mr["synthetic_scores_targets_logits_or_tensors_retained"] is False
        and mr["real_data_heldout_targets_GPU_or_scientific_training_accessed"] is False and mr["scientific_admission_established"] is False
        and mr["observed_runtime_policy"] == metric["request"]["runtime_policy"] and mr["source_manifest"] == metric["request"]["source_manifest"])
    plan = checked(metric["request"]["plan"])
    function_bindings = read("shared_message_metric_cost_v1/SOURCE_FUNCTION_BINDINGS_v1.json")
    check("Metric extracted function receipt exact", mr["function_extraction"] == function_bindings["functions"])
    for row in function_bindings["functions"]:
        source = checked(row["source_file"], False)
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == row["function"]]
        check("Metric unchanged function source/AST " + row["function"], len(nodes) == 1
            and hashlib.sha256(ast.get_source_segment(source, nodes[0]).encode()).hexdigest() == row["extracted_source_sha256"]
            and hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest() == row["extracted_AST_sha256"])
    combinations = {(task, members, pattern) for task in ("Reddit2", "ogbn-products") for members in (1, 4) for pattern in ("agreeing_argmaxes", "different_argmaxes")}
    check("All eight prospective metric shape/pattern rows present", len(mr["rows"]) == 8 and {(row["task"], row["members"], row["pattern"]) for row in mr["rows"]} == combinations)
    metric_timed = 0.0
    for row in mr["rows"]:
        shape = plan["shapes"][row["task"]]
        name = row["task"] + "/M%d/" % row["members"] + row["pattern"]
        check("Metric shape/runtime/known-outcome assertions " + name, row["nodes"] == shape["validation_nodes"] and row["classes"] == shape["classes"]
            and row["warmups_per_kernel"] == 2 and row["paired_measured_repetitions"] == 5 and row["artificial_seed"] == 20260930
            and row["artificial_CPU_metric_validation_passed"] is True and row["input_dtype"] == "float32 CPU" and row["reduction_dtype"] == "float64")
        paired = row["paired_timings"]
        check("Five alternating paired metric samples " + name, len(paired) == 5
            and all(sample["repetition"] == i + 1 and sample["order"] == (["full_split", "proxy"] if i % 2 == 0 else ["proxy", "full_split"])
                and finite(sample["full_split_seconds"], True) and finite(sample["proxy_seconds"], True)
                and sample["full_over_proxy_ratio"] == sample["full_split_seconds"] / sample["proxy_seconds"] for i, sample in enumerate(paired)))
        metric_timed += sum(sample["full_split_seconds"] + sample["proxy_seconds"] for sample in paired)
    rebuilt = {}
    for task in ("Reddit2", "ogbn-products"):
        rebuilt[task] = {}
        for members in (1, 4):
            paired = [sample for row in mr["rows"] if row["task"] == task and row["members"] == members for sample in row["paired_timings"]]
            rebuilt[task][str(members)] = dict(worst_full_split_seconds=max(sample["full_split_seconds"] for sample in paired),
                worst_proxy_seconds=max(sample["proxy_seconds"] for sample in paired),
                worst_paired_full_over_proxy_ratio=max(sample["full_over_proxy_ratio"] for sample in paired))
    check("All four elapsed metric shape bounds rebuild exactly", rebuilt == mr["shape_bounds"])
    check("Metric RSS stays below authorized2GiBcap", type(mr["peak_host_RSS_bytes"]) is int
        and max(row["peak_host_RSS_bytes"] for row in mr["rows"]) <= mr["peak_host_RSS_bytes"] <= metric["request"]["host_RSS_cap_bytes"])
    check("Metric paired timed work fits inner elapsed", metric_timed <= metric["inner_seconds"])
    accounting = mr["surcharge_accounting"]
    check("CPU surcharge accounting preserves64calls3seeds125percent", accounting["calls_per_seed_cell"] == 64
        and accounting["validation_calls"] == 60 and accounting["stored_and_new_selected_final_replay_calls"] == 4
        and accounting["seeds_per_task_arm"] == 3 and accounting["margin"] == 1.25
        and accounting["assigned_fixed_cost"] == "source_protocol_runtime_audits" and accounting["formula"] == plan["surcharge_formula"])
    ten = read(TEN)
    check("Exact completed-ten audit is qualified engineering metadata", ten["qualified_train_only_cost_metadata"] is True
        and ten["all_ten_source_request_execution_runtime_GPU_and_completed_chains_checked"] is True
        and fixed["request"]["root_ten_cost_audit"] == ref(TEN) and ten["scientific_admission_or_quality_established"] is False)
    rows = []
    for row in ten["successful_cells"]:
        value = checked(row["report"])
        check("Ten-row proxy input identity " + row["task"] + "/" + row["arm"], value["task"] == row["task"] and value["arm"] == row["arm"]
            and value["request"] == row["request"] and value["execution"] == row["execution"]
            and value["retained_weights_or_scores"] is False and value["validation_target_export_or_access"] is False and value["test_target_access"] is False)
        rows.append(dict(task=row["task"], arm=row["arm"], report=row["report"], three_CPU_label_free_reduction_seconds=value["three_CPU_label_free_reduction_seconds"]))
    tree = ast.parse(metric["source"])
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cpu_surcharge")
    arms = next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ARMS" for target in node.targets))
    namespace = dict(math=math, SHAPES={"Reddit2": None, "ogbn-products": None}, ARMS=ast.literal_eval(arms))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "unchanged_cpu_surcharge", "exec", dont_inherit=True), namespace)
    surcharge = namespace["cpu_surcharge"](rebuilt, rows)
    check("Unchanged CPU surcharge uses all ten complete rows", len(surcharge["rows"]) == 10 and surcharge["total_seconds"] == sum(row["surcharge_with125percent_margin_seconds"] for row in surcharge["rows"]))
    source_keys = ["2048_source_closure_passes_worst_sample_bound_seconds", "448_exact_quality_engineering_result_reads_parses_hashes_seconds", "5617_padded_supervisor_chain_metadata_parses_hashes_seconds"]
    io_keys = ["trace_and_FIT_END_generation_and_writes_seconds", "45000_trace_reads_parses_seconds", "150_FIT_END_reads_parses_seconds", "36120_trace_and_FIT_END_full_hashes_seconds", "18060_extra_cached_reference_path_stat_checks_seconds", "63_manifest_writes_252_reads_parses_hashes_seconds"]
    source_raw, io_raw = sum(timing[key] for key in source_keys), sum(timing[key] for key in io_keys)
    source_total = source_raw * 1.25 + 180 + surcharge["total_seconds"]
    io_total = io_raw * 1.25 + ten["suggested64child_startup_residual_bound_seconds"]
    return dict(schema="gnnm-independent-completed-cost-helper-metadata-audit-v1", UTC=datetime.now(timezone.utc).isoformat(),
        status="fetched_source_request_receipt_chains_and_engineering_timing_arithmetic_verified_remote_surrogate_hashes_and_final_disk_pending",
        checks=CHECKS, check_count=len(CHECKS), fetch_receipt=ref(FETCH), fixed_helper=fixed["refs"], metric_helper=metric["refs"],
        engineering_only=True, scientific_results_or_admission_established=False, source_or_finalized_evidence_modified=False,
        Torch_provider_data_GPU_or_SSH_used_by_auditor=False,
        actual_prior_wall_to_add_seconds=dict(fixed_helper=fixed["whole_seconds"], metric_helper=metric["whole_seconds"], total=fixed["whole_seconds"] + metric["whole_seconds"]),
        RSS=dict(fixed_helper_reported_bytes=fr["peak_host_RSS_bytes"], metric_helper_reported_bytes=mr["peak_host_RSS_bytes"], metric_authorized_cap_bytes=metric["request"]["host_RSS_cap_bytes"]),
        fixed_helper_declared_artifacts=dict(total=9103, locally_verified_START_RESULT=2, deliberately_unfetched_surrogate_files=9101,
            exact_declared_file_set_verified=True, remote_surrogate_hash_verification_pending=True,
            reported_logical_regular_file_bytes_before_report=fr["logical_regular_file_bytes_before_report"], final_allocated_disk_pending=True),
        measured_fixed_terms_seconds=timing, actual_timed_fixed_work_seconds=timed_actual,
        metrics=dict(shape_bounds=rebuilt, paired_timed_work_seconds=metric_timed, cpu_surcharge_function_source_sha256=hashlib.sha256(ast.get_source_segment(metric["source"], function).encode()).hexdigest(),
            cpu_surcharge_function_AST_sha256=hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
            preflight_rows=rows, unchanged_cpu_surcharge=surcharge),
        suggested_future_fixed_cost_terms=dict(source_protocol_runtime_audits=source_total, supervisor_startup_manifest_json_overhead=io_total),
        suggested_term_arithmetic=dict(source_audit_raw_components={key: timing[key] for key in source_keys}, source_audit_raw_sum=source_raw,
            source_audit_margin=1.25, scalar_selection_summary_integrity_reserve_seconds=180, CPU_surcharge_already_includes125percent_margin_seconds=surcharge["total_seconds"],
            metadata_raw_components={key: timing[key] for key in io_keys}, metadata_raw_sum=io_raw, metadata_margin=1.25,
            startup64child_bound_seconds=ten["suggested64child_startup_residual_bound_seconds"],
            source_sample_times_charged_as2048worst_bound_not_double_counted=True,
            existing_preflight_CPU_cost_retained_and_only_positive_excess_added=True),
        remaining_root_requirements=["Verify the declared9101surrogate hashes and full output membership remotely.",
            "Qualify prospective surrogate metadata size bounds and final allocated phase disk.",
            "Add actual helper outer walls once to updated prior accounting; include final audit/query wall reserves separately.",
            "Finalize complete-study resource ledger and actual root permission; this audit issues no launch approval."],
        limitations=["Artificial elapsed metric maxima plus engineering margin are no statistical confidence bound.",
            "No provider labels, graphs, checkpoints or scientific predictions were accessed.",
            "Buffered filesystem timings do not measure fsync or crash durability.",
            "Earlier source/protocol audits are preserved; this audit does not reopen their conclusions."])


if __name__ == "__main__":
    receipt = run()
    for name, value in (("AUDIT_RECEIPT_v1.json", receipt), ("EVIDENCE_HASHES_v1.json", dict(schema="gnnm-cost-helper-audit-source-metadata-pins-v1", files=EVIDENCE))):
        with (HERE / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(status=receipt["status"], checks=receipt["check_count"], fixed=receipt["suggested_future_fixed_cost_terms"], surcharge=receipt["metrics"]["unchanged_cpu_surcharge"]["total_seconds"])))
