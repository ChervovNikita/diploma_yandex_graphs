"""Root-gated artificial CPU metric-cost calibration; no scientific inputs.

Imports Torch only after exact source/request/outer-START admission. Extracts
unchanged function ASTs into a namespace; imports no phase package/entrypoint.
"""
import sys
import os
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import statistics
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PLAN = dict(path="shared_message_cost_admission_v1/METRIC_PROXY_QUALIFICATION_PLAN_v1.json",
            sha256="55e0911bea889334336ba194dc97279b6953424b7084de686b0414a266102422")
BOUND = "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23"
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
RUNTIME = dict(device="cpu", CPU_threads=4, torch_version="2.1.2+cu118",
               input_dtype="float32", reduction_dtype="float64", AMP=False)
SHAPES = {"Reddit2": dict(validation_nodes=23699, classes=41, member_counts=[1, 4]),
          "ogbn-products": dict(validation_nodes=39323, classes=47, member_counts=[1, 4])}
ARMS = ("exact_member", "shared_mean_message", "detached_mean_message", "single_base64", "single_parameters")


def within(path, root=PHASE):
    root, path = Path(root).resolve(), Path(os.path.abspath(path))
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path outside the declared packet")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Source/evidence/output symlinks forbidden")
    return path.resolve()


def sha(path):
    return hashlib.sha256(within(path).read_bytes()).hexdigest()


def reference(path):
    path = within(path)
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path))


def checked(value):
    if set(value) != {"path", "sha256"}:
        raise ValueError("Exact phase-relative source/metadata reference required")
    path = within(PHASE / value["path"])
    if reference(path) != value:
        raise ValueError("Source/metadata reference changed")
    return path


def write(path, value):
    with within(path, HERE).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def extract_functions():
    """Only definitions are executed; their original bodies are unchanged."""
    binding = json.loads((HERE / "SOURCE_FUNCTION_BINDINGS_v1.json").read_text())
    namespace = dict(math=math, statistics=statistics)
    receipts = []
    for row in binding["functions"]:
        path = checked(row["source_file"])
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == row["function"]]
        if len(nodes) != 1:
            raise ValueError("Exact single source function required")
        node = nodes[0]
        raw = ast.get_source_segment(text, node).encode()
        tree_sha = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        if hashlib.sha256(raw).hexdigest() != row["extracted_source_sha256"] or tree_sha != row["extracted_AST_sha256"]:
            raise ValueError("Extracted unchanged function source/AST differs")
        # No transformation, import of the whole module, or body replacement.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
        receipts.append(row)
    if {r["function"] for r in receipts} != {"full_split", "from_statistics", "label_free_reductions"}:
        raise ValueError("Only the three pinned functions may be extracted")
    return namespace, receipts


def admission(path, supervisor):
    path = within(path, HERE)
    request = json.loads(path.read_text())
    if (request.get("schema") != "gnnm-shared-message-artificial-CPU-metric-request-v1"
            or request.get("root_authorized_artificial_CPU_calibration") is not True
            or any(request.get(k) is not False for k in ("scientific_training_authorized", "real_data_or_heldout_targets_authorized", "GPU_operations_authorized", "quality_scores_or_retained_tensors_authorized"))
            or request.get("plan") != PLAN or request.get("runtime_policy") != RUNTIME
            or request.get("helper_sha256") != sha(HERE / "metric_cost.py")
            or request.get("bounded_source_sha256") != BOUND
            or type(request.get("wall_cap_seconds")) not in (int, float)
            or not math.isfinite(request["wall_cap_seconds"]) or not 0 < request["wall_cap_seconds"] <= 300
            or request.get("host_RSS_cap_bytes") != 2**31 or request.get("disk_cap_bytes") != 16*2**20):
        raise ValueError("Exact separately root-authorized artificial CPU-only request required")
    seal_path = checked(request["source_manifest"])
    if seal_path != HERE / "SOURCE_MANIFEST_v1.json":
        raise ValueError("Exact local helper source manifest required")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "gnnm-shared-message-metric-cost-source-manifest-v1":
        raise ValueError("Exact source-only helper seal required")
    for row in seal["files"]:
        file = within(HERE / row["relative_path"], HERE)
        if file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
            raise ValueError("Sealed helper source changed")
    for relative, wanted in seal["dependency_sha256"].items():
        if sha(PHASE / relative) != wanted:
            raise ValueError("Sealed helper source/metadata dependency changed")
    plan = json.loads(checked(PLAN).read_text())
    if ({task: {k: row[k] for k in ("validation_nodes", "classes", "member_counts")} for task, row in plan["shapes"].items()} != SHAPES
            or plan["synthetic_seed"] != 20260930 or plan["warmups_per_kernel_per_shape"] != 2
            or plan["paired_measured_repetitions_per_kernel_per_shape"] != 5
            or plan["source_engine_CPU_call_count_per_seed"] != 64):
        raise ValueError("Immutable calibration plan differs")
    checked(plan["source_full_split"])
    checked(plan["source_label_free_reductions"])
    metadata = json.loads(checked(plan["shape_metadata"]).read_text())
    if metadata.get("quality_scores_computed") is not False or metadata.get("heldout_targets_loaded") is not False:
        raise ValueError("Only completed train-only shape metadata permitted")
    if metadata["gpu"]["uuid"] != request["gpu_uuid"]:
        raise ValueError("Sole-allocation identity differs from root shape evidence")
    for task in metadata["tasks"]:
        spec = SHAPES[task["task"]]
        if task["plan"]["dimensions"]["classes"] != spec["classes"]:
            raise ValueError("Class metadata differs")
        for row in task["variants"]:
            if row["status"] == "qualified_bounded_training_path":
                m = 1 if row["variant"]["arm"].startswith("single_") else 4
                if row["artifact_potential"]["one_validation_member_and_pooled_logits_bytes"] != (m+1)*spec["validation_nodes"]*spec["classes"]*4:
                    raise ValueError("Official validation cardinality metadata differs")
    prior = json.loads(checked(request["root_prior_accounting"]).read_text())
    if prior.get("schema") != "gnnm-shared-message-prior-resource-accounting-root-v1":
        raise ValueError("Exact root prior accounting required")
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    supervisor = within(supervisor)
    outer_path = within(os.environ["GNNM_BOUND_START_JSON"])
    outer = json.loads(outer_path.read_text())
    command = json.loads((supervisor / "command.json").read_text())
    environment = json.loads((supervisor / "environment.json").read_text())
    argv = [str(Path(route["repository"]) / ".venv/bin/python"), str(HERE / "metric_cost.py"),
            "--request", str(path), "--supervisor", str(supervisor)]
    expected = [argv[0], str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(supervisor), "--require-idle-gpu", "--", *argv]
    if (outer_path.name != "START.json" or outer.get("schema") != "gnnm-whole-process-bound-start-v1"
            or outer.get("root_request") != reference(path) or outer.get("root_request_child_flag") != "--request"
            or outer.get("child_argv") != argv or command.get("argv") != argv or outer.get("argv") != expected
            or outer.get("inner_supervisor_directory") != str(supervisor) or outer.get("outer_output") != str(outer_path.parent)
            or outer.get("whole_cap_seconds") != request["wall_cap_seconds"]
            or outer.get("shell") is not False or command.get("shell") is not False or outer.get("start_new_session") is not True
            or outer.get("cwd") != route["repository"] or command.get("cwd") != route["repository"]
            or environment.get("repository") != route["repository"] or environment.get("phase") != str(PHASE)
            or outer.get("phase") != str(PHASE) or outer.get("ssh_destination") != DESTINATION or route["ssh_destination"] != DESTINATION
            or os.environ.get("GNNM_SSH_DESTINATION") != DESTINATION or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get("GNNM_PHASE_ROOT") != str(PHASE)
            or not 0 < outer.get("termination_grace_seconds", 0) < outer["whole_cap_seconds"]
            or outer.get("normal_child_budget_seconds") != outer["whole_cap_seconds"]-outer["termination_grace_seconds"]):
        raise ValueError("Exact corrected source-bound external START/supervisor chain required")
    for key, file in (("source_sha256", PHASE / "protocols/bounded_run_v1.py"),
            ("authorization_source_sha256", PHASE / "protocols/run_authorized_v2.py"),
            ("logging_source_sha256", PHASE / "protocols/run_logged.py"),
            ("wrapper_sha256", PHASE / "protocols/repo_env.sh"), ("research_script_sha256", HERE / "metric_cost.py")):
        if outer.get(key) != sha(file):
            raise ValueError("External source linkage changed")
    if (outer["source_sha256"] != BOUND or environment.get("script_sha256") != sha(HERE / "metric_cost.py")
            or environment.get("supervisor_sha256") != sha(PHASE / "protocols/run_logged.py")
            or environment.get("wrapper_sha256") != sha(PHASE / "protocols/repo_env.sh")):
        raise ValueError("Live source generation changed")
    for name, current in environment["cache_and_temp"].items():
        if os.environ.get(name) != current or not Path(current).resolve().is_relative_to(PHASE):
            raise ValueError("Declared cache/temp environment differs")
    start = datetime.fromisoformat(outer["start_UTC"])
    if start.tzinfo is None:
        raise ValueError("Timezone-aware whole-process START required")
    live = dict(outer_start=reference(outer_path), supervisor_directory=str(supervisor.relative_to(PHASE)),
                command=reference(supervisor / "command.json"), environment=reference(supervisor / "environment.json"),
                started=start.isoformat(), route=reference(PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json"))
    return request, reference(path), plan, live


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform.startswith("linux") else 1))


def artificial(torch, n, c, m, pattern, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    member = torch.rand((m, n, c), generator=generator, device="cpu", dtype=torch.float32) * 0.25
    target = torch.arange(n, device="cpu", dtype=torch.long) % c
    ids = torch.arange(n, device="cpu", dtype=torch.long)
    for k in range(m):
        chosen = target if pattern == "agreeing_argmaxes" else (target+k+1) % c
        member[k, ids, chosen] += 8.0
    return member, member.mean(0), target


def validate_artificial(functions, member, pooled, target, pattern):
    metrics, counts = functions["full_split"](member, pooled, target)
    n, m = target.numel(), member.shape[0]
    correct = n if pattern == "agreeing_argmaxes" else 0
    if (counts["pooled_correct"] != correct or counts["member_correct"] != [correct]*m
            or metrics != functions["from_statistics"](counts)):
        raise ValueError("Artificial CPU source metric validation failed")
    for pair in counts["pair_counts"].values():
        expected = dict(disagree=0, both_wrong=0, either_wrong=0) if pattern == "agreeing_argmaxes" else dict(disagree=n, both_wrong=n, either_wrong=n)
        if pair != expected:
            raise ValueError("Artificial member-pair source reduction validation failed")
    del metrics, counts


def cpu_surcharge(shape_bounds, preflight_rows):
    """Stdlib arithmetic only; root separately audits all input receipts."""
    if len(preflight_rows) != 10 or {(r["task"], r["arm"]) for r in preflight_rows} != {(t, a) for t in SHAPES for a in ARMS}:
        raise ValueError("All ten separately audited task/arm preflight rows required")
    rows = []
    for row in preflight_rows:
        samples = row["three_CPU_label_free_reduction_seconds"]
        if len(samples) != 3 or any(not math.isfinite(v) or v < 0 for v in samples):
            raise ValueError("Three finite actual preflight CPU proxy times required")
        m = 1 if row["arm"].startswith("single_") else 4
        full = shape_bounds[row["task"]][str(m)]["worst_full_split_seconds"]
        if not math.isfinite(full) or full < 0:
            raise ValueError("Finite completed artificial full-kernel bound required")
        excess = max(0.0, full-max(samples))
        rows.append(dict(task=row["task"], arm=row["arm"], members=m, seeds=3,
            calls_per_seed_cell=64, excess_per_call_seconds=excess,
            surcharge_with125percent_margin_seconds=3*64*excess*1.25))
    return dict(rows=rows, total_seconds=sum(r["surcharge_with125percent_margin_seconds"] for r in rows),
                assigned_fixed_cost="source_protocol_runtime_audits", root_receipt_qualification_required=True)


def run(request, request_ref, plan, live):
    output = within(PHASE / request["output_relative_to_phase"], HERE)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.fromisoformat(live["started"])
    def cap():
        if (datetime.now(timezone.utc)-started).total_seconds() >= request["wall_cap_seconds"] or rss() > request["host_RSS_cap_bytes"]:
            raise RuntimeError("Whole-process CPU wall/RSS calibration cap exceeded")
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > request["disk_cap_bytes"]:
            raise RuntimeError("Calibration metadata disk cap exceeded")
    write(output / "START.json", dict(request=request_ref, live=live, artificial_CPU_only=True,
          real_data_heldout_targets_GPU_or_scientific_training_authorized=False))
    try:
        cap()
        if importlib.metadata.version("torch") != RUNTIME["torch_version"]:
            raise ValueError("Pinned Torch runtime changed")
        import torch
        torch.set_num_threads(RUNTIME["CPU_threads"])
        if str(torch.__version__) != RUNTIME["torch_version"] or torch.get_num_threads() != RUNTIME["CPU_threads"] or torch.is_autocast_cpu_enabled():
            raise ValueError("CPU Torch version/thread/precision differs")
        functions, extraction = extract_functions()
        rows, bounds = [], {}
        for task, spec in SHAPES.items():
            bounds[task] = {}
            for m in spec["member_counts"]:
                shape_rows = []
                for pattern in ("agreeing_argmaxes", "different_argmaxes"):
                    cap()
                    seed = plan["synthetic_seed"]
                    member, pooled, targets = artificial(torch, spec["validation_nodes"], spec["classes"], m, pattern, seed)
                    validate_artificial(functions, member, pooled, targets, pattern)
                    def full():
                        result = functions["full_split"](member, pooled, targets)
                        del result
                    def proxy():
                        functions["label_free_reductions"](member, pooled)
                    for _ in range(2):
                        full(); proxy(); cap()
                    paired = []
                    for repetition in range(5):
                        sample = dict(repetition=repetition+1, order=["full_split", "proxy"] if repetition%2 == 0 else ["proxy", "full_split"])
                        for kind in sample["order"]:
                            cap()
                            begin = time.perf_counter()
                            (full if kind == "full_split" else proxy)()
                            sample[kind+"_seconds"] = time.perf_counter()-begin
                        if sample["proxy_seconds"] <= 0:
                            raise ValueError("Positive paired proxy time required")
                        sample["full_over_proxy_ratio"] = sample["full_split_seconds"]/sample["proxy_seconds"]
                        paired.append(sample)
                    row = dict(task=task, members=m, nodes=spec["validation_nodes"], classes=spec["classes"],
                               input_dtype="float32 CPU", reduction_dtype="float64", artificial_seed=seed, pattern=pattern,
                               warmups_per_kernel=2, paired_measured_repetitions=5, paired_timings=paired,
                               artificial_CPU_metric_validation_passed=True, peak_host_RSS_bytes=rss())
                    rows.append(row); shape_rows.extend(paired)
                    del member, pooled, targets, full, proxy
                    cap()
                bounds[task][str(m)] = dict(worst_full_split_seconds=max(r["full_split_seconds"] for r in shape_rows),
                    worst_proxy_seconds=max(r["proxy_seconds"] for r in shape_rows),
                    worst_paired_full_over_proxy_ratio=max(r["full_over_proxy_ratio"] for r in shape_rows))
        write(output / "RESULT.json", dict(schema="gnnm-shared-message-artificial-CPU-metric-cost-v1", status="completed_artificial_CPU_calibration",
            request=request_ref, source_manifest=request["source_manifest"], plan=PLAN, live=live,
            function_extraction=extraction, observed_runtime_policy=RUNTIME, rows=rows, shape_bounds=bounds,
            peak_host_RSS_bytes=rss(), synthetic_scores_targets_logits_or_tensors_retained=False,
            real_data_heldout_targets_GPU_or_scientific_training_accessed=False,
            surcharge_accounting=dict(calls_per_seed_cell=64, validation_calls=60, stored_and_new_selected_final_replay_calls=4,
                seeds_per_task_arm=3, margin=1.25, assigned_fixed_cost="source_protocol_runtime_audits",
                formula=plan["surcharge_formula"], preflight_rows_and_final_ledger_require_root_qualification=True),
            full_wall_and_chain_qualification_requires_successful_outer_terminal=True,
            empirical_proxy_bound_is_engineering_not_statistical_confidence=True, scientific_admission_established=False))
        write(output / "ARTIFACT_MANIFEST.json", dict(schema="gnnm-shared-message-artificial-CPU-metric-artifacts-v1",
            files=[reference(output / n) for n in ("START.json", "RESULT.json")]))
        cap()
        print(json.dumps(dict(status="completed_artificial_CPU_calibration", scores_retained=False)), flush=True)
    except BaseException as error:
        write(output / "FAILURE.json", dict(type=type(error).__name__, message=str(error), partial_metadata_preserved=True,
              real_data_heldout_targets_GPU_or_scientific_training_accessed=False, retries_or_scientific_promotion_authorized=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    request, request_ref, plan, live = admission(args.request, args.supervisor)
    run(request, request_ref, plan, live)


if __name__ == "__main__":
    main()
