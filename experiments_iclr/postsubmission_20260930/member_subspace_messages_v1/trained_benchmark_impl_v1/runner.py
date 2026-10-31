"""Closed trained-weight inference companion; no training or target scoring."""
import sys
import os
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import resource
import statistics
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
PACKAGE = str(HERE.relative_to(PHASE))
TASKS = ("Reddit2", "ogbn-products")
SEEDS = (17, 29, 43)
ROLES = ("final", "selected")
MODES = ("exact_packed", "exact_serial", "shared_packed")
SCIENTIFIC_SHA = "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b"
FIXED = dict(warmups=5, paired_rounds=50, role_order=list(ROLES), modes=list(MODES),
             atol=1e-5, rtol=1e-5, CPU_threads=4, child_cap_seconds=600,
             termination_grace_seconds=5, companion_wall_cap_seconds=3600,
             companion_artifact_cap_bytes=64 * 2**20, phase_wall_cap_seconds=21600,
             phase_disk_cap_bytes=200 * 2**30, reserved_GPU_memory_cap_bytes=60 * 2**30,
             p90_definition="nearest_rank_ceil_0.9n", residency="one_graph_both_models",
             training_authorized=False, validation_target_access_authorized=False,
             test_target_access_authorized=False, prediction_artifact_writes_authorized=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def children():
    rows = []
    for task in TASKS:
        short = "reddit2" if task == "Reddit2" else "products"
        for seed in SEEDS:
            job = "trained_shared_message_benchmark_%s_seed%d_authorized1gpu_run01" % (short, seed)
            rows.append(dict(task=task, seed=seed, supervisor_directory="protocols/logged_runs/" + job,
                             outer_directory="protocols/bounded_runs/" + job,
                             output_directory=PACKAGE + "/benchmark_authorized1gpu_run01/%s_seed%d" % (short, seed)))
    return rows


def round_order(task, seed, role):
    if task not in TASKS or seed not in SEEDS or role not in ROLES:
        raise ValueError("Outside frozen task/seed/role family")
    permutations = list(itertools.permutations(MODES))
    rounds = permutations * 8 + permutations[:2]
    random.Random(917000 + 100 * TASKS.index(task) + seed + 10000 * ROLES.index(role)).shuffle(rounds)
    return [list(row) for row in rounds]


def p90(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Finite positive samples required")
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]


def summary(samples):
    if set(samples) != set(MODES) or any(len(v) != 50 for v in samples.values()):
        raise ValueError("Every mode's complete 50 paired samples required")
    rows = {mode: dict(median_seconds=statistics.median(v), p90_seconds=p90(v)) for mode, v in samples.items()}
    stronger = min(MODES[:2], key=lambda m: (rows[m]["median_seconds"], MODES.index(m)))
    shared = rows["shared_packed"]["median_seconds"]
    reductions = {mode: 1 - shared / rows[mode]["median_seconds"] for mode in MODES[:2]}
    return dict(modes=rows, stronger_exact_mode=stronger, shared_reduction_fraction_against_exact=reductions,
                shared_reduction_fraction_against_stronger_exact=reductions[stronger],
                comparator_selected_using_timing_only=True, training_gain_inferred=False)


def bootstrap(request):
    raw = (HERE / "SOURCE_MANIFEST_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != request.get("source_manifest_sha256"):
        raise ValueError("Companion request does not bind exact local source seal")
    manifest = json.loads(raw)
    row = next(r for r in manifest["files"] if r["relative_path"] == "guard.py")
    path = HERE / "guard.py"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Companion guard changed before first phase import")
    name = "_trained_benchmark_caller_guard_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    guard = importlib.util.module_from_spec(spec)
    sys.modules[name] = guard
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), guard.__dict__)
    return guard, guard.install(request["source_manifest_sha256"])


def request_authorized(value):
    if (value.get("schema") != "gnnm-trained-inference-root-request-v1"
            or value.get("root_authorized") is not True or value.get("scientific_queue_completed") is not True
            or value.get("fixed_policy") != FIXED or value.get("children") != children()
            or value.get("output_root") != PACKAGE + "/benchmark_authorized1gpu_run01"):
        raise ValueError("Exact separately approved completed-family benchmark request required")


def admission(path, request, guard):
    request_authorized(request)
    if request.get("scientific_manifest") != dict(path="shared_message_predictive_v3/SOURCE_MANIFEST_v1.json", sha256=SCIENTIFIC_SHA):
        raise ValueError("Wrong inherited scientific generation")
    guard.checked(request["scientific_manifest"])
    manifest = json.loads((HERE / "SOURCE_MANIFEST_v1.json").read_text())
    if request["protocol_freeze"] != manifest["inherited_and_protocol_refs"]["protocol_freeze"]:
        raise ValueError("Prospectively frozen protocol differs")
    guard.checked(request["protocol_freeze"])
    approval = json.loads(guard.checked(request["root_approval"]).read_text())
    if (approval.get("schema") != "gnnm-trained-inference-root-approval-v1"
            or approval.get("qualified") is not True or approval.get("complete_family_and_source_and_resources_qualified") is not True
            or approval.get("approved_payload_sha256") != digest({k: v for k, v in request.items() if k != "root_approval"})):
        raise ValueError("Benchmark request payload is not separately root-approved")
    source_audit = json.loads(guard.checked(request["source_audit"]).read_text())
    if source_audit.get("qualified") is not True or source_audit.get("audited_source_manifest_sha256") != request["source_manifest_sha256"]:
        raise ValueError("Independent companion source audit is not adopted")
    from shared_message_predictive_v3 import contract as scientific
    from shared_message_predictive_v3.core import audit_family_lock
    admitted, binding_ref = scientific.binding(guard.checked(request["binding"]))
    if admitted["runtime_policy"]["CPU_threads"] != 4 or request["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Frozen scientific GPU/runtime differs")
    # JSON identity only: never call validation_freeze(), which hashes target bundles.
    freeze_path = guard.checked(request["validation_freeze"])
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("schema") != "gnnm-shared-message-validation-freeze-v2" or freeze.get("binding") != binding_ref
            or freeze.get("root_qualified") is not True or freeze.get("test_targets_loaded_or_referenced") is not False
            or set(freeze.get("validation_exports", {})) != set(TASKS)):
        raise ValueError("Existing completed freeze JSON identity required")
    family = audit_family_lock(guard.checked(request["family_lock"]), admitted, binding_ref, request["validation_freeze"])
    prior = json.loads(guard.checked(request["prior_accounting"]).read_text())
    keys = ("combined_phase_seconds_before_declared_children", "combined_phase_allocated_disk_bytes_before_declared_children",
            "companion_wall_seconds_before_declared_children", "companion_artifact_allocated_bytes_before_declared_children")
    if (prior.get("schema") != "gnnm-trained-inference-prior-accounting-root-v1" or prior.get("qualified") is not True
            or prior.get("updated_after_full_family_and_source_preparation") is not True
            or prior.get("declared_children_already_included") is not False
            or any(type(prior.get(k)) not in (int, float) or not math.isfinite(prior[k]) or prior[k] < 0 for k in keys)
            or prior[keys[0]] < scientific.prior_seconds(admitted)):
        raise ValueError("Current complete-phase/prior companion failure charges required")
    return scientific, admitted, binding_ref, family, prior


def allocated(paths):
    total = 0
    for root in paths:
        if root.exists():
            for file in root.rglob("*"):
                if file.is_symlink():
                    raise ValueError("Companion evidence symlink refused")
                if file.is_file():
                    total += file.stat().st_blocks * 512
    return total


def terminal_charge(start, terminal, start_sha, log_sha):
    """Validate terminal provenance without dropping over-cap failure wall."""
    seconds = terminal.get("whole_supervised_seconds")
    if (terminal.get("schema") != "gnnm-whole-process-bound-terminal-v1"
            or start["whole_cap_seconds"] != 600 or start["termination_grace_seconds"] != 5
            or terminal.get("START_sha256") != start_sha
            or terminal.get("root_request_unchanged") is not True
            or terminal.get("stdout_stderr_sha256") != log_sha
            or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError("Missing/changed prior companion terminal charge")
    return seconds


def job_evidence(request, row, scientific, *, successful):
    outer = scientific.within(PHASE / row["outer_directory"])
    start_path, terminal_path = outer / "START.json", outer / "TERMINAL.json"
    start, terminal = json.loads(start_path.read_text()), json.loads(terminal_path.read_text())
    supervisor = scientific.within(PHASE / row["supervisor_directory"])
    repo = Path(json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())["repository"])
    expected = [str(repo / ".venv/bin/python"), str(HERE / "runner.py"), "--request", str(scientific.checked(request)),
                "--supervisor", str(supervisor), "--task", row["task"], "--seed", str(row["seed"])]
    scientific.verify_outer_start(start_path, scientific.checked(request), supervisor, expected)
    seconds = terminal_charge(start, terminal, scientific.sha(start_path), scientific.sha(outer / "stdout_stderr.log"))
    if successful:
        chain = scientific.completed(supervisor, scientific.reference(start_path))
        if terminal.get("complete") is not True or terminal.get("child_exit_code") != 0:
            raise ValueError("A failed companion child cannot be retried or promoted")
        return chain
    # Failed timeout/cleanup can overrun its cap. Retain the entire actual
    # charge; only successful completion is qualified by completed() above.
    return dict(seconds=seconds, complete=terminal.get("complete"), terminal=scientific.reference(terminal_path),
                whole_cap_seconds=600, actual_wall_exceeds_cap=seconds > 600)


class Budget:
    def __init__(self, request_ref, request, prior, selected, scientific, *, context=None):
        self.request_ref, self.request, self.prior, self.scientific = request_ref, request, prior, scientific
        self.index = children().index(selected)
        self.spent = prior["companion_wall_seconds_before_declared_children"]
        self.previous = []
        failed = False
        for index, row in enumerate(children()):
            outer = PHASE / row["outer_directory"]
            if index < self.index:
                charged = job_evidence(request_ref, row, scientific, successful=False)
                self.spent += charged["seconds"]
                self.previous.append(charged)
                if charged["complete"] is not True:
                    failed = True
                else:
                    job_evidence(request_ref, row, scientific, successful=True)
            elif index > self.index and (outer.exists() or (PHASE / row["output_directory"]).exists() or (PHASE / row["supervisor_directory"]).exists()):
                raise ValueError("Outcome-independent task/seed order changed")
        if failed:
            raise ValueError("Prior companion failure charged; no continuation/retry admitted")
        if (PHASE / selected["output_directory"]).exists():
            raise ValueError("Existing output refuses automatic retry")
        if (PHASE / selected["outer_directory"] / "TERMINAL.json").exists():
            raise ValueError("A terminal current child refuses retry even without inner outputs")
        self.previous_phase_wall = prior["combined_phase_seconds_before_declared_children"] + self.spent - prior["companion_wall_seconds_before_declared_children"]
        remaining_caps = (len(children()) - self.index) * 600
        if self.spent + remaining_caps > 3600 or self.previous_phase_wall + remaining_caps > 21600:
            raise ValueError("Full remaining child caps cannot fit companion/combined phase wall")
        self.start = time.monotonic()
        if context:
            outer = json.loads(scientific.checked(context["outer_start"]).read_text())
            self.start -= max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(outer["start_UTC"])).total_seconds())
        self.paths = [PHASE / row[key] for row in children() for key in ("outer_directory", "supervisor_directory", "output_directory")]
        self.check()

    def check(self, prospective_bytes=0):
        elapsed = time.monotonic() - self.start
        if elapsed >= 595 or self.spent + elapsed >= 3600 or self.previous_phase_wall + elapsed >= 21600:
            raise RuntimeError("Frozen whole-child/companion/phase wall budget exhausted")
        disk = allocated(self.paths)
        if (self.prior["companion_artifact_allocated_bytes_before_declared_children"] + disk + prospective_bytes + 2**20 > 64 * 2**20
                or self.prior["combined_phase_allocated_disk_bytes_before_declared_children"] + disk + prospective_bytes + 2**20 > 200 * 2**30):
            raise RuntimeError("Companion/combined phase allocated disk cap would be exceeded")


def write(path, value, budget):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    budget.check(len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload)


def memory_check(torch):
    peak = torch.cuda.max_memory_reserved(0)
    if peak > 60 * 2**30 or torch.cuda.memory_reserved(0) > 60 * 2**30:
        raise RuntimeError("60 GiB reserved-memory ceiling exceeded")


def comparison(torch, packed, serial, classes):
    if packed.shape != serial.shape or packed.ndim != 2 or packed.shape[1] != classes:
        raise ValueError("Complete-node exact prediction shapes differ")
    maximum, disagreements, close = 0.0, 0, True
    for start in range(0, packed.shape[0], 65536):
        a, b = packed[start:start + 65536], serial[start:start + 65536]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite exact pooled predictions")
        maximum = max(maximum, float((a - b).abs().max()))
        disagreements += int((a.argmax(1) != b.argmax(1)).sum())
        close = close and torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    if not close:
        raise ValueError("Frozen exact packed/serial 1e-5 equivalence failed")
    return dict(atol=1e-5, rtol=1e-5, allclose=True, maximum_absolute_discrepancy=maximum,
                class_decision_disagreement_nodes=disagreements, complete_nodes=packed.shape[0],
                class_disagreement_is_reported_without_an_unfrozen_threshold=True)


def run(request, request_ref, selected, admitted, binding_ref, locked, prior, scientific, guard, guard_state, supervisor, admission_costs):
    scientific.runtime_before_import(admitted)
    Budget(request_ref, request, prior, selected, scientific)  # Full caps before GPU/live query.
    live_start = time.perf_counter()
    context = scientific.live(supervisor, HERE / "runner.py", scientific.checked(request_ref),
                              {**admitted, "per_child_wall_cap_seconds": 600})
    outer = json.loads(scientific.checked(context["outer_start"]).read_text())
    expected_argv = [str(Path(json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())["repository"]) / ".venv/bin/python"),
                     str(HERE / "runner.py"), "--request", str(scientific.checked(request_ref)), "--supervisor", str(supervisor),
                     "--task", selected["task"], "--seed", str(selected["seed"])]
    if (outer["child_argv"] != expected_argv or outer["whole_cap_seconds"] != 600 or outer["termination_grace_seconds"] != 5
            or context["supervisor_relative_to_project"] != selected["supervisor_directory"]):
        raise ValueError("Exact fresh bounded child/supervisor required")
    budget = Budget(request_ref, request, prior, selected, scientific, context=context)
    live_seconds = time.perf_counter() - live_start
    output = PHASE / selected["output_directory"]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", dict(request=request_ref, task=selected["task"], seed=selected["seed"],
          binding=binding_ref, family_lock=request["family_lock"], supervisor_directory=selected["supervisor_directory"],
          live=context, prior_charged=prior, previous_companion_charges=budget.previous), budget)
    try:
        import torch
        from shared_message_predictive_v3.core import configure, load_inputs
        from shared_message_quality_v3 import counts, data, models
        configure(admitted)
        budget.check()
        setup_start = time.perf_counter()
        export, lineage, metadata, directory = load_inputs(admitted, selected["task"])
        family = data.load_family(metadata, directory, "exact_member", device="cuda:0")
        x = export["x"].to("cuda:0")
        del export
        exact_flags = admitted["execution_policy"][selected["task"]]["exact_member"]
        shared_flags = admitted["execution_policy"][selected["task"]]["shared_mean_message"]
        exact = models.construct(selected["task"], "exact_member", seed=selected["seed"], checkpoint_blocks=exact_flags["checkpoint_blocks"]).to("cuda:0")
        shared = models.construct(selected["task"], "shared_mean", seed=selected["seed"], checkpoint_blocks=shared_flags["checkpoint_blocks"]).to("cuda:0")
        exact.eval(); shared.eval()
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        memory_check(torch)
        spec = counts.TASKS[selected["task"]]
        model_rows = {}
        for arm in ("exact_member", "shared_mean_message"):
            model_rows[arm] = next(r for r in locked["rows"] if r["task"] == selected["task"] and r["seed"] == selected["seed"] and r["arm"] == arm)

        def forward(mode, *, timed=False, copy_cpu=False, validate_finite=False):
            budget.check()
            model, execution = (shared, "packed") if mode == "shared_packed" else (exact, "packed" if mode == "exact_packed" else "serial")
            if model.training or torch.is_grad_enabled():
                raise ValueError("Only dropout-disabled no-grad inference is admitted")
            family.ledger.clear()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            values = models.pooled_inference(model, family, x, execution=execution)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            memory_check(torch)
            if values.shape != (spec["nodes"], spec["classes"]) or values.dtype != torch.float32:
                raise ValueError("Complete FP32 raw-logit pooling required")
            if validate_finite and not torch.isfinite(values).all():
                raise ValueError("Nonfinite complete pooled output")
            memory_check(torch)
            result = values.detach().cpu().clone() if copy_cpu else None
            del values
            budget.check()
            return elapsed if timed else result

        role_reports = []
        with torch.no_grad():
            for role in ROLES:
                budget.check()
                checkpoint_start = time.perf_counter()
                checkpoints = {}
                for arm, model, flags in (("exact_member", exact, exact_flags), ("shared_mean_message", shared, shared_flags)):
                    row = model_rows[arm]
                    checkpoint = row[role + "_checkpoint"]
                    epoch = 300 if role == "final" else row["selected_epoch"]
                    state = torch.load(scientific.checked(checkpoint), map_location="cpu", weights_only=True)
                    if (state.get("schema") != "gnnm-shared-message-checkpoint-v2"
                            or state["cell"] != {"task": selected["task"], "seed": selected["seed"], "arm": arm}
                            or state["epoch"] != epoch or state["binding"] != binding_ref
                            or state["validation_freeze"] != request["validation_freeze"]
                            or state["input_lineage"] != lineage or state["execution"] != flags):
                        raise ValueError("Exact complete-family checkpoint identity differs")
                    model.load_state_dict(state["model_state"], strict=True)
                    del state
                    checkpoints[arm] = dict(checkpoint=checkpoint, epoch=epoch)
                torch.cuda.synchronize()
                loading_seconds = time.perf_counter() - checkpoint_start
                memory_check(torch)
                compare_start = time.perf_counter()
                a = forward("exact_packed", copy_cpu=True)
                b = forward("exact_serial", copy_cpu=True)
                equiv = comparison(torch, a, b, spec["classes"])
                del a, b
                comparison_seconds = time.perf_counter() - compare_start
                warmup_samples = {mode: [] for mode in MODES}
                for mode in MODES:
                    for warmup_index in range(5):
                        warmup_samples[mode].append(forward(mode, timed=True, validate_finite=warmup_index == 0))
                order = round_order(selected["task"], selected["seed"], role)
                samples = {mode: [] for mode in MODES}
                paired = []
                for round_index, modes in enumerate(order):
                    row_samples = {}
                    for mode in modes:
                        row_samples[mode] = forward(mode, timed=True)
                        samples[mode].append(row_samples[mode])
                    paired.append(dict(round_index=round_index, order=modes, seconds=row_samples))
                peaks = {}
                for mode in MODES:
                    budget.check()
                    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
                    baseline = dict(allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0))
                    isolated_seconds = forward(mode, timed=True)
                    peaks[mode] = dict(baseline=baseline, total_forward_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                       total_forward_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                                       no_other_output_live=True, common_residency="one_graph_both_models",
                                       synchronized_forward_seconds=isolated_seconds)
                role_report = dict(role=role, checkpoints=checkpoints, exact_equivalence=equiv,
                                   checkpoint_hash_load_transfer_seconds=loading_seconds,
                                   numerical_comparison_and_CPU_copy_seconds=comparison_seconds,
                                   warmups_per_mode=5, warmup_samples_seconds=warmup_samples,
                                   paired_rounds=paired, samples_seconds=samples,
                                   summary=summary(samples), isolated_forward_memory=peaks)
                write(output / (role + ".json"), role_report, budget)
                role_reports.append(scientific.reference(output / (role + ".json")))
        end = dict(schema="gnnm-trained-inference-child-result-v1", status="completed", request=request_ref,
                   binding=binding_ref, validation_freeze=request["validation_freeze"], family_lock=request["family_lock"],
                   task=selected["task"], seed=selected["seed"], input_lineage=lineage,
                   prepared_manifest=admitted["tasks"][selected["task"]]["prepared_manifest"],
                   graph_fingerprint=metadata["P_edge_only_fingerprint"], graph_storage_bytes=family.full.payload_bytes(),
                   resident_model_parameter_bytes={"exact_member": sum(p.numel() * p.element_size() for p in exact.parameters()),
                                                   "shared_mean": sum(p.numel() * p.element_size() for p in shared.parameters())},
                   input_operator_and_model_setup_seconds=setup_seconds, role_reports=role_reports,
                   admission_costs_seconds=admission_costs, live_and_remaining_budget_preflight_seconds=live_seconds,
                   checkpoint_flags_preserved={"exact": exact_flags["checkpoint_blocks"], "shared": shared_flags["checkpoint_blocks"]},
                   checkpointing_inactive_eval_no_grad=True, timed_region="synchronized full-node forward plus raw-logit pooling",
                   timed_region_exclusions=["input_and_graph_setup", "checkpoint_hashes_and_loading", "source_and_family_audits", "numerical_comparison", "CPU_copy", "artifact_writes", "supervisor_overhead"],
                   training_targets_materialized_by_existing_loader=True, training_targets_indexed_or_scored=False,
                   validation_and_test_target_files_opened=False, new_quality_scores_computed=False, predictions_saved=False,
                   source_family_integrity_audit_may_parse_existing_scores=True, common_residency="one_graph_both_models",
                   maximum_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform.startswith("linux") else 1),
                   hash_work=scientific.hash_report(), inner_elapsed_since_outer_start_seconds=time.monotonic()-budget.start,
                   complete_supervised_wall_requires_outer_terminal=True, live=context, **guard.receipt(guard_state))
        write(output / "RESULT.json", end, budget)
        files = [scientific.reference(output / name) for name in ("START.json", "final.json", "selected.json", "RESULT.json")]
        write(output / "ARTIFACT_MANIFEST.json", dict(schema="gnnm-trained-inference-child-artifacts-v1", files=files), budget)
    except BaseException as error:
        # Preserve partials and outer/inner failures; never retry or relax a cap.
        if not (output / "FAILURE.json").exists():
            payload = dict(type=type(error).__name__, message=str(error), partials_preserved=True, retry_authorized=False)
            try:
                write(output / "FAILURE.json", payload, budget)
            except BaseException:
                pass  # Outer supervisor still records the terminal failure.
        raise


def audit_completed_family(request_ref, request, guard, state):
    """Consumer after six outer terminals; returns full wall costs, no scoring."""
    scientific, admitted, binding_ref, locked, prior = admission(guard.checked(request_ref), request, guard)
    from shared_message_quality_v3.counts import TASKS as SPECS
    results, wall = [], 0.0
    for row in children():
        chain = job_evidence(request_ref, row, scientific, successful=True)
        wall += chain["seconds"]
        output = PHASE / row["output_directory"]
        expected_names = {"START.json", "final.json", "selected.json", "RESULT.json", "ARTIFACT_MANIFEST.json"}
        if {p.name for p in output.iterdir()} != expected_names:
            raise ValueError("Missing/extra/failed companion artifacts")
        manifest = json.loads((output / "ARTIFACT_MANIFEST.json").read_text())
        if manifest != dict(schema="gnnm-trained-inference-child-artifacts-v1", files=[scientific.reference(output / n) for n in ("START.json", "final.json", "selected.json", "RESULT.json")]):
            raise ValueError("Companion artifact bytes differ")
        result = json.loads((output / "RESULT.json").read_text())
        start = json.loads((output / "START.json").read_text())
        if (start.get("request") != request_ref or start.get("binding") != binding_ref
                or start.get("family_lock") != request["family_lock"] or start.get("live") != chain["live"]
                or start.get("supervisor_directory") != row["supervisor_directory"]
                or start.get("task") != row["task"] or start.get("seed") != row["seed"]):
            raise ValueError("Companion START identity/source/route differs")
        flags = admitted["execution_policy"][row["task"]]
        if (result.get("schema") != "gnnm-trained-inference-child-result-v1"
                or result.get("status") != "completed" or result.get("request") != request_ref or result.get("live") != chain["live"]
                or result.get("task") != row["task"] or result.get("seed") != row["seed"]
                or result.get("family_lock") != request["family_lock"] or result.get("binding") != binding_ref
                or result.get("validation_freeze") != request["validation_freeze"]
                or result.get("prepared_manifest") != admitted["tasks"][row["task"]]["prepared_manifest"]
                or result.get("role_reports") != [scientific.reference(output / (role + ".json")) for role in ROLES]
                or result.get("common_residency") != "one_graph_both_models"
                or result.get("checkpointing_inactive_eval_no_grad") is not True
                or result.get("checkpoint_flags_preserved") != {"exact": flags["exact_member"]["checkpoint_blocks"], "shared": flags["shared_mean_message"]["checkpoint_blocks"]}
                or result.get("complete_supervised_wall_requires_outer_terminal") is not True
                or result.get("training_targets_indexed_or_scored") is not False
                or any(result.get(k) is not False for k in ("validation_and_test_target_files_opened", "new_quality_scores_computed", "predictions_saved"))):
            raise ValueError("Wrong companion result source/route/scope")
        guard.audit_receipt(result, state)
        for role in ROLES:
            report = json.loads((output / (role + ".json")).read_text())
            equiv = report["exact_equivalence"]
            maximum = equiv.get("maximum_absolute_discrepancy")
            disagreements = equiv.get("class_decision_disagreement_nodes")
            if (report["role"] != role or report["warmups_per_mode"] != 5 or equiv["allclose"] is not True
                    or equiv["atol"] != 1e-5 or equiv["rtol"] != 1e-5
                    or equiv.get("complete_nodes") != SPECS[row["task"]]["nodes"]
                    or equiv.get("class_disagreement_is_reported_without_an_unfrozen_threshold") is not True
                    or type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum < 0
                    or type(disagreements) is not int or not 0 <= disagreements <= SPECS[row["task"]]["nodes"]):
                raise ValueError("Missing frozen role/equivalence/warmup evidence")
            expected_order = round_order(row["task"], row["seed"], role)
            paired = report["paired_rounds"]
            if (len(paired) != 50 or [v["order"] for v in paired] != expected_order
                    or [v["round_index"] for v in paired] != list(range(50))
                    or any(set(v["seconds"]) != set(MODES) for v in paired)):
                raise ValueError("All frozen paired rounds required")
            samples = {m: [v["seconds"][m] for v in paired] for m in MODES}
            if report["samples_seconds"] != samples or report["summary"] != summary(samples):
                raise ValueError("All timing samples/summary must replay")
            if set(report["warmup_samples_seconds"]) != set(MODES) or any(len(v) != 5 for v in report["warmup_samples_seconds"].values()):
                raise ValueError("All five warmup samples per mode required")
            for values in report["warmup_samples_seconds"].values():
                p90(values)  # Finite positive warmup costs; excluded from summary.
            if set(report["isolated_forward_memory"]) != set(MODES):
                raise ValueError("Every mode's isolated-forward memory is required")
            for mode, peak in report["isolated_forward_memory"].items():
                memory_values = [peak["total_forward_peak_reserved_bytes"], peak["total_forward_peak_allocated_bytes"],
                                 peak["baseline"]["reserved_bytes"], peak["baseline"]["allocated_bytes"]]
                if (any(type(v) is not int or v < 0 for v in memory_values)
                        or peak["total_forward_peak_reserved_bytes"] > 60*2**30 or peak["no_other_output_live"] is not True
                        or peak["common_residency"] != "one_graph_both_models"
                        or not 0 <= peak["baseline"]["reserved_bytes"] <= peak["total_forward_peak_reserved_bytes"]
                        or not 0 <= peak["baseline"]["allocated_bytes"] <= peak["total_forward_peak_allocated_bytes"]
                        or peak["total_forward_peak_allocated_bytes"] > peak["total_forward_peak_reserved_bytes"]):
                    raise ValueError("Unqualified isolated inference memory")
                p90([peak["synchronized_forward_seconds"]])
            for arm in ("exact_member", "shared_mean_message"):
                source_row = next(v for v in locked["rows"] if v["task"] == row["task"] and v["seed"] == row["seed"] and v["arm"] == arm)
                if report["checkpoints"][arm] != dict(checkpoint=source_row[role+"_checkpoint"], epoch=300 if role=="final" else source_row["selected_epoch"]):
                    raise ValueError("Role checkpoint differs from whole-family consumer")
        results.append(dict(task=row["task"], seed=row["seed"], result=scientific.reference(output / "RESULT.json"),
                            complete_supervised_wall_seconds=chain["seconds"], outer_terminal=chain["outer_terminal"]))
    paths = [PHASE / r[k] for r in children() for k in ("outer_directory", "supervisor_directory", "output_directory")]
    disk = allocated(paths)
    if (prior["companion_wall_seconds_before_declared_children"] + wall > 3600
            or prior["combined_phase_seconds_before_declared_children"] + wall > 21600
            or prior["companion_artifact_allocated_bytes_before_declared_children"] + disk + 2**20 > 64*2**20
            or prior["combined_phase_allocated_disk_bytes_before_declared_children"] + disk + 2**20 > 200*2**30):
        raise ValueError("Completed combined companion/phase resource cap exceeded")
    return dict(schema="gnnm-complete-trained-inference-family-consumer-v1", all_six_children_and_twelve_roles_complete=True,
                request=request_ref, results=results, complete_supervised_wall_seconds=wall, companion_artifact_allocated_bytes=disk,
                combined_phase_wall_seconds=prior["combined_phase_seconds_before_declared_children"]+wall,
                combined_phase_allocated_bytes=prior["combined_phase_allocated_disk_bytes_before_declared_children"]+disk,
                no_training_or_new_quality_evaluation=True, quality_or_training_benefit_inferred=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    request_authorized(request)  # A draft fails before source/scientific imports.
    bootstrap_start = time.perf_counter()
    guard, state = bootstrap(request)
    bootstrap_seconds = time.perf_counter() - bootstrap_start
    request_path = guard.confined(args.request)
    request_ref = guard.ref(request_path)
    admission_start = time.perf_counter()
    scientific, admitted, binding_ref, family, prior = admission(request_path, request, guard)
    admission_seconds = time.perf_counter() - admission_start
    selected = next(row for row in children() if row["task"] == args.task and row["seed"] == args.seed)
    run(request, request_ref, selected, admitted, binding_ref, family, prior, scientific, guard, state, guard.confined(args.supervisor),
        dict(source_bootstrap=bootstrap_seconds, complete_family_source_and_prior_audits=admission_seconds))


if __name__ == "__main__":
    main()
