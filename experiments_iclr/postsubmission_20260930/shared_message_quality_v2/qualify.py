"""Guarded synthetic or complete training-engineering qualification only.

Never reads public provider storage, held-out labels, scores or checkpoints.
Fresh short profiling updates are discarded; no tuned weights are saved.
"""
import sys
import os
# Discard script/cwd/PYTHONPATH source paths before stdlib or package imports.
# The external source-bound wrapper has already started its whole deadline.
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import resource
import statistics
import time
import hashlib
import importlib.util

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def bootstrap(request_path):
    """Bind and verify guard source before the first phase-package import."""
    request_path = Path(request_path)
    request_value = json.loads(request_path.read_text())
    manifest_path = HERE / "SOURCE_MANIFEST_v1.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request_value.get("source_manifest_sha256"):
        raise ValueError("Bootstrap manifest differs from the exact root request")
    manifest = json.loads(raw)
    guards = [r for r in manifest["files"] if r["relative_path"] == "import_guard.py"]
    guard_path = HERE / "import_guard.py"
    if guard_path.is_symlink() or len(guards) != 1 or hashlib.sha256(guard_path.read_bytes()).hexdigest() != guards[0]["sha256"]:
        raise ValueError("Bootstrap import guard source changed")
    spec = importlib.util.spec_from_file_location("_shared_message_quality_guard_v2", guard_path)
    guard = importlib.util.module_from_spec(spec)
    # Fresh source execution avoids even a newly introduced bytecode cache.
    exec(compile(guard_path.read_bytes(), str(guard_path), "exec", dont_inherit=True), guard.__dict__)
    return guard, *guard.install(request_value["source_manifest_sha256"])


class MemoryLimitExceeded(RuntimeError):
    pass


def memory(torch, admitted, *, enforce=True):
    row = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    if enforce and row["peak_reserved_bytes"] > admitted["memory_cap_bytes"]:
        raise MemoryLimitExceeded("Declared engineering reserved-memory cap exceeded")
    return row


def payload_bytes(value, torch):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(payload_bytes(v, torch) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(payload_bytes(v, torch) for v in value)
    return 0


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def timed(call, torch, admitted, budget):
    budget.check()
    torch.cuda.synchronize(0)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cpu, start = time.process_time(), time.perf_counter()
    a.record()
    call()
    b.record()
    torch.cuda.synchronize(0)
    row = dict(synchronized_wall_seconds=time.perf_counter() - start,
               CPU_process_seconds=time.process_time() - cpu,
               CUDA_event_seconds=a.elapsed_time(b) / 1000,
               memory=memory(torch, admitted))
    budget.check()
    return row


def real(admitted, manifest_sha, live, budget):
    import torch
    from propagation_cost_engineering_v1.contract import check_gpu
    from propagation_cost_impl_v1.dropout import DropoutKey
    from shared_message_quality_v2.models import construct, make_optimizer, train_step, pooled_inference
    from shared_message_quality_v2.data import load_inputs, load_family
    reports = []
    for task in admitted["tasks"]:
        budget.check()
        phase = json.loads(checked(admitted["engineering_phases"][task]).read_text())
        if phase["task"] != task or phase["authorized_gpu_uuid"] != admitted["gpu_uuid"]:
            raise ValueError("Complete engineering input task/GPU changed")
        check_gpu(phase)
        setup_begin, setup_cpu = time.perf_counter(), time.process_time()
        export, lineage, metadata, directory = load_inputs(admitted["engineering_phases"][task])
        budget.check()
        features = export["x"].to("cuda:0")
        ids = export["training_target_node_ids"].to("cuda:0")
        targets = export["training_targets"].reshape(-1).to(dtype=torch.long, device="cuda:0")
        validation_ids = export["official_split_node_ids"]["valid"].to("cuda:0")
        torch.cuda.synchronize(0)
        input_wall, input_cpu = time.perf_counter() - setup_begin, time.process_time() - setup_cpu
        input_peak_memory = memory(torch, admitted)
        task_plan = plan(task)
        order = variants()
        random.Random(20260930 + TASKS[task]["index"]).shuffle(order)
        rows, first_gradients = [], {}
        for variant in order:
            budget.check()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            model = family = optimizer = None
            row = dict(variant=variant, id=variant_id(variant), seed=17, quality_scores_exist=False,
                       actual_training_forward_backward_AdamW=True, scientific_fit=False)
            try:
                begin, cpu = time.perf_counter(), time.process_time()
                family = load_family(metadata, directory, variant["arm"])
                model = construct(task, variant["arm"], seed=17,
                                  checkpoint_blocks=variant["checkpoint_blocks"]).to("cuda:0")
                optimizer = make_optimizer(model)
                if optimizer.state:
                    raise ValueError("Qualification optimizer must be fresh")
                torch.cuda.synchronize(0)
                row.update(setup_wall_seconds=time.perf_counter() - begin,
                           setup_CPU_process_seconds=time.process_time() - cpu,
                           parameter_plan=model.parameter_plan,
                           runtime_parameter_count=sum(p.numel() for p in model.parameters()),
                           graph_payload=family.storage_report(), setup_memory=memory(torch, admitted))
                def train(epoch):
                    model.train()
                    family.ledger.clear()
                    train_step(model, family, features, ids, targets, optimizer,
                        execution=variant["execution"], dropout_key=DropoutKey(TASKS[task]["index"], 17, epoch))
                def validation():
                    model.eval()
                    family.ledger.clear()
                    with torch.no_grad():
                        pooled = pooled_inference(model, family, features, execution=variant["execution"])
                        values = pooled[validation_ids]
                        # Same label-free reduction work for every arm. The
                        # numerical value is never reported, saved or ranked.
                        work = values.logsumexp(-1).sum() + values.sum()
                        if not bool(torch.isfinite(work)):
                            raise ValueError("Nonfinite label-free complete validation work")
                        del work, values, pooled
                row["cold_first_train"] = timed(lambda: train(1), torch, admitted, budget)
                row["cold_train_operation_summary"] = family.ledger.summary()
                # Same fresh seed/key for every execution variant; compare
                # full-input initial gradients without any validation label.
                gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
                if not all(bool(torch.isfinite(value).all()) for value in gradients.values()):
                    raise ValueError("Nonfinite complete training gradient")
                arm = variant["arm"]
                if arm not in first_gradients:
                    first_gradients[arm] = gradients
                    row["initial_gradient_reference"] = True
                else:
                    reference = first_gradients[arm]
                    if set(reference) != set(gradients):
                        raise ValueError("Execution parameter names changed")
                    tolerance = admitted["equivalence_tolerance"]
                    for name, value in gradients.items():
                        torch.testing.assert_close(value, reference[name], **tolerance)
                    row["initial_gradient_reference"] = False
                    row["maximum_initial_gradient_absolute_error"] = max(
                        float((gradients[name] - reference[name]).abs().max()) for name in gradients)
                del gradients
                row["cold_first_label_free_validation"] = timed(validation, torch, admitted, budget)
                row["warmups"], row["samples"] = [], []
                epoch = 1
                for index in range(admitted["warmups"] + admitted["samples"]):
                    epoch += 1
                    train_row = timed(lambda: train(epoch), torch, admitted, budget)
                    train_summary = family.ledger.summary()
                    validation_row = timed(validation, torch, admitted, budget)
                    sample = dict(training=train_row, label_free_validation=validation_row,
                        combined_synchronized_wall_seconds=train_row["synchronized_wall_seconds"] + validation_row["synchronized_wall_seconds"],
                        schedule_weighted_seconds=train_row["synchronized_wall_seconds"] + validation_row["synchronized_wall_seconds"] / 5,
                        training_operation_summary=train_summary, validation_operation_summary=family.ledger.summary(),
                        engineering_update_id=epoch, dropout_key_field_named_epoch=True,
                        scientific_epoch_claim=False, losses_or_predictions_recorded=False)
                    (row["warmups"] if index < admitted["warmups"] else row["samples"]).append(sample)
                row["median_train_seconds"] = statistics.median(r["training"]["synchronized_wall_seconds"] for r in row["samples"])
                row["median_label_free_validation_seconds"] = statistics.median(r["label_free_validation"]["synchronized_wall_seconds"] for r in row["samples"])
                row["median_combined_seconds"] = statistics.median(r["combined_synchronized_wall_seconds"] for r in row["samples"])
                row["median_schedule_weighted_seconds"] = statistics.median(r["schedule_weighted_seconds"] for r in row["samples"])
                state_bytes = payload_bytes(model.state_dict(), torch)
                optimizer_bytes = payload_bytes(optimizer.state_dict(), torch)
                rng_bytes = payload_bytes(torch.get_rng_state(), torch) + payload_bytes(torch.cuda.get_rng_state_all(), torch)
                members = 1 if arm.startswith("single_") else 4
                row.update(status="qualified_bounded_training_path", peak_memory=memory(torch, admitted),
                    optimizer_step_count=epoch,
                    artifact_potential=dict(weights_tensor_bytes=state_bytes, optimizer_tensor_bytes=optimizer_bytes,
                        RNG_tensor_bytes=rng_bytes,
                        one_checkpoint_tensor_payload_bytes=state_bytes + optimizer_bytes + rng_bytes,
                        one_validation_member_and_pooled_logits_bytes=(members + 1) * validation_ids.numel() * TASKS[task]["classes"] * 4,
                        one_test_member_and_pooled_logits_bytes=(members + 1) * export["official_split_node_ids"]["test"].numel() * TASKS[task]["classes"] * 4,
                        excludes_serialization_IDs_and_metadata=True,
                        no_epoch_fit_count_or_scientific_artifact_policy_bound=True),
                    peak_process_RSS_bytes=rss_bytes(), weights_or_logits_saved=False)
            except (torch.cuda.OutOfMemoryError, MemoryLimitExceeded) as error:
                row.update(status="infeasible_training_path", error_type=type(error).__name__,
                           error=str(error), peak_memory=memory(torch, admitted, enforce=False),
                           path_selection_eligible=False, smaller_graph_fallback=False)
            finally:
                del optimizer, model, family
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            write(budget.output / (("reddit2" if task == "Reddit2" else "products") + "__" + row["id"] + ".json"), row)
            budget.check()
        selected = {}
        for arm in ARMS:
            available = [r for r in rows if r["variant"]["arm"] == arm and r["status"] == "qualified_bounded_training_path"]
            selected[arm] = min(available, key=lambda r: (r["median_schedule_weighted_seconds"], r["id"]))["variant"] if available else None
        reports.append(dict(task=task, plan=task_plan, complete_data_binding=lineage,
            complete_input_replay_H2D_wall_seconds=input_wall, complete_input_replay_H2D_CPU_process_seconds=input_cpu,
            complete_input_peak_memory=input_peak_memory,
            variant_order=[variant_id(v) for v in order], variants=rows, selected_engineering_paths=selected,
            selection_rule="lowest feasible median actual train+label-free-validation/5 wall, then variant ID",
            prospective_schedule_weighting=dict(training_updates=300, validation_passes=60, scientific_schedule_admitted=False),
            every_arm_has_qualified_path=all(v is not None for v in selected.values()),
            transductive_complete_graph=True, inductive_Reddit2_claim=False))
        del first_gradients, features, ids, targets, validation_ids, export
        gc.collect()
        torch.cuda.empty_cache()
    return dict(schema="gnnm-shared-message-complete-training-qualification-v2",
        status="complete_engineering_paths" if all(r["every_arm_has_qualified_path"] for r in reports) else "infeasible_complete_engineering_family",
        source_manifest_sha256=manifest_sha, tasks=reports, actual_train_backward_AdamW_measured=True,
        prior_headroom_was_inference_only=True, prior_headroom_adopted_as_training_measurement=False,
        qualification_seed_only=17, predictive_representativeness_claim=False,
        scientific_training_admitted=False, heldout_targets_loaded=False, public_provider_storage_opened=False,
        quality_scores_computed=False, tuned_weights_or_predictions_saved=False,
        qualified_full_scientific_budget=False, future_scientific_protocol_required=True,
        supervised_wall_seconds_before_manifest=budget.seconds(), CPU_process_seconds=time.process_time(),
        peak_process_RSS_bytes=rss_bytes(), live=live)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("synthetic", "real"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    guard, import_policy, import_tree = bootstrap(args.request)
    global PHASE, Budget, checked, request, sha, write, ARMS, TASKS, plan, variants, variant_id
    from shared_message_quality_v2.contract import PHASE, Budget, checked, request, sha, write, expected_cli, outer_start
    from shared_message_quality_v2.counts import ARMS, TASKS, plan, variants, variant_id
    admitted, request_sha = request(args.request, args.operation)
    from propagation_cost_engineering_v1.contract import live_context, check_gpu
    live = live_context(args.supervisor, Path(__file__))
    argv = expected_cli(args.request, args.operation, args.supervisor)
    command = json.loads((args.supervisor / "command.json").read_text())
    if command["argv"] != argv:
        raise ValueError("Exact request/operation/supervisor CLI required")
    live["outer_start"] = outer_start(os.environ["GNNM_BOUND_START_JSON"], args.request, admitted, args.supervisor, argv)
    budget = Budget(admitted, live)
    budget.check()
    budget.output.mkdir(parents=True, exist_ok=False)
    write(budget.output / "START.json", dict(operation=args.operation, request_sha256=request_sha,
        source_manifest_sha256=admitted["source_manifest_sha256"],
        scope="source-specific synthetic or real CE/backward/AdamW engineering, no scientific fits/scores",
        live=live, heldout_targets_loaded=False))
    try:
        import torch
        policy = admitted["runtime_policy"]
        for package, version in policy["package_versions"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError("Actual frozen package changed: " + package)
        torch.set_num_threads(policy["CPU_threads"])
        torch.use_deterministic_algorithms(policy["deterministic_algorithms"], warn_only=False)
        gpu = check_gpu({"authorized_gpu_uuid": admitted["gpu_uuid"]})
        observed_policy = dict(device="cuda:0", TF32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
            AMP=bool(torch.is_autocast_enabled()), CPU_threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
            package_versions={p: importlib.metadata.version(p) for p in policy["package_versions"]})
        if observed_policy != policy:
            raise ValueError("Observed runtime policy differs from the exact root freeze")
        manifest_sha = admitted["source_manifest_sha256"]
        if args.operation == "synthetic":
            from shared_message_quality_v2.synthetic_checks import run
            torch.cuda.reset_peak_memory_stats(0)
            memory_boundaries = []
            def boundary(name):
                torch.cuda.synchronize(0)
                budget.check()
                memory_boundaries.append(dict(boundary=name, **memory(torch, admitted)))
            boundary("before_synthetic_checks")
            result = run("cuda:0", tolerance=admitted["equivalence_tolerance"], boundary=boundary)
            boundary("after_synthetic_checks")
            result.update(peak_memory=memory(torch, admitted), memory_boundaries=memory_boundaries,
                          memory_cap_bytes=admitted["memory_cap_bytes"],
                          memory_enforcement="peak reserved checked after defined CUDA-synchronized boundaries; no external hard memory limit")
            result.update(source_manifest_sha256=manifest_sha, supervised_wall_seconds=budget.seconds(), live=live)
        else:
            result = real(admitted, manifest_sha, live, budget)
        budget.check()
        result.update(request_sha256=request_sha, gpu=gpu, qualification_requires_completed_supervisor=True,
            qualification_requires_completed_outer_terminal=True, runtime_policy=admitted["runtime_policy"],
            observed_runtime_policy=observed_policy,
            equivalence_tolerance=admitted["equivalence_tolerance"], runtime_import_tree=import_tree,
            imported_phase_modules=guard.imported_modules(import_policy),
            synthetic_admission_binding=admitted.get("_synthetic_chain"))
        write(budget.output / "RESULT.json", result)
        files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(budget.output.iterdir()) if p.is_file()}
        write(budget.output / "artifact_manifest.json", dict(schema="gnnm-shared-message-engineering-artifacts-v2", files=files,
            tuned_weights_or_predictions_saved=False))
        budget.check()
        print(json.dumps(dict(status=result["status"], operation=args.operation, scores_exist=False)), flush=True)
    except BaseException as error:
        write(budget.output / "FAILURE.json", dict(type=type(error).__name__, message=str(error),
            supervised_wall_seconds=budget.seconds(), partial_metadata_preserved=True,
            peak_memory=memory(torch, admitted, enforce=False) if "torch" in locals() and torch.cuda.is_initialized() else None,
            scientific_training_admitted=False, heldout_labels_loaded=False, scores_exist=False))
        raise


if __name__ == "__main__":
    main()
