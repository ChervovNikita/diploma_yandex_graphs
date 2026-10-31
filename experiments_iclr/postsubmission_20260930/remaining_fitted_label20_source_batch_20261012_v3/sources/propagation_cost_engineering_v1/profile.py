"""Bounded complete-graph profiling; no held-out targets or saved predictions."""

from __future__ import annotations

import gc
import json
import math
import statistics
import time

import torch

from .contract import SCHEMA, TASKS, FORECAST_GPU_SECONDS, TOTAL_GPU_SECONDS, WallBudget, check_gpu, check_peak, output_path
from .complete_models import (construct_source_arm, train_step, inference_validation_work,
                              CompleteCSR, CompleteFamily, OperationLedger, detached_aggregation)
from .native_source import NativeArm
from .prepare import audit_prepared, load_family
from propagation_cost_impl_v1.dropout import DropoutKey
from propagation_cost_impl_v1.models import centered_aggregation


def variants():
    result = []
    for arm in ("candidate", "detached_current_member_mean", "exact_member", "single_base", "naive_shared_sample"):
        executions = ["packed"] if arm in ("candidate", "detached_current_member_mean") else ["single"] if arm == "single_base" else ["packed", "serial"]
        paths = ["lazy_two_colors", "cached_half"] if arm == "naive_shared_sample" else ["lazy_two_colors"]
        for execution in executions:
            for path in paths:
                for ckpt in (False, True):
                    result.append({"arm": arm, "execution": execution, "naive_path": path,
                                   "checkpoint_blocks": ckpt, "native_source": False})
    for arm in ("exact_member", "single_base"):
        result.append({"arm": arm, "execution": "serial" if arm == "exact_member" else "single",
                       "naive_path": "lazy_two_colors", "checkpoint_blocks": False, "native_source": True})
    return result


def variant_name(v):
    return "__".join((v["arm"], v["execution"], v["naive_path"],
                      "checkpoint" if v["checkpoint_blocks"] else "retained", "native" if v["native_source"] else "csr"))


def write(path, value):
    path = output_path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def timed(function, budget):
    budget.check()
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    function()
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - start
    budget.check()
    return elapsed


def detached_local_check():
    """Three-node fixed-state algebra check; no data or predictive claim."""
    ledger = OperationLedger()
    p = torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device="cuda:0")
    c = torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device="cuda:0")
    ops = {}
    for name, dense in (("full", p), ("color0", c)):
        ops[name] = CompleteCSR(name, dense.to_sparse_csr(), dense.T.to_sparse_csr(), ledger, 3)
    family = CompleteFamily(ops, {"nodes": 3, "canonical_edges": 4,
                                  "P_edge_only_fingerprint": "fixed-detached-local-algebra"}, ledger, "cuda:0")
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(441)
        z = torch.randn(3, 4, 5, device="cuda:0", requires_grad=True)
        upstream = torch.randn_like(z)
    regular = centered_aggregation(family, z, 0)
    detached = detached_aggregation(family, z, 0)
    torch.testing.assert_close(regular, detached, atol=0, rtol=0)
    regular_grad = torch.autograd.grad((regular * upstream).sum(), z)[0]
    detached_grad = torch.autograd.grad((detached * upstream).sum(), z)[0]
    q = c * 4.0
    expected_detached = torch.stack([q.T @ upstream[:, m] for m in range(4)], dim=1)
    expected_regular = expected_detached + ((p - q).T @ upstream.mean(dim=1))[:, None, :]
    torch.testing.assert_close(detached_grad, expected_detached, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(regular_grad, expected_regular, atol=3e-6, rtol=3e-6)
    difference = float((regular_grad - detached_grad).abs().max().cpu())
    if difference <= 0:
        raise RuntimeError("Ablation fixture failed to distinguish the local adjoints")
    # A detached full-mean branch contributes no full-P backward sparse call.
    full_backwards = [r for r in ledger.sparse if r["operator"] == "full" and r["direction"] == "backward"]
    if len(full_backwards) != 1:
        raise RuntimeError("Detached branch unexpectedly backpropagated through full P")
    return {"status": "passed", "nodes": 3, "channels": 5, "members": 4,
            "identical_fixed_state_forward": True, "local_adjoint_max_difference": difference,
            "full_P_backward_calls_two_arms_total": 1, "dtype": "float32",
            "predictive_claim": False, "sparse_operations": ledger.sparse}


@torch.no_grad()
def final_unlabeled_work(model, family, features, test_indices, execution):
    # Real test-node IDs are structural inputs. No target values or numerical
    # quality summaries are read, produced or written by this timing proxy.
    if model.arm == "single_base":
        logits = model(family, features, colors=(0, 2))
    else:
        logits = model(family, features, colors=(0, 2), execution=execution)
    pooled = logits.mean(dim=0)
    member_probs = logits[:, test_indices, :].softmax(dim=-1)
    pooled_values = pooled[test_indices]
    pooled_probs = pooled_values.softmax(dim=-1)
    work = pooled_values.logsumexp(-1).sum() + pooled_probs.square().sum()
    confidence, predicted = pooled_probs.max(dim=-1)
    bins = torch.bincount(torch.clamp((confidence * 15).to(torch.long), max=14), minlength=15)
    for a in range(member_probs.shape[0]):
        for b in range(a):
            work = work + (member_probs[a] - member_probs[b]).square().sum()
    del logits, pooled, member_probs, pooled_values, pooled_probs, work, confidence, predicted, bins


@torch.no_grad()
def full_source_equivalence(export, metadata, directory, features, task, budget, output):
    """Literal source inference, one member at a time, streamed reductions."""
    results = []
    raw_edges = export["edge_index"].to("cuda:0")
    family = load_family(metadata, directory, "exact_member", "lazy_two_colors")
    for arm in ("exact_member", "single_base"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        native = NativeArm(task, arm, raw_edges).to("cuda:0").eval()
        optimized = construct_source_arm(task, arm).to("cuda:0").eval()
        row = {"arm": arm, "kind": "complete_features_full_width_source_equivalence",
               "dropout": False, "members": 4 if arm == "exact_member" else 1,
               "atol": 0.0002, "rtol": 0.0002, "all_parameter_gradients_checked": False}
        original = exact = ref = value = error = None
        member_rows = []
        try:
            for member in range(row["members"]):
                budget.check()
                torch.cuda.synchronize(0)
                start = time.perf_counter()
                original = native.forward_member(family, features, member)
                torch.cuda.synchronize(0)
                original_seconds = time.perf_counter() - start
                start = time.perf_counter()
                exact = (optimized.forward_member(family, features, member) if arm == "exact_member"
                         else optimized(family, features)[0])
                torch.cuda.synchronize(0)
                exact_seconds = time.perf_counter() - start
                maximum, squared_error, squared_reference, elements = 0.0, 0.0, 0.0, 0
                for begin in range(0, task["nodes"], 65536):
                    ref, value = original[begin:begin + 65536], exact[begin:begin + 65536]
                    if not torch.isfinite(ref).all() or not torch.isfinite(value).all():
                        raise RuntimeError("Nonfinite complete source-equivalence logits")
                    if not torch.isclose(ref, value, atol=row["atol"], rtol=row["rtol"]).all():
                        raise RuntimeError("Complete frozen source logits disagree")
                    error = (ref - value).double()
                    maximum = max(maximum, float(error.abs().max().cpu()))
                    squared_error += float(error.square().sum().cpu())
                    squared_reference += float(ref.double().square().sum().cpu())
                    elements += ref.numel()
                member_rows.append({"member": member, "max_absolute_error": maximum,
                                    "normalized_L2_error": math.sqrt(squared_error / max(squared_reference, 1.0)),
                                    "elements_compared": elements,
                                    "source_forward_seconds": original_seconds,
                                    "optimized_serial_forward_seconds": exact_seconds})
                del original, exact, ref, value, error
                check_peak()
            row.update(status="passed", member_reductions=member_rows, **check_peak())
        except torch.cuda.OutOfMemoryError as error:
            row.update(status="resource_failed_OOM", error_type=type(error).__name__, message=str(error),
                       completed_member_reductions=member_rows,
                       complete_source_equivalence_claim_allowed=False)
        except RuntimeError as error:
            if "reserved-memory ceiling" not in str(error):
                raise
            row.update(status="resource_failed_memory_cap", message=str(error),
                       completed_member_reductions=member_rows,
                       complete_source_equivalence_claim_allowed=False)
        results.append(row)
        write(output / ("SOURCE_EQUIVALENCE_" + arm + ".json"), row)
        original = exact = ref = value = error = None
        del native, optimized
        gc.collect()
        torch.cuda.empty_cache()
    del raw_edges, family
    gc.collect()
    torch.cuda.empty_cache()
    return results


def profile_complete(phase, export, data_binding, output, phase_sha, live):
    output = output_path(output)
    output.mkdir(parents=True)
    gpu = check_gpu(phase)
    budget = WallBudget(min(phase["profile_gpu_seconds_cap"],
                            TOTAL_GPU_SECONDS - phase["prior_gpu_seconds"]), phase.get("deadline_utc"))
    write(output / "DETACHED_LOCAL_ADJOINT_CHECK.json", detached_local_check())
    prepared = output_path(phase["repo_root"] + "/" + phase["prepared_manifest_relative_to_repo"])
    metadata, directory = audit_prepared(prepared, phase, data_binding)
    task = TASKS[phase["task"]]
    if metadata["nodes"] != task["nodes"] or metadata["canonical_edges"] <= 0:
        raise ValueError("Complete prepared graph does not match adopted task")
    cpu_input_audit_seconds = budget.elapsed()
    copy_started = time.perf_counter()
    features = export["x"].to("cuda:0")
    train_indices = export["training_target_node_ids"].to("cuda:0")
    train_labels = export["training_targets"].reshape(-1).to(dtype=torch.long, device="cuda:0")
    validation_indices = export["official_split_node_ids"]["valid"].to("cuda:0")
    test_indices = export["official_split_node_ids"]["test"].to("cuda:0")
    torch.cuda.synchronize(0)
    complete_input_H2D_seconds = time.perf_counter() - copy_started
    rows = []
    write(output / "PROFILE_START.json", {"schema": SCHEMA, "phase_sha256": phase_sha,
          "live": live, "gpu": gpu, "data_binding": data_binding,
          "prepared_manifest_sha256": phase["prepared_manifest_sha256"],
          "variant_count": len(variants()), "scientific_fits": 0,
          "training_steps_per_variant_max": 5, "heldout_targets": False})
    source_equivalence = full_source_equivalence(export, metadata, directory, features, task, budget, output)
    for v in variants():
        budget.check()
        name = variant_name(v)
        row = {**v, "name": name, "task": phase["task"], "status": "started"}
        model = family = optimizer = raw_edges = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        try:
            if v["native_source"]:
                family = CompleteFamily({}, metadata, OperationLedger(), "cuda:0")
                raw_edges = export["edge_index"].to("cuda:0")
                model = NativeArm(task, v["arm"], raw_edges).to("cuda:0")
            else:
                family = load_family(metadata, directory, v["arm"], v["naive_path"])
                model = construct_source_arm(task, v["arm"], checkpoint_blocks=v["checkpoint_blocks"],
                                             naive_path=v["naive_path"]).to("cuda:0")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001,
                                         betas=(0.9, 0.999), eps=1e-8)
            torch.cuda.synchronize(0)
            row["operator_model_optimizer_H2D_setup_seconds"] = time.perf_counter() - start
            row["graph_GPU_payload"] = family.storage_report()
            row["native_raw_edge_GPU_payload_bytes"] = 0 if raw_edges is None else raw_edges.numel() * raw_edges.element_size()
            model.eval()
            row["cold_first_prediction_seconds"] = timed(
                lambda: inference_validation_work(model, family, features, validation_indices, v["execution"]), budget)
            check_peak()
            row["inference_setup_upper_seconds"] = row["operator_model_optimizer_H2D_setup_seconds"]
            timed(lambda: inference_validation_work(model, family, features, validation_indices, v["execution"]), budget)
            infer_times = [timed(lambda: inference_validation_work(
                model, family, features, validation_indices, v["execution"]), budget) for _ in range(3)]
            final_time = timed(lambda: final_unlabeled_work(model, family, features, test_indices, v["execution"]), budget)
            check_peak()
            row.update(inference_status="passed", inference_validation_proxy_wall_seconds=infer_times,
                       final_test_metric_proxy_wall_seconds=final_time,
                       inference_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                       inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
            check_peak()
            torch.cuda.reset_peak_memory_stats(0)
            model.train()
            # One warm step then all four color stress cases. These are
            # score-blind engineering updates, NOT trajectories in the study.
            timed(lambda: train_step(model, family, features, train_indices, train_labels, optimizer,
                                    colors=(0, 2), dropout_key=DropoutKey(task["index"], 0, 0),
                                    execution=v["execution"]), budget)
            train_times = []
            for color in range(4):
                train_times.append(timed(
                    lambda color=color: train_step(
                        model, family, features, train_indices, train_labels, optimizer,
                        colors=(color, color), dropout_key=DropoutKey(task["index"], 0, color + 1),
                        execution=v["execution"]), budget))
                check_peak()
            row.update(status="passed", train_step_wall_seconds=train_times,
                       inference_validation_proxy_wall_seconds=infer_times,
                       final_test_metric_proxy_wall_seconds=final_time,
                       training_steps=5, parameter_count=sum(p.numel() for p in model.parameters()),
                       training_profile_scope="seed0 complete official training nodes; outputs discarded",
                       stress_layer_colors=[[c, c] for c in range(4)],
                       operation_summary=family.ledger.summary(), **check_peak())
            write(output / (name + "__OPERATIONS.json"), {"sparse": family.ledger.sparse, "dense": family.ledger.dense})
        except torch.cuda.OutOfMemoryError as error:
            row.update(status="resource_failed_OOM", error_type=type(error).__name__, message=str(error),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
        except RuntimeError as error:
            if "reserved-memory ceiling" not in str(error):
                write(output / (name + "__FAILURE.json"), {**row, "status": "failed", "message": str(error)})
                raise
            row.update(status="resource_failed_memory_cap", message=str(error),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
        finally:
            row["variant_total_wall_seconds"] = time.perf_counter() - start
            del model, family, optimizer, raw_edges
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        write(output / (name + "__RESULT.json"), row)
    result = {"schema": SCHEMA, "kind": "complete_graph_score_blind_profile",
              "status": "completed_engineering_profiles", "task": phase["task"],
              "phase_sha256": phase_sha, "live": live, "data_binding": data_binding,
              "prepared_manifest_sha256": phase["prepared_manifest_sha256"],
              "gpu": gpu, "variants": rows, "source_equivalence": source_equivalence,
              "actual_canonical_edges": metadata["canonical_edges"],
              "operator_nnz": metadata["operator_nnz"],
              "actual_prepared_file_bytes": metadata["actual_prepared_file_bytes"],
              "preparation_wall_seconds": metadata["preparation_wall_seconds"],
              "preparation_peak_process_RSS_bytes": metadata["preparation_peak_process_RSS_bytes"],
              "prior_cpu_seconds": phase["prior_cpu_seconds"], "prior_disk_bytes": phase["prior_disk_bytes"],
              "input_audit_CPU_wall_seconds": cpu_input_audit_seconds,
              "complete_input_H2D_seconds": complete_input_H2D_seconds,
              "actual_engineering_GPU_phase_wall_seconds": budget.elapsed(),
              "profiled_optimizer_seeds": [0], "timing_units_are_scientific_fits": False,
              "quality_scores_written": False, "heldout_target_values_loaded": False,
              "models_checkpoints_or_logits_saved": False,
              "validation_and_final_metric_timings_are_label_free_proxies": True,
              "scientific_study_admission": False}
    write(output / "PROFILE_RESULT.json", result)
    return result


def choose_paths(result):
    """Conservative max unit times; score-blind choices, no predictive inputs."""
    chosen = {}
    for arm in ("candidate", "detached_current_member_mean", "exact_member", "single_base", "naive_shared_sample"):
        viable = [row for row in result["variants"] if row["arm"] == arm and row["status"] == "passed"]
        if not viable:
            return {"status": "incomplete_no_feasible_arm", "arm": arm}
        inference_viable = [row for row in result["variants"] if row["arm"] == arm
                            and row.get("inference_status") == "passed"
                            and row.get("inference_peak_reserved_bytes", float("inf")) <= 60 * 2**30]
        training = min(viable, key=lambda r: (max(r["train_step_wall_seconds"])
                                             + max(r["inference_validation_proxy_wall_seconds"]), r["name"]))
        inference = min(inference_viable, key=lambda r: (max(r["inference_validation_proxy_wall_seconds"]), r["name"]))
        chosen[arm] = {"training_and_validation_path": training["name"], "fastest_inference_path": inference["name"],
                       "train_seconds_upper_unit": max(training["train_step_wall_seconds"]),
                       "training_setup_seconds": training["operator_model_optimizer_H2D_setup_seconds"] + result["complete_input_H2D_seconds"],
                       "inference_switch_setup_seconds": (0.0 if inference["name"] == training["name"]
                                                           else inference["operator_model_optimizer_H2D_setup_seconds"]),
                       "validation_seconds_upper_unit": max(training["inference_validation_proxy_wall_seconds"]),
                       "inference_seconds_upper_unit": max(inference["inference_validation_proxy_wall_seconds"]),
                       "final_metric_proxy_seconds": max(training["final_test_metric_proxy_wall_seconds"],
                                                         inference["final_test_metric_proxy_wall_seconds"]),
                       "training_peak_reserved_bytes": training["peak_reserved_bytes"],
                       "inference_peak_reserved_bytes": inference["inference_peak_reserved_bytes"]}
    return {"status": "all_five_arms_have_feasible_execution", "choices": chosen}


def forecast_family(results, prior_gpu_seconds):
    if {r["task"] for r in results} != set(TASKS) or len(results) != 2:
        raise ValueError("Complete 30-fit forecast requires both complete adopted graphs")
    selected = {r["task"]: choose_paths(r) for r in results}
    if any(v["status"] != "all_five_arms_have_feasible_execution" for v in selected.values()):
        return {"schema": SCHEMA, "status": "incomplete", "selected": selected,
                "scientific_training_admitted": False}
    work, profiling = 0.0, sum(r["actual_engineering_GPU_phase_wall_seconds"] for r in results)
    breakdown = []
    for task, choice in selected.items():
        for arm, row in choice["choices"].items():
            # 3 fits x100 epochs/full validations, complete final evaluation,
            # 10 warmups+50 required timed predictions per trained seed.
            seconds = 1.25 * 3 * (100 * (row["train_seconds_upper_unit"] + row["validation_seconds_upper_unit"])
                                 + row["training_setup_seconds"] + row["inference_switch_setup_seconds"]
                                 + row["final_metric_proxy_seconds"] + 60 * row["inference_seconds_upper_unit"])
            work += seconds
            breakdown.append({"task": task, "arm": arm, "seeds": 3, "epochs": 100,
                              "forecast_seconds_with_25_percent_margin": seconds})
    total = float(prior_gpu_seconds) + profiling + work
    CPU_total = sum(r["preparation_wall_seconds"] + r["input_audit_CPU_wall_seconds"] for r in results) + max(r["prior_cpu_seconds"] for r in results)
    disk_total = sum(r["actual_prepared_file_bytes"] for r in results) + max(r["prior_disk_bytes"] for r in results)
    ancillary_pass = CPU_total <= 16 * 3600 and disk_total <= 200 * 2**30
    return {"schema": SCHEMA, "kind": "complete_30fit_score_blind_budget_forecast",
            "status": "resource_forecast_pass" if total <= FORECAST_GPU_SECONDS and ancillary_pass else "resource_forecast_fail",
            "selected": selected, "breakdown": breakdown, "fit_count": 30,
            "prior_engineering_GPU_seconds": prior_gpu_seconds,
            "complete_graph_profiling_GPU_phase_wall_seconds": profiling,
            "forecast_future_fits_validations_final_and_timings_seconds": work,
            "total_GPU_seconds_forecast": total, "ceiling_GPU_seconds": FORECAST_GPU_SECONDS,
            "CPU_preparation_and_audit_wall_seconds_charged": CPU_total,
            "extra_disk_bytes_charged": disk_total, "CPU_and_disk_caps_pass": ancillary_pass,
            "uncertainty_margin_fraction": 0.25,
            "measurement_scope": "seed0 short complete-graph timing units; not long-run convergence",
            "heldout_target_values_or_predictive_scores_used": False,
            "scientific_training_admitted": False,
            "remaining_root_gates": ["completed supervisor audit", "source-equivalence scope",
                                     "frozen execution choices", "CPU/disk/time caps", "separate scientific admission"]}
