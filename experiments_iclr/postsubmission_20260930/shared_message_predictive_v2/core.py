"""Fresh fixed-budget fits, compact evidence, validation replays and complete lock."""
import json
import math
import os
from pathlib import Path
import statistics
import time
from .contract import (ARMS, EPOCHS, HERE, MODEL_ARMS, PHASE, SEEDS, TASKS, Budget,
                       audit_manifest, audit_runtime_import_receipt, cells, checked, completed, digest, hash_report,
                       manifest, prescribed, reference, runtime_import_receipt, within, write)
from .metrics import choose_member, from_statistics, summarize


def configure(admitted):
    import importlib.metadata
    import torch
    policy = admitted["runtime_policy"]
    if policy.get("device") != "cuda:0" or policy.get("AMP") is not False or policy.get("TF32") is not False:
        raise ValueError("Float32 CUDA0 precision/runtime policy differs")
    for name, version in policy["package_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Actual frozen runtime package changed: " + name)
    torch.set_num_threads(policy["CPU_threads"])
    torch.use_deterministic_algorithms(policy["deterministic_algorithms"], warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from propagation_cost_engineering_v1.contract import check_gpu
    gpu = check_gpu({"authorized_gpu_uuid": admitted["gpu_uuid"]})
    if torch.cuda.memory_allocated(0) or torch.cuda.memory_reserved(0):
        raise ValueError("Scientific cell must start in a fresh isolated GPU process")
    return gpu


def validation_freeze(path, admitted, binding_ref):
    value = json.loads(within(path, HERE).read_text())
    if (value.get("schema") != "gnnm-shared-message-validation-freeze-v2" or value.get("binding") != binding_ref
            or value.get("root_qualified") is not True or value.get("test_targets_loaded_or_referenced") is not False
            or set(value.get("validation_exports", {})) != set(TASKS)):
        raise ValueError("Both completed separately exported validation targets must be frozen")
    for task, row in value["validation_exports"].items():
        checked(row["bundle"])
        checked(row["receipt"])
        chain = completed(PHASE / row["supervisor_directory"], row["outer_start"])
        if chain["gpu_uuid"] != admitted["gpu_uuid"]:
            raise ValueError("Validation exporter used another frozen runtime")
    return value, reference(path)


def load_inputs(admitted, task):
    from shared_message_quality_v3.data import load_inputs as load
    export, lineage, metadata, directory = load(admitted["tasks"][task]["engineering_phase"])
    if lineage["acquisition_sha256"] != admitted["tasks"][task]["acquisition"]["sha256"]:
        raise ValueError("Full input lineage differs from actual root binding")
    return export, lineage, metadata, directory


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def save_tensor(path, value, budget):
    import torch
    budget.check(tensor_bytes(value) + 2**20)
    with within(path, HERE).open("xb") as stream:
        torch.save(value, stream)
    budget.committed(path)
    ref = reference(path)
    budget.check()
    return ref


def fingerprint(state):
    from propagation_cost_engineering_v1.data_contract import _tensor_record
    return digest({k: _tensor_record(v.contiguous()) for k, v in sorted(state.items())})


def setup(admitted, cell, export, metadata, directory):
    from shared_message_quality_v3 import models, data, counts
    variant = admitted["execution_policy"][cell["task"]][cell["arm"]]
    internal_arm = MODEL_ARMS[cell["arm"]]
    family = data.load_family(metadata, directory, internal_arm, device="cuda:0")
    model = models.construct(cell["task"], internal_arm, seed=cell["seed"], checkpoint_blocks=variant["checkpoint_blocks"]).to("cuda:0")
    actual = sum(p.numel() for p in model.parameters())
    plan = counts.plan(cell["task"])
    if actual != plan["parameter_counts"][internal_arm]:
        raise ValueError("Actual source model parameter count differs from analytic control plan")
    parameter_receipt = json.loads(checked(admitted["tasks"][cell["task"]]["parameter_control_report"]).read_text())
    if parameter_receipt.get("source_plan_sha256") != digest(plan) or parameter_receipt.get("qualified") is not True:
        raise ValueError("Root parameter control report did not bind the actual deterministic source plan")
    return (model, family, export["x"].to("cuda:0"), variant,
            {"source_plan": plan, "actual_parameters": actual,
             "internal_arm": internal_arm})


def snapshot(model, optimizer, cell, epoch, binding_ref, freeze_ref, lineage, variant):
    import torch
    from shared_message_quality_v3.counts import TASKS as SPECS
    return {"schema": "gnnm-shared-message-checkpoint-v2", "cell": cell, "epoch": epoch,
            "model_state": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "dropout_stream": {"type": "stateless DropoutKey", "task_index": SPECS[cell["task"]]["index"],
                               "optimizer_seed": cell["seed"], "epoch": epoch, "colors": "none; complete P only"},
            "binding": binding_ref, "validation_freeze": freeze_ref, "input_lineage": lineage, "execution": variant}


def fit(admitted, binding_ref, freeze, freeze_ref, cell, context):
    import torch
    from shared_message_quality_v3 import models, counts
    from propagation_cost_impl_v1.dropout import DropoutKey
    from .labels import load_validation
    from .metrics import full_split
    output = prescribed(admitted, "fits", cell)
    if output.exists():
        raise ValueError("Exclusive fresh cell required; no retries/resume/overwrite")
    require_order(admitted, cell, "fits")
    budget = Budget(admitted, context)
    gpu = configure(admitted)
    export, lineage, metadata, directory = load_inputs(admitted, cell["task"])
    targets = load_validation(freeze["validation_exports"][cell["task"]], cell["task"], export, admitted, binding_ref)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "epochs", "validation_logits"):
        (output / name).mkdir()
    write(output / "START.json", {"binding": binding_ref, "cell": cell, "validation_freeze": freeze_ref,
          "supervisor_directory": context["supervisor_relative_to_project"], "live": context,
          "provider_storage_opened": False, "test_targets_loaded": False})
    torch.cuda.reset_peak_memory_stats(0)
    model, family, x, variant, model_info = setup(admitted, cell, export, metadata, directory)
    optimizer = models.make_optimizer(model)
    if optimizer.state:
        raise ValueError("Own-CE AdamW optimizer must be fresh")
    train_ids = export["training_target_node_ids"].to("cuda:0")
    train_targets = export["training_targets"].reshape(-1).to(dtype=torch.long, device="cuda:0")
    valid_ids = export["official_split_node_ids"]["valid"].to("cuda:0")
    initial = snapshot(model, optimizer, cell, 0, binding_ref, freeze_ref, lineage, variant)
    initial_hash = fingerprint(initial["model_state"])
    initial_ref = save_tensor(output / "INITIAL.pt", initial, budget)
    del initial
    trace, selected, best_logits, final_logits = [], None, None, None
    for epoch in range(1, EPOCHS + 1):
        budget.check()
        model.train()
        torch.cuda.synchronize(0)
        start = time.perf_counter()
        loss = models.train_step(model, family, x, train_ids, train_targets, optimizer,
             execution=variant["training"], dropout_key=DropoutKey(counts.TASKS[cell["task"]]["index"], cell["seed"], epoch), return_loss=True)
        torch.cuda.synchronize(0)
        train_seconds = time.perf_counter() - start
        value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)
        if not math.isfinite(value):
            raise ValueError("Nonfinite fresh own-member training CE")
        row = {"epoch": epoch, "training_own_CE_preupdate": value, "train_update_seconds": train_seconds,
               "validation": None, "sufficient_statistics": None, "improved_checkpoint": None}
        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            member, pooled = models.split_logits(model, family, x, valid_ids, execution=variant["validation"])
            torch.cuda.synchronize(0)
            row["validation_forward_and_CPU_transfer_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            measured, stats = full_split(member, pooled, targets)
            row["CPU_metric_seconds"] = time.perf_counter() - start
            row["validation"], row["sufficient_statistics"] = measured, stats
            if selected is None or (measured["NLL"], epoch) < (selected["validation"]["NLL"], selected["epoch"]):
                start = time.perf_counter()
                row["improved_checkpoint"] = save_tensor(output / "checkpoints" / f"improved_epoch{epoch:03d}.pt",
                    snapshot(model, optimizer, cell, epoch, binding_ref, freeze_ref, lineage, variant), budget)
                row["checkpoint_transfer_write_hash_seconds"] = time.perf_counter() - start
                selected = row
                best_logits = {"member_logits": member, "pooled_logits": pooled, "epoch": epoch}
            if epoch == EPOCHS:
                final_logits = {"member_logits": member, "pooled_logits": pooled, "epoch": epoch}
            del member, pooled
        write(output / "epochs" / f"epoch{epoch:03d}.json", row)
        trace.append(row)
    final_ref = save_tensor(output / "FINAL.pt", snapshot(model, optimizer, cell, EPOCHS, binding_ref, freeze_ref, lineage, variant), budget)
    logits_refs = {}
    for role, payload in (("selected", best_logits), ("final", final_logits)):
        logits_refs[role] = save_tensor(output / "validation_logits" / (role + ".pt"),
            {"schema": "gnnm-shared-message-validation-logits-v2", "cell": cell, "node_ids": export["official_split_node_ids"]["valid"], **payload}, budget)
    budget.check()
    end = {"schema": "gnnm-shared-message-complete300-fit-v2", "status": "completed", "cell": cell,
           "epochs": EPOCHS, "binding": binding_ref, "validation_freeze": freeze_ref, "initial": initial_ref,
           "initial_parameter_fingerprint": initial_hash, "selected_epoch": selected["epoch"],
           "selected_checkpoint": selected["improved_checkpoint"], "final_checkpoint": final_ref,
           "validation_logits": logits_refs, "trace": trace, "execution": variant, "model_info": model_info,
           "selected_validation": selected["validation"], "final_validation": trace[-1]["validation"],
           "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)},
           "test_targets_loaded": False, "public_provider_storage_opened": False,
           "full_official_training_targets_each_update": True, "validation_cadence": 5,
           "no_early_stopping_HPO_scheduler_or_resume": True, "gpu": gpu, "live": context,
           "pre_end_scientific_wall_seconds": time.monotonic() - budget.start,
           "authoritative_full_wall_cost": "successful bounded outer terminal includes bootstrap, setup, metrics, transfers, writes, hashes and supervisor teardown",
           "hash_work": hash_report(), "qualification": "requires completed supervisor and selected/final replay"}
    if end["memory"]["peak_reserved_bytes"] > 60 * 2**30:
        raise RuntimeError("Isolated complete training memory exceeds admission cap")
    end.update(runtime_import_receipt(require_generated=True))
    write(output / "FIT_END.json", end)
    manifest(output)


def audit_fit(admitted, binding_ref, freeze_ref, cell):
    output = prescribed(admitted, "fits", cell)
    audit_manifest(output)
    end = json.loads((output / "FIT_END.json").read_text())
    start = json.loads((output / "START.json").read_text())
    chain = completed(PHASE / start["supervisor_directory"], start["live"]["outer_start"])
    if (Path(chain["argv"][1]).resolve() != HERE / "runner.py" or chain["argv"][2] != "fit"
            or chain["gpu_uuid"] != admitted["gpu_uuid"] or end.get("schema") != "gnnm-shared-message-complete300-fit-v2"
            or end.get("status") != "completed" or end.get("cell") != cell or end.get("binding") != binding_ref
            or end.get("validation_freeze") != freeze_ref or end.get("epochs") != EPOCHS
            or end.get("test_targets_loaded") is not False or end.get("execution") != admitted["execution_policy"][cell["task"]][cell["arm"]]):
        raise ValueError("Incomplete/unrelated/wrong-route fit chain")
    if end.get("live") != chain["live"] or start.get("live") != chain["live"]:
        raise ValueError("Fit source-bound live/outer chain differs")
    audit_runtime_import_receipt(end, require_generated=True)
    trace = end["trace"]
    if [r["epoch"] for r in trace] != list(range(1, EPOCHS + 1)):
        raise ValueError("All 300 ordered full-graph updates required")
    improved, best = [], None
    for row in trace:
        if not math.isfinite(row["training_own_CE_preupdate"]):
            raise ValueError("Invalid training trace")
        if json.loads((output / "epochs" / f"epoch{row['epoch']:03d}.json").read_text()) != row:
            raise ValueError("Duplicated trace record differs")
        expected_validation = row["epoch"] % 5 == 0 or row["epoch"] == EPOCHS
        if not expected_validation:
            if row["validation"] is not None or row["sufficient_statistics"] is not None or row["improved_checkpoint"] is not None:
                raise ValueError("Unprescribed validation/selection epoch")
            continue
        if from_statistics(row["sufficient_statistics"]) != row["validation"]:
            raise ValueError("Scalar sufficient-statistic replay differs")
        is_best = best is None or (row["validation"]["NLL"], row["epoch"]) < (best["validation"]["NLL"], best["epoch"])
        if is_best != (row["improved_checkpoint"] is not None):
            raise ValueError("Every and only improved secondary state must be preserved")
        if is_best:
            checked(row["improved_checkpoint"])
            improved.append(row["improved_checkpoint"])
            best = row
    if best["epoch"] != end["selected_epoch"] or best["improved_checkpoint"] != end["selected_checkpoint"] or trace[-1]["validation"] != end["final_validation"]:
        raise ValueError("Earliest-tie secondary or final primary selection differs")
    expected = {Path(checked(r)).name for r in improved}
    if {p.name for p in (output / "checkpoints").iterdir()} != expected:
        raise ValueError("Omitted or alternative checkpoint state")
    checked(end["initial"])
    checked(end["final_checkpoint"])
    for ref in end["validation_logits"].values():
        checked(ref)
    return end, chain


def replay(admitted, binding_ref, freeze, freeze_ref, cell, context):
    import torch
    from shared_message_quality_v3 import models
    from .labels import load_validation
    from .metrics import full_split
    fit_result, fit_chain = audit_fit(admitted, binding_ref, freeze_ref, cell)
    require_order(admitted, cell, "replays")
    budget = Budget(admitted, context)
    configure(admitted)
    export, lineage, metadata, directory = load_inputs(admitted, cell["task"])
    targets = load_validation(freeze["validation_exports"][cell["task"]], cell["task"], export, admitted, binding_ref)
    output = prescribed(admitted, "replays", cell)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"cell": cell, "binding": binding_ref, "supervisor_directory": context["supervisor_relative_to_project"], "live": context})
    model, family, x, variant, model_info = setup(admitted, cell, export, metadata, directory)
    initial = torch.load(checked(fit_result["initial"]), map_location="cpu", weights_only=True)
    if fingerprint(initial["model_state"]) != fit_result["initial_parameter_fingerprint"] or fingerprint(cpu_tree(model.state_dict())) != fit_result["initial_parameter_fingerprint"]:
        raise ValueError("Saved initial state or fresh paired construction differs")
    if initial["optimizer"]["state"]:
        raise ValueError("Initial optimizer has inherited moments")
    del initial
    indices = export["official_split_node_ids"]["valid"].to("cuda:0")
    reports = []
    for role, epoch, checkpoint_ref in (("selected", fit_result["selected_epoch"], fit_result["selected_checkpoint"]), ("final", EPOCHS, fit_result["final_checkpoint"])):
        budget.check()
        state = torch.load(checked(checkpoint_ref), map_location="cpu", weights_only=True)
        if state["cell"] != cell or state["epoch"] != epoch or state["binding"] != binding_ref or state["validation_freeze"] != freeze_ref or state["input_lineage"] != lineage:
            raise ValueError("Checkpoint identity/source/input/runtime binding differs")
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()
        member, pooled = models.split_logits(model, family, x, indices, execution=variant["deployment"])
        stored = torch.load(checked(fit_result["validation_logits"][role]), map_location="cpu", weights_only=True)
        if stored["cell"] != cell or stored["epoch"] != epoch or not torch.equal(stored["node_ids"], export["official_split_node_ids"]["valid"]):
            raise ValueError("Stored split evidence identity/order differs")
        tol = admitted["execution_equivalence_policy"]
        torch.testing.assert_close(member, stored["member_logits"], **tol)
        torch.testing.assert_close(pooled, stored["pooled_logits"], **tol)
        stored_metrics, stored_stats = full_split(stored["member_logits"], stored["pooled_logits"], targets)
        expected = fit_result["selected_validation"] if role == "selected" else fit_result["final_validation"]
        if stored_metrics != expected:
            raise ValueError("Stored selected/final scalar numerical replay differs")
        measured, stats = full_split(member, pooled, targets)
        for metric in ("NLL", "accuracy_percent"):
            if abs(measured[metric] - expected[metric]) > (1e-5 if metric == "NLL" else 1e-4):
                raise ValueError("Deployment numerical metrics differ beyond frozen tolerance")
        evidence = save_tensor(output / (role + "_validation_replay.pt"), {"node_ids": export["official_split_node_ids"]["valid"], "member_logits": member, "pooled_logits": pooled, "epoch": epoch, "checkpoint": checkpoint_ref}, budget)
        reports.append({"role": role, "epoch": epoch, "checkpoint": checkpoint_ref, "validation_logits": evidence,
                        "validation": measured, "sufficient_statistics": stats, "stored_numerical_metrics": stored_metrics,
                        "chosen_member": choose_member(stored_metrics), "execution_tolerance": tol})
        del state, stored, member, pooled
    # The final state remains loaded: timings are trained, final-primary and isolated.
    warmups, samples = admitted["timing_policy"]["warmups"], admitted["timing_policy"]["samples"]
    with torch.no_grad():
        for _ in range(warmups):
            values = models.pooled_inference(model, family, x, execution=variant["deployment"])
            del values
        torch.cuda.synchronize(0)
        times = []
        for _ in range(samples):
            budget.check()
            torch.cuda.synchronize(0)
            begin = time.perf_counter()
            values = models.pooled_inference(model, family, x, execution=variant["deployment"])
            torch.cuda.synchronize(0)
            times.append(time.perf_counter() - begin)
            del values
    selected_member = reports[0]["chosen_member"]
    end = {"schema": "gnnm-shared-message-validation-replay-v2", "status": "passed", "cell": cell,
           "binding": binding_ref, "validation_freeze": freeze_ref, "fit": reference(prescribed(admitted, "fits", cell) / "FIT_END.json"),
           "completed_fit_chain": fit_chain, "reports": reports, "selected_member": selected_member,
           "all_scalar_records_and_selection_rules_replayed": True,
           "unimproved_epoch_full_forward_replay_performed": False, "selected_and_final_separate_forwards_replayed": True,
           "final_primary_validation": from_statistics(fit_result["trace"][-1]["sufficient_statistics"], selected_member),
           "secondary_validation": from_statistics(next(r for r in fit_result["trace"] if r["epoch"] == fit_result["selected_epoch"])["sufficient_statistics"], selected_member),
           "trained_final_inference": {"warmups": warmups, "samples_seconds": times, "median_seconds": statistics.median(times), "execution": variant["deployment"]},
           "initial_parameter_fingerprint": fit_result["initial_parameter_fingerprint"], "model_info": model_info,
           "test_targets_loaded": False, "live": context, "hash_work": hash_report()}
    end.update(runtime_import_receipt(require_generated=True))
    write(output / "REPLAY_END.json", end)
    manifest(output)


def audit_replay(admitted, binding_ref, freeze_ref, cell):
    output = prescribed(admitted, "replays", cell)
    audit_manifest(output)
    end = json.loads((output / "REPLAY_END.json").read_text())
    start = json.loads((output / "START.json").read_text())
    chain = completed(PHASE / start["supervisor_directory"], start["live"]["outer_start"])
    fit_result, _ = audit_fit(admitted, binding_ref, freeze_ref, cell)
    if (Path(chain["argv"][1]).resolve() != HERE / "runner.py" or chain["argv"][2] != "replay"
            or chain["gpu_uuid"] != admitted["gpu_uuid"] or end.get("schema") != "gnnm-shared-message-validation-replay-v2"
            or end.get("status") != "passed" or end.get("cell") != cell or end.get("binding") != binding_ref
            or end.get("validation_freeze") != freeze_ref or end.get("test_targets_loaded") is not False
            or end.get("all_scalar_records_and_selection_rules_replayed") is not True
            or end.get("selected_and_final_separate_forwards_replayed") is not True
            or end.get("fit") != reference(prescribed(admitted, "fits", cell) / "FIT_END.json")
            or [(r["role"], r["epoch"]) for r in end["reports"]] != [("selected", fit_result["selected_epoch"]), ("final", EPOCHS)]):
        raise ValueError("Incomplete or unrelated replay chain")
    if end.get("live") != chain["live"] or start.get("live") != chain["live"]:
        raise ValueError("Replay source-bound live/outer chain differs")
    audit_runtime_import_receipt(end, require_generated=True)
    for row in end["reports"]:
        checked(row["validation_logits"])
        expected_ref = fit_result["selected_checkpoint"] if row["role"] == "selected" else fit_result["final_checkpoint"]
        if row["checkpoint"] != expected_ref or from_statistics(row["sufficient_statistics"]) != row["validation"]:
            raise ValueError("Replay state or scalar records changed")
    selected_metrics = fit_result["selected_validation"]
    selected_member = choose_member(selected_metrics)
    if end["selected_member"] != selected_member or end["initial_parameter_fingerprint"] != fit_result["initial_parameter_fingerprint"]:
        raise ValueError("Member selector or initialization changed")
    if end["final_primary_validation"] != from_statistics(fit_result["trace"][-1]["sufficient_statistics"], selected_member):
        raise ValueError("Final-primary record differs")
    selected_stats = next(r for r in fit_result["trace"] if r["epoch"] == fit_result["selected_epoch"])["sufficient_statistics"]
    if end["secondary_validation"] != from_statistics(selected_stats, selected_member):
        raise ValueError("Secondary selected-checkpoint record differs")
    timing = end["trained_final_inference"]
    if (len(timing["samples_seconds"]) != admitted["timing_policy"]["samples"]
            or timing["warmups"] != admitted["timing_policy"]["warmups"]
            or timing["median_seconds"] != statistics.median(timing["samples_seconds"])
            or any(not math.isfinite(v) or v <= 0 for v in timing["samples_seconds"])
            or timing["execution"] != admitted["execution_policy"][cell["task"]][cell["arm"]]["deployment"]):
        raise ValueError("Incomplete trained timing evidence")
    return end, chain


def lock_family(admitted, binding_ref, freeze_ref, context):
    budget = Budget(admitted, context)
    root = within(PHASE / admitted["output_root"], HERE)
    names = {p.name for p in (root / "fits").iterdir()}
    replay_names = {p.name for p in (root / "replays").iterdir()}
    from .contract import cell_id
    if names != {cell_id(c) for c in cells()} or replay_names != names:
        raise ValueError("Missing, alternative or additional fit/replay cells")
    rows, paired_initial = [], {}
    for cell in cells():
        budget.check()
        fit_result, fit_chain = audit_fit(admitted, binding_ref, freeze_ref, cell)
        result, replay_chain = audit_replay(admitted, binding_ref, freeze_ref, cell)
        if cell["arm"] in ARMS[:3]:
            key = (cell["task"], cell["seed"])
            initial = fit_result["initial_parameter_fingerprint"]
            if key in paired_initial and paired_initial[key] != initial:
                raise ValueError("Exact/shared/detached ensemble initialization not identical")
            paired_initial[key] = initial
        rows.append({**cell, "fit": reference(prescribed(admitted, "fits", cell) / "FIT_END.json"),
                     "replay": reference(prescribed(admitted, "replays", cell) / "REPLAY_END.json"),
                     "selected_checkpoint": fit_result["selected_checkpoint"], "final_checkpoint": fit_result["final_checkpoint"],
                     "selected_epoch": fit_result["selected_epoch"], "selected_member": result["selected_member"],
                     "final_validation": result["final_primary_validation"], "secondary_validation": result["secondary_validation"],
                     "full_fit_supervised_wall_seconds": fit_chain["seconds"], "full_replay_timing_supervised_wall_seconds": replay_chain["seconds"],
                     "trained_final_inference": result["trained_final_inference"], "isolated_training_peak_memory": fit_result["memory"],
                     "model_info": fit_result["model_info"], "last20_training": [{"epoch": r["epoch"], "CE_preupdate": r["training_own_CE_preupdate"]} for r in fit_result["trace"][-20:]],
                     "last30_epoch_validation": [{"epoch": r["epoch"], "validation": r["validation"]} for r in fit_result["trace"] if r["epoch"] > EPOCHS - 30 and r["validation"] is not None]})
    output = prescribed(admitted, "family_lock")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"binding": binding_ref, "supervisor_directory": context["supervisor_relative_to_project"], "live": context})
    write(output / "FAMILY_LOCK.json", {"schema": "gnnm-shared-message-complete30-validation-family-lock-v2",
          "binding": binding_ref, "validation_freeze": freeze_ref, "all30_fits_and_replays_qualified": True,
          "fixed_updates": 9000, "cells": cells(), "rows": rows, "locked_rows_sha256": digest(rows),
          "test_target_export_or_evaluation_authorized": False, "live": context,
          **runtime_import_receipt()})
    write(output / "COMPLETE_VALIDATION_REPORT.json", summarize(rows))
    manifest(output)


def audit_family_lock(path, admitted, binding_ref, freeze_ref):
    """Stdlib verification of complete fixed paths and every locked row."""
    path = within(path, HERE)
    if path != prescribed(admitted, "family_lock") / "FAMILY_LOCK.json":
        raise ValueError("Alternative family lock refused")
    audit_manifest(path.parent)
    value = json.loads(path.read_text())
    start = json.loads((path.parent / "START.json").read_text())
    chain = completed(PHASE / start["supervisor_directory"], start["live"]["outer_start"])
    if (value.get("schema") != "gnnm-shared-message-complete30-validation-family-lock-v2"
            or value.get("binding") != binding_ref or value.get("validation_freeze") != freeze_ref
            or value.get("cells") != cells() or value.get("fixed_updates") != 9000
            or value.get("all30_fits_and_replays_qualified") is not True
            or value.get("test_target_export_or_evaluation_authorized") is not False
            or value.get("locked_rows_sha256") != digest(value["rows"])
            or Path(chain["argv"][1]).resolve() != HERE / "runner.py" or chain["argv"][2] != "lock-family"
            or chain["gpu_uuid"] != admitted["gpu_uuid"] or len(value["rows"]) != 30):
        raise ValueError("Incomplete/wrong-route/changed family lock")
    if value.get("live") != chain["live"] or start.get("live") != chain["live"]:
        raise ValueError("Family-lock source-bound live/outer chain differs")
    audit_runtime_import_receipt(value)
    for row, cell in zip(value["rows"], cells()):
        fit_result, fit_chain = audit_fit(admitted, binding_ref, freeze_ref, cell)
        replay_result, replay_chain = audit_replay(admitted, binding_ref, freeze_ref, cell)
        if ({k: row[k] for k in ("task", "arm", "seed")} != cell
                or row["fit"] != reference(prescribed(admitted, "fits", cell) / "FIT_END.json")
                or row["replay"] != reference(prescribed(admitted, "replays", cell) / "REPLAY_END.json")
                or row["final_validation"] != replay_result["final_primary_validation"]
                or row["secondary_validation"] != replay_result["secondary_validation"]
                or row["selected_checkpoint"] != fit_result["selected_checkpoint"]
                or row["final_checkpoint"] != fit_result["final_checkpoint"]
                or row["selected_member"] != replay_result["selected_member"]
                or row["full_fit_supervised_wall_seconds"] != fit_chain["seconds"]
                or row["full_replay_timing_supervised_wall_seconds"] != replay_chain["seconds"]):
            raise ValueError("Locked fixed-cell evidence changed")
    if json.loads((path.parent / "COMPLETE_VALIDATION_REPORT.json").read_text()) != summarize(value["rows"]):
        raise ValueError("Every-cell descriptive report changed")
    return value


def require_order(admitted, cell, stage):
    """No outcome-dependent reordering: fits and replays each follow root order."""
    from .contract import cell_id
    order = admitted["fit_order"]
    position = order.index(cell)
    root = within(PHASE / admitted["output_root"], HERE) / stage
    actual = {p.name for p in root.iterdir()} if root.exists() else set()
    if actual != {cell_id(c) for c in order[:position]}:
        raise ValueError("Prescribed root cell order differs; alternate/partial retry refused")
    for previous in order[:position]:
        start = json.loads((prescribed(admitted, stage, previous) / "START.json").read_text())
        completed(PHASE / start["supervisor_directory"], start["live"]["outer_start"])
