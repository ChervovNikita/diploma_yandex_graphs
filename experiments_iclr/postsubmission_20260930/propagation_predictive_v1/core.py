"""Fixed complete-graph fits and selection replay; imports only after admission."""
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

from .contract import (ARMS, TASKS, RECIPE, HERE, PHASE, Budget, audit_manifest, cells,
                      cell_id, checked, completed, digest, hash_cache_report, manifest, reference, sha, within, write)


def validation_freeze(path, admitted, binding_ref):
    freeze_path = within(path, HERE)
    record = json.loads(freeze_path.read_text())
    if (record.get("schema") != "gnnm-propagation-root-validation-target-freeze-v1"
            or record.get("method_binding") != binding_ref or record.get("root_qualified") is not True
            or set(record.get("validation_exports", {})) != set(TASKS)
            or record.get("test_targets_loaded_or_referenced") is not False):
        raise ValueError("Both completed separately admitted validation exports must be root-frozen before fits")
    for row in record["validation_exports"].values():
        checked(row["bundle"])
        checked(row["receipt"])
        completed(PHASE / row["supervisor_directory"])
    return record, reference(freeze_path)


def configure(admitted):
    import importlib.metadata
    policy = admitted["runtime_policy"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != policy["CUBLAS_WORKSPACE_CONFIG"]:
        raise ValueError("Frozen CUBLAS policy must be set before importing Torch")
    import torch
    from propagation_cost_engineering_v1.contract import check_gpu
    if policy["TF32"] is not False or policy["AMP"] is not False or policy["device"] != "cuda:0":
        raise ValueError("Scientific precision/device policy differs")
    for name, version in policy["package_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen actual runtime package changed: " + name)
    torch.set_num_threads(policy["CPU_threads"])
    torch.use_deterministic_algorithms(policy["deterministic_algorithms"], warn_only=False)
    phase = {"authorized_gpu_uuid": admitted["gpu_uuid"]}
    return check_gpu(phase)


def prior_seconds(admitted):
    root = within(PHASE / admitted["output_root"], HERE)
    elapsed = float(admitted["remaining_resource_policy"]["prior_GPU_seconds"])
    if root.exists():
        for start in sorted(root.rglob("START.json")):
            value = json.loads(start.read_text())
            if "supervisor_directory" not in value:
                continue
            # Any previous failed/incomplete scientific child closes this
            # source generation. No partial restart/outcome-based scheduling.
            receipt = completed(PHASE / value["supervisor_directory"])
            elapsed += receipt["seconds"]
    return elapsed


def load_inputs(admitted, task):
    from propagation_cost_engineering_v1.data_contract import load_acquired
    from propagation_cost_engineering_v1.prepare import audit_prepared
    phase = admitted["tasks"][task]["engineering_phase"]
    export, lineage = load_acquired(phase)
    if lineage["acquisition_sha256"] != admitted["tasks"][task]["acquisition"]["sha256"]:
        raise ValueError("Engineering phase acquisition differs from exact root binding")
    metadata, directory = audit_prepared(checked(admitted["tasks"][task]["prepared_manifest"]), phase, lineage)
    return export, lineage, metadata, directory


def prescribed(admitted, stage, cell=None):
    root = within(PHASE / admitted["output_root"], HERE)
    if stage in ("fits", "replays"):
        return within(root / stage / cell_id(cell), HERE)
    if stage in ("family_lock", "final_test"):
        return within(root / stage, HERE)
    raise ValueError("Unregistered scientific output stage")


def load_execution_family(metadata, directory, arm, variant):
    # The sealed profiler supplies no irrelevant GPU CSR storage to native
    # source controls. Scientific execution must use the same memory policy.
    from propagation_cost_engineering_v1.complete_models import CompleteFamily, OperationLedger
    from propagation_cost_engineering_v1.prepare import load_family
    if variant["native_source"]:
        return CompleteFamily({}, metadata, OperationLedger(), "cuda:0")
    return load_family(metadata, directory, arm, variant["naive_path"])


def build_model(task, arm, variant, seed, raw_edges):
    from propagation_cost_engineering_v1.contract import TASKS as SPECS
    from propagation_cost_engineering_v1.complete_models import construct_source_arm
    from propagation_cost_engineering_v1.native_source import NativeArm
    if variant["native_source"]:
        return NativeArm(SPECS[task], arm, raw_edges, optimizer_seed=seed).to("cuda:0")
    return construct_source_arm(SPECS[task], arm, optimizer_seed=seed,
             checkpoint_blocks=variant["checkpoint_blocks"], naive_path=variant["naive_path"]).to("cuda:0")


def canonical_state(model, task, arm, seed):
    from propagation_cost_engineering_v1.contract import TASKS as SPECS
    from propagation_cost_engineering_v1.complete_models import construct_source_arm
    from propagation_cost_engineering_v1.native_source import transfer_execution_state
    canonical = construct_source_arm(SPECS[task], arm, optimizer_seed=seed)
    transfer_execution_state(model, canonical)
    result = {name: value.detach().cpu().clone() for name, value in canonical.state_dict().items()}
    del canonical
    return result


def load_canonical(model, state, task, arm, seed):
    from propagation_cost_engineering_v1.contract import TASKS as SPECS
    from propagation_cost_engineering_v1.complete_models import construct_source_arm
    from propagation_cost_engineering_v1.native_source import transfer_execution_state
    canonical = construct_source_arm(SPECS[task], arm, optimizer_seed=seed)
    canonical.load_state_dict(state, strict=True)
    transfer_execution_state(canonical, model)
    del canonical


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


def tensor_payload_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_payload_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_payload_bytes(v) for v in value)
    return 0


def save_tensor(path, value, budget):
    import torch
    budget.check(tensor_payload_bytes(value) + 1024 * 1024)
    with within(path, HERE).open("xb") as stream:
        torch.save(value, stream)
    budget.committed(path)
    budget.check()
    return reference(path)


def initial_fingerprint(state):
    from propagation_cost_engineering_v1.data_contract import _tensor_record
    return digest({name: _tensor_record(value.contiguous()) for name, value in sorted(state.items())})


def split_forward(model, family, features, indices, execution):
    """ONE complete-node forward, streamed member evidence and exact raw pool.

    Serial accumulation matches sealed pooled_inference's first-result/add_
    member order. Slices are copied BEFORE the first output is mutated.
    """
    import torch
    with torch.no_grad():
        if (getattr(model, "native_source", False) or execution == "serial") and model.arm not in ARMS[:2]:
            members = 1 if model.arm == "single_base" else 4
            evidence, pooled = [], None
            for member in range(members):
                result = model.forward_member(family, features, member, colors=(0, 2))
                evidence.append(result[indices].detach().cpu().clone())
                if pooled is None:
                    pooled = result
                else:
                    pooled.add_(result)
                    del result
            pooled /= members
            return torch.stack(evidence), pooled[indices].detach().cpu().clone()
        kwargs = {} if model.arm == "single_base" else {"execution": execution}
        logits = model(family, features, colors=(0, 2), **kwargs)
        evidence = logits[:, indices].detach().cpu().clone()
        pooled = logits.mean(dim=0)[indices].detach().cpu().clone()
        return evidence, pooled


def fit(admitted, binding_ref, freeze, freeze_ref, cell, output, context):
    import numpy as np
    import torch
    from propagation_cost_engineering_v1.prepare import load_family
    from propagation_cost_engineering_v1.complete_models import train_step
    from propagation_cost_engineering_v1.contract import TASKS as SPECS, check_peak
    from propagation_cost_impl_v1.dropout import training_colors, DropoutKey
    from .labels import load_label_bundle
    from .metrics import full_split
    output = within(output, HERE)
    if output != prescribed(admitted, "fits", cell):
        raise ValueError("Fit output differs from complete-family fixed location")
    if output.exists():
        raise ValueError("New exclusive fit directory required; restart/resume forbidden")
    budget = Budget(admitted, prior_seconds(admitted), context)
    gpu = configure(admitted)
    export, lineage, metadata, directory = load_inputs(admitted, cell["task"])
    labels = load_label_bundle(freeze["validation_exports"][cell["task"]], "valid", cell["task"], export,
                              {"binding_sha256": binding_ref["sha256"], "gpu_uuid": admitted["gpu_uuid"]})
    variant = admitted["execution_policy"][cell["task"]][cell["arm"]]["training_validation"]
    task = SPECS[cell["task"]]
    colors = training_colors(task["index"], cell["seed"], 100, 2)
    schedule_rng = np.random.Generator(np.random.PCG64(20260930 + 1000 * task["index"] + cell["seed"]))
    if schedule_rng.integers(0, 4, size=(100, 2)).tolist() != colors:
        raise ValueError("Sealed PCG64 training-color stream changed")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "validation_logits", "epochs"):
        (output / name).mkdir()
    write(output / "START.json", {"schema": "gnnm-propagation-fixed30-fit-start-v1", "cell": cell,
          "binding": binding_ref, "validation_target_freeze": freeze_ref, "execution": variant,
          "supervisor_directory": context["supervisor_relative_to_project"], "live": context,
          "complete_data_binding": lineage, "gpu": gpu, "recipe": RECIPE,
          "test_targets_opened": False, "training_colors": colors,
          "public_provider_storage_opened_by_fit": False})
    try:
        torch.cuda.reset_peak_memory_stats(0)
        features = export["x"].to("cuda:0")
        train_ids = export["training_target_node_ids"].to("cuda:0")
        train_labels = export["training_targets"].reshape(-1).to(dtype=torch.long, device="cuda:0")
        validation_ids = export["official_split_node_ids"]["valid"].to("cuda:0")
        raw_edges = export["edge_index"].to("cuda:0") if variant["native_source"] else None
        family = load_execution_family(metadata, directory, cell["arm"], variant)
        model = build_model(cell["task"], cell["arm"], variant, cell["seed"], raw_edges)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                                      betas=(.9, .999), eps=1e-8)
        if optimizer.state:
            raise ValueError("Scientific optimizer must be freshly initialized")
        initial = canonical_state(model, cell["task"], cell["arm"], cell["seed"])
        init_hash = initial_fingerprint(initial)
        save_tensor(output / "INITIAL.pt", {"canonical_state": initial, "optimizer": cpu_tree(optimizer.state_dict()),
                    "canonical_parameter_fingerprint": init_hash, "fresh_initialization": True}, budget)
        del initial
        trace, selected = [], None
        for epoch in range(1, 101):
            budget.check()
            family.ledger.clear()
            model.train()
            torch.cuda.synchronize(0)
            begin = time.perf_counter()
            loss = train_step(model, family, features, train_ids, train_labels, optimizer,
                 colors=tuple(colors[epoch - 1]), dropout_key=DropoutKey(task["index"], cell["seed"], epoch),
                 execution=variant["execution"], return_loss=True)
            torch.cuda.synchronize(0)
            train_seconds = time.perf_counter() - begin
            train_ce = float(loss.cpu()) if isinstance(loss, torch.Tensor) else float(loss)
            if not math.isfinite(train_ce):
                raise ValueError("Nonfinite full official training own-member objective")
            model.eval()
            begin = time.perf_counter()
            member, pooled = split_forward(model, family, features, validation_ids, variant["execution"])
            torch.cuda.synchronize(0)
            validation_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            measured = full_split(member, labels, pooled_logits=pooled)
            val_ref = save_tensor(output / "validation_logits" / ("epoch%03d.pt" % epoch),
                    {"schema": "gnnm-complete-validation-member-logits-v1", "node_ids": export["official_split_node_ids"]["valid"],
                     "member_logits": member, "pooled_logits": pooled, "epoch": epoch}, budget)
            metric_save_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            state = canonical_state(model, cell["task"], cell["arm"], cell["seed"])
            checkpoint_ref = save_tensor(output / "checkpoints" / ("epoch%03d.pt" % epoch),
                    {"schema": "gnnm-propagation-complete-epoch-checkpoint-v1", "cell": cell, "epoch": epoch,
                     "canonical_state": state, "training_layout_state": cpu_tree(model.state_dict()),
                     "optimizer": cpu_tree(optimizer.state_dict()), "cpu_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "color_schedule": colors,
                     "PCG64_schedule_final_state": schedule_rng.bit_generator.state,
                     "dropout_key": {"task_index": task["index"], "optimizer_seed": cell["seed"], "epoch": epoch},
                     "binding": binding_ref, "validation_target_freeze": freeze_ref, "execution": variant}, budget)
            checkpoint_seconds = time.perf_counter() - begin
            row = {"epoch": epoch, "training_own_CE_preupdate": train_ce, "validation": measured,
                   "training_colors": colors[epoch - 1], "checkpoint": checkpoint_ref, "validation_logits": val_ref,
                   "train_wall_seconds": train_seconds, "validation_forward_slice_copy_seconds": validation_seconds,
                   "CPU_metric_and_validation_save_seconds": metric_save_seconds,
                   "checkpoint_canonical_transfer_save_seconds": checkpoint_seconds,
                   "operation_summary": family.ledger.summary(), "memory": check_peak()}
            write(output / "epochs" / ("epoch%03d.json" % epoch), row)
            trace.append(row)
            if selected is None or (measured["CE"], epoch) < (selected["validation"]["CE"], selected["epoch"]):
                selected = row
            del member, pooled, state
        budget.check()
        write(output / "FIT_END.json", {"schema": "gnnm-propagation-complete100-fit-v1", "status": "completed",
              "cell": cell, "epochs": 100, "binding": binding_ref, "validation_target_freeze": freeze_ref,
              "initial_parameter_fingerprint": init_hash, "selected_epoch": selected["epoch"],
              "selected_checkpoint": selected["checkpoint"], "epoch100_checkpoint": trace[-1]["checkpoint"],
              "trace": trace, "selected_validation": selected["validation"], "execution": variant,
              "test_target_values_loaded": False, "provider_storage_opened": False,
              "full_train_and_validation_nodes_each_epoch": True, "checkpoint_rule": RECIPE["checkpoint"],
              "no_early_stopping_HPO_scheduler_temperature_or_resume": True,
              "fit_validation_wall_seconds": sum(r["train_wall_seconds"] + r["validation_forward_slice_copy_seconds"]
                  + r["CPU_metric_and_validation_save_seconds"] + r["checkpoint_canonical_transfer_save_seconds"] for r in trace),
              "conservative_GPU_allocation_wall_seconds": time.monotonic() - budget.start,
              "reference_hash_cache": hash_cache_report(),
              "live": context, "qualification": "requires completed supervisor plus selection/numerical replay"})
        manifest(output)
    except BaseException as error:
        write(output / "FAILURE.json", {"schema": "gnnm-propagation-scientific-fit-failure-v1", "cell": cell,
              "type": type(error).__name__, "message": str(error), "partial_outputs_preserved": True,
              "resume_retry_or_selective_family_promotion_authorized": False, "live": context})
        raise


def audit_fit(directory, admitted, binding_ref, freeze_ref, cell):
    directory = within(directory, HERE)
    if directory != prescribed(admitted, "fits", cell):
        raise ValueError("Fit differs from prescribed fixed-cell location")
    audit_manifest(directory)
    end = json.loads((directory / "FIT_END.json").read_text())
    start = json.loads((directory / "START.json").read_text())
    chain = completed(PHASE / start["supervisor_directory"])
    if len(chain["argv"]) < 3 or Path(chain["argv"][1]).resolve() != HERE / "runner.py" or chain["argv"][2] != "fit":
        raise ValueError("Fit supervisor did not execute this sealed fit entrypoint")
    if (end.get("schema") != "gnnm-propagation-complete100-fit-v1" or end.get("status") != "completed"
            or end.get("cell") != cell or end.get("binding") != binding_ref
            or end.get("validation_target_freeze") != freeze_ref or end.get("epochs") != 100
            or end.get("test_target_values_loaded") is not False or chain["gpu_uuid"] != admitted["gpu_uuid"]):
        raise ValueError("Incomplete/wrong-cell or wrong-input scientific fit chain")
    if len(end["trace"]) != 100 or [r["epoch"] for r in end["trace"]] != list(range(1, 101)):
        raise ValueError("Every one of the100 ordered epochs is required")
    selected = min(end["trace"], key=lambda r: (r["validation"]["CE"], r["epoch"]))
    if selected["epoch"] != end["selected_epoch"] or selected["checkpoint"] != end["selected_checkpoint"]:
        raise ValueError("Earliest-tie full-validation checkpoint selection differs")
    for row in end["trace"]:
        checked(row["checkpoint"])
        checked(row["validation_logits"])
    return end, chain


def replay(admitted, binding_ref, freeze, freeze_ref, cell, run, output, context):
    import torch
    from propagation_cost_engineering_v1.prepare import load_family
    from .labels import load_label_bundle
    from .metrics import full_split, choose_member
    fit_result, fit_chain = audit_fit(run, admitted, binding_ref, freeze_ref, cell)
    budget = Budget(admitted, prior_seconds(admitted), context)
    configure(admitted)
    export, lineage, metadata, directory = load_inputs(admitted, cell["task"])
    labels = load_label_bundle(freeze["validation_exports"][cell["task"]], "valid", cell["task"], export,
                              {"binding_sha256": binding_ref["sha256"], "gpu_uuid": admitted["gpu_uuid"]})
    output = within(output, HERE)
    if output != prescribed(admitted, "replays", cell) or within(run, HERE) != prescribed(admitted, "fits", cell):
        raise ValueError("Replay/fit paths differ from complete-family fixed locations")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"cell": cell, "binding": binding_ref,
          "supervisor_directory": context["supervisor_relative_to_project"], "live": context,
          "fit": reference(within(run, HERE) / "FIT_END.json")})
    # Replay every stored metric and checkpoint selection without new forwards.
    for row in fit_result["trace"]:
        budget.check()
        values = torch.load(checked(row["validation_logits"]), map_location="cpu", weights_only=True)
        if not torch.equal(values["node_ids"], export["official_split_node_ids"]["valid"]):
            raise ValueError("Stored validation logits official node order differs")
        metrics = full_split(values["member_logits"], labels, pooled_logits=values["pooled_logits"])
        if metrics != row["validation"]:
            raise ValueError("Stored complete validation metric replay differs")
        del values
    variant = admitted["execution_policy"][cell["task"]][cell["arm"]]["deployment"]
    features = export["x"].to("cuda:0")
    indices = export["official_split_node_ids"]["valid"].to("cuda:0")
    raw_edges = export["edge_index"].to("cuda:0") if variant["native_source"] else None
    family = load_execution_family(metadata, directory, cell["arm"], variant)
    model = build_model(cell["task"], cell["arm"], variant, cell["seed"], raw_edges).eval()
    reports = []
    for epoch in sorted({fit_result["selected_epoch"], 100}):
        budget.check()
        checkpoint_ref = fit_result["trace"][epoch - 1]["checkpoint"]
        checkpoint = torch.load(checked(checkpoint_ref), map_location="cpu", weights_only=True)
        load_canonical(model, checkpoint["canonical_state"], cell["task"], cell["arm"], cell["seed"])
        member, pooled = split_forward(model, family, features, indices, variant["execution"])
        stored = torch.load(checked(fit_result["trace"][epoch - 1]["validation_logits"]), map_location="cpu", weights_only=True)
        tolerance = admitted["execution_equivalence_policy"]
        torch.testing.assert_close(member, stored["member_logits"], atol=tolerance["atol"], rtol=tolerance["rtol"])
        torch.testing.assert_close(pooled, stored["pooled_logits"], atol=tolerance["atol"], rtol=tolerance["rtol"])
        measured = full_split(member, labels, pooled_logits=pooled)
        member_choice = choose_member(measured)
        saved = save_tensor(output / ("epoch%03d_deployment_validation.pt" % epoch),
                {"node_ids": export["official_split_node_ids"]["valid"], "member_logits": member,
                 "pooled_logits": pooled, "checkpoint": checkpoint_ref}, budget)
        reports.append({"epoch": epoch, "checkpoint": checkpoint_ref, "validation_logits": saved,
                        "validation": measured, "validation_chosen_member": member_choice,
                        "selected_checkpoint": epoch == fit_result["selected_epoch"], "execution_tolerance": tolerance})
    selected = next(r for r in reports if r["selected_checkpoint"])
    write(output / "VALIDATION_REPLAY_END.json", {"schema": "gnnm-propagation-complete-validation-selection-replay-v1",
          "status": "passed", "cell": cell, "binding": binding_ref, "validation_target_freeze": freeze_ref,
          "fit": reference(within(run, HERE) / "FIT_END.json"), "completed_fit_chain": fit_chain,
          "all100_stored_metrics_replayed": True, "selected_and_epoch100_full_forwards_replayed": True,
          "reports": reports, "selected": selected, "deployment": variant, "test_target_values_loaded": False,
          "initial_parameter_fingerprint": fit_result["initial_parameter_fingerprint"],
          "reference_hash_cache": hash_cache_report(), "live": context})
    manifest(output)


def audit_replay(directory, admitted, binding_ref, freeze_ref, cell, fit_ref):
    directory = within(directory, HERE)
    if directory != prescribed(admitted, "replays", cell):
        raise ValueError("Replay is outside prescribed fixed-cell directory")
    audit_manifest(directory)
    replay_path = directory / "VALIDATION_REPLAY_END.json"
    value = json.loads(replay_path.read_text())
    start = json.loads((directory / "START.json").read_text())
    chain = completed(PHASE / start["supervisor_directory"])
    if len(chain["argv"]) < 3 or Path(chain["argv"][1]).resolve() != HERE / "runner.py" or chain["argv"][2] != "replay":
        raise ValueError("Replay supervisor did not execute this sealed replay entrypoint")
    fit_result, _ = audit_fit(checked(fit_ref).parent, admitted, binding_ref, freeze_ref, cell)
    if (value.get("schema") != "gnnm-propagation-complete-validation-selection-replay-v1"
            or value.get("status") != "passed" or value.get("cell") != cell
            or value.get("fit") != fit_ref or value.get("binding") != binding_ref
            or value.get("validation_target_freeze") != freeze_ref
            or value.get("all100_stored_metrics_replayed") is not True
            or value.get("selected_and_epoch100_full_forwards_replayed") is not True
            or value.get("test_target_values_loaded") is not False
            or chain["gpu_uuid"] != admitted["gpu_uuid"]
            or value.get("deployment") != admitted["execution_policy"][cell["task"]][cell["arm"]]["deployment"]):
        raise ValueError("Incomplete, unrelated or changed validation-selection replay")
    expected_epochs = sorted({fit_result["selected_epoch"], 100})
    if [r["epoch"] for r in value["reports"]] != expected_epochs:
        raise ValueError("Selected and epoch100 numerical replay missing or duplicated")
    for row in value["reports"]:
        checked(row["validation_logits"])
        if row["checkpoint"] != fit_result["trace"][row["epoch"] - 1]["checkpoint"]:
            raise ValueError("Replay checkpoint was not the fixed selected/last fit artifact")
    selected = next(r for r in value["reports"] if r["selected_checkpoint"])
    from .metrics import choose_member
    if (value["selected"] != selected or selected["epoch"] != fit_result["selected_epoch"]
            or selected["validation_chosen_member"] != choose_member(selected["validation"])):
        raise ValueError("Pooled checkpoint or lowest-CE validation member selection changed")
    return value, chain


def collect_choices(admitted, binding_ref, freeze_ref):
    """Audit all fixed paths. No request may nominate a subset or alternative."""
    locked, initializations = [], {}
    for cell in cells():
        fit_path = prescribed(admitted, "fits", cell) / "FIT_END.json"
        replay_path = prescribed(admitted, "replays", cell) / "VALIDATION_REPLAY_END.json"
        fit_ref, replay_ref = reference(fit_path), reference(replay_path)
        fit_result, _ = audit_fit(fit_path.parent, admitted, binding_ref, freeze_ref, cell)
        replay_result, _ = audit_replay(replay_path.parent, admitted, binding_ref, freeze_ref, cell, fit_ref)
        if cell["arm"] != "single_base":
            key = (cell["task"], cell["seed"])
            initial = fit_result["initial_parameter_fingerprint"]
            if key in initializations and initializations[key] != initial:
                raise ValueError("Ensemble arms lack identical fresh canonical initialization")
            initializations[key] = initial
        selected = replay_result["selected"]
        locked.append({"cell": cell, "fit": fit_ref, "replay": replay_ref,
            "selected_checkpoint": selected["checkpoint"], "selected_epoch": selected["epoch"],
            "validation_chosen_member": selected["validation_chosen_member"],
            "deployment": replay_result["deployment"], "selected_validation": selected["validation"],
            "fit_supervisor_directory": json.loads((fit_path.parent / "START.json").read_text())["supervisor_directory"],
            "replay_supervisor_directory": json.loads((replay_path.parent / "START.json").read_text())["supervisor_directory"]})
    return locked


def audit_family_lock(path, admitted, binding_ref):
    path = within(path, HERE)
    if path != prescribed(admitted, "family_lock") / "FAMILY_LOCK.json":
        raise ValueError("Pretest lock differs from fixed family output")
    audit_manifest(path.parent)
    lock = json.loads(path.read_text())
    if (lock.get("schema") != "gnnm-propagation-complete30-pretest-family-lock-v1"
            or lock.get("binding") != binding_ref or lock.get("all30_fits_and_replays_qualified") is not True
            or lock.get("test_target_values_loaded") is not False or lock.get("fit_count") != 30
            or len(lock.get("locked_choices", [])) != 30
            or lock.get("locked_choices_payload_sha256") != digest(lock["locked_choices"])):
        raise ValueError("All30 complete choices must be locked before any test indexing")
    freeze_ref = lock["validation_target_freeze"]
    validation_freeze(checked(freeze_ref), admitted, binding_ref)
    actual = collect_choices(admitted, binding_ref, freeze_ref)
    if actual != lock["locked_choices"]:
        raise ValueError("Fit/replay/checkpoint/member/execution lock is no longer exact")
    finalizer = lock["live"]
    chain = completed(PHASE / finalizer["supervisor_relative_to_project"])
    if (len(chain["argv"]) < 3 or Path(chain["argv"][1]).resolve() != HERE / "runner.py"
            or chain["argv"][2] != "lock-family"):
        raise ValueError("Family finalizer did not execute the sealed complete lock entrypoint")
    if chain["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Pretest family finalizer lacks completed authorized chain")
    return lock


def lock_family(admitted, binding_ref, freeze_ref, request, output, context):
    budget = Budget(admitted, prior_seconds(admitted), context)
    if (request.get("schema") != "gnnm-propagation-fixed30-root-finalization-request-v1"
            or request.get("binding") != binding_ref or request.get("root_authorized") is not True
            or request.get("validation_target_freeze") != freeze_ref
            or set(request) != {"schema", "binding", "root_authorized", "validation_target_freeze"}):
        raise ValueError("Root-fixed complete-family finalization request required")
    output = within(output, HERE)
    if output != prescribed(admitted, "family_lock") or output.exists():
        raise ValueError("Only one new complete fixed-family lock may be written")
    locked = collect_choices(admitted, binding_ref, freeze_ref)
    budget.check()
    payload_hash = digest(locked)
    requests = {}
    for task in TASKS:
        template = admitted["tasks"][task]["test_label_request_template"]
        if "binding" in template or "family_lock" in template or "family_lock_payload_sha256" in template:
            raise ValueError("Method-frozen test request template contains cyclic or mutable selection references")
        requests[task] = {**template, "binding": binding_ref, "family_lock_payload_sha256": payload_hash}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"binding": binding_ref, "live": context,
          "supervisor_directory": context["supervisor_relative_to_project"], "request": request,
          "test_target_values_loaded": False})
    lock_path = output / "FAMILY_LOCK.json"
    write(lock_path, {"schema": "gnnm-propagation-complete30-pretest-family-lock-v1", "binding": binding_ref,
          "validation_target_freeze": freeze_ref, "fit_count": 30, "all30_fits_and_replays_qualified": True,
          "locked_choices": locked, "locked_choices_payload_sha256": payload_hash,
          "test_label_request_payload_sha256": {t: digest(v) for t, v in requests.items()},
          "test_target_values_loaded": False, "checkpoint_member_execution_choices_complete": True,
          "no_test_driven_selection_permitted": True, "live": context,
          "reference_hash_cache": hash_cache_report(),
          "qualification": "requires finalizer completed supervisor chain before any test export"})
    for task, payload in requests.items():
        write(output / (("reddit2" if task == "Reddit2" else "products") + "_test_request.json"),
              {**payload, "family_lock": reference(lock_path)})
    manifest(output)
