"""Prospective diagnostic only; no CLI, execution release, or fit donor.

A separately reviewed normal-host runner must admit the complete graph and
qualified runtime before calling run. Preparation only parses this source.
The sealed qualified comparator is unchanged; discrepancies are observations,
never a replacement PASS. No model, optimizer, raw prediction or RNG payload is
serialized by this module. Only scalar tensor summaries and exact digests leave
the probe. All six trajectories start from one fresh saved-V3 CPU snapshot.
"""
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_packet(root, expected_sha):
    require(file_sha(root / "MANIFEST.json") == expected_sha, "Diagnostic source packet differs")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root), "Diagnostic source path escaped")
        require(path.stat().st_size == row.get("bytes", row.get("size"))
                and file_sha(path) == row["sha256"], "Diagnostic source bytes differ: " + row["path"])
    return manifest


def admission_gate(path, expected_sha):
    # No admission is created by this source preparation.
    path = Path(path).resolve()
    require(file_sha(path) == expected_sha, "Diagnostic root admission differs")
    admission = json.loads(path.read_text())
    plan = json.loads((HERE / "PLAN.json").read_text())
    require(admission.get("schema") == "ncnc-pattern-parity-diagnostic-root-admission-v1"
            and admission.get("status") == "APPROVED"
            and admission.get("root_authorization_reference"), "Separate diagnostic root approval required")
    source_sha = file_sha(HERE / "MANIFEST.json")
    require(admission.get("diagnostic_manifest_sha256") == source_sha, "Wrong diagnostic source release")
    verify_packet(HERE, source_sha)
    require(admission.get("caps") == plan["caps"], "Diagnostic caps must remain unchanged")
    for name in ("data_authority_sha256", "runtime_authority_sha256"):
        require(admission.get(name) == plan[name], "Diagnostic authority differs: " + name)
    require(admission.get("authorized_optimizer_updates") == 12
            and admission.get("authorized_member_trajectory_updates") == 48,
            "Diagnostic work admission differs")
    require(admission.get("authorized_invocation") == plan["authorized_invocation_contract"]
            and admission.get("state_donor") is False
            and admission.get("scientific_fit_admitted") is False
            and admission.get("TEST_supported") is False, "Diagnostic scope differs")
    require(admission.get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")
            == plan["prospective_GPU_UUID"], "Diagnostic normal device identity differs")
    for pin in plan["source_packets"]:
        verify_packet((HERE.parent / pin["root"]).resolve(), pin["manifest_sha256"])
    for pin in plan["authority_files"]:
        require(file_sha(HERE.parent / pin["path"]) == pin["sha256"], "Diagnostic authority bytes differ")
    return plan, {"path": str(path), "sha256": expected_sha,
                  "diagnostic_manifest_sha256": source_sha,
                  "root_authorization_reference": admission["root_authorization_reference"]}


def bound_modules(plan, mods, teacher):
    root = (HERE.parent / plan["V4_root"]).resolve()
    names = ("pilot_state", "pilot_model", "pattern_model", "pattern_train",
             "checkpoint_parity", "pattern_teacher", "pilot_common")
    loaded = {name: importlib.import_module(name) for name in names}
    for name, module in loaded.items():
        require(Path(module.__file__).resolve() == root / (name + ".py"),
                "Diagnostic V4 helper shadowed: " + name)
    prototype_root = (HERE.parent / plan["prototype_root"]).resolve()
    for name in ("prototype", "graph_ops"):
        require(Path(mods[name].__file__).resolve() == prototype_root / (name + ".py"),
                "Diagnostic prototype helper shadowed: " + name)
        require(importlib.import_module(name) is mods[name], "Diagnostic prototype module binding differs: " + name)
    require(Path(mods["design"].__file__).resolve()
            == HERE.parent / plan["design_root"] / "design_spec.py", "Diagnostic design helper shadowed")
    require(type(teacher) is loaded["pattern_teacher"].ObservationTeacher,
            "Diagnostic requires the unchanged TRAIN-only observation teacher")
    return loaded


def tensor_difference(actual, reference, torch, state):
    row = {"reference_shape": list(reference.shape), "actual_shape": list(actual.shape),
           "reference_dtype": str(reference.dtype), "actual_dtype": str(actual.dtype),
           "reference_sha256": state.state_digest(reference), "actual_sha256": state.state_digest(actual)}
    row["typed_bytes_equal"] = row["reference_sha256"] == row["actual_sha256"]
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        row["structural_mismatch"] = True
        return row
    row["elements"] = actual.numel()
    if not actual.is_floating_point():
        row["exact_equal"] = bool(torch.equal(actual, reference))
        return row
    row["actual_finite"] = bool(torch.isfinite(actual).all())
    row["reference_finite"] = bool(torch.isfinite(reference).all())
    if not row["actual_finite"] or not row["reference_finite"]:
        row["nonfinite_comparison"] = True
        return row
    eps = torch.finfo(actual.dtype).eps
    tolerance = 128 * eps
    # Preserve the qualified predicate's original dtype and order exactly.
    passed = (actual - reference).abs() <= tolerance + tolerance * reference.abs()
    a, r = actual.to(torch.float64), reference.to(torch.float64)
    difference = (a - r).abs()
    relative = difference / r.abs().clamp_min(torch.finfo(reference.dtype).tiny)
    scaled = difference / (tolerance + tolerance * r.abs())
    maximum = lambda value: float(value.max()) if value.numel() else 0.0
    row.update({"qualified_atol_rtol": tolerance,
                "qualified_predicate_dtype": str(actual.dtype),
                "exceeds_unchanged_qualified_threshold_count": int((~passed).sum()),
                "within_unchanged_qualified_threshold": bool(passed.all()),
                "max_absolute_difference": maximum(difference),
                "max_relative_difference": maximum(relative),
                "relative_denominator": "max(abs(reference), finfo(reference.dtype).tiny)",
                "max_scaled_difference_float64_summary": maximum(scaled),
                "scaled_summary_denominator": "128*eps + 128*eps*abs(reference), evaluated in float64 for summary only",
                "reference_L2_norm": float(torch.linalg.vector_norm(r.reshape(-1))),
                "actual_L2_norm": float(torch.linalg.vector_norm(a.reshape(-1))),
                "reference_Linf_norm": maximum(r.abs()), "actual_Linf_norm": maximum(a.abs()),
                "reference_zero_count": int((r == 0).sum()), "actual_zero_count": int((a == 0).sum())})
    return row


def nested_differences(actual, reference, torch, state, path=""):
    """Walk all leaves without stopping at the first arithmetic discrepancy."""
    rows = []
    def visit(a, r, label):
        if torch.is_tensor(a) and torch.is_tensor(r):
            rows.append({"path": label, "kind": "tensor", **tensor_difference(a, r, torch, state)})
        elif isinstance(a, dict) and isinstance(r, dict):
            if set(a) != set(r):
                rows.append({"path": label, "kind": "keys", "exact_equal": False,
                             "actual_keys": [repr(k) for k in a], "reference_keys": [repr(k) for k in r]})
            for key in r:
                if key in a:
                    visit(a[key], r[key], label + "." + str(key))
        elif isinstance(a, (tuple, list)) and type(a) is type(r):
            if len(a) != len(r):
                rows.append({"path": label, "kind": "length", "exact_equal": False,
                             "actual_length": len(a), "reference_length": len(r)})
            for index, (x, y) in enumerate(zip(a, r)):
                visit(x, y, label + "." + str(index))
        else:
            rows.append({"path": label, "kind": "typed_scalar_or_container",
                         "exact_equal": state.state_digest(a) == state.state_digest(r)})
    visit(actual, reference, path)
    return rows


def rng_comparison(actual, reference, state):
    return {"exact_equal": state.rng_digest(actual) == state.rng_digest(reference),
            "reference_sha256": state.rng_digest(reference), "actual_sha256": state.rng_digest(actual),
            "components": {name: {"reference_sha256": state.state_digest(reference[name]),
                                   "actual_sha256": state.state_digest(actual[name]),
                                   "exact_equal": state.state_digest(actual[name]) == state.state_digest(reference[name])}
                           for name in reference}}


def optimizer_binding(model, optimizer, torch):
    require(type(optimizer) is torch.optim.Adam, "Unchanged native Adam required")
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = []
    for index, (group, saved) in enumerate(zip(optimizer.param_groups, optimizer.state_dict()["param_groups"])):
        require(group["lr"] == (0.0082, 0.0037)[index]
                and group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
                and group["weight_decay"] == 0 and group["amsgrad"] is False
                and group["maximize"] is False and group["capturable"] is False
                and group["differentiable"] is False and group["foreach"] is None
                and group["fused"] is None, "Qualified native Adam options differ")
        result.append({"group": index, "configuration": {key: value for key, value in saved.items() if key != "params"},
                       "parameter_ids_to_names": {str(slot): names[id(parameter)]
                                                  for parameter, slot in zip(group["params"], saved["params"])}})
    require(len(result) == 2, "Qualified Adam group count differs")
    return result


def gradient_conditioning(actual, reference, binding, torch, state):
    rows = []
    for group in binding:
        eps = group["configuration"]["eps"]
        lr = group["configuration"]["lr"]
        for slot, name in group["parameter_ids_to_names"].items():
            a, r = actual["gradients"][name], reference["gradients"][name]
            if a is None or r is None:
                rows.append({"parameter": name, "group": group["group"], "gradient_missing": True})
                continue
            a, r = a.to(torch.float64), r.to(torch.float64)
            row = {"parameter": name, "group": group["group"], "Adam_eps": eps, "lr": lr,
                   "actual_abs_gradient_le_eps_count": int((a.abs() <= eps).sum()),
                   "reference_abs_gradient_le_eps_count": int((r.abs() <= eps).sum()),
                   "nonzero_opposite_sign_count": int(((a * r) < 0).sum())}
            moments = []
            for step in (actual, reference):
                saved = step["next_state"]["optimizer"]["state"].get(int(slot))
                if saved is None:
                    break
                beta1, beta2 = group["configuration"]["betas"]
                number = float(saved["step"])
                m = saved["exp_avg"].to(torch.float64) / (1 - beta1 ** number)
                denominator = (saved["exp_avg_sq"].to(torch.float64) / (1 - beta2 ** number)).sqrt() + eps
                moments.append((m / denominator, denominator))
            if len(moments) == 2:
                (adirection, adenominator), (rdirection, rdenominator) = moments
                row["actual_Adam_denominator_le_2eps_count"] = int((adenominator <= 2 * eps).sum())
                row["reference_Adam_denominator_le_2eps_count"] = int((rdenominator <= 2 * eps).sum())
                direction_difference = (adirection - rdirection).abs()
                row["max_absolute_Adam_direction_difference_float64"] = float(direction_difference.max()) if direction_difference.numel() else 0.0
                row["max_lr_scaled_derived_Adam_direction_difference_float64"] = lr * row["max_absolute_Adam_direction_difference_float64"]
                row["derived_direction_note"] = "Float64 bias-corrected m/(sqrt(v)+eps) summary; not an emulation of native Adam rounding"
            rows.append(row)
    return rows


def runtime_settings(torch):
    return {"torch_version": torch.__version__, "CUDA_version": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "TF32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "autocast_CUDA": torch.is_autocast_enabled("cuda"),
            "autocast_CPU": torch.is_autocast_enabled("cpu"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")}


def run(mods, data, teacher, device, *, admission_path, admission_sha256, publish):
    """publish atomically journals only this JSON-compatible scalar report."""
    started = perf_counter()
    plan, admission = admission_gate(admission_path, admission_sha256)
    bound = bound_modules(plan, mods, teacher)
    import torch
    state, parity = bound["pilot_state"], bound["checkpoint_parity"]
    settings = runtime_settings(torch)
    require(settings["torch_version"] == "2.7.1" and settings["CUDA_version"] == "12.6"
            and settings["default_dtype"] == "torch.float32" and settings["deterministic_algorithms"] is False
            and settings["TF32_matmul"] is False and settings["TF32_cudnn"] is False
            and settings["float32_matmul_precision"] == "highest"
            and not settings["autocast_CUDA"] and not settings["autocast_CPU"], "Qualified runtime flags differ")
    require(str(device) == "cuda:0" and torch.cuda.device_count() == 1, "Normal logical GPU0 required")
    require(data["x"].shape == (235868, 128) and data["x"].dtype == torch.float32
            and data["pairs"].shape == (1179052, 2) and data["pairs"].dtype == torch.int64
            and data["x"].device == device and data["pairs"].device == device
            and teacher.nodes == 235868, "Unchanged complete real graph required")
    caller_rng = state.cpu_clone(state.rng_state())
    report = {"schema": "ncnc-pattern-parity-discrepancy-diagnostic-v1", "status": "IN_PROGRESS",
              "admission": admission, "runtime_flags": settings,
              "complete_real_graph": True, "teacher": teacher.receipt(),
              "query_records": list(parity.PROBE_RECORD_IDS),
              "negative_pairs": [list(pair) for pair in parity.PROBE_NEGATIVE_PAIRS],
              "planned_optimizer_updates": 12, "planned_member_trajectory_updates": 48,
              "executed_optimizer_updates": 0, "executed_member_trajectory_updates": 0,
              "trajectories": [], "comparisons": [], "current_phase": "fresh_saved_V3_initial_state",
              "state_donor": False, "TEST_opened": False, "project_metric_computed": False,
              "raw_predictions_or_states_saved": False, "qualification_or_fit_admission": False}
    def journal():
        report["diagnostic_function_elapsed_seconds"] = perf_counter() - started
        publish(json.loads(json.dumps(report, allow_nan=False)))
    def measured_step(model, optimizer, negatives, ids, arm):
        # Same native math/order as sealed checkpoint_parity.actual_step.
        optimizer.zero_grad(set_to_none=True)
        main, records = bound["pattern_train"].batch_forward(model, data, teacher, negatives, ids)
        scores = [value for row in records for value in row["detail"]["raw_score_tensors"]]
        isolated = torch.autograd.grad(main, scores, allow_unused=True, retain_graph=True)
        require(all(value is None for value in isolated), "Diagnostic target detach boundary differs")
        auxiliary = sum(row["values"][arm].mean() for row in records)
        total = main + auxiliary
        require(bool(torch.isfinite(total)), "Nonfinite diagnostic objective")
        forward = parity.forward_image(main, records)
        forward["selected_auxiliary"] = auxiliary.detach().cpu().clone()
        forward["total"] = total.detach().cpu().clone()
        support = {name: teacher.support_digest(row["query"], row["detail"]["neighbors"], row["labels"])
                   for name, row in zip(("positive", "negative"), records)}
        before_backward = state.cpu_clone(state.rng_state())
        report["current_phase"] = "backward"
        total.backward()
        after_backward = state.cpu_clone(state.rng_state())
        bound["pattern_train"].finite(model, optimizer, gradients=True)
        gradients = parity.gradient_image(model)
        report["current_phase"] = "native_Adam_step_in_progress"
        optimizer.step()
        report["executed_optimizer_updates"] += 1
        report["executed_member_trajectory_updates"] += 4
        report["current_phase"] = "native_Adam_step_returned"
        journal()
        bound["pattern_train"].finite(model, optimizer, gradients=False)
        return {"forward": forward, "support": support, "forward_RNG": before_backward,
                "backward_RNG": after_backward, "gradients": gradients,
                "next_state": state.snapshot(model, optimizer)}
    try:
        reference_factory, report["saved_V3_source"] = parity.saved_v3_factory()
        model, optimizer = reference_factory(mods, 20261003, device, engineering_sign_seed=2026100301)
        initial = state.snapshot(model, optimizer)
        binding = optimizer_binding(model, optimizer, torch)
        report["initial_state_sha256"] = state.state_digest(initial)
        report["initial_rng_component_sha256"] = {key: state.state_digest(value) for key, value in initial["rng"].items()}
        report["initial_flags"] = initial["flags"]
        report["native_Adam_groups_and_parameter_binding"] = binding
        del model, optimizer
        torch.cuda.synchronize(0)
        for arm in ("J", "F"):
            trajectories = {}
            for label, factory in (("V3_A", reference_factory), ("V3_B", reference_factory),
                                   ("V4_C", bound["pattern_model"].make_pattern)):
                report["current_trajectory"] = {"arm": arm, "variant": label}
                report["current_phase"] = "construct_then_restore_common_initial_state"
                model, optimizer = factory(mods, 20261003, device, engineering_sign_seed=2026100301)
                require(optimizer_binding(model, optimizer, torch) == binding, "Diagnostic parameter binding differs")
                ids = torch.tensor(parity.PROBE_RECORD_IDS, device=device, dtype=torch.long)
                negatives = data["pairs"].clone()
                negatives[ids] = torch.tensor(parity.PROBE_NEGATIVE_PAIRS, device=device, dtype=torch.long)
                state.restore_snapshot(model, optimizer, initial)
                restored = state.snapshot(model, optimizer)
                require(state.state_digest(restored) == report["initial_state_sha256"], "Diagnostic initial restore differs")
                bound["pilot_model"].train_flag(model, True)
                trajectory = {"initial": restored, "steps": []}
                receipt = {"arm": arm, "variant": label, "initial_state_sha256": state.state_digest(restored), "steps": []}
                report["trajectories"].append(receipt)
                for number in (1, 2):
                    report["current_step"] = number
                    report["current_phase"] = "native_forward"
                    before = restored if number == 1 else trajectory["steps"][-1]["next_state"]
                    step = measured_step(model, optimizer, negatives, ids, arm)
                    trajectory["steps"].append(step)
                    receipt["steps"].append({"number": number,
                        "before_forward_rng_sha256": state.rng_digest(before["rng"]),
                        "forward_rng_sha256": state.rng_digest(step["forward_RNG"]),
                        "backward_rng": rng_comparison(step["backward_RNG"], step["forward_RNG"], state),
                        "next_rng_sha256": state.rng_digest(step["next_state"]["rng"]),
                        "before_flags": before["flags"], "next_flags": step["next_state"]["flags"]})
                    journal()
                trajectories[label] = trajectory
                del model, optimizer, negatives, ids, restored, step, before, trajectory
                torch.cuda.synchronize(0)
            report["current_phase"] = "scalar_discrepancy_summaries"
            for label in ("V3_B", "V4_C"):
                comparison = {"arm": arm, "reference": "V3_A", "actual": label, "steps": []}
                for number, (reference, actual) in enumerate(zip(trajectories["V3_A"]["steps"], trajectories[label]["steps"]), 1):
                    reference_before = trajectories["V3_A"]["initial"] if number == 1 else trajectories["V3_A"]["steps"][number-2]["next_state"]
                    actual_before = trajectories[label]["initial"] if number == 1 else trajectories[label]["steps"][number-2]["next_state"]
                    tensors = {name: nested_differences(actual[name], reference[name], torch, state, name)
                               for name in ("forward", "gradients")}
                    for name in ("models", "optimizer"):
                        tensors["next_" + name] = nested_differences(actual["next_state"][name], reference["next_state"][name], torch, state, "next_" + name)
                    update_a = {name: actual["next_state"]["models"][0][name] - actual_before["models"][0][name]
                                for name in actual["gradients"]}
                    update_r = {name: reference["next_state"]["models"][0][name] - reference_before["models"][0][name]
                                for name in reference["gradients"]}
                    tensors["parameter_updates"] = nested_differences(update_a, update_r, torch, state, "parameter_updates")
                    comparison["steps"].append({"number": number, "per_tensor_discrepancies": tensors,
                        "gradient_Adam_conditioning": gradient_conditioning(actual, reference, binding, torch, state),
                        "support": {"exact_equal": actual["support"] == reference["support"],
                                    "reference": reference["support"], "actual": actual["support"]},
                        "before_forward_RNG": rng_comparison(actual_before["rng"], reference_before["rng"], state),
                        "forward_RNG": rng_comparison(actual["forward_RNG"], reference["forward_RNG"], state),
                        "backward_RNG": rng_comparison(actual["backward_RNG"], reference["backward_RNG"], state),
                        "next_RNG": rng_comparison(actual["next_state"]["rng"], reference["next_state"]["rng"], state),
                        "flags_exact": actual["next_state"]["flags"] == reference["next_state"]["flags"],
                        "reference_flags": reference["next_state"]["flags"], "actual_flags": actual["next_state"]["flags"]})
                report["comparisons"].append(comparison)
                journal()
            del trajectories, reference, actual, reference_before, actual_before, update_a, update_r, tensors
        require(report["executed_optimizer_updates"] == 12, "Diagnostic work count differs")
        report["status"] = "DIAGNOSTIC_COMPLETE"
        report["arithmetic_discrepancies_are_recorded_without_qualification_PASS"] = True
    except Exception as error:
        report["status"] = "DIAGNOSTIC_FAILED"
        report["failure"] = {"type": type(error).__name__, "condition": str(error)}
        report["current_optimizer_call_may_be_partial"] = report["current_phase"] == "native_Adam_step_in_progress"
        raise
    finally:
        state.restore_rng(caller_rng)
        report["caller_rng_restored_exactly"] = state.rng_digest(state.rng_state()) == state.rng_digest(caller_rng)
        report["caller_rng_sha256"] = state.rng_digest(caller_rng)
        report["runtime_flags_after"] = runtime_settings(torch)
        report["runtime_flags_unchanged"] = report["runtime_flags_after"] == settings
        journal()
        require(report["caller_rng_restored_exactly"], "Diagnostic donated caller RNG")
    return report
