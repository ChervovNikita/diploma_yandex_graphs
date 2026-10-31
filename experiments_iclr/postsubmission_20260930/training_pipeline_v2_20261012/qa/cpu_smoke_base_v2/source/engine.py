"""Full-budget training and checkpoint custody, with no numerical imports.

The caller injects torch. An adapter owns the model, optimizer, loss reduction,
training procedure, validation fusion, selector, and every RNG stream. Epoch
adapters must return sample-weighted epoch metrics, not their last minibatch.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path


class AdapterStepError(RuntimeError):
    """A failed adapter call with its known partial operation counts.

    Report completed operations even when the call failed. A plain exception is
    retained too, but the engine then explicitly marks partial costs unknown.
    """

    def __init__(self, message, *, operation_counts):
        super().__init__(message)
        self.operation_counts = operation_counts


def _plain(torch, value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError("Trace tensors must be scalar")
        return _plain(torch, value.detach().cpu().item())
    if isinstance(value, Mapping):
        return {str(k): _plain(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(torch, v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite trace value")
        return value
    raise TypeError("Trace values must be plain JSON values or scalar tensors")


def _clone_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        result = type(value)((copy.deepcopy(k), _clone_tree(torch, v))
                             for k, v in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(_clone_tree(torch, v) for v in value)
    if isinstance(value, list):
        return [_clone_tree(torch, v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("Checkpoint state must be a plain tensor/container tree")


def _tensor_values(torch, value):
    if value.layout == torch.strided:
        return value
    if value.layout == torch.sparse_coo:
        return value._values()
    return value.values()


def _finite_tree(torch, value, name):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(_tensor_values(torch, value)).all().item()):
                raise FloatingPointError("Nonfinite " + name)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(torch, item, name + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(torch, item, name + "." + str(index))
    elif type(value) is float and not math.isfinite(value):
        raise FloatingPointError("Nonfinite " + name)


def _equal_tree(torch, left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False
        if (left.dtype, left.layout, left.shape) != (right.dtype, right.layout, right.shape):
            return False
        if left.layout == torch.sparse_coo:
            a, b = left.coalesce(), right.coalesce()
            return (torch.equal(a.indices(), b.indices())
                    and torch.equal(a.values(), b.values()))
        if left.layout != torch.strided:
            indices = ("crow_indices", "col_indices") if "csr" in str(left.layout) or "bsr" in str(left.layout) else ("ccol_indices", "row_indices")
            return (all(torch.equal(getattr(left, key)(), getattr(right, key)()) for key in indices)
                    and torch.equal(left.values(), right.values()))
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (set(left.keys()) == set(right.keys())
                and all(_equal_tree(torch, left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal_tree(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def inspect_ownership(adapter):
    """Reject duplicate, foreign, missing, or frozen optimizer parameters."""
    live = {id(p) for p in adapter.model.parameters() if p.requires_grad}
    owned = [id(p) for group in adapter.optimizer.param_groups for p in group["params"]]
    if not live or len(owned) != len(set(owned)) or set(owned) != live:
        raise ValueError("Optimizer must own each live model parameter exactly once")
    return {"live_parameter_objects": len(live), "optimizer_parameter_objects": len(owned)}


def inspect_independence(adapters):
    """Check disjoint parameter storage/optimizers/adapters, including frozen params."""
    parameter_ids, storage_ids, optimizer_ids, adapter_ids = set(), set(), set(), set()
    for adapter in adapters:
        inspect_ownership(adapter)
        ids = {id(p) for p in adapter.model.parameters()}
        storage = {(str(p.device), p.untyped_storage().data_ptr())
                   for p in adapter.model.parameters() if p.untyped_storage().nbytes() > 0}
        if (parameter_ids & ids or storage_ids & storage or id(adapter.optimizer) in optimizer_ids
                or id(adapter) in adapter_ids):
            raise ValueError("Independent trajectories share an adapter, optimizer or parameter storage")
        parameter_ids.update(ids)
        storage_ids.update(storage)
        optimizer_ids.add(id(adapter.optimizer))
        adapter_ids.add(id(adapter))
    return {"independent_adapters": len(adapter_ids)}


def capture_checkpoint(torch, adapter, *, fit_id, step, validation, decision):
    inspect_ownership(adapter)
    payload = {
        "schema_version": 1, "fit_id": fit_id, "step": step,
        "validation": _plain(torch, validation), "decision": _plain(torch, decision),
        "model_state": _clone_tree(torch, adapter.model.state_dict()),
        "optimizer_state": _clone_tree(torch, adapter.optimizer.state_dict()),
        "module_modes": {name: bool(module.training) for name, module in adapter.model.named_modules()},
        "requires_grad": {name: bool(p.requires_grad) for name, p in adapter.model.named_parameters()},
        "adapter_state": _clone_tree(torch, adapter.capture_state()),
    }
    _finite_tree(torch, payload, "checkpoint")
    return payload


def restore_checkpoint(torch, path, adapter, *, expected_fit_id=None, verify=True):
    """Restore tensors, optimizer, mixed modes, flags, selector and all RNG.

    Adapter capture/restore methods must include global and private RNG streams,
    selector state and required nonpersistent input custody. They must not draw
    randomness themselves. The checkpoint is an ordinary trusted torch file.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint")
    if expected_fit_id is not None and payload["fit_id"] != expected_fit_id:
        raise ValueError("Checkpoint belongs to another paid fit")
    modules = dict(adapter.model.named_modules())
    parameters = dict(adapter.model.named_parameters())
    if set(modules) != set(payload["module_modes"]) or set(parameters) != set(payload["requires_grad"]):
        raise ValueError("Checkpoint module/parameter topology differs")
    adapter.model.load_state_dict(payload["model_state"], strict=True)
    for name, flag in payload["requires_grad"].items():
        parameters[name].requires_grad_(flag)
    adapter.optimizer.load_state_dict(payload["optimizer_state"])
    for parameter in adapter.model.parameters():
        parameter.grad = None
    # .train() recursively changes children and destroys mixed frozen-stem modes.
    for name, flag in payload["module_modes"].items():
        modules[name].training = flag
    adapter.restore_state(_clone_tree(torch, payload["adapter_state"]))
    inspect_ownership(adapter)
    if verify:
        actual = capture_checkpoint(torch, adapter, fit_id=payload["fit_id"], step=payload["step"],
                                    validation=payload["validation"], decision=payload["decision"])
        if not _equal_tree(torch, actual, payload):
            raise RuntimeError("Checkpoint restore did not reproduce the exact saved state")
    return {"fit_id": payload["fit_id"], "step": payload["step"],
            "validation": payload["validation"], "decision": payload["decision"]}


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(torch, path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        size = temporary.stat().st_size
        temporary.replace(path)
        return size
    finally:
        temporary.unlink(missing_ok=True)


def _validate_spec(spec):
    # Copy once: this prospective contract cannot be changed by a later adapter call.
    spec = json.loads(json.dumps(spec, allow_nan=False))
    for key in ("fit_id", "budget_steps", "budget_unit", "validation_interval", "validate_initial",
                "prefixes", "inherited_fit_ids", "adapter_spec"):
        if key not in spec:
            raise ValueError("Missing engine contract field: " + key)
    if not isinstance(spec["fit_id"], str) or not spec["fit_id"]:
        raise ValueError("A fresh paid fit ID is required")
    for key in ("budget_steps", "validation_interval"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if spec["budget_unit"] not in {"optimizer_updates", "epochs"} or type(spec["validate_initial"]) is not bool:
        raise ValueError("Declare budget unit and checkpoint0 validation")
    if not isinstance(spec["adapter_spec"], dict):
        raise ValueError("Declare adapter-specific loss, fusion, selector and RNG semantics")
    parents = spec["inherited_fit_ids"]
    if (not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents)
            or len(parents) != len(set(parents)) or spec["fit_id"] in parents):
        raise ValueError("Invalid inherited paid fit IDs")
    if not isinstance(spec["prefixes"], list):
        raise ValueError("Prefix roster must be a list")
    ids = set()
    for prefix in spec["prefixes"]:
        identifier, cap = prefix.get("prefix_id"), prefix.get("cap_step")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier)
                or identifier in ids or type(cap) is not int or not 1 <= cap <= spec["budget_steps"]
                or type(prefix.get("close_on_selector_stop")) is not bool):
            raise ValueError("Invalid or duplicate planned prefix")
        ids.add(identifier)
    window = spec.get("convergence_window", 10)
    if type(window) is not int or window < 1:
        raise ValueError("convergence_window must be a positive integer")
    return spec


def _metrics(torch, raw):
    if not isinstance(raw, Mapping):
        raise TypeError("Adapter metrics must be a mapping")
    metrics = _plain(torch, {k: v for k, v in raw.items() if k != "operation_counts"})
    for key in ("objective", "ce", "regularization"):
        if key not in metrics or type(metrics[key]) not in (int, float):
            raise ValueError("Adapter must report finite objective, ce and regularization")
    return metrics


def _counts(raw, *, positive_updates):
    if not isinstance(raw, Mapping) or "optimizer_updates" not in raw:
        raise ValueError("Adapter must report actual optimizer_updates")
    if any(type(v) is not int or v < 0 for v in raw.values()):
        raise ValueError("Operation counts must be nonnegative integers")
    if positive_updates and raw["optimizer_updates"] < 1:
        raise ValueError("A successful training call must perform at least one optimizer update")
    return dict(raw)


def run_trajectory(torch, adapter, spec, output_dir, *, resource_check=None):
    """Run the complete declared budget; selector stops close only virtual prefixes.

    train_step(step) performs one declared unit and returns finite objective/CE/
    regularization plus actual operation_counts. Epoch units may contain many
    optimizer updates. Validation occurs at interval, every prefix cap and the
    last step, plus step zero when declared. observe_validation receives only
    validation metrics and returns save_selected/stop_requested booleans.

    A failed run retains spec.json, trace.jsonl, result.json and existing ordinary
    checkpoints, then propagates the original exception. Never rerun in the same
    output directory. The caller/registry owns construction and inherited costs.
    """
    spec = _validate_spec(spec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("A trajectory output directory must be empty; retain previous attempts")
    _atomic_json(output / "spec.json", spec)
    started = time.perf_counter()
    result = {
        "schema_version": 1, "fit_id": spec["fit_id"], "inherited_fit_ids": spec["inherited_fit_ids"],
        "status": "running", "budget_steps": spec["budget_steps"], "budget_unit": spec["budget_unit"],
        "completed_steps": 0, "attempted_step": 0, "selected": None, "first_selector_stop": None,
        "prefixes": [{**p, "closed": False, "selected": None} for p in spec["prefixes"]],
        "operation_counts": {"optimizer_updates": 0}, "partial_operation_counts_unknown": False,
        "costs": {key: 0.0 for key in ("train_seconds", "validation_seconds", "selector_seconds",
                                         "checkpoint_seconds", "restore_seconds", "resource_seconds")},
        "validation_calls": 0, "checkpoint_writes": 0, "checkpoint_bytes_written": 0,
        "resource_observations": [],
    }
    train_series = {key: [] for key in ("objective", "ce", "regularization")}
    phase, step, counts_recorded = "admission", 0, False
    last_validation, last_decision = None, None

    def add_counts(counts):
        for key, count in counts.items():
            result["operation_counts"][key] = result["operation_counts"].get(key, 0) + count

    def resource(boundary):
        if resource_check is not None:
            before = time.perf_counter()
            try:
                observation = resource_check(boundary, step, copy.deepcopy(result))
                if observation is not None:
                    result["resource_observations"].append({"phase": boundary, "step": step,
                                                            "value": _plain(torch, observation)})
            finally:
                result["costs"]["resource_seconds"] += time.perf_counter() - before

    def save(path, payload):
        before = time.perf_counter()
        try:
            size = _atomic_checkpoint(torch, path, payload)
            result["checkpoint_writes"] += 1
            result["checkpoint_bytes_written"] += size
        finally:
            result["costs"]["checkpoint_seconds"] += time.perf_counter() - before

    def summary():
        rows = {}
        for key, values in train_series.items():
            tail = values[-spec.get("convergence_window", 10):]
            rows[key] = ({"first": values[0], "last": values[-1], "minimum": min(values),
                          "maximum": max(values), "recent_steps": len(tail),
                          "recent_mean": sum(tail) / len(tail),
                          "recent_change_per_step": (tail[-1] - tail[0]) / (len(tail) - 1) if len(tail) > 1 else None}
                         if values else None)
        return {"diagnostic_only": True, "training": rows, "last_validation": last_validation,
                "first_selector_stop": result["first_selector_stop"],
                "full_budget_completed": result["completed_steps"] == spec["budget_steps"]}

    with (output / "trace.jsonl").open("x") as trace:
        def record(value):
            trace.write(json.dumps(value, allow_nan=False) + "\n")
            trace.flush()

        def validate():
            nonlocal phase, last_validation, last_decision
            phase = "validation"
            resource("before_validation")
            before = time.perf_counter()
            try:
                raw = adapter.validate(step)
            finally:
                result["costs"]["validation_seconds"] += time.perf_counter() - before
            result["validation_calls"] += 1
            last_validation = _metrics(torch, raw)
            _finite_tree(torch, adapter.model.state_dict(), "model after validation")
            phase = "selector"
            before = time.perf_counter()
            try:
                last_decision = _plain(torch, adapter.observe_validation(step, copy.deepcopy(last_validation)))
            finally:
                result["costs"]["selector_seconds"] += time.perf_counter() - before
            if any(type(last_decision.get(key)) is not bool for key in ("save_selected", "stop_requested")):
                raise ValueError("Selector must declare save_selected and stop_requested booleans")
            if last_decision["stop_requested"] and result["first_selector_stop"] is None:
                result["first_selector_stop"] = step
            if last_decision["save_selected"]:
                phase = "checkpoint"
                payload = capture_checkpoint(torch, adapter, fit_id=spec["fit_id"], step=step,
                                             validation=last_validation, decision=last_decision)
                save(output / "selected.pt", payload)
                result["selected"] = {"step": step, "path": "selected.pt", "validation": last_validation}
                for prefix in result["prefixes"]:
                    if not prefix["closed"]:
                        path = Path("prefixes") / prefix["prefix_id"] / "selected.pt"
                        save(output / path, payload)
                        prefix["selected"] = {"step": step, "path": str(path), "validation": last_validation}
            for prefix in result["prefixes"]:
                if prefix["closed"]:
                    continue
                reason = ("selector_stop" if prefix["close_on_selector_stop"] and last_decision["stop_requested"]
                          else "cap" if step == prefix["cap_step"] else None)
                if reason:
                    prefix.update(closed=True, closure_step=step, closure_reason=reason)
                    if prefix["selected"] is None:
                        raise RuntimeError("Prefix closed without an eligible validation checkpoint")
                    _atomic_json(output / "prefixes" / prefix["prefix_id"] / "certificate.json", prefix)
            record({"kind": "validation", "step": step, "metrics": last_validation, "decision": last_decision})

        try:
            result["ownership"] = inspect_ownership(adapter)
            _finite_tree(torch, adapter.model.state_dict(), "initial model")
            resource("before_training")
            if spec["validate_initial"]:
                validate()
            caps = {p["cap_step"] for p in spec["prefixes"]}
            for step in range(1, spec["budget_steps"] + 1):
                result["attempted_step"] = step
                phase, counts_recorded = "resource", False
                resource("before_step")
                phase = "training"
                before = time.perf_counter()
                try:
                    raw = adapter.train_step(step)
                finally:
                    result["costs"]["train_seconds"] += time.perf_counter() - before
                add_counts(_counts(raw.get("operation_counts"), positive_updates=True))
                counts_recorded = True
                metrics = _metrics(torch, raw)
                _finite_tree(torch, adapter.model.state_dict(), "trained model")
                _finite_tree(torch, adapter.optimizer.state_dict(), "trained optimizer")
                result["completed_steps"] = step
                for key in train_series:
                    train_series[key].append(metrics[key])
                record({"kind": "training", "step": step, "metrics": metrics,
                        "operation_counts": raw["operation_counts"]})
                if step % spec["validation_interval"] == 0 or step in caps or step == spec["budget_steps"]:
                    validate()
                phase = "resource"
                resource("after_step")
            phase = "checkpoint"
            save(output / "final_training.pt", capture_checkpoint(
                torch, adapter, fit_id=spec["fit_id"], step=step,
                validation=last_validation, decision=last_decision))
            if (spec["budget_unit"] == "optimizer_updates"
                    and result["operation_counts"]["optimizer_updates"] != spec["budget_steps"]):
                raise ValueError("Reported optimizer updates differ from the declared update budget")
            if result["selected"] is None:
                raise RuntimeError("Full trajectory has no eligible validation-selected checkpoint")
            phase = "restore"
            resource("before_restore")
            before = time.perf_counter()
            try:
                restore_checkpoint(torch, output / "selected.pt", adapter, expected_fit_id=spec["fit_id"])
            finally:
                result["costs"]["restore_seconds"] += time.perf_counter() - before
            resource("after_restore")
            result["status"] = "complete"
        except BaseException as exc:
            if phase == "training" and not counts_recorded:
                if hasattr(exc, "operation_counts"):
                    try:
                        add_counts(_counts(exc.operation_counts, positive_updates=False))
                    except (TypeError, ValueError) as accounting_error:
                        result["partial_operation_counts_unknown"] = True
                        result["failure_accounting_error"] = str(accounting_error)
                else:
                    result["partial_operation_counts_unknown"] = True
            result.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc),
                                                      "phase": phase, "step": step})
            record({"kind": "failure", **result["failure"]})
            raise
        finally:
            result["costs"]["wall_seconds"] = time.perf_counter() - started
            result["convergence"] = summary()
            _atomic_json(output / "result.json", result)
    return result
