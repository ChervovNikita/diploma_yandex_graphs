"""Full-budget training and checkpoint custody, with no numerical imports.

The caller injects torch. An adapter owns the model, optimizer, loss reduction,
training procedure, validation fusion, selector, and every RNG stream. Epoch
adapters must return sample-weighted epoch metrics, not their last minibatch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path


def checkpoint_receipt(path, *, fit_id, step):
    """Bind a completed selected checkpoint without decoding its tensors."""
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": path.name, "bytes": size, "sha256": digest.hexdigest(),
            "fit_id": fit_id, "step": step}


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
    # A shared module has distinct state_dict keys pointing at the same tensor.
    # Memoize only identical dense views; different views and independent bodies
    # retain separate clones. Every top-level call owns a fresh CPU snapshot.
    dense = {}

    def clone(item):
        if torch.is_tensor(item):
            if item.layout == torch.strided and not item.is_quantized:
                storage = item.untyped_storage()
                key = (item.device, item.dtype, storage.data_ptr(), storage.nbytes(),
                       item.storage_offset(), tuple(item.shape), tuple(item.stride()),
                       item.is_conj(), item.is_neg())
                if key not in dense:
                    dense[key] = item.detach().cpu().clone()
                return dense[key]
            return item.detach().cpu().clone()
        if isinstance(item, Mapping):
            result = type(item)((copy.deepcopy(k), clone(v)) for k, v in item.items())
            if hasattr(item, "_metadata"):
                result._metadata = copy.deepcopy(item._metadata)
            return result
        if isinstance(item, tuple):
            return tuple(clone(v) for v in item)
        if isinstance(item, list):
            return [clone(v) for v in item]
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise TypeError("Checkpoint state must be a plain tensor/container tree")

    return clone(value)


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
    frozen = adapter.source_frozen_factor_ownership() if hasattr(adapter, 'source_frozen_factor_ownership') else set()
    model_frozen = {id(p) for p in adapter.model.parameters() if not p.requires_grad}
    if not isinstance(frozen, set) or not frozen <= model_frozen or frozen & live:
        raise ValueError('Only declared exact source model frozen factors may retain optimizer ownership')
    if not live or len(owned) != len(set(owned)) or set(owned) != live | frozen:
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
        expected = {key: payload[key] for key in actual}
        if not _equal_tree(torch, actual, expected):
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


def _atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
        return len(payload)
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
    interval = spec.get("recovery_interval", 100)
    if type(interval) is not int or interval < 1:
        raise ValueError("recovery_interval must be a positive integer")
    policy=spec.get('native_amp_cycle_policy')
    if policy is not None:
        run=spec['adapter_spec'].get('run',{});c=run.get('config',{})
        arms={'native_single','factorized_native_single','original_gnnm','shared_own_only','native_pool_credit','source_view_supervision','uncoupled_source_contrast','COMMON_cycle','assigned_source_supply'}
        if (policy!={'schema':'native-SeHGNN-AMP-complete-cycle-v1','optimizer_owner_attempts':1}
                or spec['budget_unit']!='epochs' or spec['budget_steps']!=1000
                or c.get('sehgnn_arm') not in arms or c.get('sehgnn_epochs')!=1000
                or c.get('source_optimizer_count')!=1 or c.get('native_AMP_overflow_skip_allowed') is not True
                or c.get('historical_state_inherited') is not False
                or c.get('backbone')!='native_SeHGNN' or c.get('width')!=512 or c.get('classes')!=5
                or c.get('feature_channels')!=25 or c.get('label_channels')!=12
                or not run.get('adapter','').endswith('/adapters/sehgnn.py:make_adapter')):
            raise ValueError('Only explicitly typed complete native SeHGNN AMP epochs may skip optimizer maps')
    return spec


def _training_counts(raw,spec):
    policy=spec.get('native_amp_cycle_policy')
    if policy is None:return _counts(raw,positive_updates=True)
    counts=_counts(raw,positive_updates=False);attempts=counts.get('source_optimizer_attempts');skips=counts.get('source_AMP_skipped_optimizer_maps');maps=counts['optimizer_updates']
    if (attempts!=policy['optimizer_owner_attempts'] or skips is None or maps+skips!=attempts
            or counts.get('source_native_complete_cycles')!=1
            or counts.get('source_native_AMP_enabled') not in (0,1)
            or skips and counts.get('source_native_AMP_enabled')!=1
            or counts.get('model_forwards',0)<1 or counts.get('backward_calls',0)<1
            or counts.get('supervised_examples',0)<1
            or skips and counts.get('accepted_private_corrections',0)!=0):
        raise ValueError('Native AMP epochs must report complete supervised work, every attempt, and committed/skipped maps')
    return counts


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


def run_trajectory(torch, adapter, spec, output_dir, *, resource_check=None, _resume_admission=None):
    """Run the complete declared budget; selector stops close only virtual prefixes.

    train_step(step) performs one declared unit and returns finite objective/CE/
    regularization plus actual operation_counts. Epoch units may contain many
    optimizer updates. Validation occurs at interval, every prefix cap and the
    last step, plus step zero when declared. observe_validation receives only
    validation metrics and returns save_selected/stop_requested booleans.

    A failed run retains its ordinary files, then propagates the original error.
    Only resume_trajectory may continue an existing attempt. The caller/registry
    owns construction, inherited costs and proof that the previous owner ended.
    """
    spec = _validate_spec(spec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if _resume_admission is None and any(output.iterdir()):
        raise FileExistsError("A trajectory output directory must be empty; retain previous attempts")
    if _resume_admission is None:
        _atomic_json(output / "spec.json", spec)
    started = time.perf_counter()
    result = {
        "schema_version": 1, "fit_id": spec["fit_id"], "inherited_fit_ids": spec["inherited_fit_ids"],
        "status": "running", "budget_steps": spec["budget_steps"], "budget_unit": spec["budget_unit"],
        "completed_steps": 0, "attempted_step": 0, "selected": None, "first_selector_stop": None,
        "prefixes": [{**p, "closed": False, "selected": None} for p in spec["prefixes"]],
        "operation_counts": {"optimizer_updates": 0}, "partial_operation_counts_unknown": False,
        "trajectory_operation_counts": {"optimizer_updates": 0}, "actual_costs_incomplete": False,
        "costs": {key: 0.0 for key in ("train_seconds", "validation_seconds", "selector_seconds",
                                         "checkpoint_seconds", "restore_seconds", "resource_seconds")},
        "validation_calls": 0, "validation_attempts": 0,
        "checkpoint_writes": 0, "checkpoint_bytes_written": 0,
        "resource_observations": [],
        "attempts": [], "attempt_index": 1, "resume_admissions": [],
    }
    train_series = {key: [] for key in ("objective", "ce", "regularization")}
    phase, step, counts_recorded = "admission", 0, False
    last_validation, last_decision = None, None
    start_step, prior_wall = 0, 0.0
    attempt_start_counts, attempt_start_costs = None, None

    if _resume_admission is not None:
        if json.loads((output / "spec.json").read_text()) != spec:
            raise ValueError("Resume must use the unchanged prospective specification")
        recovery_path = output / "latest_trajectory.pt"
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        saved = recovery.get("recovery")
        if not saved or saved["spec"] != spec or recovery["fit_id"] != spec["fit_id"]:
            raise ValueError("Recovery checkpoint belongs to a different contract")
        result = copy.deepcopy(saved["result"])
        previous = json.loads((output / "result.json").read_text()) if (output / "result.json").is_file() else None
        if previous and previous.get("attempt_index") == result["attempt_index"]:
            if previous["status"] != "failed":
                raise ValueError("A completed or nonterminal run cannot be resumed")
            # Retain all paid operations/costs, including work after the latest
            # recoverable state. Logical trajectory progress comes from recovery.
            for key in ("operation_counts", "costs", "attempts", "checkpoint_writes",
                        "checkpoint_bytes_written", "validation_calls", "resource_observations",
                        "partial_operation_counts_unknown", "actual_costs_incomplete"):
                result[key] = copy.deepcopy(previous[key])
            result["replayed_completed_steps"] = previous["completed_steps"] - result["completed_steps"]
        else:
            # A killed owner might not have written a terminal result. Preserve
            # the checkpoint's known lower bound and declare unknown later work.
            result["partial_operation_counts_unknown"] = True
            result["actual_costs_incomplete"] = True
            result["attempts"].append({"attempt_index": result["attempt_index"],
                                       "status": "owner_ended_without_terminal_result",
                                       "known_completed_step": result["completed_steps"]})
            result["replayed_completed_steps"] = None
        start_step = result["completed_steps"]
        prior_wall = result["costs"].get("wall_seconds", 0.0)
        attempt_start_counts = dict(result["operation_counts"])
        attempt_start_costs = dict(result["costs"])
        # Preserve the prior attempt's actual endpoint files before recovering
        # older states. There are no hashes/manifests or implicit new paid fits.
        archive = output / "attempts" / str(result["attempt_index"])
        if archive.exists():
            raise FileExistsError("Prior attempt was already archived; inspect ownership before retrying")
        archive.mkdir(parents=True)
        before = time.perf_counter()
        for relative in ("result.json", "selected.pt", "final_training.pt", "latest_trajectory.pt"):
            source = output / relative
            if source.is_file():
                shutil.copy2(source, archive / relative)
                result["checkpoint_bytes_written"] += source.stat().st_size
        if (output / "prefixes").exists():
            shutil.copytree(output / "prefixes", archive / "prefixes")
            result["checkpoint_bytes_written"] += sum(p.stat().st_size for p in (archive / "prefixes").rglob("*") if p.is_file())
        result["costs"]["checkpoint_seconds"] += time.perf_counter() - before
        result["attempt_index"] += 1
        result["resume_admissions"].append(copy.deepcopy(_resume_admission))
        result["status"] = "running"
        result.pop("failure", None)
        train_series = copy.deepcopy(saved["train_series"])
        last_validation, last_decision = saved["last_validation"], saved["last_decision"]
        before = time.perf_counter()
        restore_checkpoint(torch, recovery_path, adapter, expected_fit_id=spec["fit_id"])
        result["costs"]["restore_seconds"] += time.perf_counter() - before
        # Full/prefix files may have advanced after the recovery save. Recover
        # those exact states too; closed prefixes remain frozen from this point.
        before = time.perf_counter()
        for relative, payload in saved["selected_snapshots"].items():
            target = output / relative
            if target.is_file() and target.read_bytes() == payload:
                continue
            size = _atomic_bytes(target, payload)
            result["checkpoint_writes"] += 1
            result["checkpoint_bytes_written"] += size
        known_endpoints = ["selected.pt"] + [str(Path("prefixes") / p["prefix_id"] / "selected.pt")
                                              for p in result["prefixes"]]
        for relative in known_endpoints:
            if relative not in saved["selected_snapshots"]:
                (output / relative).unlink(missing_ok=True)
        for prefix in result["prefixes"]:
            certificate = output / "prefixes" / prefix["prefix_id"] / "certificate.json"
            if prefix["closed"]:
                _atomic_json(certificate, prefix)
            else:
                certificate.unlink(missing_ok=True)
        result["costs"]["checkpoint_seconds"] += time.perf_counter() - before
        step = start_step
    if attempt_start_counts is None:
        attempt_start_counts = dict(result["operation_counts"])
        attempt_start_costs = dict(result["costs"])

    def add_counts(counts, *, trajectory=False):
        for key, count in counts.items():
            result["operation_counts"][key] = result["operation_counts"].get(key, 0) + count
            if trajectory:
                result["trajectory_operation_counts"][key] = result["trajectory_operation_counts"].get(key, 0) + count

    def resource(boundary):
        if resource_check is not None:
            before = time.perf_counter()
            try:
                # Historical observations grow with the run. Copy only current
                # status/counters here, keeping callback work linear in steps.
                current = {key: value for key, value in result.items()
                           if key not in {"resource_observations", "attempts", "resume_admissions", "convergence"}}
                current["resource_observation_count"] = len(result["resource_observations"])
                observation = resource_check(boundary, step, copy.deepcopy(current))
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

    def save_recovery():
        payload = capture_checkpoint(torch, adapter, fit_id=spec["fit_id"], step=result["completed_steps"],
                                     validation=last_validation, decision=last_decision)
        snapshots = {}
        endpoints = [result["selected"]] + [p["selected"] for p in result["prefixes"]]
        for endpoint in endpoints:
            if endpoint is not None and endpoint["path"] not in snapshots:
                snapshots[endpoint["path"]] = (output / endpoint["path"]).read_bytes()
        result["costs"]["wall_seconds"] = prior_wall + time.perf_counter() - started
        payload["recovery"] = {"spec": spec, "result": copy.deepcopy(result),
                               "train_series": copy.deepcopy(train_series),
                               "last_validation": last_validation, "last_decision": last_decision,
                               "selected_snapshots": snapshots}
        save(output / "latest_trajectory.pt", payload)

    with (output / "trace.jsonl").open("a" if _resume_admission is not None else "x") as trace:
        def record(value):
            value = {"attempt": result["attempt_index"], **value}
            trace.write(json.dumps(value, allow_nan=False) + "\n")
            trace.flush()

        validation_counts_recorded = False

        def validate():
            nonlocal phase, last_validation, last_decision, validation_counts_recorded
            phase = "validation"
            validation_counts_recorded = False
            resource("before_validation")
            before = time.perf_counter()
            result["validation_attempts"] = result.get("validation_attempts", 0) + 1
            try:
                raw = adapter.validate(step)
            finally:
                result["costs"]["validation_seconds"] += time.perf_counter() - before
            if "operation_counts" in raw:
                counts = _counts(raw["operation_counts"], positive_updates=False)
                add_counts(counts)
                validation_counts_recorded = True
                if counts["optimizer_updates"] != 0:
                    raise ValueError("Validation must not perform optimizer updates")
            validation_counts_recorded = True
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
            record({"kind": "validation", "step": step, "metrics": last_validation,
                    "decision": last_decision, "operation_counts": raw.get("operation_counts")})

        try:
            result["ownership"] = inspect_ownership(adapter)
            _finite_tree(torch, adapter.model.state_dict(), "initial model")
            resource("before_training")
            if _resume_admission is None:
                if spec["validate_initial"]:
                    validate()
                phase = "checkpoint"
                save_recovery()
            else:
                record({"kind": "resume", "step": start_step, "admission": _resume_admission})
            caps = {p["cap_step"] for p in spec["prefixes"]}
            for step in range(start_step + 1, spec["budget_steps"] + 1):
                result["attempted_step"] = step
                phase, counts_recorded = "resource", False
                resource("before_step")
                phase = "training"
                before = time.perf_counter()
                try:
                    raw = adapter.train_step(step)
                finally:
                    result["costs"]["train_seconds"] += time.perf_counter() - before
                counts = _training_counts(raw.get("operation_counts"),spec)
                add_counts(counts)
                counts_recorded = True
                metrics = _metrics(torch, raw)
                _finite_tree(torch, adapter.model.state_dict(), "trained model")
                _finite_tree(torch, adapter.optimizer.state_dict(), "trained optimizer")
                for key, count in counts.items():
                    result["trajectory_operation_counts"][key] = result["trajectory_operation_counts"].get(key, 0) + count
                result["completed_steps"] = step
                for key in train_series:
                    train_series[key].append(metrics[key])
                record({"kind": "training", "step": step, "metrics": metrics,
                        "operation_counts": raw["operation_counts"]})
                if step % spec["validation_interval"] == 0 or step in caps or step == spec["budget_steps"]:
                    validate()
                if (step % spec.get("recovery_interval", 100) == 0
                        or step in caps or step == spec["budget_steps"]
                        or any(p.get("closure_step") == step for p in result["prefixes"])):
                    phase = "checkpoint"
                    save_recovery()
                phase = "resource"
                resource("after_step")
            phase = "checkpoint"
            save(output / "final_training.pt", capture_checkpoint(
                torch, adapter, fit_id=spec["fit_id"], step=step,
                validation=last_validation, decision=last_decision))
            result['final_training_checkpoint'] = checkpoint_receipt(output / 'final_training.pt', fit_id=spec['fit_id'], step=step)
            if (spec["budget_unit"] == "optimizer_updates"
                    and result["trajectory_operation_counts"]["optimizer_updates"] != spec["budget_steps"]):
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
            before = time.perf_counter()
            result["selected_checkpoint"] = checkpoint_receipt(
                output / "selected.pt", fit_id=spec["fit_id"], step=result["selected"]["step"])
            result["costs"]["checkpoint_seconds"] += time.perf_counter() - before
            result["checkpoint_bytes_hashed"] = result["selected_checkpoint"]["bytes"]
            result["status"] = "complete"
        except BaseException as exc:
            if phase == "validation" and not validation_counts_recorded:
                if hasattr(exc, "operation_counts"):
                    try:
                        add_counts(_counts(exc.operation_counts, positive_updates=False))
                    except (TypeError, ValueError) as accounting_error:
                        result["partial_validation_operation_counts_unknown"] = True
                        result["failure_accounting_error"] = str(accounting_error)
                else:
                    result["partial_validation_operation_counts_unknown"] = True
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
            # Boundary callbacks can request a controlled interruption after a
            # complete unit. A mid-training/validation interruption is unsafe:
            # retain the previous atomic recovery state and report partial work.
            if phase == "resource":
                try:
                    save_recovery()
                    result["recovery_saved_at_interrupt"] = True
                except BaseException as recovery_error:
                    result["recovery_save_error"] = str(recovery_error)
            record({"kind": "failure", **result["failure"]})
            raise
        finally:
            result["costs"]["wall_seconds"] = prior_wall + time.perf_counter() - started
            result["convergence"] = summary()
            result["attempts"].append({
                "attempt_index": result["attempt_index"], "status": result["status"],
                "resume_step": start_step, "completed_step": result["completed_steps"],
                "operation_counts": {key: count - attempt_start_counts.get(key, 0)
                                     for key, count in result["operation_counts"].items()},
                "costs": {key: value - attempt_start_costs.get(key, 0.0)
                          for key, value in result["costs"].items()},
                "failure": result.get("failure"),
            })
            _atomic_json(output / "result.json", result)
    return result


def resume_trajectory(torch, adapter, spec, output_dir, *, admission, resource_check=None):
    """Explicit continuation after the caller has proven the former owner ended.

    Admission is a plain receipt with previous_owner_terminal=True and a bound
    previous_owner identity. Registry/launcher owns proof; a timeout is not proof.
    Resume restores latest trajectory state, never the validation-selected best.
    It preserves old trace/attempts and all known paid counts. Interval replay is
    charged again; killed-owner work with no terminal record is marked unknown.
    """
    admission = _plain(torch, admission)
    if (not isinstance(admission, dict) or admission.get("previous_owner_terminal") is not True
            or not admission.get("previous_owner")):
        raise ValueError("Resume requires an explicit terminal-owner admission receipt")
    return run_trajectory(torch, adapter, spec, output_dir, resource_check=resource_check,
                          _resume_admission=admission)
