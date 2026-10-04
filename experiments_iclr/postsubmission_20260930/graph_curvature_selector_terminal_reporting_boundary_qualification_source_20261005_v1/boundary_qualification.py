"""SOURCE ONLY: one native continuation plus bounded, disposable boundary probes.

Preparation did not import numerical packages or execute this module. Runtime is
caller-owned and must already be authorized for the exact fresh native start.
No loader, server, shortened training loop, score-based decision or retry exists.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

PHASE = Path(__file__).resolve().parent.parent
SOURCE_NAME = "graph_curvature_selector_terminal_reporting_source_20261004_v2"
SOURCE_MANIFEST = "faace8c4ba59e3233a92809c3052c3dc02b37ddf77159c43e3573360818fbdf1"
REVIEW_NAME = "graph_curvature_selector_terminal_reporting_v2_fresh_review_20261004_v1"
REVIEW_MANIFEST = "5d78c2346cf00d1372d1108b35fad4975cc79f6f7edb50a845433439826d01f2"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _torch():
    import torch
    return torch


def _draw_default_rngs(adapter):
    """Runtime-only disposable probe of the unchanged canonical RNG inventory."""
    import numpy as np
    adapter.random.random()
    np.random.random()
    torch = _torch()
    torch.rand(1)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.rand(1, device="cuda:"+str(index))


def _verified_packet(name, expected):
    directory = PHASE / name
    data = (directory / "MANIFEST.json").read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "Manifest custody differs")
    manifest = json.loads(data)
    for record in manifest["files"]:
        payload = (directory / record["path"]).read_bytes()
        require(len(payload) == record["bytes"] and hashlib.sha256(payload).hexdigest() == record["sha256"],
                "Manifest payload differs: " + record["path"])
    return directory


def _load_reporter():
    source = _verified_packet(SOURCE_NAME, SOURCE_MANIFEST)
    _verified_packet(REVIEW_NAME, REVIEW_MANIFEST)
    path = source / "terminal_reporting.py"
    data = path.read_bytes()
    name = "terminal_reporting_v2_boundary_qualification_verified"
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, "__boundary_executed_sha256__", None) == hashlib.sha256(data).hexdigest()
                and getattr(module, "__boundary_executed_path__", None) == str(path),
                "Occupied qualification reporter module lacks exact executed-source custody")
        return module
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    module.__boundary_executed_sha256__ = hashlib.sha256(data).hexdigest()
    module.__boundary_executed_path__ = str(path)
    return module


def _require_canonical_custody(adapter, reporting):
    record = json.loads((reporting.HERE / "SOURCE_BINDINGS.json").read_text())["original_integration"]
    path, data = reporting._verified_bytes(record)
    require(Path(adapter.__file__).resolve() == path
            and getattr(adapter, "__graph_curvature_executed_sha256__", None) == record["sha256"]
            and getattr(adapter, "__graph_curvature_executed_path__", None) == str(path),
            "Caller must supply exact canonical adapter returned by the frozen v3 verified loader")
    require(adapter.continuation.__code__.co_filename == str(path), "Canonical function custody differs")


def _sync():
    torch = _torch()
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def _memory():
    torch = _torch()
    units = "bytes" if sys.platform == "darwin" else "KiB"
    result = dict(process_lifetime_peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  process_lifetime_peak_RSS_units=units, CUDA=[])
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            result["CUDA"].append(dict(device=index, allocated=torch.cuda.memory_allocated(index),
                reserved=torch.cuda.memory_reserved(index),
                process_lifetime_peak_allocated=torch.cuda.max_memory_allocated(index),
                process_lifetime_peak_reserved=torch.cuda.max_memory_reserved(index)))
    return result


class Costs:
    def __init__(self):
        self.records = []

    def call(self, name, function, *args, **kwargs):
        wall, cpu = time.perf_counter(), time.process_time()
        failed = True
        try:
            _sync()
            result = function(*args, **kwargs)
            _sync()
            failed = False
            return result
        finally:
            try:
                memory = _memory()
            except BaseException as error:
                memory = dict(status="UNAVAILABLE", error_type=type(error).__name__)
            self.records.append(dict(component=name, wall_seconds=time.perf_counter()-wall,
                process_CPU_seconds=time.process_time()-cpu, failed=failed,
                boundary="Includes leading/trailing admitted-device synchronization where reached",
                memory=memory))


def _equal(left, right):
    torch = _torch()
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                and left.layout == right.layout and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _replay_logits_match(left, right, adapter):
    # Native replay is a fresh forward, not tensor copying. Keep the already
    # bound native identity-logit tolerances; make no bitwise replay claim.
    torch = _torch()
    return (left.dtype == right.dtype and left.shape == right.shape
        and bool(torch.allclose(left.cpu(), right.cpu(), atol=adapter.TOLERANCES["logits_atol"],
                               rtol=adapter.TOLERANCES["logits_rtol"], equal_nan=False)))


def _digest(tensor):
    torch = _torch()
    require(tensor.layout == torch.strided, "Only actual dense native tensor layouts are covered")
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                sha256_raw_bytes=hashlib.sha256(memoryview(data)).hexdigest())


def _summarize(value):
    torch = _torch()
    if torch.is_tensor(value):
        return _digest(value)
    if isinstance(value, dict):
        score_keys = {"validation_nll", "primary_validation_nll", "source_native_validation_nll",
                      "nll_nats", "accuracy_fraction", "correct_nodes", "brier_sum_over_classes"}
        return {str(key): (dict(sha256_JSON_scalar=hashlib.sha256(json.dumps(item, allow_nan=False).encode()).hexdigest())
                          if key in score_keys else _summarize(item)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summarize(item) for item in value]
    return value


def _tensor_metadata(tensor):
    if tensor is None:
        return None
    return dict(identity=id(tensor), version=tensor._version, dtype=str(tensor.dtype),
                device=str(tensor.device), shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), data_pointer=tensor.data_ptr(),
                storage_pointer=tensor.untyped_storage().data_ptr(),
                requires_grad=tensor.requires_grad, layout=str(tensor.layout))


def _inputs(graph, train, validation):
    result = {}
    for owner_name, owner, names in (("graph", graph, ("teacher_input", "teacher_edge_index")),
            ("train", train, ("nodes", "labels")), ("validation", validation, ("nodes", "labels"))):
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                result[owner_name + "." + name] = dict(metadata=_tensor_metadata(value), digest=_digest(value))
    return result


def _state(adapter, raw, optimizer, graph, train, validation):
    before_rng = adapter.rng_snapshot()
    result = dict(model=adapter.cpu_copy(raw.state_dict()), optimizer=adapter.named_optimizer_snapshot(raw, optimizer),
        gradients={name: adapter.cpu_copy(parameter.grad) for name, parameter in raw.named_parameters()},
        modes={name: module.training for name, module in raw.named_modules(remove_duplicate=False)},
        module_identities={name: id(module) for name, module in raw.named_modules(remove_duplicate=False)},
        parameter_metadata={name: _tensor_metadata(parameter)
                            for name, parameter in raw.named_parameters(remove_duplicate=False)},
        gradient_metadata={name: _tensor_metadata(parameter.grad) for name, parameter in raw.named_parameters()},
        buffer_metadata={name: _tensor_metadata(buffer) for name, buffer in raw.named_buffers(remove_duplicate=False)},
        optimizer_identity=id(optimizer), optimizer_parameter_links=[[id(p) for p in group["params"]]
                                                            for group in optimizer.param_groups],
        specification=copy.deepcopy(getattr(raw, "specification", None)),
        global_stage=getattr(raw, "global_stage", None), members=raw.members,
        inputs=_inputs(graph, train, validation), RNG=before_rng)
    require(_equal(before_rng, adapter.rng_snapshot()), "Witness sampling changed default RNG")
    return result


def _without_rng(state):
    return {key: item for key, item in state.items() if key != "RNG"}


def _clone_terminal(adapter, raw_after_primary, terminal):
    raw = copy.deepcopy(raw_after_primary)
    raw.load_state_dict(terminal["model"])
    for name, module in raw.named_modules(remove_duplicate=False):
        module.training = terminal["modes"][name]
    for name, parameter in raw.named_parameters():
        grad = terminal["gradients"][name]
        parameter.grad = None if grad is None else grad.detach().clone().to(device=parameter.device, dtype=parameter.dtype)
    optimizer = adapter.restore_named_optimizer(raw, terminal["optimizer"])
    require(_equal(adapter.cpu_copy(raw.state_dict()), terminal["model"]), "Disposable terminal model differs")
    require(_equal(adapter.named_optimizer_snapshot(raw, optimizer), terminal["optimizer"]),
            "Disposable terminal Adam differs")
    return raw, optimizer


def _extract_boundaries(adapter, reporting):
    bindings = json.loads((reporting.HERE / "SOURCE_BINDINGS.json").read_text())
    path, data = reporting._verified_bytes(bindings["reporting_integration"])
    module = ast.parse(data, filename=str(path))
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "continuation")
    hook = next(n for n in function.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "terminal_observer is not None")
    index = function.body.index(hook)
    tail = function.body[index+1:]
    require(ast.unparse(tail[0].value.func) == "raw.load_state_dict", "Bound selected tail changed")
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "train_update"
                    for statement in [hook, *tail] for n in ast.walk(statement)), "Isolated boundary contains training")
    def build(name, arguments, statements, replacements=None):
        # Only the function wrapper is new. Bound statements are unmodified AST.
        body = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in arguments],
            vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=copy.deepcopy(statements), decorator_list=[])
        isolated = ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[]))
        namespace = dict(adapter.__dict__)
        namespace.update(replacements or {})
        exec(compile(isolated, str(path)+"::ISOLATED_BOUNDARY_TEST", "exec"), namespace)
        return namespace[name]
    hook_args = ["raw", "logits", "value", "best", "completed", "cap", "patience", "backbone", "terminal_observer", "terminal_context"]
    tail_args = ["raw", "graph", "validation", "save_logits", "best", "completed", "cap", "patience", "midpoint_saved", "midpoint_epoch", "backbone"]
    return dict(hook_line=hook.lineno, hook_AST_sha256=hashlib.sha256(ast.dump(hook).encode()).hexdigest(),
        tail_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=tail, type_ignores=[])).encode()).hexdigest(),
        hook=lambda replacements=None: build("_isolated_exact_hook", hook_args, [hook], replacements),
        tail=lambda replacements=None: build("_isolated_exact_tail", tail_args, tail, replacements))


def _extract_wrapper(reporting, replacements):
    source = ast.parse((reporting.HERE / "terminal_reporting.py").read_bytes())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "continue_one_arm_with_terminal_reporting")
    namespace = dict(reporting.__dict__)
    namespace.update(replacements)
    isolated = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]))
    exec(compile(isolated, str(reporting.HERE / "terminal_reporting.py")+"::ISOLATED_FAILURE_LEDGER_TEST", "exec"), namespace)
    return namespace[function.name]


def _verify_terminal_artifacts(reporting, reporter, witness, identity):
    torch = _torch()
    data = (reporter.output / "terminal_logits.pt").read_bytes()
    stored = torch.load(reporter.output / "terminal_logits.pt", map_location="cpu", weights_only=True)
    report = json.loads((reporter.output / "terminal_report.json").read_text())
    require(len(report["member_validation_competence"]) == 4
            and [item["member"] for item in report["member_validation_competence"]] == [0,1,2,3],
            "Saved member competence count/order differs")
    require(_equal(stored["member_raw_logits"], witness["logits"]), "Saved terminal members differ")
    require(_equal(stored["served_mean_raw_logits"], witness["served"]), "Saved native mean differs")
    require(_equal(stored["validation_nodes"], reporter.nodes) and _equal(stored["validation_labels"], reporter.labels),
            "Saved authorized VAL pack differs")
    require(report["tensors"]["sha256"] == hashlib.sha256(data).hexdigest()
            and report["tensors"]["bytes"] == len(data) and report["identity"] == identity, "Terminal receipt custody differs")
    for key, expected in witness["endpoint"].items():
        require(report["endpoint"][key] == expected, "Terminal endpoint differs: " + key)
    def independent(z):
        z = z[reporter.nodes].to(dtype=torch.float64)
        labels = reporter.labels
        logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
        p = logp.exp()
        error = p.clone()
        error[torch.arange(labels.numel()), labels] -= 1
        correct = z.argmax(-1).eq(labels)
        return dict(correct_nodes=int(correct.sum()), nodes=int(labels.numel()),
            accuracy_fraction=float(correct.to(torch.float64).mean()),
            nll_nats=float(-logp[torch.arange(labels.numel()), labels].mean()),
            brier_sum_over_classes=float(error.square().sum(-1).mean()))
    for expected, actual in [(independent(witness["served"]), report["served_validation"]),
            *[(independent(z), actual) for z, actual in zip(witness["logits"], report["member_validation_competence"])]]:
        for key in ["correct_nodes", "nodes", "accuracy_fraction"]:
            require(actual[key] == expected[key], "Independent accuracy/support differs")
        for key in ["nll_nats", "brier_sum_over_classes"]:
            require(abs(actual[key]-expected[key]) <= 1e-12, "Independent CPU float64 metric differs")
    return dict(terminal_tensors_equal=True, native_mean_equal=True, endpoint_equal=True,
                authorized_VAL_equal=True, receipt_hash_equal=True, independent_metric_checks_passed=True)


def _endpoint(locals_):
    completed, patience = locals_["completed"], locals_["patience"]
    best = locals_["best"]
    return dict(teacher_backbone=locals_["backbone"], members=locals_["raw"].members,
        continuation_updates_completed=completed, terminal_actual_update=completed+(50 if patience else 250),
        terminal_native_stage_epoch=completed+50, stage="native" if patience else "global",
        update_cap=locals_["cap"], patience=patience, full_native_cap_reached=completed == locals_["cap"],
        stop_reason="native_update_cap" if completed == locals_["cap"] else "native_patience",
        source_native_validation_nll=locals_["value"], terminal_before_selected_checkpoint_restore=True,
        selected_continuation_epoch=best["continuation_epoch"], selected_actual_update=best["actual_update"])


class ProbeError(RuntimeError):
    pass


class _FaultPath(type(Path())):
    """Disposable probe-only filesystem adapter; production paths are untouched."""
    _failure = None
    _failed_name = None
    _inventory_failure = False

    def open(self, *args, **kwargs):
        if self.name == self._failed_name:
            raise self._failure
        return super().open(*args, **kwargs)

    def iterdir(self):
        if self._inventory_failure:
            raise OSError("injected unavailable probe inventory")
        return super().iterdir()


def _failure_probes(reporting, adapter, boundaries, witness, terminal, raw_after_primary,
                    graph, train, validation, identity, run_root, relative, costs):
    cases = []
    ambient_rng = adapter.rng_snapshot()
    relative = Path(relative) / "terminal_reporting_boundary_qualification" / "probes"

    def one(case, expected_error, expected_stage, *, load_error=None, restore_error=None,
            hook_replacements=None, observer_error=None, tail_error=None, fault_file=None,
            inventory_failure=False, bad_identity=False, collision=False, pre_hook_trace_error=None):
        raw, optimizer = _clone_terminal(adapter, raw_after_primary, terminal)
        initialized = dict(model=raw, optimizer=optimizer, rng=terminal["RNG"])
        probe_relative = relative / case
        base_reporter = reporting.TerminalReporter
        def constructor(*args, **kwargs):
            reporter = base_reporter(*args, **kwargs)
            if collision:
                reporter.output.mkdir(parents=True, exist_ok=False)
                (reporter.output / "EXISTING_EVIDENCE").write_bytes(b"must remain unchanged\n")
            if fault_file:
                cls = type("FaultPath_"+case, (_FaultPath,), dict(_failure=expected_error,
                    _failed_name=fault_file, _inventory_failure=inventory_failure))
                reporter.output = cls(str(reporter.output))
            return reporter
        def continuation(raw_, optimizer_, graph_, train_, validation_, trace_, save_, *, terminal_observer, terminal_context):
            if pre_hook_trace_error is not None:
                def reject_epoch_zero(event):
                    raise pre_hook_trace_error
                return adapter.continuation(raw_, optimizer_, graph_, train_, validation_, reject_epoch_zero,
                    save_, terminal_observer=terminal_observer, terminal_context=terminal_context)
            def observer(payload_):
                if observer_error is not None:
                    _draw_default_rngs(adapter)
                    raise observer_error
                terminal_observer(payload_)
            replacements = dict(hook_replacements or {})
            hook = boundaries["hook"](replacements)
            native_logits = witness["logits"].to(next(raw_.parameters()).device)
            hook(raw_, native_logits, witness["value"], adapter.cpu_copy(witness["best"]),
                witness["completed"], witness["cap"], witness["patience"], witness["backbone"], observer, terminal_context)
            if tail_error is not None:
                def selected_callback(name, logits):
                    raise tail_error
                return boundaries["tail"]()(raw_, graph_, validation_, selected_callback,
                    adapter.cpu_copy(witness["best"]), witness["completed"], witness["cap"], witness["patience"],
                    witness["midpoint_saved"], witness["midpoint_epoch"], witness["backbone"])
            raise expected_error  # Explicit probe terminal; never used for scientific return.
        def loader():
            if load_error is not None:
                raise load_error
            def restore(state):
                if restore_error is not None:
                    raise restore_error
                adapter.rng_restore(state)
            return SimpleNamespace(rng_restore=restore, continuation=continuation)
        wrapper = _extract_wrapper(reporting, dict(TerminalReporter=constructor, load_reporting_adapter=loader))
        probe_tag = dict(test_only=True, case=case, isolated_boundary_or_ledger=True, predictive_evidence=False)
        supplied_identity = {"qualification_probe": probe_tag} if bad_identity else dict(identity, qualification_probe=probe_tag)
        try:
            wrapper(initialized, graph, train, validation, lambda event: None, lambda name, logits: None,
                report_root=run_root, arm_relative_directory=str(probe_relative), identity=supplied_identity)
        except BaseException as error:
            if expected_error is not None:
                require(error is expected_error, "Probe replaced original exception: " + case)
            else:
                require(isinstance(error, (ValueError, FileExistsError)), "Unexpected precondition/collision error")
        else:
            raise AssertionError("Failure probe returned successfully: " + case)
        key = hashlib.sha256(str(probe_relative).encode()).hexdigest()
        failed_path = Path(run_root) / ("terminal_reporting_attempt_"+key+"_FAILED.json")
        receipt = json.loads(failed_path.read_text())
        require(receipt["failure_stage"] == expected_stage, "Failure wrapper stage differs: " + case)
        if receipt["endpoint"] is not None:
            for key_, value_ in witness["endpoint"].items():
                require(receipt["endpoint"][key_] == value_, "Available failure endpoint lost: " + case)
        if collision:
            output = Path(run_root) / probe_relative / "terminal_reporting"
            require(list(p.name for p in output.iterdir()) == ["EXISTING_EVIDENCE"], "Collision wrote into prior evidence")
        if inventory_failure:
            inner = json.loads((Path(run_root)/probe_relative/"terminal_reporting"/"REPORT_FAILED.json").read_text())
            require(inner["partial_file_inventory_status"] == "UNAVAILABLE_DIRECTORY_INVENTORY", "Inventory fallback differs")
        if hook_replacements and "rng_snapshot" in hook_replacements:
            require(receipt["terminal_context"]["rng_restoration_status"] == "UNAVAILABLE_NO_COMPLETE_SNAPSHOT", "Failed snapshot claimed restore")
        if hook_replacements and "rng_restore" in hook_replacements:
            require(receipt["terminal_context"]["rng_restoration_status"] == "FAILED", "Restore failure status missing")
        cases.append(dict(case=case, primary_error_preserved=True, run_receipt_retained=True,
            endpoint_available=receipt["endpoint"] is not None, scientifically_used=False))
        adapter.rng_restore(ambient_rng)
        require(_equal(adapter.rng_snapshot(), ambient_rng), "Probe cleanup RNG differs")

    def fail(error):
        def operation(*args, **kwargs):
            raise error
        return operation

    cases_spec = []
    cases_spec.append(("constructor", None, "reporter_constructor", dict(bad_identity=True)))
    error = ProbeError("injected source loading failure")
    cases_spec.append(("loading", error, "bound_source_loading", dict(load_error=error)))
    error = ProbeError("injected initial restore failure")
    cases_spec.append(("initial_restore", error, "initial_continuation_rng_restore", dict(restore_error=error)))
    error = ProbeError("injected actual native epoch-zero trace failure")
    cases_spec.append(("native_before_hook", error, "native_continuation", dict(pre_hook_trace_error=error)))
    error = ProbeError("injected snapshot failure")
    cases_spec.append(("snapshot", error, "native_continuation", dict(hook_replacements=dict(rng_snapshot=fail(error)))))
    error = ProbeError("injected capture failure")
    cases_spec.append(("capture", error, "native_continuation", dict(hook_replacements=dict(cpu_copy=fail(error)))))
    primary, secondary = ProbeError("injected observer primary"), ProbeError("injected restoration secondary")
    cases_spec.append(("observer_and_restore", primary, "native_continuation", dict(observer_error=primary,
        hook_replacements=dict(rng_restore=fail(secondary)))))
    error = ProbeError("injected restoration after reporting")
    cases_spec.append(("restore_after_report", error, "native_continuation", dict(hook_replacements=dict(rng_restore=fail(error)))))
    error = ProbeError("injected selected callback failure")
    cases_spec.append(("selected_tail", error, "native_continuation", dict(tail_error=error)))
    error = ProbeError("injected terminal tensor IO failure")
    cases_spec.append(("tensor_IO", error, "native_continuation", dict(fault_file="terminal_logits.pt")))
    error = ProbeError("injected final JSON failure")
    cases_spec.append(("final_JSON", error, "native_continuation", dict(fault_file="terminal_report.json")))
    error = ProbeError("injected tensor IO plus unavailable inventory")
    cases_spec.append(("inventory", error, "native_continuation", dict(fault_file="terminal_logits.pt", inventory_failure=True)))
    cases_spec.append(("directory_collision", None, "native_continuation", dict(collision=True)))
    try:
        for case, error, stage, options in cases_spec:
            costs.call("isolated_failure_"+case, one, case, error, stage, **options)
        # Existing marker: exact wrapper path validation/open, no continuation.
        used = relative / "constructor"
        key = hashlib.sha256(str(used).encode()).hexdigest()
        marker = Path(run_root)/("terminal_reporting_attempt_"+key+".json")
        failure = Path(run_root)/("terminal_reporting_attempt_"+key+"_FAILED.json")
        original = (marker.read_bytes(), failure.read_bytes())
        try:
            reporting.continue_one_arm_with_terminal_reporting({}, graph, train, validation, lambda event: None,
                lambda name, logits: None, report_root=run_root, arm_relative_directory=str(used), identity=identity)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Marker collision did not propagate")
        require(original == (marker.read_bytes(), failure.read_bytes()), "Marker collision overwrote evidence")
        cases.append(dict(case="exclusive_marker_collision", existing_evidence_unchanged=True, scientifically_used=False))
        # Complete best-effort receipt: safely unavailable identity and inventory;
        # then a receipt write collision must preserve the pending original error.
        class NoInventory:
            def iterdir(self):
                raise OSError("injected missing inventory")
        receipt_path = Path(run_root)/relative/"best_effort_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        primary = ProbeError("retained primary for best-effort receipt")
        reporting._failure_exclusive(receipt_path, primary, identity={"unserializable": object()}, endpoint=None,
            terminal_context={}, stage="isolated_receipt_probe", wall_started=time.perf_counter(),
            cpu_started=time.process_time(), cost_scope="isolated_receipt_only", partial_directory=NoInventory())
        receipt = json.loads(receipt_path.read_text())
        require(receipt["identity_status"] == "UNAVAILABLE_JSON_IDENTITY"
                and receipt["partial_file_inventory_status"] == "UNAVAILABLE_DIRECTORY_INVENTORY", "Receipt fallback differs")
        previous = receipt_path.read_bytes()
        try:
            try:
                raise primary
            except BaseException:
                reporting._failure_exclusive(receipt_path, primary, identity=identity, endpoint=None, terminal_context={},
                    stage="isolated_receipt_write_collision", wall_started=time.perf_counter(), cpu_started=time.process_time(),
                    cost_scope="isolated_receipt_only")
                raise
        except BaseException as error:
            require(error is primary and receipt_path.read_bytes() == previous, "Receipt failure masked primary or overwrote")
        cases.append(dict(case="complete_best_effort_receipt_and_write_collision", primary_error_preserved=True,
            unavailable_fields_explicit=True, existing_evidence_unchanged=True, scientifically_used=False))
    finally:
        adapter.rng_restore(ambient_rng)
    return cases


def run_once(initialized, graph, train, validation, trace, save_logits, *, canonical_adapter,
             report_root, arm_relative_directory, identity, root_custody, retain_native_result,
             record_attempt_accounting):
    """One authorized full continuation; return its original result after checks.

    Root must verify fresh native warm/initialization, actual-state transport/
    selector qualification, runtime versions/device and VAL provenance already.
    ``retain_native_result`` must preserve the successful primary result before
    disposable probes; a failed probe does not authorize rerunning that primary.
    """
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    costs = Costs()
    audit = dict(schema="terminal-reporting-boundary-qualification-runtime-v1", source_manifest_sha256=SOURCE_MANIFEST,
        review_manifest_sha256=REVIEW_MANIFEST, primary_full_continuations_attempted=0, primary_completed=False,
        qualification_passed=False, failure_probes=[], costs=costs.records,
        cost_intervals_overlap_do_not_sum=True, predictive_metric_values_omitted_from_qualification_receipt=True,
        no_scientific_retry_authorized=True)
    owned, output, reporting, reporter = False, None, None, None
    context = {}
    try:
        require(sys.gettrace() is None, "An existing tracer requires separate reviewed integration")
        for key in ["fresh_actual_native_warm_verified", "actual_initialization_state_verified",
                    "continuation_RNG_verified", "native_runtime_and_device_verified", "authorized_VAL_provenance_verified",
                    "exact_captured_preprocessing_bound_to_warm_verified", "native_trace_and_save_callbacks_observational_verified"]:
            require(root_custody.get(key) is True, "Missing caller custody prerequisite: " + key)
        require(root_custody.get("source_manifest_sha256") == SOURCE_MANIFEST
                and root_custody.get("old_qualification_inherited") is False, "Caller custody/source scope differs")
        require(isinstance(root_custody.get("verification_receipts"), dict) and root_custody["verification_receipts"],
                "Caller must supply exact root-owned actual-state/runtime/provenance verification receipt descriptors")
        require(callable(retain_native_result), "Root must retain primary result before probes")
        require(callable(record_attempt_accounting), "Root must charge successful/failed complete qualification attempts")
        audit["root_custody_declarations"] = json.loads(json.dumps(root_custody, allow_nan=False))
        root, relative = Path(report_root).resolve(), Path(arm_relative_directory)
        require(root.is_relative_to(PHASE.resolve()) and root != PHASE.resolve(), "Run root must stay inside this phase")
        require(not relative.is_absolute() and relative.parts and ".." not in relative.parts, "Nonempty relative arm path required")
        output = (root/relative/"terminal_reporting_boundary_qualification").resolve()
        require(output.is_relative_to(root), "Qualification path leaves run root")
        output.mkdir(parents=True, exist_ok=False)
        owned = True
        (output/"ATTEMPT.json").write_text(json.dumps(dict(schema=audit["schema"], retry_authorized=False,
            source_manifest_sha256=SOURCE_MANIFEST, stage="BEFORE_LOADING_CONSTRUCTOR_AND_PRIMARY"), indent=2)+"\n")
        reporting = costs.call("verify_and_load_reporting_source", _load_reporter)
        _require_canonical_custody(canonical_adapter, reporting)
        raw, optimizer = initialized["model"], initialized["optimizer"]
        require(graph.teacher_backbone == reporting.BACKBONES[identity["graph"]] and raw.members == 4,
                "Supplied native initialization/backbone/member custody differs")
        baseline = costs.call("caller_state_before_reporter_constructor", _state, canonical_adapter,
                             raw, optimizer, graph, train, validation)
        reporter = costs.call("reporter_constructor", reporting.TerminalReporter, root, str(relative), identity, validation)
        adapter = costs.call("verify_and_load_reporting_adapter", reporting.load_reporting_adapter)
        boundaries = _extract_boundaries(adapter, reporting)
        audit["isolated_boundaries"] = {key: boundaries[key] for key in ["hook_line", "hook_AST_sha256", "tail_AST_sha256"]}
        costs.call("initial_native_continuation_RNG_restore", adapter.rng_restore, initialized["rng"])
        entry = costs.call("entry_witness", _state, adapter, raw, optimizer, graph, train, validation)
        require(_equal(_without_rng(baseline), _without_rng(entry)), "Reporter construction/loading changed native entry state")
        require(_equal(entry["RNG"], initialized["rng"]), "Native initial RNG differs")
        audit["entry_state_digests"] = _summarize(entry)
        del baseline, entry
        witness, observed_selected, callbacks = {}, {}, []
        forward_count = {"raw": 0, "train_update": 0, "evaluate": 0}
        def raw_forward(module, args):
            forward_count["raw"] += 1
        handle = raw.register_forward_pre_hook(raw_forward)
        def tracer(frame, event, arg):
            if event == "call":
                if frame.f_code is adapter.train_update.__code__:
                    forward_count["train_update"] += 1
                if frame.f_code is adapter.evaluate.__code__:
                    forward_count["evaluate"] += 1
                return tracer if frame.f_code is adapter.continuation.__code__ else None
            if frame.f_code is adapter.continuation.__code__ and event == "line" and frame.f_lineno == boundaries["hook_line"]:
                require(not witness, "Terminal boundary reached twice")
                live = frame.f_locals
                witness.update({key: live[key] for key in ["value", "completed", "cap", "patience", "backbone",
                    "midpoint_saved", "midpoint_epoch"]})
                witness["state"] = costs.call("actual_terminal_state_witness", _state, adapter, raw, optimizer, graph, train, validation)
                witness["best"] = adapter.cpu_copy(live["best"])
                witness["logits"] = adapter.cpu_copy(live["logits"])
                witness["native_logits_reference"] = live["logits"]
                witness["served"] = adapter.cpu_copy(live["logits"].mean(0))
                witness["endpoint"] = _endpoint(live)
                witness["native_logit_metadata"] = _tensor_metadata(live["logits"])
            return tracer
        def observer(payload):
            require(witness, "Observer dispatched without actual terminal witness")
            require(_equal(payload["member_logits"], witness["logits"])
                    and _equal(payload["served_logits"], witness["served"]), "Hook payload differs from native witness")
            require(not payload["member_logits"].requires_grad and not payload["served_logits"].requires_grad,
                    "Hook payload requires gradient")
            require(payload["member_logits"].untyped_storage().data_ptr()
                    != payload["served_logits"].untyped_storage().data_ptr(), "Member/served copies alias storage")
            if witness["native_logit_metadata"]["device"] == "cpu":
                require(payload["member_logits"].untyped_storage().data_ptr()
                        != witness["native_logit_metadata"]["storage_pointer"], "CPU hook payload aliases native logits")
            before = costs.call("pre_observer_state_witness", _state, adapter, raw, optimizer, graph, train, validation)
            require(_equal(before, witness["state"]), "Hook capture changed actual native state/RNG")
            costs.call("actual_supplied_terminal_reporter", reporter, payload)
            after = costs.call("post_observer_state_witness", _state, adapter, raw, optimizer, graph, train, validation)
            require(_equal(before, after), "Supplied observer changed native state/Adam/gradients/modes/inputs/RNG")
            require(_equal(witness["native_logits_reference"], witness["logits"])
                    and _tensor_metadata(witness["native_logits_reference"]) == witness["native_logit_metadata"],
                    "Supplied observer/capture changed existing native logits")
            audit["supplied_observer_native_state_unchanged"] = True
        def selected_callback(name, logits):
            callbacks.append(name)
            if name == "selected":
                observed_selected["logits"] = adapter.cpu_copy(logits)
                observed_selected["RNG"] = adapter.rng_snapshot()
            return save_logits(name, logits)
        audit["primary_full_continuations_attempted"] = 1
        try:
            sys.settrace(tracer)
            result = costs.call("one_unchanged_native_continuation_with_measured_observer", adapter.continuation,
                raw, optimizer, graph, train, validation, trace, selected_callback,
                terminal_observer=observer, terminal_context=context)
        finally:
            sys.settrace(None)
            handle.remove()
        audit["primary_completed"] = True
        costs.call("root_retains_original_primary_result", retain_native_result, *result)
        best, metadata = result
        terminal = witness["state"]
        require(_equal(best, witness["best"]), "Returned selected best differs from boundary best")
        require(_equal(adapter.cpu_copy(raw.state_dict()), best["state"]), "Final live model is not selected")
        require(all(not module.training for module in raw.modules()), "Final live model is not eval")
        require(_equal(adapter.named_optimizer_snapshot(raw, optimizer), terminal["optimizer"]), "Live Adam is not terminal history")
        require(_equal({n: adapter.cpu_copy(p.grad) for n,p in raw.named_parameters()}, terminal["gradients"]),
                "Live gradients are not terminal history")
        require(_equal(adapter.rng_snapshot(), terminal["RNG"]), "Post-return default RNG differs")
        require(_equal(_inputs(graph, train, validation), terminal["inputs"]), "Inputs changed across selected tail")
        require(context.get("rng_restoration_status") == "RESTORED", "Actual hook RNG restoration not established")
        U = witness["completed"]
        require(forward_count == dict(raw=2*U+2, train_update=U, evaluate=U+2), "Observed model forward counts differ")
        require(callbacks == (["native_midpoint"] if witness["midpoint_saved"] else [])+["selected"], "Callback names/order differ")
        audit.update(forward_counts=forward_count, callbacks=callbacks,
            primary_return_best_equal=True, selected_live_model_verified=True,
            terminal_live_Adam_and_gradients_verified=True, selected_best_Adam_snapshot_verified=True,
            actual_default_RNG_restored=True, actual_hook_added_model_forwards=0)
        audit["terminal_artifact_checks"] = costs.call("actual_terminal_artifact_and_metric_checks",
            _verify_terminal_artifacts, reporting, reporter, witness, identity)
        # Paired selected tail from the same real boundary; no training statements.
        ambient_rng = adapter.rng_snapshot()
        try:
            tail_values = []
            for enabled in [False, True]:
                def tail_case():
                    disposable, disposable_optimizer = _clone_terminal(adapter, raw, terminal)
                    adapter.rng_restore(terminal["RNG"])
                    invocations = [0]
                    def count_model(module, args):
                        invocations[0] += 1
                    isolated_handle = disposable.register_forward_pre_hook(count_model)
                    if enabled:
                        before = _state(adapter, disposable, disposable_optimizer, graph, train, validation)
                        def mutate_copy(payload):
                            payload["member_logits"].zero_()
                            payload["served_logits"].zero_()
                            _draw_default_rngs(adapter)
                        isolated_context = {}
                        native_logits = witness["logits"].to(next(disposable.parameters()).device).clone()
                        boundaries["hook"]()(disposable, native_logits,
                            witness["value"], adapter.cpu_copy(witness["best"]), U, witness["cap"], witness["patience"],
                            witness["backbone"], mutate_copy, isolated_context)
                        require(isolated_context.get("rng_restoration_status") == "RESTORED", "Isolated mutation/draw did not restore")
                        require(_equal(before, _state(adapter, disposable, disposable_optimizer, graph, train, validation)),
                                "Isolated payload mutation/draw changed native state")
                        require(_equal(native_logits, witness["logits"]), "Payload mutation changed isolated native logits")
                        require(invocations[0] == 0, "Isolated hook added a native model forward")
                    saved = []
                    returned = boundaries["tail"]()(disposable, graph, validation,
                        lambda name, logits: saved.append((name, adapter.cpu_copy(logits))),
                        adapter.cpu_copy(witness["best"]), U, witness["cap"], witness["patience"],
                        witness["midpoint_saved"], witness["midpoint_epoch"], witness["backbone"])
                    require(_equal(adapter.named_optimizer_snapshot(disposable, disposable_optimizer), terminal["optimizer"]),
                            "Isolated selected tail changed terminal Adam")
                    require(invocations[0] == 1, "Isolated selected tail model count differs")
                    isolated_handle.remove()
                    return dict(result=returned, model=adapter.cpu_copy(disposable.state_dict()),
                        gradients={n:adapter.cpu_copy(p.grad) for n,p in disposable.named_parameters()},
                        modes={n:m.training for n,m in disposable.named_modules(remove_duplicate=False)},
                        saved=saved, RNG=adapter.rng_snapshot())
                tail_values.append(costs.call("isolated_selected_tail_"+str(enabled), tail_case))
            require(_equal({k:v for k,v in tail_values[0].items() if k != "saved"},
                           {k:v for k,v in tail_values[1].items() if k != "saved"})
                    and _equal(tail_values[0]["result"], result), "Paired selected state/RNG/return differs")
            require(tail_values[0]["saved"][0][0] == tail_values[1]["saved"][0][0] == "selected"
                    and len(tail_values[0]["saved"]) == len(tail_values[1]["saved"]) == 1,
                    "Paired selected callback differs")
            require(_replay_logits_match(tail_values[0]["saved"][0][1], tail_values[1]["saved"][0][1], adapter)
                    and _replay_logits_match(tail_values[0]["saved"][0][1], observed_selected["logits"], adapter),
                    "Actual/paired selected replay logits exceed bound native tolerance")
            audit["paired_actual_boundary_selected_tail_checks_passed"] = True
            audit["selected_replay_logit_tolerances"] = {key:adapter.TOLERANCES[key] for key in ["logits_atol","logits_rtol"]}
            audit["selected_replay_bitwise_identity_claim"] = False
        finally:
            costs.call("paired_tail_cleanup_RNG_restore", adapter.rng_restore, ambient_rng)
        audit["failure_probes"] = costs.call("all_bounded_isolated_failure_probes", _failure_probes,
            reporting, adapter, boundaries, witness, terminal, raw, graph, train, validation, identity,
            root, relative, costs)
        audit["post_probe_default_RNG_equal"] = _equal(adapter.rng_snapshot(), terminal["RNG"])
        require(audit["post_probe_default_RNG_equal"], "Disposable probes changed primary post-return RNG")
        audit["qualification_passed"] = True
        audit["coverage"] = dict(graph=identity["graph"], seed=identity["seed"], arm=identity["arm"],
            actual_stop_reason=witness["endpoint"]["stop_reason"], full_cap_reached=witness["endpoint"]["full_native_cap_reached"],
            native_noncommon_arm=root_custody.get("naturally_eligible_noncommon_arm"),
            other_backbone_or_stop_paths_not_inherited=True, full_trajectory_numerical_equivalence_claim=False,
            primary_instrumentation_and_all_probe_costs_included=True)
        audit["terminal_state_digests"] = _summarize(terminal)
        audit["selected_best_state_digests"] = _summarize(best)
        audit["terminal_logits_digests"] = dict(members=_digest(witness["logits"]), served=_digest(witness["served"]))
        audit["successful_RNG_restoration_context"] = _summarize(context)
        audit["wrapper_wall_seconds_before_final_audit_write"] = time.perf_counter()-wall_started
        audit["wrapper_process_CPU_seconds_before_final_audit_write"] = time.process_time()-cpu_started
        audit["final_audit_JSON_write_excluded_from_saved_wrapper_boundary"] = True
        audit["memory_interpretation"] = "Process-lifetime peaks; snapshots/probes included; not bare training peaks or independent component maxima"
        costs.call("final_qualification_receipt_write", reporting._json_exclusive, output/"QUALIFICATION.json", audit)
        # The final JSON cannot contain its own completed write time. Return that
        # outer measurement to the caller for its existing attempt accounting.
        accounting = dict(wall_seconds_through_final_audit_write=time.perf_counter()-wall_started,
            process_CPU_seconds_through_final_audit_write=time.process_time()-cpu_started,
            final_audit_write_cost=costs.records[-1], memory=_memory(), nested_intervals_do_not_sum=True)
        record_attempt_accounting(dict(primary_completed=True, qualification_passed=True, **accounting))
        return result
    except BaseException as error:
        audit["qualification_passed"] = False
        audit["wrapper_wall_seconds_before_failure_receipt"] = time.perf_counter()-wall_started
        audit["wrapper_process_CPU_seconds_before_failure_receipt"] = time.process_time()-cpu_started
        if owned and reporting is not None:
            reporting._failure_exclusive(output/"QUALIFICATION_FAILED.json", error, identity=identity,
                endpoint=context.get("endpoint"), terminal_context=dict(native_context=context, qualification_audit=audit),
                stage="boundary_qualification", wall_started=wall_started, cpu_started=cpu_started,
                cost_scope="all_qualification_work_before_best_effort_failure_receipt", partial_directory=output)
        elif owned:
            try:
                with (output/"QUALIFICATION_FAILED.json").open("x") as handle:
                    json.dump(dict(error_type=type(error).__name__, stage="verified_source_loading",
                        original_exception_propagated=True, qualification_audit=audit), handle, indent=2, allow_nan=False)
            except BaseException:
                pass
        try:
            record_attempt_accounting(dict(primary_completed=audit["primary_completed"], qualification_passed=False,
                wall_seconds_through_failure_receipt=time.perf_counter()-wall_started,
                process_CPU_seconds_through_failure_receipt=time.process_time()-cpu_started,
                costs=costs.records, nested_intervals_do_not_sum=True,
                cost_scope="Includes attempted failure receipts; excludes this root accounting callback"))
        except BaseException:
            pass  # Caller-owned outer accounting must cover unavailable bookkeeping.
        raise
