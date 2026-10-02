"""Runnable control fitter for a future root-admitted runtime.

Importing this module uses only the standard library. Scientific imports happen
inside fit_control AFTER source and root-admission validation. No loader,
preprocessing, final labels, hyperparameter grid, retry or backend fallback.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
from types import SimpleNamespace

from control_protocol import (CompactRole, PreparedGraph, PROTOCOL, canonical_spec,
    digest, qualification_checks, validate_admission)
import control_protocol as _protocol_module


PACKET = Path(__file__).resolve().parent


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verified_manifest():
    if Path(_protocol_module.__file__).resolve() != PACKET / "control_protocol.py":
        raise RuntimeError("Different protocol source was imported")
    manifest_path = PACKET / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = PACKET / entry["path"]
        if not path.resolve().is_relative_to(PACKET) or path.is_symlink():
            raise ValueError("Source manifest escapes the packet")
        if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise ValueError(f"Source seal mismatch: {entry['path']}")
    return _sha(manifest_path)


def _runtime():
    # Absolute imports in unchanged native/port sources need this local root.
    # Reject a previously loaded namesake; never silently use different bodies.
    if str(PACKET) not in sys.path:
        sys.path.insert(0, str(PACKET))
    for name in ("ports", "qualifications", "native_polyformer", "native_polyformer_outer",
                 "native_polynormer"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != PACKET / (name + ".py"):
            raise RuntimeError(f"Different source already imported as {name}")
        found = importlib.util.find_spec(name)
        if found is None or Path(found.origin).resolve() != PACKET / (name + ".py"):
            raise RuntimeError(f"Import path does not select sealed local {name}")
    torch = importlib.import_module("torch")
    pyg = importlib.import_module("torch_geometric")
    ports = importlib.import_module("ports")
    return torch, pyg, ports


def _runtime_binding(torch, pyg, device):
    """Root freezes this same environment fingerprint during qualification."""
    gat_source = Path(inspect.getsourcefile(pyg.nn.GATConv)).resolve()
    environment = dict(python_version=sys.version, torch_version=str(torch.__version__),
        pyg_version=str(pyg.__version__), torch_cuda_version=torch.version.cuda,
        pyg_gat_source_sha256=_sha(gat_source),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        cudnn_version=torch.backends.cudnn.version(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), device=str(device))
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment["cuda_device"] = dict(name=properties.name,
            capability=[properties.major, properties.minor], total_memory=properties.total_memory)
    binding = dict(torch_version=str(torch.__version__), pyg_version=str(pyg.__version__),
        device=str(device), environment_sha256=digest(environment))
    return binding, environment


class _Recorder:
    """Inclusive fitter costs; root must also charge preparation/outer process.

    Wall intervals synchronize the one input device. CPU max RSS is the process
    lifetime high-water mark. CUDA peaks are reset for this fit, after import.
    """
    def __init__(self, out, spec, admission, mode, manifest_sha256):
        self.out, self.torch, self.device = out, None, None
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.intervals, self.active_stage = [], None
        self.summary = dict(schema="efficient-graph-control-costs-v1", mode=mode,
            spec_sha256=digest(spec), attempt_id=admission["attempt_id"],
            packet_manifest_sha256=manifest_sha256, stages=[], updates_completed=0,
            status="running", cost_scope="inclusive fitter including construction, input binding, epoch0, selection, snapshots, transition and selected logits serialization",
            external_costs_required=["existing prepared graph/preprocessing/cache materialization",
                "outer process/startup/qualification and failed attempts"],
            gpu_timing="one input device synchronization at interval boundaries",
            cpu_peak_scope="process lifetime high-water RSS, not incremental fit memory")
        _json(out / "START.json", dict(schema="efficient-graph-control-start-v1",
            specification=spec, admission=admission, mode=mode,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    def synchronize(self):
        if self.torch is not None and self.device is not None and self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def charge(self, category):
        self.synchronize()
        start, cpu = time.perf_counter(), time.process_time()
        body_error = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            synchronization_error = None
            try:
                self.synchronize()
            except BaseException as error:
                synchronization_error = error
            self.intervals.append(dict(category=category, stage=self.active_stage,
                wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu,
                body_error=f"{type(body_error).__name__}: {body_error}" if body_error is not None else None,
                synchronization_error=f"{type(synchronization_error).__name__}: {synchronization_error}"
                    if synchronization_error is not None else None))
            if synchronization_error is not None:
                if body_error is not None:
                    raise synchronization_error from body_error
                raise synchronization_error

    def flush(self, status):
        self.summary.update(status=status, wall_seconds=time.perf_counter() - self.started,
            process_cpu_seconds=time.process_time() - self.cpu_started,
            intervals=self.intervals)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.summary["process_max_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        if self.torch is not None and self.device is not None and self.device.type == "cuda":
            try:
                self.summary.update(cuda_peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated(self.device)),
                    cuda_peak_reserved_bytes=int(self.torch.cuda.max_memory_reserved(self.device)))
            except BaseException as error:
                self.summary["cuda_peak_read_error"] = f"{type(error).__name__}: {error}"
        _json(self.out / "COSTS.json", self.summary)


def _tensor_hash(torch, tensor):
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps(dict(shape=list(tensor.shape), dtype=str(tensor.dtype)),
        sort_keys=True, separators=(",", ":")).encode()
    # Tensor.numpy is a runtime conversion only; no dataset or global label read.
    return hashlib.sha256(header + b"\0" + tensor.numpy().tobytes()).hexdigest()


def _validate_inputs(torch, ports, spec, graph, train, validation, admission):
    if type(graph) is not PreparedGraph or type(train) is not CompactRole or type(validation) is not CompactRole:
        raise TypeError("Use PreparedGraph and exactly two CompactRole packs")
    inputs, edges = ports.graph_inputs(graph, spec["backbone"])
    context = spec["context"]
    expected_shape = ((context["nodes"], 13, context["features"])
        if spec["backbone"] == "polyformer_mono" else (context["nodes"], context["features"]))
    if tuple(inputs.shape) != expected_shape or inputs.dtype != torch.float32:
        raise ValueError("Prepared graph must have the retained full-context FP32 shape")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("Nonfinite prepared inputs")
    if edges is not None:
        if edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2 or edges.device != inputs.device:
            raise ValueError("Photo edges must be coherent int64 [2,E] on the input device")
        if not bool(((edges >= 0) & (edges < context["nodes"])).all()):
            raise ValueError("Photo edge identity out of range")
    for role in (train, validation):
        ports.check_compact_role(role, context["nodes"])
        if role.nodes.device != inputs.device or role.labels.device != inputs.device:
            raise ValueError("Compact roles and prepared inputs must use one execution device")
        if len(role.nodes.unique()) != len(role.nodes):
            raise ValueError("A compact role contains repeated node identities")
        if not bool(((role.labels >= 0) & (role.labels < context["classes"])).all()):
            raise ValueError("Compact role label out of class range")
    if bool(torch.isin(train.nodes, validation.nodes).any()):
        raise ValueError("TRAIN and validation roles overlap")
    bindings = dict(prepared_graph_sha256=digest(dict(backbone=graph.teacher_backbone,
        inputs=_tensor_hash(torch, inputs), edges=_tensor_hash(torch, edges) if edges is not None else None)),
        train_pack_sha256=digest(dict(nodes=_tensor_hash(torch, train.nodes), labels=_tensor_hash(torch, train.labels))),
        validation_pack_sha256=digest(dict(nodes=_tensor_hash(torch, validation.nodes), labels=_tensor_hash(torch, validation.labels))))
    if any(admission[key] != value for key, value in bindings.items()):
        raise ValueError("Caller inputs differ from the root-frozen graph/compact role bindings")
    return bindings


def _new_model(torch, ports, spec):
    if spec["arm"] == "fixed_cap_packed_m4":
        return ports.IndependentMembers.build(spec["backbone"], spec["seed"], backend=spec["backend"])
    if spec["arm"] == "cached_token_mimo_m4":
        return ports.PolyFormerMIMO(spec["seed"])

    class NativeFamily(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.members, self.global_stage = spec["backbone"], spec["members"], False
            self.models = torch.nn.ModuleList([ports.NativeInput(self.backbone,
                ports.new_native(self.backbone, spec["width"], member_seed))
                for member_seed in spec["member_seeds"]])
            self.specification = copy.deepcopy(spec)

        def forward(self, supplied_graph):
            inputs, edges = ports.graph_inputs(supplied_graph, self.backbone)
            return torch.stack([member(inputs, edges) for member in self.models])

        def set_global_stage(self, enabled):
            if self.backbone != "polynormer_r" and enabled:
                raise ValueError("PolyFormer has no global stage")
            self.global_stage = bool(enabled)
            if self.backbone == "polynormer_r":
                for member in self.models:
                    member.core._global = self.global_stage

    return NativeFamily()


def _cpu_copy(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(torch, v) for v in value)
    return copy.deepcopy(value)


def _save(torch, path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _set_stage(model, spec, stage):
    if hasattr(model, "set_global_stage"):
        model.set_global_stage(stage == "global")
    elif spec["backbone"] == "polynormer_r":
        raise ValueError("Photo control lacks explicit local/global stage propagation")


def _validate_logits(torch, spec, logits):
    expected = (spec["members"], spec["context"]["nodes"], spec["context"]["classes"])
    if tuple(logits.shape) != expected or not bool(torch.isfinite(logits).all()):
        raise ValueError("Full coherent graph member logits have wrong shape or nonfinite values")


def _selected_restore_check(torch, ports, model, optimizer, selected, graph, validation, spec):
    model.load_state_dict(selected["model_state"], strict=True)
    optimizer.load_state_dict(selected["adam_state"])
    _assert_state_equal(torch, model.state_dict(), selected["model_state"])
    _assert_state_equal(torch, optimizer.state_dict(), selected["adam_state"])
    _set_stage(model, spec, selected["stage"])
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        _validate_logits(torch, spec, logits)
        nll = float(ports.validation_nll(logits, validation, reducer="common_logit"))
    if not math.isfinite(nll) or not math.isclose(nll, selected["validation_nll"], rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("Selected state does not restore its primary validation NLL")
    return logits, nll


def _assert_state_equal(torch, actual, expected):
    """Exact model and Adam tensor/metadata handoff, including inactive slots."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.dtype != expected.dtype or not torch.equal(
                actual.detach().cpu(), expected.detach().cpu()):
            raise RuntimeError("Selected model/Adam tensor state was not restored exactly")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError("Selected state mapping differs")
        for key in expected:
            _assert_state_equal(torch, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise RuntimeError("Selected state sequence differs")
        for a, e in zip(actual, expected):
            _assert_state_equal(torch, a, e)
    elif actual != expected:
        raise RuntimeError("Selected Adam metadata differs")


def fit_control(spec, graph, train, validation, *, admission, mode, out):
    """Fit ONE fixed cell; API consumes no labels outside TRAIN/validation.

    `out` must be a fresh directory outside this sealed source packet. The root
    supplies authentic frozen admission/qualification assertions and supervises
    process termination, outer cap, failures/retries and final report admission.
    This function never creates or changes a registry or a scientific decision.
    """
    spec = canonical_spec(spec)
    manifest_sha256 = _verified_manifest()
    admission = validate_admission(spec, admission, mode=mode, manifest_sha256=manifest_sha256)
    out = Path(out).resolve()
    if out.is_relative_to(PACKET):
        raise ValueError("Fit outputs cannot mutate the sealed source packet")
    out.mkdir(parents=True, exist_ok=False)
    recorder = _Recorder(out, spec, admission, mode, manifest_sha256)
    updates, selected, stage_summaries = 0, None, []
    trace = None
    try:
        with recorder.charge("scientific_import"):
            torch, pyg, ports = _runtime()
        recorder.torch, recorder.device = torch, graph.teacher_input.device
        actual_runtime, environment = _runtime_binding(torch, pyg, recorder.device)
        if actual_runtime != admission["runtime_binding"]:
            raise RuntimeError("Installed scientific runtime/device differs from root qualification binding")
        _json(out / "RUNTIME_BINDING.json", dict(binding=actual_runtime, environment=environment))
        if recorder.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(recorder.device)
        with recorder.charge("construction_and_complete_count"):
            model = _new_model(torch, ports, spec).to(recorder.device)
            actual_count = sum(p.numel() for p in model.parameters())
            expected_count = (ports.mimo_parameter_count(spec["width"])
                if spec["arm"] == "cached_token_mimo_m4" else
                spec["members"] * ports.native_parameter_count(spec["backbone"], spec["width"]))
            if actual_count != expected_count:
                raise RuntimeError("Constructed complete native module count differs from source algebra")
            budget = dict(actual_parameters=actual_count, expected_parameters=expected_count)
            if spec["arm"] in ("fixed_cap_packed_m4", "cached_token_mimo_m4"):
                qualification_module = importlib.import_module("qualifications")
                budget.update(qualification_module.check_parameter_budget(model))
            optimizer = ports.optimizer_for(model)
            random.seed(spec["seed"] + 70000)
            torch.random.default_generator.manual_seed(spec["seed"] + 70000)
            if recorder.device.type == "cuda":
                with torch.cuda.device(recorder.device):
                    torch.cuda.manual_seed(spec["seed"] + 70000)
            tuple_generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + 90000)
        with recorder.charge("prepared_input_and_compact_role_binding"):
            bindings = _validate_inputs(torch, ports, spec, graph, train, validation, admission)
            _json(out / "INPUT_BINDINGS.json", bindings)
        context = spec["context"]
        stages = ([('local', 2 if mode == "qualify" else context["local_epochs"]),
                   ('global', 2 if mode == "qualify" else context["global_epochs"])]
            if spec["backbone"] == "polynormer_r" else
            [('native', 3 if mode == "qualify" else context["max_epochs"])])
        trace = (out / "TRACE.jsonl").open("x")
        for stage, cap in stages:
            recorder.active_stage = stage
            stage_summary = dict(stage=stage, update_cap=cap, updates_completed=0,
                selected_epoch=None, selected_validation_nll=None, status="running")
            recorder.summary["stages"].append(stage_summary)
            with recorder.charge("stage_transition"):
                if stage == "global":
                    if selected is None or selected["stage"] != "local":
                        raise RuntimeError("No selected local state for Photo handoff")
                    # Both states are restored before activating the global stage.
                    _selected_restore_check(torch, ports, model, optimizer, selected, graph, validation, spec)
                _set_stage(model, spec, stage)
            # A fresh selector is mandatory for Photo global-only final selection.
            best_nll, best_epoch, selected, completed = float("inf"), None, None, 0
            for epoch in range(cap + 1):
                loss_value, nonzero_gradients, tuple_binding = None, None, None
                if epoch > 0:
                    with recorder.charge("train_update"):
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        if spec["arm"] == "cached_token_mimo_m4":
                            positions = ports.independent_tuple_positions(len(train.nodes),
                                generator=tuple_generator, device="cpu").to(recorder.device)
                            # Exactly one full exposure per compact TRAIN position in each slot.
                            expected = torch.arange(len(train.nodes), device=recorder.device)
                            if positions.shape != (len(train.nodes), 4) or any(
                                    not torch.equal(positions[:, slot].sort().values, expected) for slot in range(4)):
                                raise RuntimeError("MIMO update does not expose every TRAIN position once per slot")
                            tuples = ports.make_mimo_tuples(graph, train, positions)
                            # The unchanged hook verifies complete token rows and matched compact labels.
                            importlib.import_module("qualifications").check_mimo_correspondence(
                                graph, train, positions, tuples)
                            loss = ports.mimo_loss(model.forward_tuples(tuples), tuples)
                            tuple_binding = dict(positions_sha256=_tensor_hash(torch, positions),
                                tuple_rows=len(positions), exposures_per_slot=len(train.nodes),
                                slots=4, repetition=0, loss_reduction="mean tuples, sum matching heads")
                        else:
                            logits = model(graph)
                            _validate_logits(torch, spec, logits)
                            loss = ports.member_loss(logits, train)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite compact TRAIN loss")
                        loss.backward()
                        nonzero_gradients = 0
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                if not bool(torch.isfinite(parameter.grad).all()):
                                    raise RuntimeError("Nonfinite training derivative")
                                nonzero_gradients += int(bool((parameter.grad != 0).any()))
                        if nonzero_gradients == 0:
                            raise RuntimeError("No nonzero training derivative")
                        optimizer.step()
                        loss_value = float(loss.detach())
                        updates += 1
                        completed = epoch
                        recorder.summary["updates_completed"] = updates
                        stage_summary["updates_completed"] = completed
                with recorder.charge("primary_validation"):
                    model.eval()
                    with torch.no_grad():
                        logits = model(graph)
                        _validate_logits(torch, spec, logits)
                        value = float(ports.validation_nll(logits, validation, reducer="common_logit"))
                    if not math.isfinite(value):
                        raise RuntimeError("Nonfinite primary validation NLL")
                improved = value < best_nll
                if improved:
                    with recorder.charge("selected_state_snapshot_and_serialization"):
                        best_nll, best_epoch = value, epoch
                        selected = dict(schema="efficient-graph-control-selected-state-v1",
                            specification=spec, input_bindings=bindings, mode=mode,
                            stage=stage, global_stage=stage == "global", stage_epoch=epoch,
                            actual_updates_completed=updates, validation_nll=value,
                            model_state=_cpu_copy(torch, model.state_dict()),
                            adam_state=_cpu_copy(torch, optimizer.state_dict()),
                            torch_rng_state=torch.get_rng_state().clone(),
                            tuple_rng_state=tuple_generator.get_state().clone(),
                            python_rng_state=random.getstate(),
                            cuda_rng_state=torch.cuda.get_rng_state(recorder.device).cpu()
                                if recorder.device.type == "cuda" else None,
                            rng_policy="saved for custody; local-to-global handoff retains continuing training RNG",
                            epoch_zero_status=PROTOCOL["epoch_zero_status"])
                        _save(torch, out / f"SELECTED_{stage}.pt", selected)
                        stage_summary.update(selected_epoch=epoch, selected_validation_nll=value)
                trace.write(json.dumps(dict(stage=stage, stage_epoch=epoch,
                    actual_updates_completed=updates, train_loss=loss_value,
                    nonzero_gradient_tensors=nonzero_gradients, validation_nll=value,
                    selected_strict_improvement=improved, tuple_binding=tuple_binding), allow_nan=False) + "\n")
                trace.flush()
                # Qualification uses its short cap; full Squirrel keeps native patience.
                if stage == "native" and mode == "full" and epoch - best_epoch >= context["patience"]:
                    stage_summary["stop_reason"] = "patience250"
                    break
            if selected is None:
                raise RuntimeError("No finite selected checkpoint")
            stage_summary.update(status="success", updates_completed=completed,
                selected_epoch=best_epoch, selected_validation_nll=best_nll)
            stage_summaries.append(copy.deepcopy(stage_summary))
            recorder.flush("running")
        if spec["backbone"] == "polynormer_r" and selected["stage"] != "global":
            raise RuntimeError("Final Photo control must come from the fresh global selector")
        recorder.active_stage = "finalization"
        with recorder.charge("selected_state_restore_and_logits"):
            logits, restored_nll = _selected_restore_check(torch, ports, model, optimizer,
                selected, graph, validation, spec)
            logits = logits.detach().cpu().clone()
        with recorder.charge("final_artifact_serialization"):
            _save(torch, out / "SELECTED_CHECKPOINT.pt", selected)
            _save(torch, out / "SELECTED_MEMBER_LOGITS.pt", dict(
                schema="efficient-graph-control-member-logits-v1", specification=spec,
                input_bindings=bindings, logits=logits,
                nodes=torch.arange(context["nodes"], dtype=torch.int64),
                logits_shape=list(logits.shape), selected_stage=selected["stage"],
                selected_epoch=selected["stage_epoch"], label_scope=["train", "validation"],
                final_labels_present=False, pooling="common_logit"))
            selection = dict(schema="efficient-graph-control-selection-v1", specification=spec,
                input_bindings=bindings, stages=stage_summaries,
                selected_stage=selected["stage"], selected_epoch=selected["stage_epoch"],
                primary_validation_nll=selected["validation_nll"], restored_validation_nll=restored_nll,
                updates_completed=updates, budget=budget,
                primary_pooling=PROTOCOL["primary_pooling"], secondary_selected=False,
                secondary_pooling=PROTOCOL["secondary_pooling"],
                published_probability_pooling_adaptation=spec["members"] > 1,
                epoch_zero_status=PROTOCOL["epoch_zero_status"],
                train_reduction=PROTOCOL["mimo_training_loss"] if spec["arm"] == "cached_token_mimo_m4"
                    else PROTOCOL["member_training_loss"],
                label_scope=["train", "validation"], mode=mode,
                qualification_only=mode == "qualify", report_eligible=False,
                report_status="requires separate root cohort closure and report admission",
                runtime_binding=actual_runtime,
                executed_backend=spec["backend"],
                efficient_backend_claim="not established by this fit",
                qualification_required_checks=qualification_checks(spec),
                root_qualification=admission.get("root_qualification"))
            _json(out / "SELECTION.json", selection)
        trace.close()
        trace = None
        recorder.flush("success")
        files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": _sha(p)}
            for p in sorted(out.iterdir()) if p.is_file() and p.name != "TERMINAL.json"]
        _json(out / "TERMINAL.json", dict(schema="efficient-graph-control-terminal-v1",
            status="success", attempt_id=admission["attempt_id"], spec_sha256=digest(spec),
            start_sha256=_sha(out / "START.json"), files=files,
            report_eligible=False, qualification_pass_issued=False,
            completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        return selection
    except BaseException as error:
        if trace is not None:
            trace.close()
        if recorder.summary["stages"] and recorder.summary["stages"][-1]["status"] == "running":
            recorder.summary["stages"][-1]["status"] = "failed"
        failure = dict(schema="efficient-graph-control-failure-v1", status="failed",
            error_type=type(error).__name__, error=str(error), active_stage=recorder.active_stage,
            updates_completed=updates, spec_sha256=digest(spec),
            attempt_id=admission["attempt_id"], automatic_retry=False,
            automatic_backend_fallback=False, last_selected_stage=selected["stage"] if selected else None,
            last_selected_epoch=selected["stage_epoch"] if selected else None)
        _json(out / "FAILURE.json", failure)
        recorder.flush("failed")
        _json(out / "TERMINAL.json", dict(failure,
            schema="efficient-graph-control-terminal-v1", start_sha256=_sha(out / "START.json"),
            failure_sha256=_sha(out / "FAILURE.json"), costs_sha256=_sha(out / "COSTS.json"),
            report_eligible=False, qualification_pass_issued=False))
        raise
