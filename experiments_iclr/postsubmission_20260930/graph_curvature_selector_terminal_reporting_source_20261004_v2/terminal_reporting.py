"""Observation-only continuation reporting; source preparation is not execution.

No numerical package is imported at module import. No data loader, training CLI,
selector, schedule, checkpoint rule, or server launcher is introduced.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ARMS = ("common_only", "fixed_first_graph_pair", "selected_graph_pair",
        "selected_permuted_span", "selected_random_span")
SEEDS = (17, 29, 43)
BACKBONES = {"Squirrel": "polyformer_mono", "Photo": "polynormer_r"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _verified_bytes(record):
    relative = Path(record["path"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "A phase-relative source descriptor is required")
    path = (PHASE / relative).resolve()
    require(path.is_relative_to(PHASE.resolve()), "Source leaves the research phase")
    data = path.read_bytes()
    require(len(data) == record["bytes"]
            and hashlib.sha256(data).hexdigest() == record["sha256"],
            "Bound source bytes differ: " + str(path))
    return path, data


def load_reporting_adapter():
    """Compile the bound separate copy; never replace the canonical module."""
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    _verified_bytes(bindings["original_integration"])
    for record in bindings["frozen_source_records"]:
        _verified_bytes(record)
    path, data = _verified_bytes(bindings["reporting_integration"])
    name = "graph_init_training_adapter_terminal_reporting_v2"
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, "__executed_sha256__", None)
                == bindings["reporting_integration"]["sha256"]
                and getattr(module, "__executed_path__", None) == str(path),
                "Cached reporting adapter lacks exact executed-source custody")
        return module
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(name, path))
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
        module.__executed_sha256__ = bindings["reporting_integration"]["sha256"]
        module.__executed_path__ = str(path)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _json_exclusive(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _failure_exclusive(path, error, *, identity, endpoint, terminal_context,
        stage, wall_started, cpu_started, cost_scope, partial_directory=None,
        source_bindings_verified=False):
    """Entire receipt construction/write is best effort; never replace error."""
    try:
        wall_before_receipt = time.perf_counter()-wall_started
        cpu_before_receipt = time.process_time()-cpu_started
        try:
            identity = json.loads(json.dumps(identity, allow_nan=False))
            identity_status = "AVAILABLE"
        except BaseException:
            identity, identity_status = None, "UNAVAILABLE_JSON_IDENTITY"
        try:
            endpoint = json.loads(json.dumps(endpoint, allow_nan=False))
            endpoint_status = "AVAILABLE" if endpoint is not None else "NOT_CAPTURED"
        except BaseException:
            endpoint, endpoint_status = None, "UNAVAILABLE_JSON_ENDPOINT"
        endpoint_fields = ("teacher_backbone", "members", "continuation_updates_completed",
            "terminal_actual_update", "terminal_native_stage_epoch", "stage", "update_cap",
            "patience", "full_native_cap_reached", "stop_reason", "source_native_validation_nll",
            "terminal_before_selected_checkpoint_restore", "selected_continuation_epoch",
            "selected_actual_update", "capture_wall_seconds_before_observer",
            "capture_process_cpu_seconds_before_observer")
        available_endpoint_fields = sorted(endpoint) if isinstance(endpoint, dict) else []
        unknown_endpoint_fields = [name for name in endpoint_fields if name not in available_endpoint_fields]
        if endpoint_status == "AVAILABLE" and unknown_endpoint_fields:
            endpoint_status = "PARTIAL_METADATA_AVAILABLE"
        endpoint_values = endpoint if isinstance(endpoint, dict) else {}
        partial_files, inventory_status = None, "DIRECTORY_NOT_OWNED_OR_NOT_CREATED"
        if partial_directory is not None:
            try:
                partial_files = sorted(item.name for item in partial_directory.iterdir())
                inventory_status = "AVAILABLE"
            except BaseException:
                inventory_status = "UNAVAILABLE_DIRECTORY_INVENTORY"
        try:
            source_bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
            source_bindings_status = "AVAILABLE"
        except BaseException:
            source_bindings, source_bindings_status = None, "UNAVAILABLE_SOURCE_BINDINGS"
        try:
            error_text = str(error)
        except BaseException:
            error_text = "UNAVAILABLE_EXCEPTION_STRING"
        failure = dict(schema="graph-curvature-terminal-report-failure-v2",
            identity=identity, identity_status=identity_status,
            error_type=type(error).__name__, error=error_text, failure_stage=stage,
            endpoint=endpoint, endpoint_status=endpoint_status,
            available_endpoint_fields=available_endpoint_fields,
            unknown_endpoint_fields=unknown_endpoint_fields,
            capture_wall_seconds_before_observer=endpoint_values.get("capture_wall_seconds_before_observer"),
            capture_process_cpu_seconds_before_observer=endpoint_values.get("capture_process_cpu_seconds_before_observer"),
            terminal_context=terminal_context,
            partial_files_preserved=partial_files, partial_file_inventory_status=inventory_status,
            source_bindings=source_bindings, source_bindings_status=source_bindings_status,
            source_bindings_verified=source_bindings_verified,
            attempt_wall_seconds_before_failure_receipt=wall_before_receipt,
            attempt_process_cpu_seconds_before_failure_receipt=cpu_before_receipt,
            cost_scope=cost_scope, failure_receipt_construction_and_write_excluded=True,
            cost_intervals_overlap_do_not_sum=True,
            retry_authorized=False)
        _json_exclusive(path, failure)
    except BaseException:
        pass  # Preserve the original exception even if inventory or receipt IO fails.


def _tensor_digest(tensor):
    data = tensor.contiguous().numpy().tobytes(order="C")
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                sha256_raw_contiguous_bytes=hashlib.sha256(data).hexdigest())


def _classification_metrics(logits, labels):
    """CPU float64 reporting from captured logits; never used by selection."""
    import torch
    import torch.nn.functional as F
    z = logits.to(dtype=torch.float64)
    log_probabilities = F.log_softmax(z, dim=-1)
    probabilities = log_probabilities.exp()
    truth = F.one_hot(labels, num_classes=z.shape[-1]).to(dtype=torch.float64)
    nll = -log_probabilities.gather(1, labels[:, None]).mean()
    brier = (probabilities - truth).square().sum(-1).mean()
    correct = z.argmax(-1).eq(labels)
    return dict(accuracy_fraction=float(correct.to(torch.float64).mean()),
                correct_nodes=int(correct.sum()), nodes=int(labels.numel()),
                nll_nats=float(nll), brier_sum_over_classes=float(brier))


class TerminalReporter:
    """One exclusive artifact for one frozen arm/block, using VALIDATION only.

    The caller owns verified acquisition and role/label provenance. This observer
    receives no model, optimizer, graph, TRAIN labels, or heldout labels. It does
    not certify the caller's data descriptors merely by copying them.
    """

    def __init__(self, report_root, arm_relative_directory, identity, validation):
        import torch
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        graph = self.identity.get("graph")
        seed = self.identity.get("seed")
        require(graph in BACKBONES and seed in SEEDS
                and self.identity.get("source_split_index") == SEEDS.index(seed)
                and self.identity.get("configuration") == 0
                and self.identity.get("arm") in ARMS,
                "Exact frozen graph/seed/split/cfg0/arm identity is required")
        require(isinstance(self.identity.get("validation_provenance"), dict)
                and bool(self.identity["validation_provenance"]),
                "Caller must bind the verified VALIDATION role and compact-label descriptors")
        root = Path(report_root).resolve()
        relative = Path(arm_relative_directory)
        require(root.is_relative_to(PHASE.resolve()) and root != PHASE.resolve(),
                "Reporting root must be inside the project research phase")
        require(not relative.is_absolute() and relative.parts
                and ".." not in relative.parts,
                "A nonempty arm-relative output directory is required")
        self.output = (root / relative / "terminal_reporting").resolve()
        require(self.output.is_relative_to(root), "Reporting output leaves its run root")
        self.nodes = validation.nodes.detach().cpu().clone()
        self.labels = validation.labels.detach().cpu().clone()
        require(self.nodes.dtype == self.labels.dtype == torch.int64
                and self.nodes.ndim == self.labels.ndim == 1
                and self.nodes.shape == self.labels.shape and self.nodes.numel() > 0
                and torch.unique(self.nodes).numel() == self.nodes.numel(),
                "Nonempty paired unique int64 VALIDATION nodes/labels are required")
        self.called = False
        self.completed = False
        self.output_created = False
        self.endpoint = None
        self.failure_stage = None
        self.failure_body_cost = None

    def __call__(self, payload):
        import torch
        require(not self.called, "A terminal observer cannot be called twice or retried")
        self.called = True
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        try:
            self.failure_stage = "endpoint_metadata"
            endpoint = json.loads(json.dumps(payload["metadata"], allow_nan=False))
            self.endpoint = endpoint
            self.failure_stage = "terminal_directory_creation"
            self.output.mkdir(parents=True, exist_ok=False)
            self.output_created = True
            self.failure_stage = "payload_validation"
            member, served = payload["member_logits"], payload["served_logits"]
            require(endpoint["teacher_backbone"] == BACKBONES[self.identity["graph"]]
                    and endpoint["members"] == 4,
                    "Captured endpoint backbone/member count differs")
            require(member.device.type == served.device.type == "cpu"
                    and member.dtype == served.dtype == torch.float32
                    and not member.requires_grad and not served.requires_grad
                    and member.ndim == 3 and member.shape[0] == 4
                    and served.shape == member.shape[1:],
                    "Captured native FP32 member/served logit shapes differ")
            require(bool(torch.isfinite(member).all()) and bool(torch.isfinite(served).all())
                    and bool((self.nodes >= 0).all())
                    and bool((self.nodes < served.shape[0]).all())
                    and bool((self.labels >= 0).all())
                    and bool((self.labels < served.shape[1]).all()),
                    "Captured logits or declared VALIDATION support are invalid")
            require(math.isfinite(endpoint["source_native_validation_nll"]),
                    "Native terminal VALIDATION NLL is nonfinite")
            self.failure_stage = "reporting_metrics"
            with torch.no_grad():
                pooled = _classification_metrics(served[self.nodes], self.labels)
                competence = [dict(member=index,
                    **_classification_metrics(z[self.nodes], self.labels))
                    for index, z in enumerate(member)]
            tensors = dict(schema="graph-curvature-terminal-logits-v1",
                           member_raw_logits=member, served_mean_raw_logits=served,
                           validation_nodes=self.nodes, validation_labels=self.labels)
            tensor_path = self.output / "terminal_logits.pt"
            self.failure_stage = "terminal_tensor_write"
            with tensor_path.open("xb") as handle:
                torch.save(tensors, handle)
            self.failure_stage = "terminal_tensor_hash_read"
            tensor_bytes = tensor_path.read_bytes()
            self.failure_stage = "source_bindings_read"
            source_bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
            self.failure_stage = "report_construction"
            report = dict(schema="graph-curvature-terminal-report-v1",
                identity=self.identity, endpoint=endpoint,
                served_validation=pooled, member_validation_competence=competence,
                native_nll_minus_cpu_float64_nll=(
                    endpoint["source_native_validation_nll"] - pooled["nll_nats"]),
                reporting_arithmetic="CPU float64 log_softmax/exp from captured FP32 logits",
                served_pool="native-device arithmetic mean of raw member logits, then softmax",
                brier_convention="sum class-wise squared probability error, then mean over nodes",
                source_native_nll_is_checkpoint_metric=True,
                cpu_reporting_metrics_do_not_select_checkpoint_or_change_screen=True,
                input_tensor_identities=dict(nodes=_tensor_digest(self.nodes),
                                            labels=_tensor_digest(self.labels)),
                tensors=dict(path=tensor_path.name, bytes=len(tensor_bytes),
                    sha256=hashlib.sha256(tensor_bytes).hexdigest(),
                    member_shape=list(member.shape), served_shape=list(served.shape)),
                source_bindings=source_bindings,
                reporting_body_wall_seconds=time.perf_counter()-wall_started,
                reporting_body_process_cpu_seconds=time.process_time()-cpu_started,
                reporting_body_cost_excludes_hook_capture_and_RNG_restoration=True,
                reporting_body_cost_excludes_final_report_json_write=True,
                heldout_labels_accessed=False, predictive_gain_claim=False)
            self.failure_stage = "terminal_report_json_write"
            _json_exclusive(self.output / "terminal_report.json", report)
            self.completed = True
            self.failure_stage = None
        except BaseException as error:
            try:
                self.failure_body_cost = dict(wall_seconds=time.perf_counter()-wall_started,
                    process_cpu_seconds=time.process_time()-cpu_started)
                if self.output_created:
                    _failure_exclusive(self.output / "REPORT_FAILED.json", error,
                        identity=self.identity, endpoint=self.endpoint, terminal_context=None,
                        stage=self.failure_stage, wall_started=wall_started, cpu_started=cpu_started,
                        cost_scope="observer_body_until_failure_including_attempted_final_JSON_write",
                        partial_directory=self.output)
            except BaseException:
                pass
            raise


def continue_one_arm_with_terminal_reporting(initialized, graph, train, validation,
        trace, save_logits, *, report_root, arm_relative_directory, identity):
    """Replace only the caller's continuation invocation; leave selector intact.

    Returns the original continuation's (best, metadata). Existing midpoint and
    selected save_logits calls remain unchanged. Failures are not suppressed.
    """
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    stage, attempt_owned, bindings_verified = "run_path_validation", False, False
    reporter, failure_path, terminal_context = None, None, {}
    try:
        root, relative = Path(report_root).resolve(), Path(arm_relative_directory)
        require(root.is_relative_to(PHASE.resolve()) and root != PHASE.resolve(),
                "Reporting root must be inside the project research phase")
        require(not relative.is_absolute() and relative.parts and ".." not in relative.parts,
                "A nonempty arm-relative output directory is required")
        key = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()
        attempt_path = root / ("terminal_reporting_attempt_" + key + ".json")
        failure_path = root / ("terminal_reporting_attempt_" + key + "_FAILED.json")
        terminal_context.update(arm_relative_directory=str(relative),
            attempt_marker=attempt_path.name, run_failure_record=failure_path.name)
        stage = "attempt_root_creation"
        root.mkdir(parents=True, exist_ok=True)
        stage = "exclusive_attempt_marker"
        with attempt_path.open("x", encoding="utf-8") as handle:
            attempt_owned = True
            json.dump(dict(schema="graph-curvature-terminal-reporting-attempt-v2",
                arm_relative_directory=str(relative), retry_authorized=False,
                stage="STARTED_BEFORE_CONSTRUCTOR_OR_BOUND_SOURCE_LOADING"), handle,
                indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        stage = "reporter_constructor"
        reporter = TerminalReporter(report_root, arm_relative_directory, identity, validation)
        stage = "bound_source_loading"
        adapter = load_reporting_adapter()
        bindings_verified = True
        stage = "initial_continuation_rng_restore"
        adapter.rng_restore(initialized["rng"])
        stage = "native_continuation"
        return adapter.continuation(initialized["model"], initialized["optimizer"],
            graph, train, validation, trace, save_logits, terminal_observer=reporter,
            terminal_context=terminal_context)
    except BaseException as error:
        if attempt_owned:
            try:
                terminal_context["reporter_failure_stage"] = reporter.failure_stage if reporter else None
                terminal_context["reporter_failure_body_cost"] = reporter.failure_body_cost if reporter else None
                terminal_context["reporter_completed"] = reporter.completed if reporter else False
                _failure_exclusive(failure_path, error, identity=identity,
                    endpoint=terminal_context.get("endpoint"), terminal_context=terminal_context,
                    stage=stage, wall_started=wall_started, cpu_started=cpu_started,
                    cost_scope="wrapper_constructor_loading_rng_restore_continuation_and_failure_unwinding_before_run_failure_receipt",
                    partial_directory=reporter.output if reporter and reporter.output_created else None,
                    source_bindings_verified=bindings_verified)
            except BaseException:
                pass
        raise
