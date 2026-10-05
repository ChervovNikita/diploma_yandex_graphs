"""Disabled fixed W-only acquisition and six serial G0 H16 continuations.

No A label input, A scoring, stock loader, old checkpoint or selector. The original
operator and native port stay disabled and byte-identical. Only a future root
fit admission may call their pure callback/pair helpers and the public sequential
episode through separately reviewed flag-only successors. Actual all-six full
FP32 qualification is required before W access. This supplies no authorization.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import resource
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

SOURCE_RELEASED = False
ARMS = ("live", "uniform", "margins", "graph_free", "permuted", "stop_q")
CONTROLS = {arm: ("live" if arm == "permuted" else arm) for arm in ARMS}
HORIZON = 16
PACKET = Path(__file__).resolve().parent


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o444)


def _descriptor(path):
    path = Path(path)
    return {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}


def _source_path(project_root, row):
    root = Path(project_root).resolve()
    relative = Path(row["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "Source path escape")
    path = root / relative
    _require(not path.is_symlink() and path.resolve().is_relative_to(root)
             and _sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
             "Pinned source differs: " + row["path"])
    return path


def _module(project_root, bindings, key):
    path = _source_path(project_root, bindings["files"][key])
    name = "_amazon_G0_WSR_sequential_v1_" + key
    _require(name not in sys.modules, "Require a fresh process for this source binding")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    _require(value is None or type(value) in (bool, str, int, float), "Unsafe state leaf")
    return value


def _rng(device):
    import numpy as np
    import torch
    value = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"name": value[0], "keys": value[1].tolist(), "position": int(value[2]),
                      "has_gauss": int(value[3]), "cached_gaussian": float(value[4])},
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(torch.device(device)).clone()
                    if device.startswith("cuda:") else None}


def _fresh_family(native, boundary, device):
    """Exact V6 seed/constructor/to/reset/wrap order; no post-wrap reset."""
    import numpy as np
    import torch
    _require(torch.get_default_dtype() == torch.float32
             and str(torch.get_default_device()) == "cpu", "Native construction defaults differ")
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    if device.startswith("cuda:"):
        torch.cuda.manual_seed_all(17)
    construction = {"seed": 17, "after_seed": _rng(device)}
    model = native.Polynormer(300, 256, 5, local_layers=10, global_layers=1,
                             in_dropout=0.2, dropout=0.3, global_dropout=0.3,
                             heads=2, beta=-1, pre_ln=False)
    construction["after_constructor_cpu"] = _rng(device)
    model.to(device)
    model.reset_parameters()
    model._global = False
    construction["after_device_reset"] = _rng(device)
    family = boundary.PolynormerBoundaryFamily(model, members=4)
    construction["after_optional_factor_wrap"] = _rng(device)
    return family, construction


def _inactive(name, global_stage):
    return (name.startswith("local_head.") if global_stage else
            name.startswith(("core.global_attn.", "core.ln.", "global_head.")))


def _finite(value, message):
    import torch
    _require(bool(torch.isfinite(value).all()), message)


def _finite_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        _finite(value, "Nonfinite G0 diagnostic")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_tree(item)


def _warm_step(family, optimizer, data, audit):
    import torch
    import torch.nn.functional as F
    family.train()
    optimizer.zero_grad(set_to_none=True)
    logits = family(data["features"], data["edge_index"])
    _require(logits.dtype == torch.float32 and logits.shape == (4, 24492, 5),
             "Complete native FP32 own-member forward required")
    _finite(logits, "Nonfinite W acquisition logits")
    loss = torch.stack([F.nll_loss(F.log_softmax(logits[m], dim=1).index_select(0, data["W_ids"]),
                                   data["W_labels"]) for m in range(4)]).mean()
    _finite(loss, "Nonfinite W own-CE acquisition loss")
    audit.acquisition_backward_attempts += 1
    audit.event("ordinary_acquisition_backward_attempt")
    loss.backward()
    for name, parameter in family.named_parameters():
        if _inactive(name, bool(family.core._global)):
            _require(parameter.grad is None, "Inactive acquisition gradient: " + name)
        else:
            _require(parameter.grad is not None, "Disconnected acquisition parameter: " + name)
            _finite(parameter.grad, "Nonfinite acquisition gradient: " + name)
    audit.acquisition_optimizer_attempts += 1
    audit.event("ordinary_acquisition_optimizer_attempt")
    optimizer.step()
    for parameter in family.parameters():
        _finite(parameter, "Nonfinite acquisition state")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                _finite(value, "Nonfinite acquisition Adam state")
    return float(loss.detach().cpu())


def _save_state(path, value):
    import torch
    with Path(path).open("xb") as stream:
        torch.save(value, stream)
    Path(path).chmod(0o444)
    return _descriptor(path)


def _permutation(data):
    import torch
    node_ids = data["inner_indices"].cpu().tolist()
    labels = data["inner_labels"].cpu().tolist()
    result = list(range(len(node_ids)))
    for cls in range(5):
        increasing = sorted((i for i, label in enumerate(labels) if label == cls),
                            key=lambda i: node_ids[i])
        hashed = sorted(increasing, key=lambda i: (
            hashlib.sha256(f"amazon-response-G0|split=0|seed=17|perm|{cls}|{node_ids[i]}".
                           encode("utf-8")).digest(), node_ids[i]))
        for destination, source in zip(increasing, hashed):
            result[destination] = source
    return torch.tensor(result, dtype=torch.long, device=data["features"].device)


def _check_commit(theta, phis, theta_next, phis_next):
    import torch
    _require(set(theta) == set(theta_next) and len(phis_next) == len(phis) == 4,
             "Incomplete committed core/private state")
    for name, previous in theta.items():
        _require(theta_next[name].shape == previous.shape, "Core shape changed")
        _finite(theta_next[name], "Nonfinite committed core: " + name)
        if name.startswith("local_head."):
            _require(torch.equal(previous, theta_next[name]), "Dormant global-stage core moved")
    for before, after in zip(phis, phis_next):
        _require(set(before) == set(after), "Incomplete private commit")
        for name, previous in before.items():
            _require(after[name].shape == previous.shape, "Private shape changed")
            _finite(after[name], "Nonfinite committed private: " + name)
            if name.startswith("local_head."):
                _require(torch.equal(previous, after[name]), "Dormant global-stage private moved")


def _install(family, theta, phis, private_names):
    import torch
    parameters = dict(family.named_parameters())
    _require(set(parameters) == set(theta) | set(private_names), "Complete parameter bank required")
    with torch.no_grad():
        for name, value in theta.items():
            parameters[name].copy_(value)
        for name in private_names:
            parameters[name].copy_(torch.stack([phi[name] for phi in phis]))
    family.set_global_stage(True)
    family.eval()


def _distances(state, common):
    """State displacement is B-only evidence, never an A selector."""
    import torch
    shared, private = 0.0, [0.0] * 4
    names = {f"{owner}.{factor}" for owner in ("stem", "local_head", "global_head")
             for factor in ("R", "S", "B")}
    for name, value in state.items():
        delta = value.double() - common[name].double()
        if name in names:
            for member in range(4):
                private[member] += float(delta[member].square().sum())
        else:
            shared += float(delta.square().sum())
    return {"shared_l2": shared ** 0.5, "private_member_l2": [v ** 0.5 for v in private]}



def _bound_document(project_root, row):
    path = _source_path(project_root, row)
    _require(path.stat().st_mode & 0o222 == 0, "Prerequisite evidence must be immutable")
    return _read(path)


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _resource_evidence(row, cuda_required):
    _require(_positive_number(row["elapsed_seconds"])
             and _positive_number(row["process_peak_rss_bytes"]), "Actual time/RSS required")
    _require(_positive_number(row["fixed_deadline_seconds"])
             and row["elapsed_seconds"] <= row["fixed_deadline_seconds"], "Fixed resource limit not passed")
    if cuda_required:
        _require(_positive_number(row["cuda_peak_allocated_bytes"])
                 and _positive_number(row["cuda_peak_reserved_bytes"])
                 and row["cuda_peak_reserved_bytes"] >= row["cuda_peak_allocated_bytes"],
                 "Actual whole-process CUDA peaks required")


def _prerequisites(project_root, bindings, queue, device):
    """Root normalizes and pins actual evidence; a oneLIVE receipt cannot admit fit."""
    path = PACKET / "PREREQUISITES.json"
    _require(path.stat().st_mode & 0o222 == 0, "Frozen admission record required")
    gate = _read(path)
    _require(gate["schema"] == "amazon_sequential_actual_all_six_fit_admission_v1"
             and gate["status"] == "ACTUAL_ALL_SIX_QUALIFIED_EXPLICIT_ROOT_FIT_ADMISSION"
             and gate["root_explicit_fit_authorized"] is True
             and gate["A_evaluator_authorized"] is False
             and gate["VALID_TEST_authorized"] is False,
             "Actual all-six qualification and explicit root fit admission required")
    _require(gate["queue_sha256"] == _sha(PACKET / "QUEUE.json")
             and gate["source_bindings_sha256"] == _sha(PACKET / "SOURCE_BINDINGS.json")
             and bindings["files"]["worker"]["sha256"] == _sha(Path(__file__))
             and gate["device"] == device, "Admission recipe/device differs")
    for key in ("worker", "accessor", "warm_response_probe", "sequential"):
        _require(bindings["files"][key]["sha256"] ==
                 bindings["future_flag_only_releases"][key]["anticipated_sha256"],
                 "Only reviewed flag-only release bindings may be admitted: " + key)
    _require(bindings["files"]["G0V2"]["sha256"] ==
             "fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3"
             and bindings["files"]["native_port"]["sha256"] ==
             "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86",
             "Original disabled operator/port bindings changed")
    reviews = gate["independent_review_evidence"]
    _require({row["role"] for row in reviews} == {"candidate_source", "scientific_source"}
             and len(reviews) == 2, "Candidate and scientific successor independent reviews required")
    for row in reviews:
        evidence_path = _source_path(project_root, row["evidence"])
        expected_reviewed = (bindings["sequential_qualified_predecessor"]["sha256"]
                            if row["role"] == "candidate_source" else
                            bindings["future_flag_only_releases"]["worker"]["disabled_sha256"])
        _require(evidence_path.stat().st_mode & 0o222 == 0
                 and row["reviewed_source_sha256"] == expected_reviewed
                 and row["unresolved_blocking_source_defects"] == 0,
                 "Independent review evidence/source or verdict differs")
    scope = _bound_document(project_root, gate["all_six_full_scope_decision"])
    _require(scope["controls"] == list(ARMS) and scope["fixed_before_execution"] is True
             and scope["full_context_FP32"] is True, "Separately fixed all-six full scope required")
    synthetic = _bound_document(project_root, gate["synthetic_result"])
    _require(synthetic["status"] == "PASS_SYNTHETIC_SEQUENTIAL_PARITY_ONLY"
             and synthetic["mode"] == "synthetic"
             and synthetic["all_six_controls_compared"] is True
             and synthetic["caller_states_and_flags_and_gates_unchanged"] is True
             and synthetic["candidate_pin"][1] == bindings["sequential_qualified_predecessor"]["sha256"]
             and synthetic["model_fits"] == 0 and synthetic["persistent_updates"] == 0
             and synthetic["A_scoring"] is False and synthetic["VALID_TEST_access"] is False,
             "Actual all-control synthetic parity/restoration required")
    full = _bound_document(project_root, gate["all_six_full_certificate"])
    _require(full["schema"] == "root_actual_all_six_full_FP32_certificate_v1"
             and full["status"] == "PASS_ALL_SIX_FULL_FP32_COMMIT_RESOURCE"
             and full["qualified_controls"] == list(ARMS)
             and full["candidate_sha256"] == bindings["sequential_qualified_predecessor"]["sha256"]
             and full["operator_sha256"] == bindings["files"]["G0V2"]["sha256"]
             and full["port_sha256"] == bindings["files"]["native_port"]["sha256"]
             and full["native_model_sha256"] == bindings["files"]["native_model"]["sha256"]
             and full["boundary_model_sha256"] == bindings["files"]["boundary_model"]["sha256"]
             and full["nodes"] == 24492 and full["features"] == 300 and full["classes"] == 5
             and full["members"] == 4 and full["dtype"] == "torch.float32"
             and full["device"] == device and full["original_phi_recompute_qualified"] is True
             and full["state_input_RNG_alias_gate_restoration_passed"] is True
             and full["model_fits"] == 0 and full["persistent_updates"] == 0
             and full["A_scoring"] is False and full["VALID_TEST_access"] is False,
             "Actual complete all-six FP32 commitment/context/restoration required")
    _resource_evidence(full["whole_process_resources"], device.startswith("cuda:"))
    _require(set(full["per_control"]) == set(ARMS), "All six actual resource rows required")
    for arm in ARMS:
        row = full["per_control"][arm]
        _require(row["complete_public_episode_passed"] is True
                 and row["original_phi_commit_passed"] is True
                 and row["finite_values_states_diagnostics_and_feasibility_passed"] is True
                 and row["constructed_states_discarded"] is True
                 and row["actual_episode_counts"] == queue["expected_episode_counts"][arm],
                 "Unqualified full FP32 control: " + arm)
    limits = gate["scientific_resource_limits"]
    _require(limits["fixed_before_any_fit"] is True
             and gate["external_watchdog_required"] is True
             and _positive_number(limits["max_elapsed_seconds"])
             and _positive_number(limits["max_process_rss_bytes"]),
             "Fixed prospective scientific resource limits/watchdog required")
    if device.startswith("cuda:"):
        _require(_positive_number(limits["max_cuda_allocated_bytes"])
                 and _positive_number(limits["max_cuda_reserved_bytes"]),
                 "Fixed prospective CUDA resource limits required")
    _require(bool(full["raw_result_evidence"]), "Actual raw full-result evidence required")
    for row in full["raw_result_evidence"]:
        raw = _bound_document(project_root, row)
        _require(str(raw["status"]).startswith("PASS_")
                 and raw["model_fits"] == 0 and raw["persistent_updates"] == 0
                 and raw["A_scoring"] is False and raw["VALID_TEST_access"] is False,
                 "Raw full qualification evidence failed or performed scientific work")
    return gate, full


class _Audit:
    """Durable attempted operation events; never records tensor payloads."""

    def __init__(self, output):
        self.output = output
        self.phase, self.arm, self.episode = "preflight", None, None
        self.callback_counts = {"acquisition": 0, "warm_diagnostic": 0, "continuation": 0}
        self.callback_seconds = dict.fromkeys(self.callback_counts, 0.0)
        self.warm_diagnostic = {"private_gradient_calls": 0, "q_map_primal_calls": 0}
        self.acquisition_backward_attempts = 0
        self.acquisition_optimizer_attempts = 0
        self.completed_acquisition_updates = 0
        self.completed_episodes = dict.fromkeys(ARMS, 0)
        self.meters = {}

    def event(self, kind, **fields):
        row = {"kind": kind, "phase": self.phase, "arm": self.arm,
               "episode": self.episode, "monotonic_seconds": time.monotonic(), **fields}
        with (self.output / "ATTEMPTED_OPERATIONS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def snapshot(self):
        return {"independent_native_callback_attempts": dict(self.callback_counts),
                "independent_native_callback_seconds": dict(self.callback_seconds),
                "warm_diagnostic_attempts": dict(self.warm_diagnostic),
                "acquisition_backward_attempts": self.acquisition_backward_attempts,
                "acquisition_optimizer_attempts": self.acquisition_optimizer_attempts,
                "completed_acquisition_updates": self.completed_acquisition_updates,
                "completed_episodes": dict(self.completed_episodes),
                "sequential_by_arm": {arm: meter.snapshot() for arm, meter in self.meters.items()}}


@contextmanager
def _count_native_callbacks(family, audit):
    """Transparent complete native member callback wrapper; restored in finally."""
    had_attribute = "forward_member" in family.__dict__
    own_attribute = family.__dict__.get("forward_member")
    original = family.forward_member
    def counted(*args, **kwargs):
        scope = ("acquisition" if audit.phase == "common_acquisition" else
                 "warm_diagnostic" if audit.phase == "initial_response_diagnostic" else "continuation")
        audit.callback_counts[scope] += 1
        audit.event("independent_native_callback_attempt", scope=scope)
        begun = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            audit.callback_seconds[scope] += time.monotonic() - begun
    family.forward_member = counted
    try:
        yield
    finally:
        if had_attribute:
            family.forward_member = own_attribute
        else:
            delattr(family, "forward_member")


@contextmanager
def _count_original_warm_private_and_maps(operator, audit):
    """Count original warm partial/map bodies; preserve values and restore APIs."""
    original = {name: getattr(operator, name)
                for name in ("_own_ce", "_main_loss", "_assignment_map")}
    def wrap(name):
        def counted(*args, **kwargs):
            key = "q_map_primal_calls" if name == "_assignment_map" else "private_gradient_calls"
            audit.warm_diagnostic[key] += 1
            audit.event("original_warm_operation_attempt", operation=key, original_function=name)
            return original[name](*args, **kwargs)
        return counted
    try:
        for name in original:
            setattr(operator, name, wrap(name))
        yield
    finally:
        for name, function in original.items():
            setattr(operator, name, function)


def _meter(sequential, audit):
    class DurableCounters(sequential.Counters):
        def add(self, key, stage):
            super().add(key, stage)
            audit.event("sequential_operation_attempt", operation=key, stage=stage)
    return DurableCounters()


def _resources(device, start):
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row = {"elapsed_seconds": time.monotonic() - start,
           "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
           "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
           "scope": "Whole current process; CUDA peaks are not reset to hide earlier work"}
    if device.startswith("cuda:") and torch.cuda.is_initialized():
        row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return row


def _check_resource_limits(admission, device, start):
    row = _resources(device, start)
    limits = admission["scientific_resource_limits"]
    _require(row["elapsed_seconds"] <= limits["max_elapsed_seconds"]
             and row["process_peak_rss_bytes"] <= limits["max_process_rss_bytes"],
             "Fixed scientific time/RSS limit exceeded")
    if device.startswith("cuda:") and row["cuda_peak_allocated_bytes"] is not None:
        _require(row["cuda_peak_allocated_bytes"] <= limits["max_cuda_allocated_bytes"]
                 and row["cuda_peak_reserved_bytes"] <= limits["max_cuda_reserved_bytes"],
                 "Fixed scientific CUDA resource limit exceeded")
    return row


def _freeze_events(output):
    path = output / "ATTEMPTED_OPERATIONS.jsonl"
    if path.exists():
        path.chmod(0o444)


def _continuation_counts(audit):
    keys = ("native_forward_calls", "private_gradient_calls", "native_vjp_calls",
            "q_map_primal_calls", "q_map_vjp_calls", "small_query_vjp_calls")
    return {key: sum(meter.total[key] for meter in audit.meters.values()) for key in keys}


def _public_identity_before_w(accessor, project_root, public_b_dir, qualification):
    """IDs/public features/edges only; no W/B/S/R/A label reader is called."""
    import numpy as np
    manifest_path, manifest, roles = accessor._projection(public_b_dir)
    features, edges, preprocessing = accessor._public_context(np, project_root)
    _require(qualification["native_edge_logical_sha256"] == preprocessing["edge_logical_sha256"]
             and qualification["public_graph_sha256"] == manifest["public_graph"]["sha256"]
             and qualification["roles_sha256"] == manifest["roles"]["sha256"],
             "Actual qualified graph/roles differ before W-label access")
    identity = {"public_b_manifest_sha256": accessor._descriptor(manifest_path, str(manifest_path))["sha256"],
                "public_graph_sha256": manifest["public_graph"]["sha256"],
                "roles_sha256": manifest["roles"]["sha256"], "preprocessing": preprocessing}
    del features, edges, roles
    return identity


def run_six(project_root, public_b_dir, output_dir, *, device="cpu"):
    """A future admission prepares common400 and all six H16 endpoints, without A.

    All attempted events survive Python failures. External termination leaves
    the durable event stream and no COMPLETE.json; no shorter replacement runs.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only sequential six-arm scientific worker")
    start = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=False, exist_ok=False)
    audit = _Audit(output)
    common_row, endpoints, diagnostics = None, {}, {}
    initial_response_row, permutation_row = None, None
    recipe, admission, qualification = None, None, None
    try:
        _require(device == "cpu" or device.startswith("cuda:"), "Explicit native device required")
        bindings = _read(PACKET / "SOURCE_BINDINGS.json")
        queue = _read(PACKET / "QUEUE.json")
        _require(queue["arms"] == [{"id": arm, "control": CONTROLS[arm],
                                    "affinity": "K_perm" if arm == "permuted" else "K",
                                    "episodes": HORIZON} for arm in ARMS]
                 and queue["common"] == {"seed": 17, "split": 0, "local_updates": 200,
                                          "global_updates": 200, "all_W_own_CE4": True}
                 and queue["coefficients"] == {"eta_probe": 0.01, "eta_private": 0.01,
                     "eta_core": 0.001, "extra_margin": 0.1, "pool_fraction": 0.5,
                     "response_epsilon": 0.001, "entropy": 1.0, "graph": 1.0,
                     "assignment_steps": 8, "assignment_rate": 1.0,
                     "ratio_smoothing": 0.01, "reciprocal_smoothing": 0.01}
                 and queue["planned_total_member_forwards_including_evaluation"] == 5740
                 and queue["planned_fit_member_forwards_including_warm_diagnostic"] == 5712,
                 "Fixed sequential queue differs")
        admission, qualification = _prerequisites(project_root, bindings, queue, device)
        audit.event("actual_all_six_prerequisites_verified_before_W_access")
        accessor = _module(project_root, bindings, "accessor")
        port = _module(project_root, bindings, "native_port")
        operator = _module(project_root, bindings, "G0V2")
        warm_probe = _module(project_root, bindings, "warm_response_probe")
        sequential = _module(project_root, bindings, "sequential")
        _require(accessor.SOURCE_RELEASED is True and warm_probe.SOURCE_RELEASED is True
                 and sequential.SOURCE_RELEASED is True and port.PORT_RELEASED is False
                 and operator.SOURCE_RELEASED is False,
                 "Released reviewed caller/sequential successors and original false operator/port required")
        native = _module(project_root, bindings, "native_model")
        boundary = _module(project_root, bindings, "boundary_model")
        import torch
        qualified_identity = _public_identity_before_w(accessor, project_root, public_b_dir, qualification)
        audit.event("qualified_public_identity_verified_before_W_access",
                    public_b_manifest_sha256=qualified_identity["public_b_manifest_sha256"],
                    public_graph_sha256=qualified_identity["public_graph_sha256"],
                    roles_sha256=qualified_identity["roles_sha256"],
                    native_edge_logical_sha256=qualified_identity["preprocessing"]["edge_logical_sha256"])
        warm_data = accessor.load_public_w(project_root, public_b_dir, device=device)
        _require(set(warm_data) == {"features", "edge_index", "W_ids", "W_labels", "provenance"},
                 "Only W labels and public context may enter acquisition")
        preprocessing_identity = warm_data["provenance"]["preprocessing"]
        _require(warm_data["provenance"]["public_b_manifest"]["sha256"] ==
                 qualified_identity["public_b_manifest_sha256"]
                 and preprocessing_identity == qualified_identity["preprocessing"],
                 "Preflight and W-reader public identity changed")
        _require(qualification["native_edge_logical_sha256"] == preprocessing_identity["edge_logical_sha256"]
                 and qualification["public_graph_sha256"] == warm_data["provenance"]["public_graph"]["sha256"]
                 and qualification["roles_sha256"] == warm_data["provenance"]["roles"]["sha256"],
                 "Actual qualified graph/roles differ from scientific context")
        recipe = {"queue_sha256": _sha(PACKET / "QUEUE.json"),
                  "bindings_sha256": _sha(PACKET / "SOURCE_BINDINGS.json"),
                  "worker_source_sha256": _sha(Path(__file__)),
                  "prerequisites_sha256": _sha(PACKET / "PREREQUISITES.json"),
                  "all_six_full_certificate_sha256": admission["all_six_full_certificate"]["sha256"],
                  "sequential_source_sha256": bindings["files"]["sequential"]["sha256"],
                  "sequential_qualified_predecessor_sha256": bindings["sequential_qualified_predecessor"]["sha256"],
                  "public_b_manifest_sha256": warm_data["provenance"]["public_b_manifest"]["sha256"],
                  "roles_sha256": warm_data["provenance"]["roles"]["sha256"],
                  "public_graph_sha256": warm_data["provenance"]["public_graph"]["sha256"],
                  "native_edge_logical_sha256": preprocessing_identity["edge_logical_sha256"]}
        _write(output / "RUN.json", {"schema": "amazon_G0_six_arm_run_v1", "recipe": recipe,
                                     "provenance": warm_data["provenance"], "device": device,
                                     "A_labels_received": False, "A_scoring_performed": False,
                                     "actual_all_six_FP32_prerequisites_checked": True})
        family, construction = _fresh_family(native, boundary, device)
        with _count_native_callbacks(family, audit):
            audit.phase = "common_acquisition"
            audit.event("phase_started")
            optimizer = torch.optim.Adam(family.parameters(), lr=0.001, weight_decay=0.0)
            construction["after_Adam_constructor"] = _rng(device)
            group = optimizer.param_groups[0]
            _require(len(optimizer.param_groups) == 1 and group["betas"] == (0.9, 0.999)
                     and group["eps"] == 1e-8 and not group["amsgrad"]
                     and len({id(p) for p in group["params"]}) == len(list(family.parameters())),
                     "Adam defaults/complete ownership differ")
            warm_losses = []
            for update in range(400):
                _check_resource_limits(admission, device, start)
                if update == 200:
                    family.set_global_stage(True)
                warm_losses.append(_warm_step(family, optimizer, warm_data, audit))
                audit.completed_acquisition_updates = update + 1
                audit.event("acquisition_update_completed", update=update + 1)
            _require(audit.callback_counts["acquisition"] == 1600
                     and audit.acquisition_backward_attempts == 400
                     and audit.acquisition_optimizer_attempts == 400,
                     "Actual acquisition work differs")
            family.eval()
            common = _cpu_tree(family.state_dict())
            common_payload = {"schema": "amazon_G0_frozen_state_v1", "id": "initial",
                              "recipe": recipe, "episodes": 0, "warm_updates": 400,
                              "global_stage": True, "eval_mode": True, "family_state": common,
                              "Adam": _cpu_tree(optimizer.state_dict()), "rng": _cpu_tree(_rng(device)),
                              "construction": _cpu_tree(construction), "W_own_CE_trace": warm_losses,
                              "warm_role": "W"}
            common_row = _save_state(output / "initial.pt", common_payload)
            del common_payload, optimizer, warm_data
            audit.phase = "continuation_context"
            data = accessor.load_public_b(project_root, public_b_dir, device=device)
            _require(set(data) == {"features", "edge_index", "B_ids", "B_labels", "W_ids", "W_labels",
                                   "inner_indices", "inner_labels", "query_indices", "query_labels",
                                   "A_ids", "provenance"}, "Exact W/S/R episode accessor required")
            _require(data["provenance"]["public_b_manifest"]["sha256"] == recipe["public_b_manifest_sha256"],
                     "Common and continuation role custody differs")
            _require(data["provenance"]["preprocessing"] == preprocessing_identity,
                     "Warm and continuation native edge identities differ")
            frozen_permutation = _permutation(data)
            permutation_row = _save_state(output / "permutation.pt", _cpu_tree(frozen_permutation))
            for arm in ARMS:
                audit.arm, audit.episode, audit.phase = arm, None, "continuation_binding"
                family.load_state_dict(common, strict=True)
                family.set_global_stage(True)
                family.eval()
                # These are original pure helpers under the explicit root admission above.
                # The original port/operator process and file gates remain false.
                forward, theta, phis, _ = port._native_callback_and_state(
                    family, data["features"], data["edge_index"], expected_nodes=24492, global_stage=True)
                port._check_operator(operator, bindings["files"]["G0V2"]["sha256"])
                pairs = port._sparse_pairs(operator, data["inner_indices"], data["inner_labels"],
                    data["edge_index"], node_count=24492, dtype=data["features"].dtype,
                    affinity_permutation=frozen_permutation if arm == "permuted" else None)
                history = []
                if arm == "live":
                    audit.phase = "initial_response_diagnostic"
                    audit.event("phase_started")
                    with _count_original_warm_private_and_maps(operator, audit):
                        initial_response = warm_probe.inspect_common_response(
                            operator, bindings["files"]["G0V2"]["sha256"], theta, phis,
                            forward, pairs, data["inner_indices"], data["inner_labels"],
                            common_state_sha256=common_row["sha256"],
                            warm_metadata={"id": "initial", "warm_updates": 400, "episodes": 0,
                                           "global_stage": True, "eval_mode": True})
                    _require(audit.callback_counts["warm_diagnostic"] == 16
                             and audit.warm_diagnostic == {"private_gradient_calls": 8, "q_map_primal_calls": 10},
                             "Original warm diagnostic actual work differs")
                    _finite_tree(initial_response)
                    initial_response_row = _save_state(output / "initial_response.pt", _cpu_tree(initial_response))
                    del initial_response
                audit.meters[arm] = _meter(sequential, audit)
                for episode in range(HORIZON):
                    _check_resource_limits(admission, device, start)
                    audit.phase, audit.episode = "continuation", episode + 1
                    audit.event("episode_started")
                    before_callbacks = audit.callback_counts["continuation"]
                    begun = time.monotonic()
                    theta_next, phis_next, info = sequential.sequential_native_episode(
                        operator, bindings["files"]["G0V2"]["sha256"],
                        port, bindings["files"]["native_port"]["sha256"], theta, phis, forward, pairs,
                        data["inner_indices"], data["inner_labels"],
                        data["query_indices"], data["query_labels"], control=CONTROLS[arm],
                        counters=audit.meters[arm])
                    _require(info["sequential_episode_counts"] == queue["expected_episode_counts"][arm]
                             and audit.callback_counts["continuation"] - before_callbacks ==
                             info["sequential_episode_counts"]["native_forward_calls"],
                             "Independent callbacks and complete episode counters differ")
                    _check_commit(theta, phis, theta_next, phis_next)
                    _finite_tree(info)
                    theta, phis = theta_next, phis_next
                    history.append(_cpu_tree(info))
                    audit.completed_episodes[arm] = episode + 1
                    audit.event("episode_completed", elapsed_seconds=time.monotonic() - begun,
                                counters=info["sequential_episode_counts"])
                _install(family, theta, phis, port.PRIVATE_NAMES)
                endpoint_state = _cpu_tree(family.state_dict())
                endpoints[arm] = _save_state(output / (arm + ".pt"),
                    {"schema": "amazon_G0_frozen_state_v1", "id": arm, "recipe": recipe,
                     "episodes": HORIZON, "warm_updates": 400, "warm_role": "W", "common_state": common_row,
                     "global_stage": True, "eval_mode": True, "family_state": endpoint_state})
                diagnostics[arm] = _save_state(output / (arm + "_B_diagnostics.pt"),
                    {"episodes": tuple(history), "endpoint_vs_common": _distances(endpoint_state, common),
                     "interpretation": "Finite G0 support diagnostics, never an A selector"})
                del history, endpoint_state, theta, phis, theta_next, phis_next, forward, pairs
            _require(port.PORT_RELEASED is False and operator.SOURCE_RELEASED is False,
                     "Original operator/port gates moved")
        actual = _continuation_counts(audit)
        _require(actual == queue["expected_continuation_counts"]
                 and audit.callback_counts == {"acquisition": 1600, "warm_diagnostic": 16, "continuation": 4096}
                 and sum(audit.callback_counts.values()) == 5712
                 and audit.completed_episodes == dict.fromkeys(ARMS, HORIZON),
                 "Actual complete scientific work differs from prospective amendment")
        audit.phase = "complete"
        audit.event("all_six_endpoints_completed", counters=actual)
        _freeze_events(output)
        complete = {"schema": "amazon_G0_seven_states_complete_v1", "recipe": recipe,
                    "common": common_row, "endpoints": endpoints, "B_diagnostics": diagnostics,
                    "permutation": permutation_row, "initial_response": initial_response_row,
                    "diagnostic_extra_member_forwards": 16, "diagnostic_extra_private_gradients": 8,
                    "warm_updates": 400, "warm_role": "W", "episodes_per_arm": HORIZON,
                    "arms_in_order": list(ARMS), "A_labels_received": False, "A_scoring_performed": False,
                    "elapsed_fit_seconds": time.monotonic() - start,
                    "planned_fit_member_forwards_including_warm_diagnostic": 5712,
                    "planned_total_member_forwards_including_evaluation": 5740,
                    "served_evaluation_forwards_reserved_not_executed": 28,
                    "actual_operation_accounting": audit.snapshot(),
                    "actual_resources": _check_resource_limits(admission, device, start),
                    "attempted_operation_events": _descriptor(output / "ATTEMPTED_OPERATIONS.jsonl"),
                    "actual_all_six_FP32_prerequisites_checked": True,
                    "new_runtime_qualification_claim": False}
        _write(output / "COMPLETE.json", complete)
        return _descriptor(output / "COMPLETE.json")
    except BaseException as error:
        # Capture attempted counters even for a timeout, OOM or interrupted Python call.
        # SIGKILL cannot run this handler; its durable events and external receipt remain.
        try:
            actual_resources = _resources(device, start)
        except BaseException as resource_error:
            actual_resources = {"elapsed_seconds": time.monotonic() - start,
                                "resource_collection_error": type(resource_error).__name__ + ": " + str(resource_error)}
        _freeze_events(output)
        _write(output / "FAILURE.json", {"schema": "amazon_G0_incomplete_run_v1", "phase": audit.phase,
            "arm": audit.arm, "episode": audit.episode,
            "error_type": type(error).__name__, "error": str(error), "common": common_row,
            "endpoints": endpoints, "initial_response": initial_response_row, "recipe": recipe,
            "actual_operation_accounting": audit.snapshot(), "actual_resources": actual_resources,
            "elapsed_fit_seconds": time.monotonic() - start,
            "A_labels_received": False, "A_scoring_performed": False,
            "replacement_seed_or_arm_or_shortened_H_authorized": False})
        raise
