"""Disabled strict native FP32 utility episode/support qualification; no fit.

Synthetic monolithic parity is a separate immutable prerequisite. This caller
uses the original native callback/cold state/public+B projection and qualifies
one complete utility episode, a fresh original-phi recommit and an isolated
one-forward product-rule support control. No full-graph monolithic oracle/FD.
"""
import time
STARTED = time.monotonic()
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import socket
import subprocess
import sys
import traceback

SOURCE_RELEASED = False
BASE = ("learnability_responsibility_native_numerical_worker_preparation_20261005_v3/qualify.py", "baeac626bade87bd286fe7a93a95602d8dc1f57d3257d0eeeb7e4a8e3823c688")
UTILITY = ("matched_first_order_private_gradient_utility_control_source_preparation_20261006_v3/utility_control.py", "b55ba7cc7ad3d8446ef86e6ca7e7d6a739526650d1b867047541c3414ed11bb5")
SEQUENTIAL = ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py", "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167")
ACCESSOR = ("amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py", "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85")
SYNTHETIC_SOURCE_SHA256 = "1d31a71e1e2b938c0752ff18635d1e2003706347567178fad39f39e709ad4574"
SYNTHETIC_RESULT = None  # Pending actual new PASS; exact descriptor comes from immutable root scope.
PUBLIC_B = "learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b"
PUBLIC_B_SHA256, PUBLIC_B_BYTES = "da03b6c14615da934443c9b8f36b68f9d643e2b23e04ebe17dab7dacf630df55", 1568
HOST_REPOSITORIES = {"anogena-2-0": "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs",
                     "peptide": "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git"}
WORKSPACE_CONFIG, ATOL, RTOL, NONTRIVIAL = ":4096:8", 2e-6, 2e-5, 1e-12
EPISODE = {"native_forward_calls": 44, "private_gradient_calls": 28, "native_vjp_calls": 12,
           "q_map_primal_calls": 30, "q_map_vjp_calls": 10, "small_query_vjp_calls": 1,
           "utility_margin_phi_vjp_calls": 8, "utility_dummy_cotangent_reverse_calls": 8,
           "utility_weighted_margin_private_gradient_calls": 4}
RECOMMIT = {"native_forward_calls": 12, "private_gradient_calls": 8, "native_vjp_calls": 0,
           "q_map_primal_calls": 10, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0,
           "utility_margin_phi_vjp_calls": 4, "utility_dummy_cotangent_reverse_calls": 4,
           "utility_weighted_margin_private_gradient_calls": 0}
CREDIT = {"native_forward_calls": 2, "private_gradient_calls": 2, "native_vjp_calls": 2,
          "q_map_primal_calls": 0, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0,
          "utility_margin_phi_vjp_calls": 0, "utility_dummy_cotangent_reverse_calls": 0,
          "utility_weighted_margin_private_gradient_calls": 1}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_json(phase, row):
    root, relative = Path(phase).resolve(), Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Root metadata path escape")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.stat().st_mode & 0o222 == 0,
            "Root-frozen in-phase metadata required")
    raw = path.read_bytes()
    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "Root metadata bytes differ")
    return json.loads(raw)


def load(phase, label, pin, loaded):
    path = Path(phase) / pin[0]
    require(sha(path) == pin[1], "Exact source differs: " + pin[0])
    name = "_native_utility_qualification_" + label
    require(name not in sys.modules, "Fresh prepared-module identity required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loaded.append(name)
    spec.loader.exec_module(module)
    return module


def read_scope(args, phase):
    path = Path(args.execution_scope).resolve()
    require(not path.is_symlink() and path.is_relative_to(phase) and path.stat().st_mode & 0o222 == 0
            and sha(path) == args.execution_scope_sha256, "Exact immutable root execution scope required")
    scope = json.loads(path.read_text())
    require(scope["schema"] == "root_native_utility_qualification_execution_scope_v1"
            and scope["root_engineering_invocation_authorized"] is True and scope["fixed_before_execution"] is True
            and scope["qualifier_sha256"] == sha(__file__) and scope["utility_sha256"] == UTILITY[1]
            and scope["no_fit_or_scoring"] is True and scope["closed_A_VALID_TEST"] is True,
            "Root synthetic/native engineering scope is not fixed or differs")
    host, repo = scope["hostname"], Path(scope["repository"]).resolve()
    require(host in HOST_REPOSITORIES and str(repo) == HOST_REPOSITORIES[host]
            and socket.gethostname() == host and Path.cwd().resolve() == repo
            and Path(sys.executable).absolute() == Path(scope["python_executable"]).absolute()
            and sys.dont_write_bytecode, "Wrong root-chosen authorized host/repository/cwd/interpreter/-B")
    require(scope["device"] == "cuda:0" and scope["CUDA_VISIBLE_DEVICES"] == scope["GPU_UUID"]
            and isinstance(scope["GPU_UUID"], str) and scope["GPU_UUID"].startswith("GPU-"),
            "One root-selected UUID mapped to cuda:0 required")
    require(Path(scope["public_b_dir"]).resolve() == (phase / PUBLIC_B).resolve()
            and scope["public_b_manifest_sha256"] == PUBLIC_B_SHA256, "Exact existing public+B projection required")
    limits = scope["resource_limits"]
    for key in ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes"):
        value = limits[key]
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Positive root resource limit required: " + key)
    require(scope["external_watchdog_required"] is True
            and type(scope["external_watchdog_seconds"]) in (int, float)
            and math.isfinite(scope["external_watchdog_seconds"])
            and scope["external_watchdog_seconds"] > limits["max_elapsed_seconds"]
            and scope["external_terminal_accounting_required"] is True,
            "Root-fixed watchdog and whole-child terminal accounting required")
    synthetic_row = scope["utility_synthetic_result"]
    require(isinstance(synthetic_row, dict) and set(synthetic_row) == {"path", "bytes", "sha256"}
            and isinstance(synthetic_row["path"], str) and type(synthetic_row["bytes"]) is int and synthetic_row["bytes"] > 0
            and isinstance(synthetic_row["sha256"], str) and len(synthetic_row["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in synthetic_row["sha256"]),
            "Actual new root-frozen synthetic PASS descriptor required; pending template has no authority")
    synthetic = bound_json(phase, synthetic_row)
    require(synthetic["status"] == "PASS_CPU_FLOAT64_UTILITY_SYNTHETIC_PARITY_ONLY"
            and synthetic["worker_sha256"] == SYNTHETIC_SOURCE_SHA256
            and synthetic["source_pins"]["utility"] == list(UTILITY)
            and synthetic["source_pins"]["sequential"] == list(SEQUENTIAL)
            and synthetic["restoration_errors"] == [] and synthetic["model_fits"] == 0
            and synthetic["persistent_updates"] == 0 and synthetic["A_scoring"] is False
            and synthetic["VALID_TEST_access"] is False and synthetic["dataset_labels_read"] is False,
            "Exact utility synthetic parity prerequisite differs; it is not native qualification")
    required = {"complete_signed_cost_gradient_credit_and_original_phi_commit_parity",
                "both_factors_at_zero_own_gradient_analytic_case",
                "fixed_smooth_synthetic_shared_and_private_directional_FD"}
    require(required.issubset({c["name"] for c in synthetic["checks"]}), "Synthetic prerequisite checks incomplete")
    for name in ("qualifier_review", "utility_source_review"):
        bound_json(phase, scope[name])  # Root binds immutable source-review metadata, not a new scientific outcome.
    raw = (phase / PUBLIC_B / "PUBLIC_B_MANIFEST.json").read_bytes()
    require(len(raw) == PUBLIC_B_BYTES and hashlib.sha256(raw).hexdigest() == PUBLIC_B_SHA256,
            "Existing public+B manifest bytes differ")
    return scope


def backend_snapshot():
    result = {"deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
              "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "deterministic_debug_mode": torch.get_deterministic_debug_mode(), "flags": {}}
    for group, owner, names in (("cudnn", torch.backends.cudnn, ("enabled", "deterministic", "benchmark", "allow_tf32")),
        ("cuda_matmul", torch.backends.cuda.matmul, ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"))):
        result["flags"][group] = {n: getattr(owner, n) for n in names if hasattr(owner, n)}
    return result


def backend_restore(old):
    torch.use_deterministic_algorithms(old["deterministic_algorithms_enabled"], warn_only=old["deterministic_algorithms_warn_only"])
    for group, owner in (("cudnn", torch.backends.cudnn), ("cuda_matmul", torch.backends.cuda.matmul)):
        for name, value in old["flags"][group].items():
            setattr(owner, name, value)
    require(backend_snapshot() == old, "Backend flags not restored")


def runtime_identity():
    import torch_geometric
    extensions = {}
    for name in ("torch-scatter", "torch-sparse", "pyg-lib"):
        try:
            extensions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            extensions[name] = None
    return {"torch_version": str(torch.__version__), "torch_module_path": str(Path(torch.__file__).resolve()),
        "torch_CUDA_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "PyG_version": str(torch_geometric.__version__), "PyG_module_path": str(Path(torch_geometric.__file__).resolve()),
        "extension_versions": extensions, "GPU_name": torch.cuda.get_device_name(0),
        "GPU_capability": list(torch.cuda.get_device_capability(0))}


def resources():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row = {"elapsed_seconds_including_imports_setup_checks_and_restore": time.monotonic() - STARTED,
           "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
           "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    if "torch" in globals() and torch.cuda.is_initialized():
        row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    return row


def limits_check(limits):
    row = resources()
    require(row["elapsed_seconds_including_imports_setup_checks_and_restore"] <= limits["max_elapsed_seconds"]
            and row["process_peak_rss_bytes"] <= limits["max_process_rss_bytes"], "Root process time/RSS limit exceeded")
    if row["cuda_peak_allocated_bytes"] is not None:
        require(row["cuda_peak_allocated_bytes"] <= limits["max_cuda_allocated_bytes"]
                and row["cuda_peak_reserved_bytes"] <= limits["max_cuda_reserved_bytes"], "Root CUDA peak limit exceeded")


def compare(actual, expected):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype
                and actual.device == expected.device and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
                "Native recommit tensor structure/finite values differ")
        if not expected.is_floating_point():
            require(torch.equal(actual, expected), "Native integer tensor differs")
            return 0.0
        error = float((actual.detach() - expected.detach()).abs().max()) if expected.numel() else 0.0
        require(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), "Original native FP32 tolerance violated: " + str(error))
        return error
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Native dictionary differs")
        return max((compare(actual[k], expected[k]) for k in expected), default=0.0)
    if isinstance(expected, (tuple, list)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), "Native sequence differs")
        return max((compare(a, b) for a, b in zip(actual, expected)), default=0.0)
    require(actual == expected, "Native scalar metadata differs")
    return 0.0


def maximum(tree):
    if isinstance(tree, dict):
        return max((float(v.detach().abs().max()) for v in tree.values()), default=0.0)
    return max((maximum(v) for v in tree), default=0.0)


def add_trees(a, b):
    if isinstance(a, dict):
        return {k: a[k] + b[k] for k in a}
    return tuple(add_trees(x, y) for x, y in zip(a, b))


def validate_diagnostics(info):
    require(all(len(info[k]) == 10 for k in ("pairs", "assignments", "utility_raw_costs", "observed_finite_response_costs")),
            "Complete ten utility/finite/Q pair banks required")
    require(bool(torch.isfinite(info["probe_own_ce_change"]).all()), "Nonfinite paid finite CE diagnostic")
    for row, q, raw, finite in zip(info["pairs"], info["assignments"], info["utility_raw_costs"], info["observed_finite_response_costs"]):
        require(row["cost_kind"] == "first_order_private_gradient_utility"
                and row["observed_response_kind"] == "paid_finite_softplus_margin_response", "Utility/finite diagnostic kinds differ")
        require(q.shape == raw.shape == finite.shape and bool(torch.isfinite(q).all()) and bool((q > 0).all())
                and bool(torch.isfinite(raw).all()) and bool(torch.isfinite(finite).all()), "Native utility/finite/Q values invalid")
        for value in row.values():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), "Nonfinite native diagnostic")
        require(float(row["row_residual"]) <= 3e-6 and float(row["column_residual"]) <= 2e-3,
                "Original native FP32 assignment feasibility tolerance violated")
        centered = finite - finite.mean(dim=1, keepdim=True)
        compare(row["centered_response_rms"], centered.square().mean().sqrt())
        utility_centered = raw - raw.mean(dim=1, keepdim=True)
        compare(row["centered_cost_rms"], utility_centered.square().mean().sqrt())
    return {"all_utility_and_distinct_observed_finite_diagnostics_finite": True,
            "positive_Q_original_row_column_feasibility": True}


def observed_call(label, counters, physical, forward, function, expected):
    row = {"native_forward_callbacks": 0, "ordinary_autograd_grad_API_attempts": 0}
    physical[label] = row
    def counted(core, phi):
        row["native_forward_callbacks"] += 1
        return forward(core, phi)
    original = torch.autograd.grad
    def counted_grad(*args, **kwargs):
        row["ordinary_autograd_grad_API_attempts"] += 1
        return original(*args, **kwargs)
    torch.autograd.grad = counted_grad
    try:
        result = function(counted)
    finally:
        torch.autograd.grad = original
    require(torch.autograd.grad is original, "Ordinary-grad observer not restored")
    require(counters.total == expected, "Actual native utility construction counts differ: " + label)
    reverse_keys = ("private_gradient_calls", "native_vjp_calls", "q_map_vjp_calls", "small_query_vjp_calls",
                    "utility_margin_phi_vjp_calls", "utility_dummy_cotangent_reverse_calls", "utility_weighted_margin_private_gradient_calls")
    require(row["native_forward_callbacks"] == expected["native_forward_calls"]
            and row["ordinary_autograd_grad_API_attempts"] == sum(expected[k] for k in reverse_keys),
            "Independent native callback/ordinary-grad API observations differ: " + label)
    return result


def product_support(op, theta, phis, forward, pairs, s, ys, fixed_t, observations):
    """One native forward, two private partials, three product-rule VJPs.

    Isolated fixed-cotangent native support; not a monolithic full-graph oracle.
    Every attempted API is recorded before invocation. No input state is written.
    """
    core = {n: v.detach().requires_grad_(True) for n, v in theta.items()}
    phi = {n: v.detach().requires_grad_(True) for n, v in phis[0].items()}
    observations["native_forward_callbacks"] += 1
    with torch.enable_grad():
        logits = forward(core, phi)[s]
        own = torch.nn.functional.cross_entropy(logits, ys)
        observations["private_partial_API_attempts"] += 1
        dg = torch.autograd.grad(own, tuple(phi.values()), allow_unused=True, create_graph=True, retain_graph=True)
        g = {n: torch.zeros_like(v) if d is None else d for (n, v), d in zip(phi.items(), dg)}
        weighted = sum((t[:, 0].detach() * op._margin(logits.unsqueeze(0), pair)[:, 0]).sum()
                       for pair, t in zip(pairs, fixed_t))
        observations["private_partial_API_attempts"] += 1
        dh = torch.autograd.grad(weighted, tuple(phi.values()), allow_unused=True, create_graph=True, retain_graph=True)
        h = {n: torch.zeros_like(v) if d is None else d for (n, v), d in zip(phi.items(), dh)}
        require(maximum(g) > NONTRIVIAL and maximum(h) > NONTRIVIAL, "Fixed native support factors are trivial")
        anchor = sum(0.0 * v.sum() for v in (*core.values(), *phi.values()))
        values = (sum((g[n] * h[n]).sum() for n in phi),
                  sum((g[n].detach() * h[n]).sum() for n in phi),
                  sum((g[n] * h[n].detach()).sum() for n in phi))
        results = []
        for index, value in enumerate(values):
            observations["shared_private_VJP_API_attempts"] += 1
            derivatives = torch.autograd.grad(-op.Config().eta_probe * value + anchor,
                tuple(core.values()) + tuple(phi.values()), allow_unused=True, retain_graph=index != 2)
            result = tuple({n: torch.zeros_like(v) if d is None else d.detach()
                for (n, v), d in zip(params.items(), part)} for params, part in
                ((core, derivatives[:len(core)]), (phi, derivatives[len(core):])))
            results.append(result)
        full, h_path, g_path = results
        require(maximum(h_path[0]) > NONTRIVIAL and maximum(g_path[0]) > NONTRIVIAL,
                "Both shared native mixed-credit paths must be nontrivial")
        error = compare(full, add_trees(h_path, g_path))
    return full, {"both_differentiable_factor_paths_supported": True, "product_rule_sum_max_error": error,
                  "h_path_max_abs": maximum(h_path), "g_path_max_abs": maximum(g_path),
                  "h_path_shared_max_abs": maximum(h_path[0]), "g_path_shared_max_abs": maximum(g_path[0]),
                  "h_path_private_max_abs": maximum(h_path[1]), "g_path_private_max_abs": maximum(g_path[1]),
                  "isolated_support_counts": dict(observations), "full_monolithic_oracle": False}


def run(args, receipt, save, check):
    global torch
    phase = Path(args.source_root).resolve()
    require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Fresh process before Torch required")
    scope = read_scope(args, phase)
    receipt["execution_scope"] = {"path": str(Path(args.execution_scope).resolve().relative_to(phase)),
                                  "sha256": args.execution_scope_sha256}
    receipt["synthetic_prerequisite"] = scope["utility_synthetic_result"]
    receipt["root_frozen_resource_limits"] = scope["resource_limits"]
    loaded, modules = [], {}
    old_path, old_argv, python_rng = list(sys.path), list(sys.argv), random.getstate()
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Fresh child with no active timer required")
    old_environment = {n: (n in os.environ, os.environ.get(n)) for n in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")}
    backend_before = old_threads = rng_before = numpy = before = None
    family = x = edges = theta = phis = pairs = verify_native = None
    meters, physical = {}, {}
    support_observations = {"native_forward_callbacks": 0, "private_partial_API_attempts": 0, "shared_private_VJP_API_attempts": 0}
    receipt["independent_observations"] = physical
    receipt["isolated_support_observations"] = support_observations
    try:
        def expired(signum, frame):
            raise TimeoutError("Root-frozen native utility whole-process deadline exceeded")
        signal.signal(signal.SIGALRM, expired)
        remaining = scope["resource_limits"]["max_elapsed_seconds"] - (time.monotonic() - STARTED)
        require(remaining > 0, "Root deadline exhausted before numerical imports")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        modules["base"] = load(phase, "base", BASE, loaded)
        modules["sequential"] = load(phase, "sequential", SEQUENTIAL, loaded)
        modules["utility"] = load(phase, "utility", UTILITY, loaded)
        modules["accessor"] = load(phase, "accessor", ACCESSOR, loaded)
        require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Prepared stdlib source prematurely imported Torch")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE_CONFIG
        os.environ["CUDA_VISIBLE_DEVICES"] = scope["GPU_UUID"]
        receipt["workspace_and_visible_UUID_configured_before_Torch"] = True
        site = Path(scope["site_packages"]).resolve()
        require(site.is_dir(), "Root-chosen normal runtime site-packages missing; no fallback")
        sys.path.insert(0, str(site))
        import torch
        require(not torch.cuda.is_initialized(), "CUDA initialized before strict configuration")
        backend_before = backend_snapshot()
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
                and not torch.cuda.is_initialized(), "Strict deterministic mode must precede CUDA")
        receipt["strict_backend_configured_before_CUDA"] = True
        require(torch.get_default_dtype() == torch.float32 and str(torch.empty(0).device) == "cpu",
                "Original cold native construction defaults differ")
        old_threads = torch.get_num_threads()
        receipt["interop_before"] = torch.get_num_interop_threads()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        receipt["interop_restoration"] = "Not attempted; one-time fresh-child initialization ends with process termination."
        require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1, "Original 1/1 qualifier thread mode required")
        configured = backend_snapshot()
        require(configured == scope["strict_backend_expected"], "Root-fixed strict backend differs")
        receipt["backend_during_native"] = configured
        import numpy
        base = modules["base"]
        base.torch, base.np = torch, numpy
        # Read-only GPU identity query is only reachable on an explicitly allowed target host.
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=15)
        require(scope["GPU_UUID"] in gpu.stdout.splitlines(), "Root selected GPU UUID is absent")
        require(torch.cuda.device_count() == 1, "Exactly one visible root-selected device required")
        identity = runtime_identity()
        require(identity == scope["runtime_identity_expected"], "Root-fixed normal runtime/GPU metadata differs")
        receipt["runtime_identity"] = identity
        receipt["selected_GPU_UUID"] = scope["GPU_UUID"]
        rng_before = {"numpy": numpy.random.get_state(), "torch_cpu": torch.get_rng_state().clone(),
                      "cuda": [v.clone() for v in torch.cuda.get_rng_state_all()]}
        for name, pin in base.PINS.items():
            modules[name] = load(phase, name, pin, loaded)
        op, port, seq, utility, accessor = (modules[k] for k in ("operator", "port", "sequential", "utility", "accessor"))
        require(op.SOURCE_RELEASED is False and port.PORT_RELEASED is False and seq.SOURCE_RELEASED is False
                and utility.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True, "Original/candidate process gates differ")
        payload = accessor.load_public_b(phase, Path(scope["public_b_dir"]), device="cuda:0")
        x, edges = payload["features"], payload["edge_index"]
        s, ys, r, yr = (payload[k] for k in ("inner_indices", "inner_labels", "query_indices", "query_labels"))
        require(x.shape == (24492, 300) and x.dtype == torch.float32 and s.numel() == 2449 and r.numel() == 2450
                and payload["W_ids"].numel() == 4898 and payload["B_ids"].numel() == 9797
                and not bool(torch.isin(torch.cat((s, r)), payload["A_ids"]).any()), "Complete original public/B context differs")
        receipt["public_B_binding"] = {"directory": scope["public_b_dir"], "manifest_sha256": PUBLIC_B_SHA256,
                                       "accessor_pin": ACCESSOR, "provenance": payload["provenance"]}
        family = base.build_family(modules["native"], modules["boundary"], torch.device("cuda:0"), torch.float32, 17)
        forward, theta, phis, _ = port._native_callback_and_state(family, x, edges, expected_nodes=24492, global_stage=True)
        pairs = port._sparse_pairs(op, s, ys, edges, node_count=24492, dtype=torch.float32)
        before = base.snapshot(family, x, edges, torch.device("cuda:0"))
        caller_tensors = [*theta.values(), *(v for phi in phis for v in phi.values()), s, ys, r, yr]
        for pair in pairs:
            caller_tensors.extend([pair.nodes, pair.targets, pair.competitors, pair.laplacian.row,
                                   pair.laplacian.column, pair.laplacian.weight, pair.laplacian.degree])
        caller_before = [(id(v), v.data_ptr(), v.requires_grad, v.clone()) for v in caller_tensors]
        def verify_native():
            result = base.unchanged(before, family, x, edges, torch.device("cuda:0"))
            for old, value in zip(caller_before, caller_tensors):
                require(old[:3] == (id(value), value.data_ptr(), value.requires_grad) and torch.equal(old[3], value),
                        "Caller tensor value/object/storage/flags changed")
            require(backend_snapshot() == configured and torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
                    "Native work changed strict backend/thread mode")
            return result
        receipt["cold_state"] = {"constructor_source": BASE, "seed": 17, "global_stage": True,
                                 "warm_or_trained_checkpoint_loaded": False}
        arguments = (op, base.PINS["operator"][1], port, base.PINS["port"][1], seq, SEQUENTIAL[1])
        meters["episode"] = utility.Counters()
        def complete_episode():
            nt, np_, info, inspection = observed_call("episode", meters["episode"], physical, forward,
                lambda counted: utility._engineering_episode(*arguments, theta, phis, counted, pairs, s, ys, r, yr,
                    engineering_authorized=True, counters=meters["episode"], collect_inspection=False), EPISODE)
            require(inspection is None and bool(torch.isfinite(info["virtual_query_loss"])), "Resource episode retained inspection or nonfinite query")
            require(all(bool(torch.isfinite(v).all()) for bank in (nt, *np_) for v in bank.values()), "Nonfinite complete output state")
            diagnostics = validate_diagnostics(info)
            update = {n: (theta[n] - v) / op.Config().eta_core for n, v in nt.items()}
            require(any(bool(torch.count_nonzero(v)) for n, v in update.items() if not n.startswith("local_head.")), "Native shared update is trivial")
            base.zero_unused(update, phis, np_)
            for name, value in theta.items():
                if name.startswith("local_head."):
                    require(torch.equal(value, nt[name]), "Inactive shared state moved")
            limits_check(scope["resource_limits"])
            return {"complete_native_utility_episode": True, "diagnostics": diagnostics,
                    "same_caller_state_restored": verify_native(), "episode_counts": meters["episode"].snapshot()}, (nt, np_, info)
        _, committed_values = check("complete_native_utility_live_FP32_episode", complete_episode, auxiliary=True)
        nt, np_, info = committed_values
        meters["recommit"] = utility.Counters()
        def recommit():
            fresh = observed_call("recommit", meters["recommit"], physical, forward,
                lambda counted: utility._engineering_response(*arguments, nt, phis, counted, pairs, s, ys, r, yr,
                    engineering_authorized=True, counters=meters["recommit"]), RECOMMIT)
            fresh_info = fresh["response"].diagnostics
            errors = {"every_original_phi_committed_private_coordinate": compare(np_, fresh["adapted"]),
                      "all_committed_utility_Q_finite_CE_and_pair_diagnostics": compare({k: info[k] for k in fresh_info}, fresh_info)}
            validate_diagnostics(fresh_info)
            for gradients in (fresh["response"].probe_private_gradients, fresh["main_private_gradients"]):
                for row in gradients:
                    for name, value in row.items():
                        require(bool(torch.isfinite(value).all()), "Nonfinite native recommit partial")
                        if name.startswith("local_head."):
                            require(torch.count_nonzero(value).item() == 0, "Inactive private partial is not exact zero")
            base.zero_unused({n: (theta[n] - nt[n]) / op.Config().eta_core for n in theta}, phis, fresh["adapted"])
            limits_check(scope["resource_limits"])
            return {"max_absolute_errors": errors, "fixed_FP32_atol": ATOL, "fixed_FP32_rtol": RTOL,
                    "fresh_original_phi_not_virtual_or_probe_phi": True, "same_caller_state_restored": verify_native(),
                    "recommit_counts": meters["recommit"].snapshot(), "constructed_states_discarded": True}
        check("independent_fresh_original_phi_full_native_recommit", recommit)
        fixed_t = []
        for index, pair in enumerate(pairs):
            rows = torch.arange(pair.nodes.numel(), dtype=torch.float32, device=s.device)
            first = 0.03 * torch.sin(rows * 0.017 + 0.11 * (index + 1)) + 0.02 * torch.cos(rows * 0.031)
            fixed_t.append(torch.stack((first, torch.zeros_like(first), torch.zeros_like(first), torch.zeros_like(first)), dim=1))
        fixed_t = tuple(fixed_t)
        meters["fixed_credit"] = utility.Counters()
        def mixed_support():
            candidate = observed_call("fixed_credit", meters["fixed_credit"], physical, forward,
                lambda counted: utility._engine(*arguments, theta, phis, counted, pairs, s, ys, r, yr,
                    engineering_authorized=True, counters=meters["fixed_credit"])._utility_credit(fixed_t, 0), CREDIT)
            reference, support = product_support(op, theta, phis, forward, pairs, s, ys, fixed_t, support_observations)
            error = compare(candidate, reference)
            base.zero_unused(candidate[0])
            for name, value in candidate[1].items():
                if name.startswith("local_head."):
                    require(torch.count_nonzero(value).item() == 0, "Inactive fixed-credit private coordinate is nonzero")
            limits_check(scope["resource_limits"])
            return {"candidate_vs_one_forward_product_rule_max_error": error, "support": support,
                    "same_caller_state_restored": verify_native(), "no_full_monolithic_or_FD": True}
        check("nontrivial_native_both_factor_mixed_credit_support", mixed_support)
        require(support_observations == {"native_forward_callbacks": 1, "private_partial_API_attempts": 2,
                                         "shared_private_VJP_API_attempts": 3}, "Actual isolated support accounting differs")
        receipt["actual_native_forward_total"] = sum(v["native_forward_callbacks"] for v in physical.values()) + support_observations["native_forward_callbacks"]
        receipt["actual_ordinary_grad_API_total"] = (sum(v["ordinary_autograd_grad_API_attempts"] for v in physical.values())
            + support_observations["private_partial_API_attempts"] + support_observations["shared_private_VJP_API_attempts"])
        require(receipt["actual_native_forward_total"] == 59 and receipt["actual_ordinary_grad_API_total"] == 97,
                "Actual complete native utility support bill differs")
        native_kinds = ("private_gradient_calls", "native_vjp_calls", "utility_margin_phi_vjp_calls",
                        "utility_dummy_cotangent_reverse_calls", "utility_weighted_margin_private_gradient_calls")
        receipt["native_reverse_constructions_excluding_maps_query"] = (sum(m.total[k] for m in meters.values() for k in native_kinds)
            + support_observations["private_partial_API_attempts"] + support_observations["shared_private_VJP_API_attempts"])
        require(receipt["native_reverse_constructions_excluding_maps_query"] == 86, "Actual native reverse bill differs")
        receipt["native_restoration_on_exit"] = verify_native()
        receipt["all_constructed_states_discarded_no_parameter_commit"] = True
    except BaseException as error:
        receipt["primary_exception"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt["attempted_candidate_counters"] = {k: v.snapshot() for k, v in meters.items()}
        errors = []
        def restore(label, function):
            try:
                function()
                receipt[label] = True
            except BaseException as error:
                errors.append({"restoration": label, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        if before is not None and verify_native is not None:
            restore("native_caller_states_aliases_flags_inputs_RNG_and_modes_unchanged", verify_native)
        elif before is not None:
            restore("native_family_restoration_before_caller_snapshot_completed",
                    lambda: modules["base"].unchanged(before, family, x, edges, torch.device("cuda:0")))
        def gates():
            for name, module in modules.items():
                if name in ("base", "native", "boundary"):
                    continue
                attr = "PORT_RELEASED" if name == "port" else "SOURCE_RELEASED"
                expected = name == "accessor"
                require(getattr(module, attr) is expected, "Original/candidate process gate changed: " + name)
            for name, pin in (("base", BASE), ("sequential", SEQUENTIAL), ("utility", UTILITY), ("accessor", ACCESSOR)):
                if name in modules:
                    require(sha(modules[name].__file__) == pin[1], "Prepared source bytes changed")
            if "base" in modules:
                for name, pin in modules["base"].PINS.items():
                    if name in modules:
                        require(sha(modules[name].__file__) == pin[1], "Original mathematical/native source bytes changed")
        restore("loaded_source_bytes_and_process_gates_unchanged", gates)
        if rng_before is not None:
            def rng():
                numpy.random.set_state(rng_before["numpy"])
                torch.set_rng_state(rng_before["torch_cpu"])
                torch.cuda.set_rng_state_all(rng_before["cuda"])
                require(torch.equal(torch.get_rng_state(), rng_before["torch_cpu"]), "Torch CPU RNG not restored")
                actual = torch.cuda.get_rng_state_all()
                require(len(actual) == len(rng_before["cuda"]) and all(torch.equal(a, b) for a, b in zip(actual, rng_before["cuda"])), "Visible CUDA RNG not restored")
                a, b = numpy.random.get_state(), rng_before["numpy"]
                require(a[0] == b[0] and numpy.array_equal(a[1], b[1]) and a[2:] == b[2:], "NumPy RNG not restored")
            restore("NumPy_and_Torch_CPU_visible_CUDA_rng_restored", rng)
        if backend_before is not None:
            restore("backend_restored_exactly", lambda: backend_restore(backend_before))
        if old_threads is not None:
            def threads():
                torch.set_num_threads(old_threads)
                require(torch.get_num_threads() == old_threads, "Intra-op count not restored")
            restore("intra_op_threads_restored", threads)
            receipt["interop_after"] = torch.get_num_interop_threads()
        def environment():
            for name, (present, value) in old_environment.items():
                if present:
                    os.environ[name] = value
                else:
                    os.environ.pop(name, None)
            require(all((name in os.environ, os.environ.get(name)) == old for name, old in old_environment.items()), "Workspace/visible-device environment not restored")
        restore("workspace_and_visible_device_environment_restored", environment)
        def python_state():
            random.setstate(python_rng)
            sys.path[:], sys.argv[:] = old_path, old_argv
            for name in loaded:
                sys.modules.pop(name, None)
            require(random.getstate() == python_rng and sys.path == old_path and sys.argv == old_argv
                    and not any(n in sys.modules for n in loaded), "Python RNG/path/argv/prepared-module binding restoration failed")
        restore("Python_rng_path_argv_and_prepared_bindings_restored", python_state)
        def signals():
            signal.signal(signal.SIGALRM, old_handler)
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
            require(signal.getsignal(signal.SIGALRM) == old_handler and signal.getitimer(signal.ITIMER_REAL) == old_timer, "Signal handler/timer restoration failed")
        restore("signal_handler_timer_restored", signals)
        receipt["restoration_errors"] = errors
        save()
    require(not receipt["restoration_errors"], "Native utility caller restoration failed")
    limits_check(scope["resource_limits"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--execution-scope")
    parser.add_argument("--execution-scope-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_NATIVE_UTILITY_QUALIFICATION_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False}))
        return
    require(SOURCE_RELEASED is False, "This default-disabled engineering caller keeps its artifact marker false")
    if not all((args.execution_scope, args.execution_scope_sha256, args.output)):
        parser.error("Exact future root execution scope/sha and fresh output required")
    # Deliberate research paths only; normal libraries/runtime caches are untouched.
    host = socket.gethostname()
    require(host in HOST_REPOSITORIES, "Only named authorized numeric hosts are permitted")
    repository = Path(HOST_REPOSITORIES[host])
    argument = Path(args.source_root).absolute()
    require(".." not in argument.parts
            and argument == repository / "experiments_iclr/postsubmission_20260930"
            and Path.cwd().resolve() == repository.resolve(), "Named authorized source phase/cwd required")
    phase = argument.resolve()
    require(phase.is_dir() and phase.parent.parent == repository.resolve(), "Source phase escapes named repository")
    scope_argument = Path(args.execution_scope).absolute()
    require(".." not in scope_argument.parts and scope_argument.is_relative_to(argument)
            and not scope_argument.is_symlink() and scope_argument.resolve().is_relative_to(phase),
            "Root execution scope must be inside the named source phase")
    read_scope(args, phase)  # Validate immutable authority before any output creation.
    output_argument = Path(args.output).absolute()
    require(".." not in output_argument.parts and output_argument.is_relative_to(argument)
            and not output_argument.is_symlink() and output_argument.resolve().is_relative_to(phase),
            "Deliberate output must remain inside the named source phase")
    output = output_argument.resolve()
    require(not output.exists() and output.parent.is_dir(), "New output with existing project parent required")
    output.mkdir(parents=False, exist_ok=False)
    created = output.stat()
    output_identity = (created.st_dev, created.st_ino)
    def owned_output():
        require(not output.is_symlink() and output.is_dir(), "Created output directory changed")
        current = output.stat()
        require((current.st_dev, current.st_ino) == output_identity, "Only this invocation's output may be finalized")
    receipt = {"schema": "native_utility_strict_FP32_episode_support_result_v1", "status": "RUNNING",
        "UTC": datetime.now(timezone.utc).isoformat(), "worker_sha256": sha(__file__), "utility_pin": UTILITY,
        "checks": [], "model_fits": 0, "persistent_updates": 0, "A_scoring": False, "VALID_TEST_access": False,
        "full_monolithic_higher_order_or_FD_retry": False, "original_six_arm_pilot_changed": False,
        "cheaper_memory_runtime_FLOP_novelty_or_predictive_claim": False}
    def save():
        owned_output()
        receipt["resource_measurement_scope"] = "Worker capture through numerical checks/restoration; excludes subsequent final publication and exit."
        receipt["external_terminal_resource_closure_required"] = True
        receipt["whole_process_resources"] = resources()
        temporary = output / "RESULT.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "RESULT.json")
    def check(name, function, auxiliary=False):
        receipt["last_started_check"] = name
        save()
        begin = time.monotonic()
        result = function()
        reported = result[0] if auxiliary else result
        receipt["checks"].append({"name": name, "elapsed_seconds": time.monotonic() - begin, "result": reported})
        save()
        return result
    code = 1
    try:
        save()
        run(args, receipt, save, check)
        receipt["status"] = "PASS_NATIVE_UTILITY_FP32_EPISODE_RECOMMIT_AND_ISOLATED_SUPPORT_RESOURCE_ONLY"
        code = 0
    except BaseException as error:
        receipt.update(status="FAIL_NATIVE_UTILITY_QUALIFICATION", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        save()
        owned_output()
        (output / "RESULT.json").chmod(0o444)
    print(json.dumps({"status": receipt["status"], "checks_completed": len(receipt["checks"]), "error": receipt.get("error"), "fit_authorized": False}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
