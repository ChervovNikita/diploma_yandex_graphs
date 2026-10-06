"""Disabled normal first-order engineering qualification; never a reference fit.

Small stochastic complete-gradient/Adam/RNG checks and two discarded native
one-update replays per fixed objective. No state checkpoints or held scoring.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import traceback
from types import SimpleNamespace

SOURCE_RELEASED = False
ORDINARY = ("amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py",
            "5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b")
OBJECTIVES = ("own", "own_pool")
ATOL, RTOL = 2e-6, 2e-5
COMMON_DESCRIPTORS = {
    "common_checkpoint": {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt", "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"},
    "common_origin_run": {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/RUN.json", "bytes": 6402, "sha256": "2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5"}}


def require(value, message):
    if not value: raise RuntimeError(message)


def load_ordinary(root):
    path = root / ORDINARY[0]
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ORDINARY[1], "Final disabled ordinary source differs")
    name = "_normal_ordinary_first_order_qualification"
    require(name not in sys.modules, "Fresh normal numerical child required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    require(module.SOURCE_RELEASED is False and module.OBJECTIVES == OBJECTIVES, "Disabled fixed two-reference source required")
    return module, name


def compare(actual, expected, exact=False):
    import torch
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype,
                "Compared tensor structure differs")
        a, b = actual.detach().cpu(), expected.detach().cpu()
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Nonfinite compared value")
        require(torch.equal(a, b) if exact or not b.is_floating_point() else torch.allclose(a, b, atol=ATOL, rtol=RTOL),
                "Complete-coordinate first-order mismatch")
        return float((a.double() - b.double()).abs().max()) if b.numel() else 0.0
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Compared dictionary keys differ")
        return max((compare(actual[k], expected[k], exact) for k in expected), default=0.0)
    if isinstance(expected, (tuple, list)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), "Compared sequence differs")
        return max((compare(a, b, exact) for a, b in zip(actual, expected)), default=0.0)
    require(actual == expected, "Compared scalar/None differs")
    return 0.0


def gradients(family, helpers):
    return {n: helpers._cpu_tree(p.grad) if p.grad is not None else None for n, p in family.named_parameters()}


def record(family, optimizer, helpers, device):
    return {"parameters": helpers._cpu_tree(family.state_dict()), "Adam": helpers._cpu_tree(optimizer.state_dict()),
            "gradients": gradients(family, helpers), "rng": helpers._cpu_tree(helpers._rng(device))}


def small_fixture(ordinary, port):
    """15 logical CPU nodes; zero padding satisfies the unchanged full-row API.

    Explicitly synthetic; padding is only an API witness, never native evidence.
    Complete physical private shapes and every shared/private gradient are kept.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F
    def value(shape, scale, offset=0.0):
        return offset + scale * torch.sin(torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) * 0.17 + 0.3)
    class Boundary(nn.Module):
        def __init__(self, name, weight_shape):
            super().__init__(); self.weight = nn.Parameter(value(weight_shape, 0.13))
            for suffix in ("R", "S", "B"):
                shape = (4, *port.PRIVATE_SHAPES[name + "." + suffix])
                setattr(self, suffix, nn.Parameter(value(shape, 0.07, 1.0 if suffix != "B" else 0.0)))
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = Boundary("stem", (7, 6)); self.global_head = Boundary("global_head", (5, 7))
            self.local_head = Boundary("local_head", (5, 7))
            self.core = nn.Module(); self.core.residual = nn.Parameter(value((6, 7), 0.11)); self.core._global = True
        def set_global_stage(self, enabled): self.core._global = bool(enabled)
        def forward_member(self, x, edges, member):
            z = F.dropout(x, 0.2, self.training) * self.stem.R[member, :6]
            h = torch.tanh((z @ self.stem.weight.T) * self.stem.S[member, :7] + self.stem.B[member, :7])
            h = h + 0.15 * torch.tanh(z @ self.core.residual)
            h = F.dropout(h, 0.3, self.training)
            y = ((h * self.global_head.R[member, :7]) @ self.global_head.weight.T) * self.global_head.S[member] + self.global_head.B[member]
            y = F.dropout(y, 0.15, self.training)
            return torch.cat((y, y.new_zeros(ordinary.NODES - y.shape[0], ordinary.CLASSES)))
        def forward(self, x, edges): return torch.stack([self.forward_member(x, edges, m) for m in range(4)])
    x = value((15, 6), 0.3)
    data = {"features": x, "edge_index": torch.stack((torch.arange(15), torch.arange(15))),
            "training_ids": torch.arange(10), "training_targets": torch.arange(10) % 5}
    return Fixture, data


def initial_small_state(factory, helpers, device):
    """Nonzero synthetic Adam history, explicitly not a measured common400."""
    import torch
    model = factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    for p in model.parameters():
        optimizer.state[p] = {"step": torch.tensor(3.0), "exp_avg": 0.01 * torch.sin(torch.arange(p.numel()).reshape(p.shape).float()),
                              "exp_avg_sq": torch.full_like(p, 0.03)}
    return {"family_state": helpers._cpu_tree(model.state_dict()), "Adam": helpers._cpu_tree(optimizer.state_dict()),
            "rng": helpers._cpu_tree(helpers._rng(device))}


def independent_small_step(model, optimizer, data, kind, helpers, audit):
    """Direct standard mean CE plus explicit arithmetic softmax pool, no worker loss helper."""
    import torch
    import torch.nn.functional as F
    model.train(); model.set_global_stage(True); optimizer.zero_grad(set_to_none=True)
    logits = model(data["features"], data["edge_index"])[:, data["training_ids"]]
    own = sum(F.cross_entropy(logits[m], data["training_targets"]) for m in range(4)) / 4
    probabilities = F.softmax(logits, dim=-1).mean(dim=0)
    pool = -probabilities.gather(1, data["training_targets"][:, None]).log().mean()
    loss = own if kind == "own" else 0.5 * own + 0.5 * pool
    require(bool(torch.isfinite(loss)), "Nonfinite independent synthetic objective")
    audit.backward_attempts[kind] += 1; audit.event("independent_joint_backward_attempt")
    loss.backward()
    for n, p in model.named_parameters():
        if helpers._inactive(n, True): require(p.grad is None, "Dormant synthetic gradient appeared")
        else: require(p.grad is not None and bool(torch.isfinite(p.grad).all()), "Synthetic active gradient missing/nonfinite")
    audit.optimizer_attempts[kind] += 1; audit.event("independent_Adam_attempt")
    optimizer.step()
    return float(loss.detach())


def small_parity(ordinary, helpers, port, device, audit, check):
    import torch
    factory, data = small_fixture(ordinary, port)
    image = initial_small_state(factory, helpers, device)
    frozen = helpers._cpu_tree(image)
    results = {}
    for kind in OBJECTIVES:
        variants = {}
        for mode in ("independent", "joint", "streamed_exact"):
            audit.arm, audit.update, audit.case = kind, 1, "small_" + mode
            model = factory(); model.load_state_dict(image["family_state"], strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
            optimizer.load_state_dict(helpers._cpu_tree(image["Adam"]))
            ordinary.restore_rng(image["rng"], device)
            with ordinary.counted(model, audit):
                loss = independent_small_step(model, optimizer, data, kind, helpers, audit) if mode == "independent" else ordinary.step(
                    model, optimizer, data, kind, mode, helpers, audit, device)["optimized_loss"]
            variants[mode] = record(model, optimizer, helpers, device)
            compare(image, frozen, exact=True)
            del model, optimizer
        errors = {"joint_vs_independent": compare(variants["joint"], variants["independent"]),
                  "streamed_vs_independent": compare(variants["streamed_exact"], variants["independent"]),
                  "logical_RNG_joint_vs_streamed": compare(variants["joint"]["rng"], variants["streamed_exact"]["rng"], exact=True)}
        require(any(v is not None and bool(torch.count_nonzero(v)) for n, v in variants["joint"]["gradients"].items() if n.endswith(".weight")),
                "Synthetic shared gradients are trivial")
        require(any(v is not None and bool(torch.count_nonzero(v)) for n, v in variants["joint"]["gradients"].items() if n.endswith((".R", ".S", ".B"))),
                "Synthetic private gradients are trivial")
        results[kind] = errors
    return {"errors": results, "cached_synthetic_family_Adam_RNG_immutable": True,
            "physical_private_shapes_all_rows_compared": True, "CPU_FP32_logical_nodes": 15,
            "full_API_zero_padding_not_native_architecture": True, "synthetic_Adam_step_history": 3}


def native_replay(ordinary, helpers, accessor, native, boundary, scope, root, device, audit):
    import torch
    common_path, origin_path = ordinary.bound(root, scope["common_checkpoint"]), ordinary.bound(root, scope["common_origin_run"])
    require(common_path.name == "initial.pt" and origin_path.name == "RUN.json" and common_path.parent == origin_path.parent, "Exact common origin required")
    origin = json.loads(origin_path.read_text())
    require(origin["recipe"]["worker_source_sha256"] == ordinary.COMMON_ORIGIN_WORKER
        and origin["A_labels_received"] is False and origin["A_scoring_performed"] is False, "No old/all-TRAIN/selected state")
    public_b = (root / ordinary.PUBLIC_B_RELATIVE).resolve()
    preflight = helpers._public_identity_before_w(accessor, root, public_b, ordinary.INPUT_IDENTITY)
    image = torch.load(common_path, map_location="cpu", weights_only=False)
    require(image["schema"] == "amazon_G0_frozen_state_v1" and image["id"] == "initial" and image["warm_updates"] == 400
        and image["warm_role"] == "W" and image["episodes"] == 0 and image["global_stage"] is True and image["eval_mode"] is True
        and image["recipe"] == origin["recipe"] and image["construction"]["seed"] == 17
        and len(image["W_own_CE_trace"]) == 400, "Actual fixed W-only common400 required")
    helpers._finite_tree(image)
    frozen = helpers._cpu_tree(image)
    loaded = accessor.load_public_b(root, public_b, device=device)
    s, r = loaded["inner_indices"], loaded["query_indices"]
    require(s.numel() == 2449 and r.numel() == 2450 and not bool(torch.isin(s, r).any()), "Fixed role sizes/disjointness differ")
    ids, order = torch.sort(torch.cat((s, r))); targets = torch.cat((loaded["inner_labels"], loaded["query_labels"]))[order]
    require(ids.numel() == 4899 and not bool(torch.isin(ids, torch.cat((loaded["W_ids"], loaded["A_ids"]))).any()), "Only S/R objective labels allowed")
    require(loaded["provenance"]["preprocessing"] == preflight["preprocessing"], "Native public context changed")
    data = {"features": loaded["features"], "edge_index": loaded["edge_index"], "training_ids": ids, "training_targets": targets}
    del loaded, s, r, ids, order, targets
    inputs = {k: v.detach().clone() for k, v in data.items()}
    results = {}
    for kind in OBJECTIVES:
        records, replay_resources = [], []
        for repeat in range(2):
            audit.arm, audit.update, audit.case = kind, 1, "native_replay_" + str(repeat + 1)
            family, construction = helpers._fresh_family(native, boundary, device)
            family.load_state_dict(image["family_state"], strict=True); family.set_global_stage(True)
            optimizer = torch.optim.Adam(family.parameters(), lr=0.001, weight_decay=0.0)
            optimizer.load_state_dict(helpers._cpu_tree(image["Adam"]))
            ordinary.restore_rng(image["rng"], device)
            torch.cuda.synchronize(device); update_started = time.monotonic()
            with ordinary.counted(family, audit):
                stats = ordinary.step(family, optimizer, data, kind, "streamed_exact", helpers, audit, device)
            torch.cuda.synchronize(device)
            replay_resources.append({"repeat": repeat + 1, "streamed_update_elapsed_seconds": time.monotonic() - update_started,
                "whole_process_resources_after_update": ordinary.resources(device)})
            replay_resources[-1]["whole_process_resources_after_update"]["elapsed_seconds"] = time.monotonic() - STARTED
            records.append(record(family, optimizer, helpers, device))
            compare(image, frozen, exact=True)
            require(all(torch.equal(data[k], v) for k, v in inputs.items()), "Public/role inputs changed")
            del family, optimizer, construction
        error = compare(records[0], records[1]); compare(records[0]["rng"], records[1]["rng"], exact=True)
        require(any(v is not None and bool(torch.count_nonzero(v)) for n, v in records[0]["gradients"].items() if n not in
            {"stem.R", "stem.S", "stem.B", "local_head.R", "local_head.S", "local_head.B", "global_head.R", "global_head.S", "global_head.B"}),
            "Native shared update is trivial")
        results[kind] = {"complete_gradient_parameter_Adam_replay_max_error": error, "logical_RNG_replay_exact": True, "discarded_replay_resources": replay_resources, **stats}
    require(ordinary.sha(common_path) == scope["common_checkpoint"]["sha256"] and ordinary.sha(origin_path) == scope["common_origin_run"]["sha256"], "Common file bytes changed")
    return {"objectives": results, "cached_common_family_Adam_RNG_immutable": True,
            "complete_original_context_S_R_only": True, "virtual_Adam_updates_discarded": 4, "trained_state_saved": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true"); parser.add_argument("--source-root")
    parser.add_argument("--scope"); parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_NORMAL_ORDINARY_FIRST_ORDER_QUALIFICATION", "SOURCE_RELEASED": False, "numeric_imports": False})); return
    require(SOURCE_RELEASED is False, "Engineering caller keeps its disabled source marker")
    # Select77's literal named target before resolving any supplied path.
    literal = "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930"
    require(args.source_root == literal and socket.gethostname() == "peptide", "Exact normal77 qualification target required")
    require(args.scope and args.output, "Immutable root engineering scope and fresh output required")
    root = Path(args.source_root).resolve()
    ordinary, ordinary_name = load_ordinary(root)
    root, output, scope_path = ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root, output=args.output, admission=args.scope))
    require(scope_path.stat().st_mode & 0o222 == 0, "Immutable root qualification scope required")
    scope = json.loads(scope_path.read_text())
    require(scope["engineering_execution_authorized"] is True and scope["fit_authorized"] is False and scope["A_scoring"] is False
        and scope["VALID_TEST_access"] is False and scope["ordinary_source_review_approved"] is True
        and scope["ordinary_worker_sha256"] == ORDINARY[1] and scope["objectives"] == list(OBJECTIVES)
        and scope["implementation"] == "streamed_exact" and scope["qualifier_source_review_approved"] is True
        and scope["qualifier_worker_sha256"] == ordinary.sha(__file__)
        and all(scope[key] == row for key, row in COMMON_DESCRIPTORS.items()), "Actual fixed root engineering scope required")
    require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Fresh pre-Torch child required")
    caps = scope["resource_limits"]
    require(all(type(caps[k]) in (int, float) and math.isfinite(caps[k]) and caps[k] > 0 for k in
        ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes"))
        and type(scope["external_watchdog_seconds"]) in (int, float) and math.isfinite(scope["external_watchdog_seconds"])
        and scope["external_watchdog_seconds"] > caps["max_elapsed_seconds"]
        and type(scope["minimum_initial_cuda_free_bytes"]) is int and scope["minimum_initial_cuda_free_bytes"] > 0,
        "Root prospective limits/watchdog/free-memory floor required")
    require(isinstance(scope["CUDA_VISIBLE_DEVICES"], str) and scope["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
        and "," not in scope["CUDA_VISIBLE_DEVICES"]
        and os.environ.get("CUDA_VISIBLE_DEVICES") == scope["CUDA_VISIBLE_DEVICES"], "Root-pinned single physical GPU exposure required")
    owner = ordinary.CreatedOutput(output, root); token = owner.creator_token
    receipt = {"status": "RUNNING_NORMAL_ORDINARY_QUALIFICATION", "worker_sha256": ordinary.sha(__file__), "ordinary_source_pin": ORDINARY,
        "root_scope_sha256": ordinary.sha(scope_path), "checks": [], "model_fits": 0, "persistent_updates": 0,
        "A_scoring": False, "VALID_TEST_access": False, "implementation": "streamed_exact"}
    def save():
        receipt["elapsed_seconds_including_imports_setup_checks_restore"] = time.monotonic() - STARTED
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            row = ordinary.resources("cuda:0"); row["elapsed_seconds"] = time.monotonic() - STARTED
            receipt["whole_process_resources"] = row
        ordinary.atomic(owner.verify(token) / "RESULT.json", receipt)
    def check(name, fn):
        receipt["last_attempted_check"] = name; save(); value = fn(); receipt["checks"].append({"name": name, "result": value}); save(); return value
    class Meter(ordinary.Audit):
        case = "setup"
        def event(self, kind, **fields): super().event(kind, qualification_case=self.case, **fields)
    small_meter, native_meter = Meter(output), Meter(output)
    names, helpers, process = [], None, None
    old_path, python_rng = list(sys.path), random.getstate()
    old_env_present, old_env = "CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Fresh child timer required")
    backend_before = old_threads = rng_before = None
    code, success_metadata = 1, None
    try:
        def expired(signum, frame): raise TimeoutError("Root-frozen normal first-order qualification deadline exceeded")
        signal.signal(signal.SIGALRM, expired)
        remaining = caps["max_elapsed_seconds"] - (time.monotonic() - STARTED); require(remaining > 0, "No remaining root time")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        config = scope["process_configuration"]; os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["CUBLAS_WORKSPACE_CONFIG"]
        sys.path.insert(0, scope["runtime"]["site_packages"])
        import torch
        require(str(Path(torch.__file__).resolve()) == scope["runtime"]["torch_module_path"] and str(Path(sys.executable).resolve()) == scope["runtime"]["python_resolved"]
            and sys.dont_write_bytecode and not torch.cuda.is_initialized(), "Normal pinned runtime/-B/pre-CUDA required")
        process = ordinary.load(root, "process_helpers", names); process.torch = torch
        backend_before, old_threads = process.backend_snapshot(), torch.get_num_threads()
        torch.use_deterministic_algorithms(config["deterministic_algorithms_enabled"], warn_only=config["warn_only"])
        torch.set_num_threads(config["intra_op_threads"]); torch.set_num_interop_threads(config["interop_threads"])
        require(not torch.cuda.is_initialized(), "Qualified mode must precede CUDA")
        identity = process.runtime_identity(); require(identity == scope["expected_backend_runtime_metadata"]
            and torch.cuda.device_count() == 1, "Normal runtime/single selected GPU identity differs")
        free_bytes, total_bytes = torch.cuda.mem_get_info("cuda:0")
        require(free_bytes >= scope["minimum_initial_cuda_free_bytes"], "Selected GPU initial free-memory floor failed")
        receipt["initial_cuda_memory"] = {"free_bytes": free_bytes, "total_bytes": total_bytes,
            "required_free_bytes": scope["minimum_initial_cuda_free_bytes"]}
        helpers, accessor, native, boundary = (ordinary.load(root, k, names) for k in ("scientific_helpers", "accessor", "native", "boundary"))
        port_path = root / "learnability_responsibility_native_amazon_sparse_port_20261005_v1/native_sparse_port.py"
        require(ordinary.sha(port_path) == "a8ee35afda97afcb745c365a51f8e8647fe5e6a1350f654a5321c81e12abad86", "Physical private-shape source differs")
        spec = importlib.util.spec_from_file_location("_ordinary_qual_port_shapes", port_path); port = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = port; names.append(spec.name); spec.loader.exec_module(port)
        require(helpers.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True and port.PORT_RELEASED is False, "Source gates differ")
        rng_before = helpers._rng("cuda:0")
        check("small_stochastic_independent_joint_streamed_full_gradient_Adam_RNG", lambda: small_parity(ordinary, helpers, port, "cuda:0", small_meter, check))
        check("native_common400_two_discarded_update_replays_per_objective", lambda: native_replay(ordinary, helpers, accessor, native, boundary, scope, root, "cuda:0", native_meter))
        require(small_meter.callback_attempts == dict.fromkeys(OBJECTIVES, 16) and small_meter.backward_attempts == dict.fromkeys(OBJECTIVES, 6)
            and small_meter.small_logit_reverse_attempts == dict.fromkeys(OBJECTIVES, 1) and small_meter.optimizer_attempts == dict.fromkeys(OBJECTIVES, 3), "Complete synthetic attempt schedule differs")
        require(native_meter.callback_attempts == dict.fromkeys(OBJECTIVES, 16) and native_meter.backward_attempts == dict.fromkeys(OBJECTIVES, 8)
            and native_meter.small_logit_reverse_attempts == dict.fromkeys(OBJECTIVES, 2) and native_meter.optimizer_attempts == dict.fromkeys(OBJECTIVES, 2), "Complete native discarded replay schedule differs")
        success_metadata = dict(ordinary_first_order_supported=True, both_objectives_checked=True, same_checkpoint_context_and_RNG_checked=True,
            coupled_pool_gradient_and_Adam_update_parity_checked=True, stochastic_member_RNG_replay_checked=True,
            input_identity=ordinary.INPUT_IDENTITY, runtime={"hostname": socket.gethostname(), "python_resolved": str(Path(sys.executable).resolve()),
            "site_packages": scope["runtime"]["site_packages"], "torch_module_path": str(Path(torch.__file__).resolve())},
            process_configuration=config, backend_snapshot=process.backend_snapshot(), backend_runtime_metadata=identity)
    except BaseException as error:
        receipt.update(status="FAIL_NORMAL_ORDINARY_QUALIFICATION", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt["synthetic_attempted_counts"], receipt["native_attempted_counts"] = small_meter.snapshot(), native_meter.snapshot()
        errors = []
        def restore(name, fn):
            try: fn(); receipt[name] = True
            except BaseException as error: errors.append({"restoration": name, "error": str(error), "traceback": traceback.format_exc()})
        if rng_before is not None:
            def rng():
                ordinary.restore_rng(rng_before, "cuda:0")
                compare(helpers._rng("cuda:0"), rng_before, exact=True)
            restore("caller_RNG_restored", rng)
        if backend_before is not None: restore("backend_restored", lambda: process.backend_restore(backend_before))
        if old_threads is not None:
            def threads():
                torch.set_num_threads(old_threads)
                require(torch.get_num_threads() == old_threads, "Intra-op restoration differs")
            restore("reversible_threads_restored", threads)
        random.setstate(python_rng); sys.path[:] = old_path
        if old_env_present: os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env
        else: os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        signal.signal(signal.SIGALRM, old_handler); signal.setitimer(signal.ITIMER_REAL, *old_timer)
        for name in [*names, ordinary_name]: sys.modules.pop(name, None)
        restore("Python_path_environment_signal_state_restored", lambda: require(random.getstate() == python_rng
            and sys.path == old_path and ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == old_env_present
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == old_env
            and signal.getsignal(signal.SIGALRM) == old_handler and signal.getitimer(signal.ITIMER_REAL) == old_timer,
            "Reversible process state differs"))
        restore("original_sources_and_common_bytes_unchanged", lambda: require(all(ordinary.sha(root / p) == h for p, h in ordinary.PINS.values())
            and ordinary.sha(root / ORDINARY[0]) == ORDINARY[1], "Source bytes changed"))
        receipt["restoration_errors"] = errors
        receipt["interop_restoration"] = "One-time fresh-child initialization; not claimed reversible."
        if success_metadata is not None and not errors:
            try:
                row = ordinary.resources("cuda:0"); row["elapsed_seconds"] = time.monotonic() - STARTED
                require(all(row[m] <= caps[c] for m, c in (("elapsed_seconds", "max_elapsed_seconds"), ("process_peak_rss_bytes", "max_process_rss_bytes"),
                    ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"), ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes"))), "Root whole-process resource cap exceeded")
                receipt.update(success_metadata)
                receipt["status"] = "PASS_NORMAL_ORDINARY_FIRST_ORDER_STREAMED_RESOURCE_ONLY"; code = 0
            except BaseException as error: receipt.update(status="FAIL_NORMAL_ORDINARY_QUALIFICATION", error=str(error), traceback=traceback.format_exc())
        if code != 0: receipt["ordinary_first_order_supported"] = False
        try:
            save(); owner.freeze(token)
        except BaseException as error:
            code = 1
            receipt.update(status="FAIL_NORMAL_ORDINARY_QUALIFICATION", ordinary_first_order_supported=False,
                output_finalization_error_type=type(error).__name__, output_finalization_error=str(error),
                output_finalization_traceback=traceback.format_exc())
            try: save(); owner.freeze(token)
            except BaseException as recovery_error: receipt["output_failure_receipt_error"] = str(recovery_error)
    print(json.dumps({"status": receipt["status"], "error": receipt.get("error"), "model_fits": 0, "persistent_updates": 0, "output_finalization_error": receipt.get("output_finalization_error")}))
    raise SystemExit(code)


if __name__ == "__main__": main()
