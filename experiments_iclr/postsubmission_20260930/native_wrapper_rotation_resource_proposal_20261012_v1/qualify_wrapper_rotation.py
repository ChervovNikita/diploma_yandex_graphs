"""Ten-case common-wrapper/local-pair source proposal; root executes only.

Imports numerical APIs only after exact allocation identity and root receipt
checks. Reuses previously reviewed optimizer/role/commit denial helpers. A
fresh reviewed engineering receipt is required; this program cannot make GO.
No remote connection, fit registry, selection, evaluator or optimizer kernel.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

HOST = "anogena-2-0"
UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
ORIGINAL_FIVE = {"romanempire", "amazonratings", "minesweeper", "questions", "tolokers"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def verify_route():
    if socket.gethostname() != HOST:
        raise ValueError("Exact authorized allocation hostname required")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.split() != [UUID]:
        raise ValueError("Exact sole authorized allocation UUID required")


def local(root, relative):
    value = Path(relative)
    result = root / value
    if value.is_absolute() or ".." in value.parts or result.is_symlink() or not result.resolve().is_relative_to(root):
        raise ValueError("Confined non-symlink project path required")
    return result


def receipt(root, path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve().relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    path = local(root, row["path"])
    if receipt(root, path) != row:
        raise ValueError("Exact source/request/input binding changed")
    return path


def module(root, row, name):
    path = bound(root, row)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def state_hashes(torch, h, z, adapter):
    return {
        "parameters": h.tensor_digest(torch, adapter.model.named_parameters()),
        "moments": h.tensor_digest(torch, h.optimizer_tensors(adapter)),
        "buffers": h.tensor_digest(torch, adapter.model.named_buffers()),
        "external": z.tree_digest(torch, adapter.capture_state()),
    }


def sync(torch, model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_finite_gradients(torch, model):
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
        raise FloatingPointError("Nonfinite exact-source random-state gradient")


def node_probe(torch, h, z, root, case, study, run, output, admission):
    """Exact full factory with only unused SELECT authority explicitly replaced."""
    actual_inputs = run["config"]["node_inputs"]
    allowed = [bound(root, actual_inputs[key]) for key in ("graph", "train")]
    if run["config"].get("bootstrap_weights"):
        allowed.append(bound(root, run["config"]["bootstrap_weights"]))
    forbidden = [local(root, path) for role, value in study["data"]["roles"].items() if role != "train" for path in h.role_paths(value)]
    if not set(allowed).isdisjoint(forbidden):
        raise ValueError("Declared TRAIN inputs overlap forbidden role paths")
    runtime = local(root, admission["runtime_package"])
    sys.path.insert(0, str(runtime))
    with h.OptimizerDenial(torch) as denial, h.RoleIO(torch, root, allowed, forbidden, output) as io_guard:
        # This inherited helper reads only actual GRAPH/TRAIN, writes a one-row
        # unused engineering SELECT stub, and changes no frozen source/study.
        clone, engineering_run, stub = h.engineering_clone(torch, root, study, {"run": run}, output)
        io_guard.allowed.add(stub.resolve())
        clone["runs"] = [engineering_run]
        from run_study import load_factory
        factory = load_factory(root, engineering_run["adapter"])
        adapter = factory(torch=torch, project_root=root, study=clone, run=engineering_run, device="cuda:0")
        for name in ("validate", "observe_validation", "evaluate", "export"):
            if hasattr(adapter, name):
                setattr(adapter, name, lambda *a, **k: (_ for _ in ()).throw(h.ForbiddenRoleAccess("Selectors/targets/metrics prohibited")))
        h.allocate_adam_moments(torch, adapter)
        moments = [v for _, v in h.optimizer_tensors(adapter)]
        extra_workspace = [torch.empty_like(p) for p in adapter.model.parameters() for _ in range(2)]
        initial = copy.deepcopy(adapter.capture_state())
        modes = {name: module.training for name, module in adapter.model.named_modules()}
        before = state_hashes(torch, h, z, adapter)
        fixed_weights = getattr(adapter, "bootstrap_weights", None)
        fixed_hash = None if fixed_weights is None else h.tensor_digest(torch, [("fixed_weights", fixed_weights)])
        if len(adapter.train_ids) != case["train_rows"]:
            raise ValueError("Exact representative TRAIN shape changed")
        counts = {key: 0 for key in ("backward_calls", "autograd_grad_VJP_calls", "optimizer_attempts", "model_forwards",
            "native_body_forward_calls", "native_stem_calls", "native_residual0_calls", "native_residual1_calls",
            "native_graph_block_calls", "native_head_calls", "coupled_route_block_calls", "corrected_dense_map_calls",
            "Householder_reflection_calls")}
        cycle_counts = []
        times = []
        gradient_bytes = 0
        original_grad, original_backward = torch.autograd.grad, torch.Tensor.backward
        def counted_grad(*args, **kwargs):
            counts["autograd_grad_VJP_calls"] += 1
            return original_grad(*args, **kwargs)
        def counted_backward(value, *args, **kwargs):
            counts["backward_calls"] += 1
            return original_backward(value, *args, **kwargs)
        native_body = getattr(adapter.model, "body", adapter.model)
        if len(native_body.residual_modules) != 2:
            raise ValueError("Exact full two-residual native Model required")
        hooks = []
        def count(keys):
            def observed(*ignored):
                for key in keys: counts[key] += 1
            return observed
        # Exact exchange source uses four live native prefixes and four full
        # continuing tails; body.forward is intentionally not called there.
        # Count the complete path independently at the real native operators.
        hooks.append(native_body.register_forward_hook(count(("native_body_forward_calls",))))
        hooks.append(native_body.input_linear.register_forward_hook(count(("native_stem_calls",))))
        for index, residual in enumerate(native_body.residual_modules):
            hooks.append(residual.register_forward_hook(count(("native_residual" + str(index) + "_calls", "native_graph_block_calls"))))
        hooks.append(native_body.output_linear.register_forward_hook(count(("native_head_calls", "model_forwards"))))
        block = getattr(adapter.model, "block", None)
        if block is not None: hooks.append(block.register_forward_hook(count(("coupled_route_block_calls",))))
        corrected = [module for module in native_body.modules() if hasattr(module, "u1") and hasattr(module, "u2")]
        def count_correction(*ignored):
            counts["corrected_dense_map_calls"] += 1
            # Literal CorrectedLinear.forward contains exactly two Householder
            # applications; this count is source-formula derived per real call.
            counts["Householder_reflection_calls"] += 2
        for corrected_map in corrected: hooks.append(corrected_map.register_forward_hook(count_correction))
        if ((case["family"] == "common_native_wrapper" and (block is None or corrected))
                or (case["family"] == "private_rotation" and (block is not None or len(corrected) != 2))):
            raise ValueError("Exact exchange or two local-pair source placement required")
        with ExitStack() as stack:
            for hook in hooks: stack.callback(hook.remove)
            stack.enter_context(mock.patch.object(torch.autograd, "grad", counted_grad))
            stack.enter_context(mock.patch.object(torch.Tensor, "backward", counted_backward))
            for index in range(4):
                adapter.restore_state(copy.deepcopy(initial))
                for name, module in adapter.model.named_modules():
                    module.training = modes[name]
                counts.update({key: 0 for key in counts})
                attempts = denial.denied_step_calls
                sync(torch, adapter.model)
                tick = time.perf_counter()
                with z.TensorCommitDenial(torch, list(adapter.model.parameters()), moments) as commits:
                    ignored = adapter.train_step(index + 1)
                    del ignored  # No objective/component/nominal-update export.
                sync(torch, adapter.model)
                times.append(time.perf_counter() - tick)
                counts["optimizer_attempts"] = denial.denied_step_calls - attempts
                if commits.denied or denial.denied_functional_calls:
                    raise ValueError("Protected source action attempted")
                require_finite_gradients(torch, adapter.model)
                gradient_bytes = max(gradient_bytes, sum(p.grad.numel() * p.grad.element_size() for p in adapter.model.parameters() if p.grad is not None))
                expected = case["expected_actions_per_cycle"]
                if any(counts[key] != expected[key] for key in counts):
                    raise ValueError("Source backward/VJP/owner counts differ")
                cycle_counts.append(dict(counts))
                current = state_hashes(torch, h, z, adapter)
                if current["parameters"] != before["parameters"] or current["moments"] != before["moments"]:
                    raise ValueError("Parameter or manual-moment bytes changed")
        adapter.restore_state(copy.deepcopy(initial))
        for name, module in adapter.model.named_modules():
            module.training = modes[name]
        if before != state_hashes(torch, h, z, adapter):
            raise ValueError("Initial source state not exactly restored")
        # Restore source-derived counters/modes/streams after label-free serving.
        adapter.model.eval()
        sync(torch, adapter.model)
        tick = time.perf_counter()
        with torch.no_grad():
            logits = adapter.logits_fn(adapter.model, False, 0)
            if logits.ndim != 3 or not torch.isfinite(logits).all().item():
                raise ValueError("Finite full-graph source serving shape required")
            serving_shape = list(logits.shape)
            serving_cpu_bytes = logits.numel() * logits.element_size()
            del logits
        sync(torch, adapter.model)
        serving_seconds = time.perf_counter() - tick
        adapter.restore_state(copy.deepcopy(initial))
        for name, module in adapter.model.named_modules():
            module.training = modes[name]
        from engine import capture_checkpoint, restore_checkpoint, _atomic_checkpoint
        if Path(sys.modules["engine"].__file__).resolve() != runtime / "engine.py":
            raise ValueError("Frozen engine runtime differs")
        tick = time.perf_counter()
        payload = capture_checkpoint(torch, adapter, fit_id="ENGINEERING_RANDOM_UNTRAINED_UNSELECTED", step=0, validation={}, decision={})
        clone_seconds = time.perf_counter() - tick
        snapshot = output / "DISPOSABLE_RANDOM_ENGINEERING_STATE.pt"
        io_guard.allowed.add(snapshot.resolve())
        tick = time.perf_counter()
        size = _atomic_checkpoint(torch, snapshot, payload)
        save_seconds = time.perf_counter() - tick
        saved = receipt(root, snapshot)
        del payload
        # Move only disposable runtime state; no parameter/moment mutation.
        torch.rand(3, device="cuda:0")
        adapter.model.eval()
        tick = time.perf_counter()
        restore_checkpoint(torch, snapshot, adapter, expected_fit_id="ENGINEERING_RANDOM_UNTRAINED_UNSELECTED", verify=True)
        restore_seconds = time.perf_counter() - tick
        if before != state_hashes(torch, h, z, adapter):
            raise ValueError("Engine model/moment/source/private RNG restore mismatch")
        if fixed_weights is not None and fixed_hash != h.tensor_digest(torch, [("fixed_weights", fixed_weights)]):
            raise ValueError("Fixed untrained table changed")
        snapshot.unlink()
        role_shapes = admission.get("node_bank_role_shapes", {}).get(case["dataset"])
        CPU_bank = {"qualified": False, "scope": "root metadata role counts not supplied"}
        if role_shapes is not None:
            if (set(role_shapes) != {"cal", "evaluate", "custody_receipt"}
                    or any(type(role_shapes[key]) is not int or role_shapes[key] < 1 for key in ("cal", "evaluate"))
                    or role_shapes["cal"] + role_shapes["evaluate"] > case["model_graph_shape"]["nodes"]):
                raise ValueError("Finite shape-only CPU bank role counts required")
            custody = json.loads(bound(root, role_shapes["custody_receipt"]).read_text())
            expected_shapes = {role: max(row["role_counts"][role] for row in custody["splits"]) for role in ("cal", "evaluate")}
            if custody.get("dataset_id") != case["dataset"] or any(expected_shapes[role] != role_shapes[role] for role in expected_shapes):
                raise ValueError("CPU bank shape counts differ from frozen dataset custody metadata")
            # Mirror sequential genuine-I4 export's four retained CPU route
            # tensors and cat transient. Dummy values are never logits/evidence.
            views = {role: [torch.zeros((1, role_shapes[role], case["shape_config"]["classes"]), dtype=torch.float32) for _ in range(4)]
                     for role in ("cal", "evaluate")}
            banks = {role: torch.cat(value, dim=0) for role, value in views.items()}
            coordinates = sum(value.numel() for value in banks.values())
            bank_path = output / "DISPOSABLE_CPU_BANK_SHAPE.pt"
            io_guard.allowed.add(bank_path.resolve())
            tick = time.perf_counter()
            torch.save({"engineering_dummy_shape_only": True, "role_tensors": banks}, bank_path)
            bank_save_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            loaded = torch.load(bank_path, map_location="cpu", weights_only=True)
            bank_load_seconds = time.perf_counter() - tick
            if not loaded["engineering_dummy_shape_only"] or any(not torch.equal(banks[key], loaded["role_tensors"][key]) for key in banks):
                raise ValueError("Dummy CPU shape serializer roundtrip differs")
            bank_receipt = receipt(root, bank_path)
            bank_path.unlink()
            CPU_bank = {"qualified": True, "dummy_values_no_predictions": True, "roles": expected_shapes, "served_routes": 4,
                        "retained_route_logits_bytes": 4 * coordinates, "cat_transient_bytes": 4 * coordinates,
                        "snapshot": bank_receipt, "save_seconds": bank_save_seconds, "load_seconds": bank_load_seconds, "removed": True,
                        "actual_endpoint_ID_or_target_access": False}
            del loaded, views, banks
        stub_receipt = receipt(root, stub)
        stub.unlink()
        result = {"cold_callback_seconds": times[0], "warm_callback_seconds": times[1:], "warm_median_seconds": statistics.median(times[1:]),
                  "cycle_action_counts": cycle_counts, "source_state_hashes_before": before, "source_state_hashes_after": state_hashes(torch, h, z, adapter),
                  "full_graph_serving_shape": serving_shape, "full_graph_serving_seconds": serving_seconds, "full_graph_logits_CPU_bytes": serving_cpu_bytes,
                  "parameter_bytes": sum(p.numel() * p.element_size() for p in adapter.model.parameters()),
                  "gradient_bytes": gradient_bytes,
                  "manual_moment_bytes": sum(v.numel() * v.element_size() for v in moments),
                  "extra_workspace_bytes": sum(v.numel() * v.element_size() for v in extra_workspace),
                  "fixed_table_hash": fixed_hash, "input_load_paths": io_guard.tensor_load_paths, "denied_role_reads": io_guard.denied,
                  "unused_SELECT_stub": {**stub_receipt, "removed": True, "synthetic_one_row_never_validated": True},
                  "random_engine_restore": {"snapshot": saved, "bytes": size, "clone_seconds": clone_seconds, "save_seconds": save_seconds,
                                            "load_and_verify_restore_seconds": restore_seconds, "removed": True, "selected_checkpoint": False},
                  "optimizer_attempts": denial.denied_step_calls, "optimizer_functional_attempts": denial.denied_functional_calls,
                  "original_optimizer_kernels": denial.actual_optimizer_kernels, "actual_optimizer_updates": 0,
                  "CPU_bank_export_proxy": CPU_bank, "real_CAL_evaluate_export": False}
        return result


def ogb_probe(torch, h, z, root, case, run, output, admission):
    """Exact native GNN/train AST on three sealed TRAIN-only batch packs."""
    import ast
    import random
    import numpy as np
    from torch_geometric.data import Batch, Data
    source_files = {Path(row["path"]).name: row for row in run["cli"]["source_files"]}
    source_dir = bound(root, source_files["gnn.py"]).parent
    for name in ("main_pyg.py", "gnn.py", "conv.py"):
        bound(root, source_files[name])
    packs = admission.get("TRAIN_packs", {}).get(case["case_id"])
    if not isinstance(packs, dict) or set(packs) != {"top_nodes", "top_edges", "tail"}:
        raise ValueError("Three frozen TRAIN-only pack receipts required")
    allowed = [bound(root, row) for row in packs.values()]
    seed = case["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    sys.path.insert(0, str(source_dir))
    native = module(root, source_files["gnn.py"], "engineering_pinned_ogb_gnn")
    if Path(sys.modules["conv"].__file__).resolve() != source_dir / "conv.py":
        raise ValueError("Exact pinned native conv module required")
    constructor = case["constructor"]
    tree = ast.parse(bound(root, source_files["main_pyg.py"]).read_bytes())
    train_ast = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"]
    if len(train_ast) != 1:
        raise ValueError("One pinned native train callback required")
    ns = {"torch": torch, "cls_criterion": torch.nn.BCEWithLogitsLoss(), "reg_criterion": torch.nn.MSELoss(),
          "tqdm": lambda iterable, **ignored_display: iterable}
    exec(compile(ast.Module(body=train_ast, type_ignores=[]), str(source_dir / "main_pyg.py") + "::exact_train_AST", "exec"), ns)
    callback_AST = hashlib.sha256(ast.dump(train_ast[0]).encode()).hexdigest()
    with h.OptimizerDenial(torch) as denial, h.RoleIO(torch, root, allowed, [], output) as io_guard:
        model = native.GNN(gnn_type=constructor["gnn_type"], num_tasks=1, num_layer=5, emb_dim=300, drop_ratio=.5,
                           virtual_node=constructor["virtual_node"]).to("cuda:0")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        adapter = SimpleNamespace(model=model, optimizer=optimizer)
        h.allocate_adam_moments(torch, adapter)
        moments = [v for _, v in h.optimizer_tensors(adapter)]
        workspace = [torch.empty_like(p) for p in model.parameters() for _ in range(2)]
        initial_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
        initial_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        p0 = h.tensor_digest(torch, model.named_parameters())
        m0 = h.tensor_digest(torch, h.optimizer_tensors(adapter))
        b0 = h.tensor_digest(torch, model.named_buffers())
        observations = []
        def restore_initial():
            model.load_state_dict(initial_model, strict=True)
            torch.set_rng_state(initial_rng[0]);torch.cuda.set_rng_state(initial_rng[1])
            model.train()
        for pack_name in ("top_nodes", "top_edges", "tail"):
            pack = torch.load(bound(root, packs[pack_name]), map_location="cpu", weights_only=True)
            expected = 32 if pack_name != "tail" else case["TRAIN_pack_policy"]["tail_graphs"]
            if (set(pack) != {"schema", "dataset", "role", "pack_name", "source_bindings", "graphs"}
                    or pack["schema"] != "ogb-native-TRAIN-engineering-pack-v1" or pack["dataset"] != case["dataset"]
                    or pack["role"] != "TRAIN" or pack["pack_name"] != pack_name or len(pack["graphs"]) != expected):
                raise ValueError("Exact sealed native TRAIN pack schema/count required")
            if (not pack["source_bindings"] or pack["source_bindings"] != admission.get("TRAIN_pack_source_bindings", {}).get(case["dataset"])
                    or admission.get("TRAIN_pack_custodian_verified", {}).get(case["dataset"]) is not True):
                raise ValueError("Official TRAIN raw/provider custody required")
            data = []
            for row in pack["graphs"]:
                if set(row) != {"graph_id", "x", "edge_index", "edge_attr", "y"} or type(row["graph_id"]) is not int:
                    raise ValueError("Native TRAIN graph tensor/identity schema changed")
                if (row["x"].dtype != torch.long or row["x"].ndim != 2 or row["x"].shape[1] != 9
                        or row["edge_index"].dtype != torch.long or row["edge_index"].shape[0] != 2
                        or row["edge_attr"].dtype != torch.long or row["edge_attr"].shape != (row["edge_index"].shape[1], 3)
                        or tuple(row["y"].shape) != (1, 1)):
                    raise ValueError("Native full atom/bond/binary graph schema required")
                data.append(Data(x=row["x"], edge_index=row["edge_index"], edge_attr=row["edge_attr"], y=row["y"]))
            if len({row["graph_id"] for row in pack["graphs"]}) != expected:
                raise ValueError("No repeated graph in a representative TRAIN pack")
            batch = Batch.from_data_list(data)
            if batch.x.shape[0] == 1 or batch.batch[-1].item() == 0:
                raise ValueError("Representative source pack must exercise native non-skipped train callback")
            times = [];bn_changes = []
            for _ in range(4):
                restore_initial();attempts = denial.denied_step_calls
                sync(torch, model);tick = time.perf_counter()
                with z.TensorCommitDenial(torch, list(model.parameters()), moments) as commits:
                    ns["train"](model, torch.device("cuda:0"), [batch], optimizer, "classification")
                sync(torch, model);times.append(time.perf_counter() - tick)
                require_finite_gradients(torch, model)
                if (commits.denied or denial.denied_functional_calls or denial.denied_step_calls - attempts != 1
                        or p0 != h.tensor_digest(torch, model.named_parameters()) or m0 != h.tensor_digest(torch, h.optimizer_tensors(adapter))):
                    raise ValueError("Exact native callback/zero-step equality failed")
                changed = [name for name, value in model.named_buffers() if not torch.equal(value, initial_model[name])]
                if any(not name.endswith(("running_mean", "running_var", "num_batches_tracked")) for name in changed):
                    raise ValueError("Unexpected non-BatchNorm native buffer mutation")
                bn_changes.append(changed)
            restore_initial()
            if h.tensor_digest(torch, model.named_buffers()) != b0:
                raise ValueError("Native BatchNorm buffers not restored")
            model.eval();sync(torch, model);tick = time.perf_counter()
            with torch.no_grad():
                predictions = model(batch.to("cuda:0"))
                if tuple(predictions.shape) != (expected, 1) or not torch.isfinite(predictions).all().item():
                    raise ValueError("Native label-free serving shape differs")
                del predictions
            sync(torch, model);serving = time.perf_counter() - tick
            observations.append({"pack_name": pack_name, "graphs": expected, "nodes": int(batch.x.shape[0]), "edges": int(batch.edge_index.shape[1]),
                                 "cold_seconds": times[0], "warm_seconds": times[1:], "warm_median_seconds": statistics.median(times[1:]),
                                 "BatchNorm_buffer_names_changed_during_native_train": bn_changes,
                                 "BatchNorm_buffers_restored": True, "serving_seconds": serving})
            del batch, data, pack
        restore_initial()
        if p0 != h.tensor_digest(torch, model.named_parameters()) or m0 != h.tensor_digest(torch, h.optimizer_tensors(adapter)):
            raise ValueError("Native random-state parameter/moment mutation")
        return {"native_train_AST_sha256": callback_AST, "packs": observations, "optimizer_attempts": denial.denied_step_calls,
                "actual_optimizer_updates": 0, "original_optimizer_kernels": 0, "source_parameter_hash_unchanged": p0,
                "manual_moment_hash_unchanged": m0, "BatchNorm_buffer_hash_restored": b0,
                "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "manual_moment_bytes": sum(v.numel() * v.element_size() for v in moments),
                "extra_workspace_bytes": sum(v.numel() * v.element_size() for v in workspace),
                "native_selected_model_checkpoint_exists": False, "native_processed_cache_resource_qualified": False,
                "native_VALID_TEST_loader_geometry_qualified": False, "input_load_paths": io_guard.tensor_load_paths,
                "denied_role_reads": io_guard.denied, "all_epoch_ETA": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify_route()  # before any project reads/writes
    root = Path(args.project_root).resolve()
    output = local(root, args.output)
    admission_path = local(root, args.admission)
    admission = json.loads(admission_path.read_text())
    self_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (admission.get("schema") != "root-approved-zero-update-resource-admission-v1"
            or admission.get("root_source_review_confirmed") is not True
            or admission.get("helper_sha256") != self_hash or admission.get("scientific_GO") is not False
            or args.case_id not in admission.get("approved_case_ids", [])
            or admission.get("max_actual_optimizer_updates") != 0):
        raise ValueError("Fresh root exact-source engineering admission required; no scientific GO")
    proposal = json.loads(bound(root, admission["proposal_receipt"]).read_text())
    for dependency in admission["runtime_dependency_receipts"] + admission["source_dependency_receipts"]:
        bound(root, dependency)
    case = next(c for c in proposal["requested_cases"] if c["case_id"] == args.case_id)
    study_row = admission["case_study_receipts"][args.case_id]
    raw = bound(root, study_row).read_bytes()
    study = json.loads(gzip.decompress(raw))
    if study["dataset"].lower().replace("_", "").replace("-", "") in ORIGINAL_FIVE:
        raise ValueError("Original-paper datasets excluded for every method")
    run = next(r for r in study["runs"] if r["run_id"] == case["source_run_id"])
    if sha(run) != case["source_run_hash"]:
        raise ValueError("Exact prospective source run changed")
    if output.exists():
        raise ValueError("Fresh output required; failures never overwritten or silently retried")
    # New helper's imports/review receipts explicitly bind inherited boundaries.
    h = module(root, admission["role_optimizer_cap_helper"], "new_source_inherited_role_optimizer_cap")
    z = module(root, admission["tensor_commit_tree_helper"], "new_source_inherited_tensor_commit_tree")
    timeout = case["timeout_seconds"]
    old = signal.getsignal(signal.SIGALRM)
    def timed_out(*ignored):
        raise h.ResourceTimeout("Declared source engineering child deadline")
    signal.signal(signal.SIGALRM, timed_out);signal.alarm(timeout)
    output.mkdir(parents=True)
    owner = h.own_identity();started = time.monotonic()
    (output / "START.json").write_text(json.dumps({"case_id": args.case_id, "owner": owner, "admission": receipt(root, admission_path),
                                                 "helper_sha256": self_hash, "timeout_seconds": timeout, "scientific_fits": 0, "GO": False}, indent=2) + "\n")
    try:
        import torch
        torch.set_num_threads(2)
        cap, before_usage, after_init = h.configure_cuda_allocator(torch, output)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        if case["family"] not in ("common_native_wrapper", "private_rotation"):
            raise ValueError("Only the ten exact proposed wrapper/local-pair cases are admitted")
        result = node_probe(torch, h, z, root, case, study, run, output, admission)
        result.update(schema="new-source-zero-update-resource-result-v1", status="RANDOM_SOURCE_ZERO_UPDATE_ENGINEERING_COMPLETE", case_id=args.case_id,
                      owner=owner, constructor_and_probe_seconds=time.monotonic()-tick,
                      elapsed_wall_seconds=time.monotonic()-started, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(), current_child_allocator_cap=cap,
                      GPU_usage_before_init=before_usage, GPU_usage_after_init=after_init, GPU_usage_at_completion=h.GPU_process_usage(),
                      process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      scientific_fits=0, GO_created=False, scores_emitted=0, loss_values_emitted=0,
                      heldout_target_files_opened=0, actual_optimizer_updates=0, whole_master_admitted=False)
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"case_id": args.case_id, "status": result["status"], "actual_optimizer_updates": 0, "scientific_fits": 0}))
    except BaseException as error:
        failure = h.terminal_failure(error, owner, started)
        failure.update(case_id=args.case_id, helper_sha256=self_hash, GO_created=False,
                       optimizer_updates_authorized=0, automatic_retry=False)
        (output / "FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    main()
