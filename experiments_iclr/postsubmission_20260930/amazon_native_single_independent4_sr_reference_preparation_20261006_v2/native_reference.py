"""Disabled fresh competent native SINGLE/independent4 W400 then SR2300.

SINGLE is fixed native member0/seed17 alias. Every member has its own fresh
parameters, empty Adam, W-only acquisition and exact uninterrupted history.
No copied shared bank, old checkpoint, selector, retry, held scoring or launch.
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
import resource
import signal
import socket
import sys
import traceback
from types import SimpleNamespace

SOURCE_RELEASED = False
TARGET = "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930"
MEMBER_SEEDS = (17, 1026, 2035, 3044)
W_UPDATES, W_LOCAL, SR_UPDATES = 400, 200, 2300
NODES, FEATURES, CLASSES = 24492, 300, 5
ORDINARY_UTILITIES = ("amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py", "5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b")
PINS = {
    "native": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py", "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
    "scientific_helpers": ("amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3/six_arm_worker.py", "f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1"),
    "accessor": ("amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py", "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85"),
    "process": ("amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py", "65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb")}
PUBLIC_B_RELATIVE = "learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b"
INPUT_IDENTITY = {"public_graph_sha256": "19757299bcfd9e493e9ceae9e73248753ab1e773ccc1f6b57c6fadf8c843310f", "roles_sha256": "9cab6f2cf24dbecee59a2179e860d78b9b7b995adf0bb5347ca16dce86f32c97", "native_edge_logical_sha256": "229a8a787ef9120a4d7a1dcdd6481b973619a1c7f44a4abe9ed69d05c256e550", "public_b_manifest_sha256": "da03b6c14615da934443c9b8f36b68f9d643e2b23e04ebe17dab7dacf630df55"}


def require(value, message):
    if not value: raise RuntimeError(message)


def load(root, row, name, names):
    path = root / row[0]
    require(hashlib.sha256(path.read_bytes()).hexdigest() == row[1] and name not in sys.modules, "Exact fresh source binding required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; names.append(name)
    spec.loader.exec_module(module)
    return module


def restore_rng(value, device):
    import numpy as np
    import torch
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(value["torch_cpu"].cpu())
    require(value["cuda"] is not None and device == "cuda:0", "Qualified native CUDA RNG required")
    torch.cuda.set_rng_state(value["cuda"].cpu(), device)


def equal(actual, expected):
    """Exact immutable-state comparison; never a training loss or metric."""
    import torch
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(actual.detach().cpu(), expected.detach().cpu()), "Exact own-state/RNG differs")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Own-state keys differ")
        for key in expected: equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), "Own-state sequence differs")
        for a, b in zip(actual, expected): equal(a, b)
    else: require(actual == expected, "Own-state scalar differs")


def optimizer_ownership(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    require(len(optimizer.param_groups) == 1, "One native Adam group required")
    group = optimizer.param_groups[0]; parameters = group["params"]
    require(len(parameters) == len(names) and len({id(p) for p in parameters}) == len(parameters)
        and {id(p) for p in parameters} == set(names) and all(p.requires_grad for p in parameters)
        and group["lr"] == 0.001 and group["weight_decay"] == 0.0 and group["betas"] == (0.9, 0.999)
        and group["eps"] == 1e-8 and group["amsgrad"] is False, "Exact complete native Adam ownership/defaults required")
    return [names[id(p)] for p in parameters]


def independent_ownership(models, optimizers):
    """Every propagation/readout parameter and every live Adam tensor is private."""
    import torch
    require(len(models) == len(optimizers) == 4 and len({id(m) for m in models}) == 4
        and len({id(o) for o in optimizers}) == 4, "Four distinct native models and optimizers required")
    parameter_objects, parameter_storage, state_objects, state_storage = set(), set(), set(), set()
    for model, optimizer in zip(models, optimizers):
        optimizer_ownership(model, optimizer)
        require(not hasattr(model, "core") and not hasattr(model, "forward_member"), "Fully independent native propagation required")
        for p in model.parameters():
            storage = (str(p.device), p.untyped_storage().data_ptr())
            require(id(p) not in parameter_objects and storage not in parameter_storage, "Cross-member parameter/storage sharing")
            parameter_objects.add(id(p)); parameter_storage.add(storage)
        for entry in optimizer.state.values():
            require(id(entry) not in state_objects, "Cross-member Adam entry sharing")
            state_objects.add(id(entry))
            for tensor in entry.values():
                if isinstance(tensor, torch.Tensor):
                    storage = (str(tensor.device), tensor.untyped_storage().data_ptr())
                    require(storage not in state_storage and storage not in parameter_storage, "Cross-member Adam tensor/storage sharing")
                    state_storage.add(storage)
    require(parameter_storage.isdisjoint(state_storage), "Adam tensors alias model parameters")


def fresh(native, helpers, seed, device):
    """Exact V6 native seed/constructor/device/reset/Adam order; no boundary wrap."""
    import numpy as np
    import torch
    require(seed in MEMBER_SEEDS and device == "cuda:0" and torch.get_default_dtype() == torch.float32
        and str(torch.empty(0).device) == "cpu", "Fixed native acquisition defaults required")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    construction = {"seed": seed, "after_seed": helpers._rng(device)}
    model = native.Polynormer(300, 256, 5, local_layers=10, global_layers=1,
        in_dropout=0.2, dropout=0.3, global_dropout=0.3, heads=2, beta=-1, pre_ln=False)
    construction["after_constructor_cpu"] = helpers._rng(device)
    model.to(device); model.reset_parameters(); model._global = False
    construction["after_device_reset"] = helpers._rng(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    optimizer_ownership(model, optimizer)
    require(not optimizer.state, "Each native member must start with genuinely empty Adam history")
    construction["after_Adam_constructor"] = helpers._rng(device)
    require(type(model) is native.Polynormer and len(model.local_convs) == 10 and model.global_attn.num_layers == 1
        and model.global_attn.hidden_channels == 256 and model.global_attn.heads == 2 and model.global_attn.qk_shared is True
        and model.beta == -1 and model.pre_ln is False and model.in_drop == 0.2
        and model.dropout == 0.3 and model.global_attn.dropout == 0.3, "Complete capable nonlinear native architecture required")
    return model, optimizer, helpers._cpu_tree(construction)


def inactive(name, global_stage):
    return name.startswith("pred_local.") if global_stage else name.startswith(("global_attn.", "ln.", "pred_global."))


def history(model, optimizer, absolute_update):
    """Exact member-specific native clocks, including structurally dormant heads."""
    require(absolute_update in (W_UPDATES, W_UPDATES + SR_UPDATES), "Fixed history boundaries required")
    optimizer_ownership(model, optimizer)
    for name, parameter in model.named_parameters():
        expected = W_LOCAL if name.startswith("pred_local.") else (
            absolute_update - W_LOCAL if name.startswith(("global_attn.", "ln.", "pred_global.")) else absolute_update)
        require(parameter in optimizer.state and float(optimizer.state[parameter]["step"].item()) == expected,
            "Independently acquired native Adam clock differs: " + name)


class Audit:
    def __init__(self, output):
        self.output, self.member, self.phase, self.update = output, None, "setup", 0
        self.counts = {str(m): {phase: {key: 0 for key in ("callback_attempts", "backward_attempts", "Adam_attempts", "completed_updates")}
            for phase in ("W", "S_R")} for m in range(4)}
        self.serving_callbacks = 0
        self.checkpoint_write_attempts, self.checkpoint_completed_writes = 0, 0
        self.checkpoint_write_and_hash_elapsed_seconds = 0.0
    def event(self, kind, **fields):
        with (self.output / "ATTEMPTED_OPERATIONS.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, "member": self.member, "phase": self.phase, "update": self.update,
                "elapsed_seconds": time.monotonic() - STARTED, **fields}, allow_nan=False) + "\n")
    def attempt(self, key, kind):
        self.counts[str(self.member)][self.phase][key] += 1; self.event(kind)
    def snapshot(self): return json.loads(json.dumps(self.counts))


def step(model, optimizer, data, audit, helpers):
    """One complete train-mode native trajectory, own mean CE, one native Adam."""
    import torch
    import torch.nn.functional as F
    model.train(); optimizer.zero_grad(set_to_none=True)
    audit.attempt("callback_attempts", "complete_native_forward_attempt")
    logits = model(data["features"], data["edge_index"])
    require(logits.shape == (NODES, CLASSES) and logits.dtype == torch.float32, "Full native FP32 logits required")
    helpers._finite(logits, "Nonfinite native logits")
    loss = F.nll_loss(F.log_softmax(logits, dim=1).index_select(0, data["training_ids"]), data["training_targets"])
    helpers._finite(loss, "Nonfinite native own CE")
    audit.attempt("backward_attempts", "native_own_CE_backward_attempt"); loss.backward()
    for name, parameter in model.named_parameters():
        if inactive(name, model._global): require(parameter.grad is None, "Dormant native gradient appeared")
        else:
            require(parameter.requires_grad and parameter.grad is not None, "Active native parameter disconnected: " + name)
            helpers._finite(parameter.grad, "Nonfinite native gradient")
    audit.attempt("Adam_attempts", "native_member_Adam_attempt"); optimizer.step()
    for p in model.parameters(): helpers._finite(p, "Nonfinite native parameter")
    for entry in optimizer.state.values():
        for value in entry.values():
            if isinstance(value, torch.Tensor): helpers._finite(value, "Nonfinite native Adam state")
    return float(loss.detach())


def state(model, optimizer, helpers, device):
    return {"model": helpers._cpu_tree(model.state_dict()), "Adam": helpers._cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": optimizer_ownership(model, optimizer), "rng": helpers._cpu_tree(helpers._rng(device)),
        "global_stage": bool(model._global), "eval_mode": not model.training}


def checkpoint(output, kind, member, snapshot, construction, recipe, ordinary, audit):
    import torch
    name = "member" + str(member) + "_" + kind + ".pt"
    payload = {"schema": "fresh_native_W400_SR2300_reference_state_v1", "member": member, "seed": MEMBER_SEEDS[member],
        "kind": kind, "state": snapshot, "construction": construction, "recipe": recipe,
        "W_updates": W_UPDATES, "SR_updates": 0 if kind == "W400" else SR_UPDATES,
        "absolute_update": W_UPDATES if kind == "W400" else W_UPDATES + SR_UPDATES,
        "last_fixed_horizon": True, "A_labels_received": False, "VALID_TEST_access": False}
    path = output / name
    audit.checkpoint_write_attempts += 1
    audit.event("native_checkpoint_write_attempt", checkpoint_kind=kind, filename=name)
    started = time.monotonic()
    try:
        with path.open("xb") as stream: torch.save(payload, stream)
        path.chmod(0o444)
        row = {"path": name, "bytes": path.stat().st_size, "sha256": ordinary.sha(path)}
        audit.checkpoint_completed_writes += 1
        audit.event("native_checkpoint_frozen", checkpoint_kind=kind, **row)
        return row
    finally:
        audit.checkpoint_write_and_hash_elapsed_seconds += time.monotonic() - started


def served_probabilities(logits):
    """Fixed final native SINGLE/member0 and uniform arithmetic probability ENS4.

    Pure serving reduction for a separately admitted evaluator. No labels,
    metric, selection, routing, member drop, temperature or logits averaging.
    """
    import torch
    require(isinstance(logits, (list, tuple)) and len(logits) == 4
        and all(z.shape == (NODES, CLASSES) and z.dtype == torch.float32 and z.device == logits[0].device
            and bool(torch.isfinite(z).all()) for z in logits), "Four complete final native trajectories required")
    probabilities = torch.stack([torch.softmax(z, dim=-1) for z in logits])
    return {"SINGLE": probabilities[0], "ENS4": probabilities.mean(dim=0)}


def resources(device):
    import torch
    return {"elapsed_seconds": time.monotonic() - STARTED,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def limits(caps, device):
    row = resources(device)
    require(all(row[m] <= caps[c] for m, c in (("elapsed_seconds", "max_elapsed_seconds"),
        ("process_peak_rss_bytes", "max_process_rss_bytes"), ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"),
        ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes"))), "Fresh native whole-process resource cap exceeded")


def run(args, receipt, save):
    require(SOURCE_RELEASED is True, "Disabled native reference preparation has no fitting authority")
    require(args.source_root == TARGET and socket.gethostname() == "peptide", "Exact literal normal77 native reference target required")
    require(sys.dont_write_bytecode and not any(n == "torch" or n.startswith("torch.") for n in sys.modules),
        "Fresh pre-Torch/-B child required before loading any prepared helper")
    names = []
    root = Path(args.source_root).resolve()
    ordinary = load(root, ORDINARY_UTILITIES, "_fresh_native_reference_utilities", names)
    require(ordinary.SOURCE_RELEASED is False, "Original disabled shared reference source remains preserved")
    root, output, admission_path = ordinary.deliberate_paths(SimpleNamespace(source_root=args.source_root, output=args.output, admission=args.admission))
    require(admission_path.stat().st_mode & 0o222 == 0, "Immutable actual root native reference admission required")
    admission = json.loads(admission_path.read_text())
    require(admission["schema"] == "root_fresh_native_single_independent4_SR_admission_v1" and admission["root_fit_authorized"] is True
        and admission["worker_sha256"] == ordinary.sha(__file__) and admission["protocol_sha256"] == ordinary.sha(Path(__file__).parent / "PROTOCOL.json")
        and admission["source_review_approved"] is True and admission["member_seeds"] == list(MEMBER_SEEDS)
        and admission["W_updates"] == W_UPDATES and admission["W_local_updates"] == W_LOCAL and admission["SR_updates"] == SR_UPDATES
        and admission["single_alias_member"] == 0 and admission["all_W400_frozen_before_SR_decode"] is True
        and admission["A_scoring"] is False and admission["VALID_TEST_access"] is False
        and admission["old_checkpoint_inputs"] is False and admission["retry"] is False,
        "Exact fixed fresh native cohort admission required")
    qpath = ordinary.bound(root, admission["normal_native_context_qualification"]); qualified = json.loads(qpath.read_text())
    disabled_digest = hashlib.sha256(Path(__file__).read_text().replace("SOURCE_RELEASED = True", "SOURCE_RELEASED = False", 1).encode()).hexdigest()
    require(qualified["schema"] == "normal_native_independent_first_order_qualification_v1" and qualified["status"] == "PASS"
        and qualified["native_reference_disabled_source_sha256"] == disabled_digest
        and all(qualified[k] is True for k in ("full_native_context_supported", "both_stages_checked", "native_own_CE_gradient_Adam_RNG_checked",
            "fresh_seed_constructor_reset_checked", "independent_parameter_and_Adam_ownership_checked", "four_resident_models_and_histories_checked"))
        and qualified["input_identity"] == INPUT_IDENTITY and qualified["member_seeds"] == list(MEMBER_SEEDS)
        and qualified["CUDA_VISIBLE_DEVICES"] == admission["CUDA_VISIBLE_DEVICES"], "Actual new full-native context/gradient/state/ownership qualification required")
    require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules) and sys.dont_write_bytecode
        and str(Path(sys.executable).resolve()) == qualified["runtime"]["python_resolved"]
        and qualified["runtime"]["hostname"] == "peptide", "Fresh exact normal interpreter/-B required")
    uuid = admission["CUDA_VISIBLE_DEVICES"]
    require(isinstance(uuid, str) and uuid.startswith("GPU-") and "," not in uuid and os.environ.get("CUDA_VISIBLE_DEVICES") == uuid
        and admission["device"] == "cuda:0", "One qualified native physical UUID exposure required")
    caps, watchdog = admission["resource_limits"], admission["external_watchdog_seconds"]
    require(all(type(caps[k]) in (int, float) and math.isfinite(caps[k]) and caps[k] > 0 for k in
        ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes"))
        and admission["external_watchdog_required"] is True and type(watchdog) in (int, float) and math.isfinite(watchdog)
        and watchdog > caps["max_elapsed_seconds"], "Actual measured whole-cohort root caps and external supervisor required")
    local_seconds, global_seconds = (qualified[k] for k in ("native_local_update_elapsed_seconds_max", "native_global_update_elapsed_seconds_max"))
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (local_seconds, global_seconds)),
        "Actual synchronized normal native local/global update durations required")
    projection = 4 * (W_LOCAL * local_seconds + (W_UPDATES-W_LOCAL+SR_UPDATES) * global_seconds)
    basis = admission["resource_admission_basis"]; margin = basis["root_frozen_setup_IO_restore_margin_seconds"]
    require(basis["qualification_RESULT_sha256"] == admission["normal_native_context_qualification"]["sha256"]
        and basis["measured_native_local_max_seconds"] == local_seconds and basis["measured_native_global_max_seconds"] == global_seconds
        and basis["projected_four_member_training_seconds"] == projection
        and type(margin) in (int, float) and math.isfinite(margin) and margin > 0
        and caps["max_elapsed_seconds"] >= projection + margin, "Actual measured full-horizon root fit-resource basis required")
    owner = ordinary.CreatedOutput(output, root); token = owner.creator_token; save(owner, token, ordinary)
    audit = Audit(output); models, optimizers, constructions, common, common_frozen, warm_rows, final_rows = [], [], [], [], [], {}, {}
    old_path, python_rng = list(sys.path), random.getstate()
    old_env_present, old_env = "CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Fresh child timer required")
    backend_before = old_threads = rng_before = process = helpers = None
    data_w = data_sr = inputs = sr_inputs = None
    try:
        def expired(signum, frame): raise TimeoutError("Root-frozen fresh native cohort deadline exceeded")
        signal.signal(signal.SIGALRM, expired)
        remaining = caps["max_elapsed_seconds"] - (time.monotonic() - STARTED); require(remaining > 0, "No root time before numerical setup")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        config = qualified["process_configuration"]; os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["CUBLAS_WORKSPACE_CONFIG"]
        sys.path.insert(0, qualified["runtime"]["site_packages"])
        import torch
        require(str(Path(torch.__file__).resolve()) == qualified["runtime"]["torch_module_path"] and not torch.cuda.is_initialized(), "Normal pinned Torch/pre-CUDA required")
        process = load(root, PINS["process"], "_fresh_native_reference_process", names); process.torch = torch
        backend_before, old_threads = process.backend_snapshot(), torch.get_num_threads()
        torch.use_deterministic_algorithms(config["deterministic_algorithms_enabled"], warn_only=config["warn_only"])
        torch.set_num_threads(config["intra_op_threads"]); torch.set_num_interop_threads(config["interop_threads"])
        require(not torch.cuda.is_initialized() and process.backend_snapshot() == qualified["backend_snapshot"], "Qualified native backend configuration differs")
        require(process.runtime_identity() == qualified["backend_runtime_metadata"] and torch.cuda.device_count() == 1, "Qualified selected native GPU/runtime differs")
        device = "cuda:0"; free, total = torch.cuda.mem_get_info(device)
        require(type(admission["minimum_initial_cuda_free_bytes"]) is int and admission["minimum_initial_cuda_free_bytes"] > 0
            and free >= admission["minimum_initial_cuda_free_bytes"], "Qualified native GPU free floor failed")
        receipt["initial_cuda_memory"] = {"free_bytes": free, "total_bytes": total}
        helpers = load(root, PINS["scientific_helpers"], "_fresh_native_reference_scientific_helpers", names)
        accessor = load(root, PINS["accessor"], "_fresh_native_reference_accessor", names)
        native = load(root, PINS["native"], "_fresh_native_reference_exact_architecture", names)
        require(helpers.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True, "Exact unchanged helper/accessor gates required")
        rng_before = helpers._rng(device)
        public_b = (root / PUBLIC_B_RELATIVE).resolve()
        preflight = helpers._public_identity_before_w(accessor, root, public_b, INPUT_IDENTITY)
        require(preflight["public_b_manifest_sha256"] == INPUT_IDENTITY["public_b_manifest_sha256"],
            "Exact original public/B/W manifest identity required before W label access")
        recipe = {"source_sha256": ordinary.sha(__file__), "protocol_sha256": admission["protocol_sha256"],
            "admission_sha256": ordinary.sha(admission_path), "member_seeds": list(MEMBER_SEEDS), "single_alias_member": 0,
            "native_source_pin": PINS["native"], "input_identity": INPUT_IDENTITY, "W_local_updates": W_LOCAL,
            "W_global_updates": W_UPDATES-W_LOCAL, "SR_global_updates": SR_UPDATES,
            "new_independent_W_acquisitions": 4, "copied_shared_history": False,
            "S_R_received_only_after_all_W400_freeze": True, "A_labels_received": False, "A_scoring": False, "VALID_TEST_access": False}
        ordinary.atomic(output / "RUN.json", recipe); (output / "RUN.json").chmod(0o444)
        loaded_w = accessor.load_public_w(root, public_b, device=device)
        require(loaded_w["W_ids"].numel() == 4898 and loaded_w["provenance"]["preprocessing"] == preflight["preprocessing"]
            and loaded_w["provenance"]["public_b_manifest"]["sha256"] == INPUT_IDENTITY["public_b_manifest_sha256"],
            "Exact original W-only native public context and label-manifest identity required")
        data_w = {"features": loaded_w["features"], "edge_index": loaded_w["edge_index"],
            "training_ids": loaded_w["W_ids"], "training_targets": loaded_w["W_labels"]}
        del loaded_w
        inputs = {key: value.detach().clone() for key, value in data_w.items()}
        # Construct four native models and four EMPTY native Adam histories.
        # No torch.load, state_dict copy or shared-family construction occurs.
        for seed in MEMBER_SEEDS:
            model, optimizer, construction = fresh(native, helpers, seed, device)
            models.append(model); optimizers.append(optimizer); constructions.append(construction)
        independent_ownership(models, optimizers)
        for member, (model, optimizer, construction) in enumerate(zip(models, optimizers, constructions)):
            audit.member, audit.phase = member, "W"
            restore_rng(construction["after_Adam_constructor"], device)
            with (output / ("member" + str(member) + "_W.trace.jsonl")).open("x") as trace:
                for update in range(1, W_UPDATES + 1):
                    audit.update = update; model._global = update > W_LOCAL; limits(caps, device)
                    loss = step(model, optimizer, data_w, audit, helpers)
                    audit.counts[str(member)]["W"]["completed_updates"] = update
                    audit.event("native_W_update_completed", global_stage=model._global)
                    trace.write(json.dumps({"update": update, "global_stage": model._global, "W_own_CE": loss}, allow_nan=False)+"\n"); trace.flush()
            history(model, optimizer, W_UPDATES); model.eval()
            snapshot = state(model, optimizer, helpers, device)
            common.append(snapshot); common_frozen.append(helpers._cpu_tree(snapshot))
            warm_rows[str(member)] = checkpoint(output, "W400", member, snapshot, construction, recipe, ordinary, audit)
            equal(helpers._rng(device), snapshot["rng"])
            receipt["W400_states"] = dict(warm_rows); save()
        independent_ownership(models, optimizers)
        require(len(warm_rows) == 4 and all((output / row["path"]).stat().st_mode & 0o222 == 0
            and ordinary.sha(output / row["path"]) == row["sha256"] for row in warm_rows.values()), "All four independently acquired W400 images must freeze before SR access")
        audit.event("all_four_W400_images_frozen_before_S_R_label_access")
        receipt["all_W400_frozen_before_SR_decode"] = True; save()
        loaded = accessor.load_public_b(root, public_b, device=device)
        s, r = loaded["inner_indices"], loaded["query_indices"]
        require(s.numel() == 2449 and r.numel() == 2450 and not bool(torch.isin(s, r).any()), "Fixed disjoint S/R role sizes required")
        ids, order = torch.sort(torch.cat((s, r))); targets = torch.cat((loaded["inner_labels"], loaded["query_labels"]))[order]
        require(ids.numel() == 4899 and not bool(torch.isin(ids, torch.cat((loaded["W_ids"], loaded["A_ids"]))).any())
            and torch.equal(loaded["features"], data_w["features"]) and torch.equal(loaded["edge_index"], data_w["edge_index"])
            and loaded["provenance"]["preprocessing"] == preflight["preprocessing"]
            and loaded["provenance"]["public_b_manifest"]["sha256"] == INPUT_IDENTITY["public_b_manifest_sha256"],
            "Only original-manifest sorted SR targets on unchanged full public native context")
        data_sr = {"features": data_w["features"], "edge_index": data_w["edge_index"], "training_ids": ids, "training_targets": targets}
        del loaded, s, r, ids, order, targets
        sr_inputs = {key: value.detach().clone() for key, value in data_sr.items()}
        receipt["SR_labels_received"] = True; save()
        for member, (model, optimizer, construction) in enumerate(zip(models, optimizers, constructions)):
            audit.member, audit.phase = member, "S_R"
            # Each live native model/Adam still has only ITS OWN 400-update history.
            equal(model.state_dict(), common[member]["model"]); equal(optimizer.state_dict(), common[member]["Adam"])
            history(model, optimizer, W_UPDATES); restore_rng(common[member]["rng"], device); model._global = True
            with (output / ("member" + str(member) + "_SR.trace.jsonl")).open("x") as trace:
                for update in range(1, SR_UPDATES + 1):
                    audit.update = update; limits(caps, device)
                    loss = step(model, optimizer, data_sr, audit, helpers)
                    audit.counts[str(member)]["S_R"]["completed_updates"] = update
                    audit.event("native_SR_update_completed")
                    trace.write(json.dumps({"update": update, "global_stage": True, "SR_own_CE": loss}, allow_nan=False)+"\n"); trace.flush()
            history(model, optimizer, W_UPDATES + SR_UPDATES); model.eval()
            final_rows[str(member)] = checkpoint(output, "SR2300", member, state(model, optimizer, helpers, device), construction, recipe, ordinary, audit)
            for snapshot, frozen in zip(common, common_frozen): equal(snapshot, frozen)
            independent_ownership(models, optimizers)
            receipt["endpoints"] = dict(final_rows); save()
        require(len(final_rows) == 4 and all(all(row[key] == (W_UPDATES if phase == "W" else SR_UPDATES)
            for key in ("callback_attempts", "backward_attempts", "Adam_attempts", "completed_updates"))
            for phases in audit.counts.values() for phase, row in phases.items()), "All four full fixed native horizons and exact native work counts required")
        require(all(torch.equal(data_sr[key], value) for key, value in sr_inputs.items()), "SR/public tensors changed")
        receipt.update(single_alias={"member": 0, "seed": MEMBER_SEEDS[0], "endpoint": final_rows["0"], "additional_fit": False},
            independent4_endpoints=dict(final_rows), all_four_complete=True, comparison_complete=True,
            training_callbacks_W=1600, training_callbacks_SR=9200, total_training_callbacks=10800,
            backward_APIs=10800, native_Adam_steps=10800, serving_callbacks=0)
        limits(caps, device)
    except BaseException as error:
        receipt["primary_exception"] = {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}; raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt["operation_accounting"] = audit.snapshot(); receipt["W400_states"], receipt["endpoints"] = dict(warm_rows), dict(final_rows)
        errors = []
        def restore(name, fn):
            try: fn(); receipt[name] = True
            except BaseException as error: errors.append({"restoration": name, "error": str(error), "traceback": traceback.format_exc()})
        if inputs is not None: restore("public_and_W_inputs_unchanged", lambda: require(all(torch.equal(data_w[k], v) for k,v in inputs.items()), "Public/W tensors changed"))
        if sr_inputs is not None: restore("public_and_SR_inputs_unchanged", lambda: require(all(torch.equal(data_sr[k], v) for k,v in sr_inputs.items()), "Public/SR tensors changed"))
        if common:
            restore("cached_own_W400_model_Adam_RNG_immutable", lambda: equal(common, common_frozen))
        if rng_before is not None:
            def caller_rng(): restore_rng(rng_before, "cuda:0"); equal(helpers._rng("cuda:0"), rng_before)
            restore("caller_RNG_restored", caller_rng)
        if backend_before is not None: restore("backend_restored", lambda: process.backend_restore(backend_before))
        if old_threads is not None:
            def threads(): torch.set_num_threads(old_threads); require(torch.get_num_threads() == old_threads, "Intra-op restoration differs")
            restore("reversible_threads_restored", threads)
        random.setstate(python_rng); sys.path[:] = old_path
        if old_env_present: os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env
        else: os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        signal.signal(signal.SIGALRM, old_handler); signal.setitimer(signal.ITIMER_REAL, *old_timer)
        for name in names: sys.modules.pop(name, None)
        restore("original_sources_and_process_state_verified", lambda: require(random.getstate() == python_rng and sys.path == old_path
            and ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == old_env_present and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == old_env
            and signal.getsignal(signal.SIGALRM) == old_handler and signal.getitimer(signal.ITIMER_REAL) == old_timer
            and all(ordinary.sha(root / p) == h for p,h in [ORDINARY_UTILITIES, *PINS.values()]), "Sources or reversible process state differ"))
        restore("all_acquired_W400_file_bytes_unchanged", lambda: require(all(ordinary.sha(output / row["path"]) == row["sha256"] for row in warm_rows.values()), "Own native W400 file changed"))
        receipt["checkpoint_cost"] = {"write_attempts": audit.checkpoint_write_attempts,
            "completed_frozen_writes": audit.checkpoint_completed_writes,
            "write_and_hash_elapsed_seconds": audit.checkpoint_write_and_hash_elapsed_seconds,
            "partial_and_complete_checkpoint_bytes_observed": sum(p.stat().st_size for p in output.glob("member*.pt")
                if p.is_file() and not p.is_symlink()),
            "CPU_snapshots_hashing_and_checkpoint_work_included_in_whole_process_resources": True,
            "single_alias_extra_checkpoint_writes": 0}
        receipt["restoration_errors"] = errors; receipt["interop_restoration"] = "One-time fresh-child initialization; not reversible."
        save()
    require(not receipt["restoration_errors"], "Fresh native reference restoration failed")
    limits(caps, "cuda:0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true"); parser.add_argument("--source-root")
    parser.add_argument("--admission"); parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_FRESH_NATIVE_SINGLE_INDEPENDENT4_SR_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False})); return
    require(SOURCE_RELEASED is True, "Disabled fresh native source has no execution/fitting authority")
    receipt = {"status": "RUNNING_FRESH_NATIVE_SINGLE_INDEPENDENT4", "comparison_complete": False, "all_four_complete": False,
        "A_scoring": False, "VALID_TEST_access": False, "SR_labels_received": False, "all_W400_frozen_before_SR_decode": False,
        "W400_states": {}, "endpoints": {}, "original_paper_scores_changed": False, "old_checkpoint_inputs": False}
    owner, token, ordinary = None, None, None
    def save(created=None, creator_token=None, utilities=None):
        nonlocal owner, token, ordinary
        if created is not None:
            require(owner is None and created.verify(creator_token) == created.path, "Only this newly created output is owned")
            owner, token, ordinary = created, creator_token, utilities
            receipt["created_output_identity"] = {"path": str(owner.path), "device": owner.device, "inode": owner.inode}
        receipt["elapsed_seconds_including_imports_setup_acquisition_continuation_restore"] = time.monotonic() - STARTED
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized(): receipt["whole_process_resources"] = resources("cuda:0")
        if owner is not None: ordinary.atomic(owner.verify(token) / "RESULT.json", receipt)
    code = 1
    try:
        require(args.source_root and args.admission and args.output, "Exact source-root, prospective admission and new output required")
        run(args, receipt, save)
        receipt["status"] = "FRESH_NATIVE_SINGLE_INDEPENDENT4_COMPLETE_A_CLOSED"; code = 0
    except BaseException as error:
        receipt.update(status="FAIL_FRESH_NATIVE_SINGLE_INDEPENDENT4", comparison_complete=False, all_four_complete=False,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        try:
            save()
            if owner is not None: owner.freeze(token)
        except BaseException as error:
            code = 1; receipt.update(status="FAIL_FRESH_NATIVE_SINGLE_INDEPENDENT4", comparison_complete=False, all_four_complete=False,
                output_finalization_error=str(error))
            try: save(); owner.freeze(token) if owner is not None else None
            except BaseException as recovery_error: receipt["output_failure_receipt_error"] = str(recovery_error)
    print(json.dumps({"status": receipt["status"], "comparison_complete": receipt["comparison_complete"], "error": receipt.get("error")}))
    raise SystemExit(code)


if __name__ == "__main__": main()
