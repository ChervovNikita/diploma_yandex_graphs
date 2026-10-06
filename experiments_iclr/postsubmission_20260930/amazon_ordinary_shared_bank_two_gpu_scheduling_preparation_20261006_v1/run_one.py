"""Disabled ONE assigned ordinary continuation from exact paired common400.

No original fit, common acquisition, scoring, selector or automatic retry.
Paired supervisor must retain both endpoints; H16 is diagnostic only.
"""
import time
STARTED = time.monotonic()
import argparse
from contextlib import contextmanager
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
import stat
import sys
import traceback

SOURCE_RELEASED = False
ORIGINAL_ORDINARY = ("amazon_ordinary_shared_bank_own_pool_reference_preparation_20261006_v4/ordinary_reference.py", "5044a16f3f710aaf234057b115ab928d589af3a06a595940afbfa8876636c97b")
QUALIFIER_SUPERVISOR_SHA256 = "6106ee06d4764293cf58e77870f96a157bcc23f442936258a1d982c63dd740cf"
QUALIFIER_SHA256 = "d175dc6bf44886d874959cecd9833dc9f192f30b764bc7e9b938be83f2ad8af2"
COMMON_DESCRIPTORS = {
    "common_checkpoint": {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/initial.pt", "bytes": 109671738, "sha256": "2e0e9b44767abc12b3dc896986ea2d4faeaa667c8682b95929f927d10718154f"},
    "common_origin_run": {"path": "amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2/output/scientific_run/RUN.json", "bytes": 6402, "sha256": "2e18a77081437e47c0c4a0d59cb07b2a15500f660066cbed1c980f613c028ca5"}}
AUTHORIZED_TARGETS = {
    "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git": "peptide"}
AUTHORIZED_PHASE_RELATIVE = "experiments_iclr/postsubmission_20260930"
OBJECTIVES = ("own", "own_pool")
UPDATES, PREFIX, MEMBERS, NODES, CLASSES = 2300, 16, 4, 24492, 5
PINS = {
 "scientific_helpers": ("amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3/six_arm_worker.py", "f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1"),
 "native": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/native_polynormer.py", "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
 "boundary": ("amazon_polynormer_paired_family_source_preparation_20261003_v6/reused_models/backbone_boundary_adapter.py", "699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac"),
 "accessor": ("amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py", "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85"),
 "process_helpers": ("amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3/run_scientific.py", "65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb")}
COMMON_ORIGIN_WORKER = "ad28d1c8a05a168aeadb6825183ef8d1d1a6cffea3b488c5a33aa3c90318c8f3"
PUBLIC_B_RELATIVE = "learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b"
INPUT_IDENTITY = {"public_graph_sha256": "19757299bcfd9e493e9ceae9e73248753ab1e773ccc1f6b57c6fadf8c843310f", "roles_sha256": "9cab6f2cf24dbecee59a2179e860d78b9b7b995adf0bb5347ca16dce86f32c97", "native_edge_logical_sha256": "229a8a787ef9120a4d7a1dcdd6481b973619a1c7f44a4abe9ed69d05c256e550", "public_b_manifest_sha256": "da03b6c14615da934443c9b8f36b68f9d643e2b23e04ebe17dab7dacf630df55"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def deliberate_paths(args):
    """Assert only the named project/phase and deliberately supplied file paths.

    These checks do not constrain normal interpreter libraries, runtime caches,
    CUDA caches, temporary libraries or the user's filesystem namespace.
    """
    choices = {str(Path(repository) / AUTHORIZED_PHASE_RELATIVE): (repository, host)
               for repository, host in AUTHORIZED_TARGETS.items()}
    # Validate the supplied literal BEFORE resolving or touching any other path.
    require(isinstance(args.source_root, str) and args.source_root in choices,
            "Supply exactly one of the two named authorized project-phase spellings")
    repository, expected_host = choices[args.source_root]
    require(socket.gethostname() == expected_host, "Named repository/expected host pair differs")
    named_argument, root_argument = Path(args.source_root), Path(args.source_root)
    named_repository = Path(repository).resolve()
    root = root_argument.resolve()
    require(root.parent.parent == named_repository and root.is_dir(),
            "Resolved source phase must belong to its chosen named repository")
    def inside(argument, *, new):
        argument = Path(argument).absolute()
        require(".." not in argument.parts and any(argument.is_relative_to(prefix)
                for prefix in (root_argument, root, named_argument.absolute())),
                "Admission/output must be deliberately inside the source root")
        require(not argument.is_symlink() and argument.resolve().is_relative_to(root),
                "Admission/output symlink escape is not permitted")
        # An in-root spelling must not traverse a parent that escapes root.
        for parent in argument.parents:
            if parent in (root_argument, root, named_argument.absolute()):
                break
            if parent.exists() or parent.is_symlink():
                require(parent.resolve().is_relative_to(root), "Parent path escapes the source root")
        resolved = argument.resolve()
        if new:
            require(not resolved.exists() and not argument.exists() and resolved.parent.is_dir(),
                    "Output must be new with an existing in-root parent")
        else:
            require(resolved.is_file(), "Existing in-root admission file required")
        return resolved
    return root, inside(args.output, new=True), inside(args.admission, new=False)


class CreatedOutput:
    """Creator token and directory identity for this invocation's fresh output."""
    def __init__(self, path, root):
        self.path, self.root, self.creator_token = path, root, object()
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                "Verified output path escaped before creation")
        path.mkdir(parents=False, exist_ok=False)
        information = path.lstat()
        require(stat.S_ISDIR(information.st_mode), "Created output is not a real directory")
        self.device, self.inode = information.st_dev, information.st_ino

    def verify(self, creator_token):
        require(creator_token is self.creator_token and not self.path.is_symlink()
                and self.path.resolve().is_relative_to(self.root),
                "Output creator token/path no longer belongs to this invocation")
        information = self.path.lstat()
        require(stat.S_ISDIR(information.st_mode)
                and (information.st_dev, information.st_ino) == (self.device, self.inode),
                "Created output directory identity changed")
        return self.path

    def freeze(self, creator_token):
        path = self.verify(creator_token)
        for child in path.iterdir():
            if child.is_file() and not child.is_symlink():
                child.chmod(0o444)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def bound(root, row):
    root, relative = Path(root).resolve(), Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "In-root evidence required")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root)
            and path.stat().st_mode & 0o222 == 0 and path.stat().st_size == row["bytes"]
            and sha(path) == row["sha256"], "Immutable evidence bytes differ")
    return path


def load(root, key, names):
    path, digest = PINS[key]
    path = Path(root) / path
    require(sha(path) == digest, "Pinned source differs: " + key)
    name = "_ordinary_shared_bank_reference_" + key
    require(name not in sys.modules, "Fresh source module required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; names.append(name)
    spec.loader.exec_module(module)
    return module


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def objective(logits, targets, kind):
    """Complete native own CE and the same arithmetic-probability served pool."""
    import torch
    import torch.nn.functional as F
    require(kind in OBJECTIVES and logits.shape == (MEMBERS, targets.numel(), CLASSES), "Objective shape/enum differs")
    log_probs = F.log_softmax(logits, dim=-1)
    own = -log_probs.gather(2, targets[None, :, None].expand(MEMBERS, -1, 1)).mean()
    pooled = torch.logsumexp(log_probs, dim=0) - math.log(MEMBERS)
    pool = -pooled.gather(1, targets[:, None]).mean()
    total = own if kind == "own" else 0.5 * own + 0.5 * pool
    require(bool(torch.isfinite(total)) and bool(torch.isfinite(own)) and bool(torch.isfinite(pool)), "Nonfinite training objective")
    return total, {"own_CE": float(own.detach()), "probability_pool_NLL": float(pool.detach()), "optimized_loss": float(total.detach())}


def restore_rng(value, device):
    import numpy as np
    import torch
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(value["torch_cpu"].cpu())
    torch.cuda.set_rng_state(value["cuda"].cpu(), device)


class Audit:
    def __init__(self, output):
        self.output, self.arm, self.update = output, None, 0
        self.callback_attempts = {k: 0 for k in OBJECTIVES}
        self.backward_attempts = {k: 0 for k in OBJECTIVES}
        self.small_logit_reverse_attempts = {k: 0 for k in OBJECTIVES}
        self.optimizer_attempts = {k: 0 for k in OBJECTIVES}
        self.completed_updates = {k: 0 for k in OBJECTIVES}

    def event(self, kind, **fields):
        with (self.output / "ATTEMPTED_OPERATIONS.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, "objective": self.arm, "update": self.update,
                "elapsed_seconds": time.monotonic() - STARTED, **fields}, allow_nan=False) + "\n")

    def snapshot(self):
        return {name: dict(getattr(self, name)) for name in ("callback_attempts", "backward_attempts", "small_logit_reverse_attempts", "optimizer_attempts", "completed_updates")}


@contextmanager
def counted(family, audit):
    old, present = family.forward_member, "forward_member" in family.__dict__
    def callback(features, edges, member):
        audit.callback_attempts[audit.arm] += 1
        audit.event("complete_native_callback_attempt", member=member)
        return old(features, edges, member)
    family.forward_member = callback
    try:
        yield
    finally:
        if present:
            family.forward_member = old
        else:
            del family.forward_member


def step(family, optimizer, data, kind, implementation, helpers, audit, device):
    """Joint baseline, or qualified exact stochastic first-order decomposition."""
    import torch
    family.train(); family.set_global_stage(True)
    optimizer.zero_grad(set_to_none=True)
    x, edges, ids, targets = (data[k] for k in ("features", "edge_index", "training_ids", "training_targets"))
    if implementation == "joint":
        logits = family(x, edges)
        require(logits.shape == (MEMBERS, NODES, CLASSES) and logits.dtype == torch.float32, "Full native FP32 trajectories required")
        helpers._finite(logits, "Nonfinite complete native logits")
        loss, stats = objective(logits[:, ids], targets, kind)
        audit.backward_attempts[kind] += 1; audit.event("joint_ordinary_backward_attempt")
        loss.backward()
        del logits, loss
    else:
        require(implementation == "streamed_exact", "No alternate/approximate implementation")
        # First complete value forwards capture the exact four stochastic paths.
        # Every gradient replay restores its own pre-member RNG. Pool cotangents
        # couple all four members; no memberwise or factor-only surrogate is used.
        states, values = [], []
        with torch.no_grad():
            for member in range(MEMBERS):
                states.append(helpers._rng(device))
                logits = family.forward_member(x, edges, member)
                require(logits.shape == (NODES, CLASSES) and logits.dtype == torch.float32, "Full native value path required")
                helpers._finite(logits, "Nonfinite native value path")
                values.append(logits[ids].detach().clone()); del logits
        after_values = helpers._rng(device)
        bank = torch.stack(values).requires_grad_(True)
        loss, stats = objective(bank, targets, kind)
        audit.small_logit_reverse_attempts[kind] += 1; audit.event("joint_logit_cotangent_reverse_attempt")
        cotangents, = torch.autograd.grad(loss, bank)
        for member in range(MEMBERS):
            restore_rng(states[member], device)
            logits = family.forward_member(x, edges, member)
            helpers._finite(logits, "Nonfinite native gradient replay")
            require(torch.allclose(logits[ids].detach(), values[member], atol=2e-6, rtol=2e-5), "Stochastic value/gradient replay differs")
            audit.backward_attempts[kind] += 1; audit.event("native_logit_VJP_backward_attempt", member=member)
            logits[ids].backward(cotangents[member])
            del logits
        restore_rng(after_values, device)
        del bank, cotangents, loss, values, states, after_values
    for name, parameter in family.named_parameters():
        if helpers._inactive(name, True):
            require(parameter.grad is None, "Dormant local-head gradient appeared: " + name)
        else:
            require(parameter.requires_grad and parameter.grad is not None, "Active shared/private parameter disconnected: " + name)
            helpers._finite(parameter.grad, "Nonfinite ordinary gradient: " + name)
    audit.optimizer_attempts[kind] += 1; audit.event("ordinary_Adam_step_attempt")
    optimizer.step()
    for parameter in family.parameters():
        helpers._finite(parameter, "Nonfinite ordinary parameter")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                helpers._finite(value, "Nonfinite ordinary Adam state")
    return stats


def resources(device):
    import torch
    return {"elapsed_seconds": time.monotonic() - STARTED,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def limits(admission, device):
    row = resources(device)
    for measured, cap in (("elapsed_seconds", "max_elapsed_seconds"), ("process_peak_rss_bytes", "max_process_rss_bytes"),
            ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"), ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes")):
        require(row[measured] <= admission["resource_limits"][cap], "Ordinary whole-process resource cap exceeded: " + measured)


def checkpoint(path, family, optimizer, helpers, recipe, kind, update, device):
    import torch
    # Checkpoint serialization is not a served prediction or endpoint selection.
    payload = {"schema": "amazon_fixed_ordinary_reference_state_v1", "objective": kind, "update": update,
        "diagnostic_prefix_only": update == PREFIX, "competence_endpoint": update == UPDATES,
        "global_stage": True, "eval_mode": True, "recipe": recipe,
        "family_state": helpers._cpu_tree(family.state_dict()), "Adam": helpers._cpu_tree(optimizer.state_dict()),
        "rng": helpers._cpu_tree(helpers._rng(device))}
    with path.open("xb") as stream:
        torch.save(payload, stream)
    path.chmod(0o444)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def run(args, receipt, save):
    require(SOURCE_RELEASED is True, "Disabled ordinary references require independent source/runtime/resource review and root release")
    root, output, admission_path = deliberate_paths(args)
    require(not admission_path.is_symlink() and admission_path.stat().st_mode & 0o222 == 0, "Immutable root admission required")
    registry = json.loads(admission_path.read_text())
    require(registry["schema"] == "root_ordinary_two_gpu_once_only_registry_v1"
        and isinstance(registry["pair_id"], str) and registry["pair_id"]
        and registry["paired_supervisor_required"] is True and registry["automatic_retry"] is False
        and registry["implementation"] == "streamed_exact" and registry["objectives"] == list(OBJECTIVES)
        and len(registry["jobs"]) == 2
        and [j["objective"] for j in registry["jobs"]] == list(OBJECTIVES)
        and [j["job_id"] for j in registry["jobs"]] == list(OBJECTIVES)
        and len({j["CUDA_VISIBLE_DEVICES"] for j in registry["jobs"]}) == 2
        and all(isinstance(j["CUDA_VISIBLE_DEVICES"], str) and j["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
            and "," not in j["CUDA_VISIBLE_DEVICES"] for j in registry["jobs"])
        and all(registry[key] == row for key, row in COMMON_DESCRIPTORS.items())
        and sha(root / ORIGINAL_ORDINARY[0]) == ORIGINAL_ORDINARY[1], "Exact fixed paired registry/source/common required")
    require(args.objective in OBJECTIVES, "Exactly one assigned objective required")
    assigned_kind = args.objective
    job = next(j for j in registry["jobs"] if j["objective"] == assigned_kind)
    relative = Path(job["output_relative"])
    require(not relative.is_absolute() and ".." not in relative.parts and output == (root / relative).resolve(),
        "Child output must be its once-only assigned registry directory")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == job["CUDA_VISIBLE_DEVICES"], "Assigned physical UUID exposure required before Torch")
    admission = dict(registry, normal_runtime_context_qualification=job["normal_runtime_context_qualification"],
        resource_limits=job["resource_limits"], external_watchdog_seconds=job["external_watchdog_seconds"], device="cuda:0")
    receipt.update(pair_id=registry["pair_id"], paired_registry_sha256=sha(admission_path), assigned_objective=assigned_kind,
        CUDA_VISIBLE_DEVICES=job["CUDA_VISIBLE_DEVICES"], worker_sha256=sha(__file__))
    require(admission["root_fit_authorized"] is True and admission["A_scoring"] is False and admission["VALID_TEST_access"] is False
        and admission["worker_sha256"] == sha(__file__) and admission["objectives"] == list(OBJECTIVES)
        and admission["continuation_updates"] == UPDATES and admission["diagnostic_prefix"] == PREFIX
        and admission["source_review"]["approved"] is True, "Exact two-reference root admission required")
    require(admission["protocol_sha256"] == sha(Path(__file__).parent / "PROTOCOL.json"), "Frozen protocol differs")
    qual_path = bound(root, admission["normal_runtime_context_qualification"])
    qualified = json.loads(qual_path.read_text())
    qualification_scope = json.loads(bound(root, job["qualification_scope"]).read_text())
    require(qualified["status"] == "PASS_NORMAL_ORDINARY_FIRST_ORDER_STREAMED_RESOURCE_ONLY"
        and qualified["worker_sha256"] == QUALIFIER_SHA256 and qualified["ordinary_source_pin"] == list(ORIGINAL_ORDINARY)
        and qualified["root_scope_sha256"] == job["qualification_scope"]["sha256"]
        and qualification_scope["CUDA_VISIBLE_DEVICES"] == job["CUDA_VISIBLE_DEVICES"]
        and qualification_scope["qualifier_worker_sha256"] == QUALIFIER_SHA256
        and qualified["model_fits"] == 0 and qualified["persistent_updates"] == 0
        and job["qualification_supervisor_exit_code"] == 0,
        "Successful exact per-UUID first-order qualification and wait4 closure required")
    qualifier_supervision = json.loads(bound(root, job["qualification_supervisor_receipt"]).read_text())
    require(qualifier_supervision["status"] == "PASS_OWNED_QUALIFIER_WHOLE_CHILD_CLOSED"
        and qualifier_supervision["supervisor_worker_sha256"] == QUALIFIER_SUPERVISOR_SHA256
        and qualifier_supervision["scope_sha256"] == job["qualification_scope"]["sha256"]
        and qualifier_supervision["child_exit_code"] == 0 and qualifier_supervision["qualifier_result"]["sha256"] == job["normal_runtime_context_qualification"]["sha256"],
        "Exact immutable external qualifier closure required")
    implementation = admission["implementation"]
    require(implementation in ("joint", "streamed_exact") and qualified["ordinary_first_order_supported"] is True
        and qualified["implementation"] == implementation and qualified["both_objectives_checked"] is True
        and qualified["same_checkpoint_context_and_RNG_checked"] is True, "Actual normal first-order/context/memory qualification required")
    if implementation == "streamed_exact":
        require(qualified["coupled_pool_gradient_and_Adam_update_parity_checked"] is True
            and qualified["stochastic_member_RNG_replay_checked"] is True, "Actual exact-decomposition qualification required")
    require(qualified["input_identity"] == INPUT_IDENTITY, "Exact public graph/roles identity required")
    common_path, origin_path = bound(root, admission["common_checkpoint"]), bound(root, admission["common_origin_run"])
    require(common_path.name == "initial.pt" and origin_path.name == "RUN.json" and common_path.parent == origin_path.parent, "Exact pilot common400 origin required")
    origin = json.loads(origin_path.read_text())
    require(origin["recipe"]["worker_source_sha256"] == COMMON_ORIGIN_WORKER
        and origin["A_labels_received"] is False and origin["A_scoring_performed"] is False, "Old/full-TRAIN/selected checkpoints are ineligible")
    require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Fresh normal runtime child required")
    require(socket.gethostname() == qualified["runtime"]["hostname"] and str(Path(sys.executable).resolve()) == qualified["runtime"]["python_resolved"]
        and sys.dont_write_bytecode, "Qualified normal host/interpreter/-B required")
    for path, pin in PINS.values():
        require(sha(root / path) == pin, "Mathematical/native source binding differs")
    device = admission["device"]
    require(device == "cuda:0", "Fixed qualified native CUDA0 reference required")
    caps = admission["resource_limits"]
    require(all(type(caps[k]) in (int, float) and math.isfinite(caps[k]) and caps[k] > 0 for k in
        ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes"))
        and admission["external_watchdog_required"] is True
        and type(admission["external_watchdog_seconds"]) in (int, float)
        and math.isfinite(admission["external_watchdog_seconds"])
        and admission["external_watchdog_seconds"] > caps["max_elapsed_seconds"], "Prospectively frozen root resource limits/watchdog required")
    output_owner = CreatedOutput(output, root)
    save(output_owner, output_owner.creator_token)
    audit, names, endpoints, prefixes = Audit(output), [], {}, {}
    old_env_present, old_env = "CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_path, python_rng = list(sys.path), random.getstate()
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Do not replace an active process timer")
    backend_before = old_threads = rng_before = helpers = process = accessor = None
    data_before = data = None
    try:
        config = qualified["process_configuration"]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["CUBLAS_WORKSPACE_CONFIG"]
        def expired(signum, frame):
            raise TimeoutError("Root-frozen ordinary-reference whole-process deadline exceeded")
        signal.signal(signal.SIGALRM, expired)
        remaining = caps["max_elapsed_seconds"] - (time.monotonic() - STARTED)
        require(remaining > 0, "Process time exhausted before normal numerical setup")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        site = Path(qualified["runtime"]["site_packages"]).resolve()
        require(site.is_dir(), "Qualified normal runtime unavailable; no upgrade/fallback")
        sys.path.insert(0, str(site))
        import torch
        import numpy as np
        require(str(Path(torch.__file__).resolve()) == qualified["runtime"]["torch_module_path"] and not torch.cuda.is_initialized(), "Wrong Torch identity or premature CUDA")
        process = load(root, "process_helpers", names); process.torch = torch
        backend_before = process.backend_snapshot(); old_threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(config["deterministic_algorithms_enabled"], warn_only=config["warn_only"])
        require(not torch.cuda.is_initialized(), "Backend setup must precede CUDA initialization")
        torch.set_num_threads(config["intra_op_threads"]); torch.set_num_interop_threads(config["interop_threads"])
        require(torch.get_num_threads() == config["intra_op_threads"] and torch.get_num_interop_threads() == config["interop_threads"], "Qualified thread mode differs")
        require(process.backend_snapshot() == qualified["backend_snapshot"], "Qualified backend flags differ")
        receipt["admission_and_actual_normal_qualification_before_checkpoint_or_labels"] = True
        receipt["qualified_process_configuration"] = config
        identity = process.runtime_identity()
        require(identity == qualified["backend_runtime_metadata"] and torch.cuda.device_count() == 1,
            "Qualified normal runtime/single assigned GPU differs")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        require(type(job["minimum_initial_cuda_free_bytes"]) is int and job["minimum_initial_cuda_free_bytes"] > 0
            and free_bytes >= job["minimum_initial_cuda_free_bytes"], "Assigned GPU initial free floor failed")
        receipt["initial_cuda_memory"] = {"free_bytes": free_bytes, "total_bytes": total_bytes,
            "required_free_bytes": job["minimum_initial_cuda_free_bytes"]}
        helpers, accessor, native, boundary = (load(root, k, names) for k in ("scientific_helpers", "accessor", "native", "boundary"))
        require(helpers.SOURCE_RELEASED is False and accessor.SOURCE_RELEASED is True, "Pinned helper/accessor gates differ")
        rng_before = helpers._rng(device)
        public_b = Path(args.public_b_dir).resolve()
        require(public_b == (root / PUBLIC_B_RELATIVE).resolve(), "Fixed original public+B projection required")
        preflight = helpers._public_identity_before_w(accessor, root, public_b, INPUT_IDENTITY)
        require(preflight["public_b_manifest_sha256"] == INPUT_IDENTITY["public_b_manifest_sha256"], "Public manifest differs before label access")
        image = torch.load(common_path, map_location="cpu", weights_only=False)
        require(image["schema"] == "amazon_G0_frozen_state_v1" and image["id"] == "initial"
            and image["warm_updates"] == 400 and image["episodes"] == 0 and image["warm_role"] == "W"
            and image["global_stage"] is True and image["eval_mode"] is True
            and image["recipe"] == origin["recipe"] and image["construction"]["seed"] == 17
            and len(image["W_own_CE_trace"]) == 400, "Only the exact last W-only common400 state is eligible")
        helpers._finite_tree(image)
        loaded = accessor.load_public_b(root, public_b, device=device)
        s, r = loaded["inner_indices"], loaded["query_indices"]
        require(s.numel() == 2449 and r.numel() == 2450 and not bool(torch.isin(s, r).any()), "Fixed original S/R roles differ")
        ids, order = torch.sort(torch.cat((s, r)))
        targets = torch.cat((loaded["inner_labels"], loaded["query_labels"]))[order]
        require(ids.numel() == 4899 and not bool(torch.isin(ids, torch.cat((loaded["W_ids"], loaded["A_ids"]))).any()), "Only S/R labels may supervise continuation")
        provenance = loaded["provenance"]
        require(provenance["public_b_manifest"]["sha256"] == INPUT_IDENTITY["public_b_manifest_sha256"] and provenance["preprocessing"] == preflight["preprocessing"], "Reader/preflight context differs")
        data = {"features": loaded["features"], "edge_index": loaded["edge_index"], "training_ids": ids, "training_targets": targets}
        # B/W label arrays decoded by the existing B-only reader are discarded;
        # W is never included in the ordinary loss. A labels are never opened.
        del loaded, s, r, ids, targets, order
        data_before = {k: v.detach().clone() for k, v in data.items()}
        recipe = {"common_checkpoint": admission["common_checkpoint"], "common_origin_run": admission["common_origin_run"],
            "protocol_sha256": admission["protocol_sha256"], "normal_qualification": admission["normal_runtime_context_qualification"],
            "implementation": implementation, "objectives": list(OBJECTIVES), "updates_per_objective": UPDATES,
            "source_sha256": sha(__file__), "native_source_pins": PINS, "input_identity": INPUT_IDENTITY,
            "A_labels_received": False, "A_scoring": False, "VALID_TEST_access": False,
            "paired_registry_sha256": sha(admission_path), "pair_id": registry["pair_id"], "assigned_objective": assigned_kind,
            "CUDA_VISIBLE_DEVICES": job["CUDA_VISIBLE_DEVICES"]}
        atomic(output / "RUN.json", recipe)
        (output / "RUN.json").chmod(0o444)
        for kind in (assigned_kind,):
            audit.arm, audit.update = kind, 0; audit.event("objective_started")
            family, construction = helpers._fresh_family(native, boundary, device)
            family.load_state_dict(image["family_state"], strict=True); family.set_global_stage(True)
            optimizer = torch.optim.Adam(family.parameters(), lr=0.001, weight_decay=0.0)
            optimizer.load_state_dict(helpers._cpu_tree(image["Adam"]))
            group = optimizer.param_groups[0]
            require(len(optimizer.param_groups) == 1 and group["lr"] == 0.001 and group["weight_decay"] == 0
                and group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8 and not group["amsgrad"]
                and len({id(p) for p in group["params"]}) == len(list(family.parameters()))
                and all(p.requires_grad for p in family.parameters()), "Complete original Adam ownership/history required")
            restore_rng(image["rng"], device)
            with counted(family, audit), (output / (kind + ".trace.jsonl")).open("x") as trace:
                for update in range(1, UPDATES + 1):
                    audit.update = update; limits(admission, device)
                    stats = step(family, optimizer, data, kind, implementation, helpers, audit, device)
                    audit.completed_updates[kind] = update
                    audit.event("ordinary_update_completed")
                    trace.write(json.dumps({"update": update, "objective": kind, **stats}, allow_nan=False) + "\n"); trace.flush()
                    if update in (PREFIX, UPDATES):
                        row = checkpoint(output / (kind + "_" + str(update) + ".pt"), family, optimizer, helpers, recipe, kind, update, device)
                        (prefixes if update == PREFIX else endpoints)[kind] = row
                        receipt["prefixes"], receipt["endpoints"] = dict(prefixes), dict(endpoints); save()
            expected = 4 * UPDATES if implementation == "joint" else 8 * UPDATES
            require(audit.callback_attempts[kind] == expected and audit.completed_updates[kind] == UPDATES
                and audit.optimizer_attempts[kind] == UPDATES and audit.backward_attempts[kind] == (UPDATES if implementation == "joint" else 4 * UPDATES)
                and audit.small_logit_reverse_attempts[kind] == (0 if implementation == "joint" else UPDATES), "Complete ordinary work counts differ")
            del family, optimizer, construction
        require(set(endpoints) == {assigned_kind} and set(prefixes) == {assigned_kind}, "Exactly the assigned complete outcome required")
        unassigned = next(k for k in OBJECTIVES if k != assigned_kind)
        require(all(values[unassigned] == 0 for values in audit.snapshot().values()), "Unassigned objective work must remain zero")
        receipt["assigned_objective_complete"] = True
        receipt["comparison_complete"] = False; receipt["A_scoring"] = False
        limits(admission, device)
    except BaseException as error:
        receipt["primary_exception"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt["operation_accounting"] = audit.snapshot(); receipt["prefixes"], receipt["endpoints"] = dict(prefixes), dict(endpoints)
        errors = []
        def restore(name, function):
            try:
                function(); receipt[name] = True
            except BaseException as error:
                errors.append({"restoration": name, "error": str(error), "traceback": traceback.format_exc()})
        if data is not None and data_before is not None:
            def input_check():
                require(all(torch.equal(data[k], v) for k, v in data_before.items()), "Public/role input tensors changed")
            restore("public_and_role_inputs_unchanged", input_check)
        if rng_before is not None:
            def rng():
                restore_rng(rng_before, device)
                current = helpers._rng(device)
                require(current["python"] == rng_before["python"] and current["numpy"] == rng_before["numpy"]
                    and torch.equal(current["torch_cpu"], rng_before["torch_cpu"])
                    and torch.equal(current["cuda"], rng_before["cuda"]), "Caller RNG restoration differs")
            restore("caller_RNG_restored", rng)
        if backend_before is not None:
            restore("backend_restored", lambda: process.backend_restore(backend_before))
        if old_threads is not None:
            def threads():
                torch.set_num_threads(old_threads)
                require(torch.get_num_threads() == old_threads, "Intra-op thread restoration differs")
            restore("reversible_threads_restored", threads)
        # Interop is a qualified one-time fresh-child setting, not reversible.
        random.setstate(python_rng); sys.path[:] = old_path
        if old_env_present: os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env
        else: os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        signal.signal(signal.SIGALRM, old_handler); signal.setitimer(signal.ITIMER_REAL, *old_timer)
        for name in names: sys.modules.pop(name, None)
        def process_state():
            require(random.getstate() == python_rng and sys.path == old_path
                and ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == old_env_present
                and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == old_env
                and signal.getsignal(signal.SIGALRM) == old_handler and signal.getitimer(signal.ITIMER_REAL) == old_timer
                and not any(name in sys.modules for name in names), "Reversible process state differs")
            for relative, digest in PINS.values():
                require(sha(root / relative) == digest, "Original source bytes changed")
            if helpers is not None:
                require(helpers.SOURCE_RELEASED is False, "Scientific helper source gate changed")
            if accessor is not None:
                require(accessor.SOURCE_RELEASED is True, "Original B-only accessor gate changed")
        restore("original_source_and_reversible_process_state_verified", process_state)
        restore("common_and_origin_bytes_unchanged", lambda: require(sha(common_path) == admission["common_checkpoint"]["sha256"] and sha(origin_path) == admission["common_origin_run"]["sha256"], "Common input bytes changed"))
        receipt["restoration_errors"] = errors
        receipt["interop_restoration"] = "Fresh-child lifetime only; no reversible claim."
        save()
    require(not receipt["restoration_errors"], "Ordinary reference restoration failed")
    limits(admission, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--objective", choices=OBJECTIVES)
    parser.add_argument("--public-b-dir"); parser.add_argument("--admission"); parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_ONE_ASSIGNED_ORDINARY_REFERENCE_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False})); return
    require(SOURCE_RELEASED is True, "Disabled preparation has no fitting authority")
    receipt = {"status": "RUNNING_ONE_ASSIGNED_ORDINARY_REFERENCE", "A_scoring": False, "VALID_TEST_access": False,
        "original_six_arm_pilot_changed": False, "comparison_complete": False, "endpoints": {}, "prefixes": {}}
    created_output, creator_token = None, None
    def save(owner=None, token=None):
        nonlocal created_output, creator_token
        if owner is not None:
            require(isinstance(owner, CreatedOutput) and (created_output is None or owner is created_output),
                    "Receipt output must be the verified directory created by this invocation")
            owner.verify(token)
            created_output, creator_token = owner, token
            receipt["created_output_identity"] = {"path": str(owner.path), "device": owner.device, "inode": owner.inode}
        receipt["elapsed_seconds_including_imports_setup_fit_restore"] = time.monotonic() - STARTED
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            receipt["cuda_peak_allocated_bytes"] = sys.modules["torch"].cuda.max_memory_allocated("cuda:0")
            receipt["cuda_peak_reserved_bytes"] = sys.modules["torch"].cuda.max_memory_reserved("cuda:0")
            receipt["whole_process_resources"] = resources("cuda:0")
        if created_output is not None:
            output = created_output.verify(creator_token)
            atomic(output / "RESULT.json", receipt)
    code = 1
    try:
        require(all((args.source_root, args.public_b_dir, args.admission, args.output, args.objective)),
                "Exact named source-root and future root admission/context/output required")
        run(args, receipt, save)
        receipt["status"] = "ONE_ASSIGNED_ORDINARY_REFERENCE_COMPLETE_A_CLOSED"; code = 0
    except BaseException as error:
        receipt.update(status="FAIL_ONE_ASSIGNED_ORDINARY_REFERENCE", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        # Before a verified successful mkdir there is no output authority.
        # Rejected/existing/outside/symlink paths stay untouched; print failure.
        try:
            save()
            if created_output is not None:
                created_output.freeze(creator_token)
        except BaseException as error:
            code = 1
            receipt.update(status="FAIL_ONE_ASSIGNED_ORDINARY_REFERENCE", output_finalization_error_type=type(error).__name__,
                           output_finalization_error=str(error), output_finalization_traceback=traceback.format_exc())
    print(json.dumps({"status": receipt["status"], "A_scoring": False, "error": receipt.get("error"),
        "output_finalization_error": receipt.get("output_finalization_error"),
        "verified_created_output": created_output is not None}))
    raise SystemExit(code)

if __name__ == "__main__":
    main()
