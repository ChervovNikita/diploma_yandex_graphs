"""Disabled fixed B-only acquisition and six serial G0 H16 continuations.

No A label input, A scoring, stock loader, old checkpoint or selector. The exact
disabled native port is bound through its PUBLIC gated APIs only. A future
reviewed successor must release/rebind that port and the accessor before this
worker can reach acquisition. This draft supplies no execution authorization.
"""

from __future__ import annotations

import hashlib
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
    name = "_amazon_G0_preparation_" + key
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


def _warm_step(family, optimizer, data):
    import torch
    import torch.nn.functional as F
    family.train()
    optimizer.zero_grad(set_to_none=True)
    logits = family(data["features"], data["edge_index"])
    _require(logits.dtype == torch.float32 and logits.shape == (4, 24492, 5),
             "Complete native FP32 own-member forward required")
    _finite(logits, "Nonfinite B acquisition logits")
    loss = torch.stack([F.nll_loss(F.log_softmax(logits[m], dim=1).index_select(0, data["B_ids"]),
                                   data["B_labels"]) for m in range(4)]).mean()
    _finite(loss, "Nonfinite B own-CE acquisition loss")
    loss.backward()
    for name, parameter in family.named_parameters():
        if _inactive(name, bool(family.core._global)):
            _require(parameter.grad is None, "Inactive acquisition gradient: " + name)
        else:
            _require(parameter.grad is not None, "Disconnected acquisition parameter: " + name)
            _finite(parameter.grad, "Nonfinite acquisition gradient: " + name)
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


def run_six(project_root, public_b_dir, output_dir, *, device="cpu"):
    """Prepare common400 and six immutable H16 endpoints; never evaluates A.

    A new output directory is mandatory. Failure retains partial states and
    prevents COMPLETE.json, hence the separate evaluator cannot open A.
    """
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled source-only six-arm scientific worker")
    bindings = _read(PACKET / "SOURCE_BINDINGS.json")
    queue = _read(PACKET / "QUEUE.json")
    _require(queue["arms"] == [{"id": arm, "control": CONTROLS[arm],
                                "affinity": "K_perm" if arm == "permuted" else "K",
                                "episodes": HORIZON} for arm in ARMS]
             and queue["common"] == {"seed": 17, "split": 0, "local_updates": 200,
                                      "global_updates": 200, "all_B_own_CE4": True}
             and queue["coefficients"] == {"eta_probe": 0.01, "eta_private": 0.01,
                 "eta_core": 0.001, "extra_margin": 0.1, "pool_fraction": 0.5,
                 "response_epsilon": 0.001, "entropy": 1.0, "graph": 1.0,
                 "assignment_steps": 8, "assignment_rate": 1.0,
                 "ratio_smoothing": 0.01, "reciprocal_smoothing": 0.01}, "Fixed queue differs")
    # Fail before data/acquisition if any requisite source is still disabled.
    accessor = _module(project_root, bindings, "accessor")
    port = _module(project_root, bindings, "native_port")
    operator = _module(project_root, bindings, "G0V2")
    _require(accessor.SOURCE_RELEASED is True and port.PORT_RELEASED is True
             and operator.SOURCE_RELEASED is False,
             "Require separately reviewed released accessor/port and exact disabled G0V2")
    native = _module(project_root, bindings, "native_model")
    boundary = _module(project_root, bindings, "boundary_model")
    import torch
    _require(device == "cpu" or device.startswith("cuda:"), "Explicit native device required")
    data = accessor.load_public_b(project_root, public_b_dir, device=device)
    _require(set(data) == {"features", "edge_index", "B_ids", "B_labels", "inner_indices",
                           "inner_labels", "query_indices", "query_labels", "A_ids", "provenance"},
             "Only the sealed public+B accessor output is accepted")
    output = Path(output_dir).resolve()
    output.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    phase, arm, done = "common_acquisition", None, 0
    common_row, endpoints, diagnostics = None, {}, {}
    recipe = {"queue_sha256": _sha(PACKET / "QUEUE.json"),
              "bindings_sha256": _sha(PACKET / "SOURCE_BINDINGS.json"),
              "worker_source_sha256": _sha(Path(__file__)),
              "public_b_manifest_sha256": data["provenance"]["public_b_manifest"]["sha256"],
              "roles_sha256": data["provenance"]["roles"]["sha256"],
              "public_graph_sha256": data["provenance"]["public_graph"]["sha256"]}
    _write(output / "RUN.json", {"schema": "amazon_G0_six_arm_run_v1", "recipe": recipe,
                                 "provenance": data["provenance"], "device": device,
                                 "A_labels_received": False, "A_scoring_performed": False})
    try:
        family, construction = _fresh_family(native, boundary, device)
        optimizer = torch.optim.Adam(family.parameters(), lr=0.001, weight_decay=0.0)
        construction["after_Adam_constructor"] = _rng(device)
        group = optimizer.param_groups[0]
        _require(len(optimizer.param_groups) == 1 and group["betas"] == (0.9, 0.999)
                 and group["eps"] == 1e-8 and not group["amsgrad"]
                 and len({id(p) for p in group["params"]}) == len(list(family.parameters())),
                 "Adam defaults/complete ownership differ")
        warm_losses = []
        for update in range(400):
            if update == 200:
                family.set_global_stage(True)  # Same weights/Adam/RNG; no restore/reset.
            warm_losses.append(_warm_step(family, optimizer, data))
            done = update + 1
        family.eval()
        common = _cpu_tree(family.state_dict())
        common_payload = {"schema": "amazon_G0_frozen_state_v1", "id": "initial",
                          "recipe": recipe, "episodes": 0, "warm_updates": 400,
                          "global_stage": True, "eval_mode": True, "family_state": common,
                          "Adam": _cpu_tree(optimizer.state_dict()), "rng": _cpu_tree(_rng(device)),
                          "construction": _cpu_tree(construction), "B_own_CE_trace": warm_losses}
        common_row = _save_state(output / "initial.pt", common_payload)
        del common_payload, optimizer
        frozen_permutation = _permutation(data)
        permutation_row = _save_state(output / "permutation.pt", _cpu_tree(frozen_permutation))
        phase = "continuation"
        for arm in ARMS:
            done = 0
            family.load_state_dict(common, strict=True)  # Exact common clone, no constructor/reset.
            family.set_global_stage(True)
            family.eval()
            forward, theta, phis, _ = port.bind_native_callback(
                family, data["features"], data["edge_index"], expected_nodes=24492, global_stage=True)
            pairs = port.prepare_sparse_pairs(operator, bindings["files"]["G0V2"]["sha256"],
                data["inner_indices"], data["inner_labels"], data["edge_index"],
                node_count=24492, dtype=data["features"].dtype,
                affinity_permutation=frozen_permutation if arm == "permuted" else None)
            history = []
            for episode in range(HORIZON):
                theta_next, phis_next, info = port.native_sparse_episode(
                    operator, bindings["files"]["G0V2"]["sha256"], theta, phis, forward, pairs,
                    data["inner_indices"], data["inner_labels"],
                    data["query_indices"], data["query_labels"], control=CONTROLS[arm])
                _check_commit(theta, phis, theta_next, phis_next)
                _finite_tree(info)
                theta, phis = theta_next, phis_next  # Port detaches each committed state.
                history.append(_cpu_tree(info))
                done = episode + 1
            _install(family, theta, phis, port.PRIVATE_NAMES)
            endpoint_state = _cpu_tree(family.state_dict())
            endpoints[arm] = _save_state(output / (arm + ".pt"),
                {"schema": "amazon_G0_frozen_state_v1", "id": arm, "recipe": recipe,
                 "episodes": HORIZON, "warm_updates": 400, "common_state": common_row,
                 "global_stage": True, "eval_mode": True, "family_state": endpoint_state})
            diagnostics[arm] = _save_state(output / (arm + "_B_diagnostics.pt"),
                {"episodes": tuple(history), "endpoint_vs_common": _distances(endpoint_state, common),
                 "interpretation": "Finite G0 support diagnostics, never an A selector"})
            del history, endpoint_state, theta, phis, theta_next, phis_next, forward, pairs
        complete = {"schema": "amazon_G0_seven_states_complete_v1", "recipe": recipe,
                    "common": common_row, "endpoints": endpoints, "B_diagnostics": diagnostics,
                    "permutation": permutation_row, "warm_updates": 400,
                    "episodes_per_arm": HORIZON, "arms_in_order": list(ARMS),
                    "A_labels_received": False, "A_scoring_performed": False,
                    "elapsed_fit_seconds": time.monotonic() - start,
                    "planned_fit_member_forwards": 5056,
                    "peak_native_bytes": None, "new_runtime_qualification_claim": False}
        _write(output / "COMPLETE.json", complete)
        return _descriptor(output / "COMPLETE.json")
    except Exception as error:
        _write(output / "FAILURE.json", {"schema": "amazon_G0_incomplete_run_v1", "phase": phase,
            "arm": arm, "completed_updates_or_episodes_in_phase": done,
            "error_type": type(error).__name__, "error": str(error), "common": common_row,
            "endpoints": endpoints, "elapsed_fit_seconds": time.monotonic() - start,
            "A_labels_received": False, "A_scoring_performed": False,
            "replacement_seed_or_arm_or_shortened_H_authorized": False})
        raise
