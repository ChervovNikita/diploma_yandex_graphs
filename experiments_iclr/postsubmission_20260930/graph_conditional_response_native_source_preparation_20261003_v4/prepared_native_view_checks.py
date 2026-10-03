"""Prospective bound runtime, three-view normalization and input-custody checks."""
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import inspect
import importlib
import json
from core.groups import LabelRoles, raw_group_keys
from core.transaction import ExternalStateHook
from native_source.tokens import CacheIdentity, CompleteViewBank, build_uncached_mono_tokens
from resolved_groups import build_fit_plan, build_control_plan

ROOT = Path(__file__).resolve().parent


def require_runtime_receipt(authorization):
    """Deferred runtime binding; receipt must be supplied by root before execution.

    These checks bind actual source bytes, dependency versions and declared
    binaries. They do not establish PyG2.3 parity or numerical admission by name.
    """
    receipt = authorization.get("runtime_dependency_receipt")
    if not isinstance(receipt, dict) or not receipt.get("root_binding_reference"):
        raise PermissionError("Explicit actual runtime/source/binary receipt required")
    rows = receipt.get("files", [])
    paths = {str(Path(r["path"]).resolve()): r for r in rows}
    if len(paths) != len(rows) or not paths:
        raise ValueError("Missing or duplicate actual-runtime file binding")
    # Reject missing/stale declared bytes before loading numerical packages.
    for path, row in paths.items():
        p = Path(path)
        if sha256(p.read_bytes()).hexdigest() != row["sha256"] or p.stat().st_size != row["size"]:
            raise RuntimeError("Actual runtime source/binary custody mismatch")
    import torch
    import numpy
    import scipy
    import torch_geometric
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops, to_undirected, scatter
    from torch.optim import _functional as optimizer_functional
    from torch_geometric.datasets import HeterophilousGraphDataset
    modules = (torch, numpy, scipy, torch_geometric)
    actual_versions = {m.__name__: str(m.__version__) for m in modules}
    if receipt.get("versions") != actual_versions or torch_geometric.__version__ != "2.7.0":
        raise RuntimeError("Actual dependency versions differ from the root-bound fixture runtime")
    required_sources = {str(Path(m.__file__).resolve()) for m in modules}
    required_sources.add(str(Path(optimizer_functional.__file__).resolve()))
    required_sources.update(str(Path(inspect.getfile(f)).resolve()) for f in (
        torch.optim.AdamW, inspect.unwrap(torch.optim.AdamW.step), torch.func.functional_call, scatter))
    functions = (gcn_norm, to_scipy_sparse_matrix, add_remaining_self_loops, to_undirected, HeterophilousGraphDataset)
    required_sources.update(str(Path(inspect.getfile(f)).resolve()) for f in functions)
    numpy_binary = importlib.import_module("numpy._core._multiarray_umath" if hasattr(numpy, "_core") else "numpy.core._multiarray_umath")
    scipy_binary = importlib.import_module("scipy.sparse._sparsetools")
    required_binaries = {str(Path(m.__file__).resolve()) for m in (torch._C, numpy_binary, scipy_binary)}
    if not required_sources <= paths.keys() or not required_binaries <= paths.keys() or any(
            paths[p].get("kind") != "binary" for p in required_binaries):
        raise ValueError("Runtime binding omits imported dependencies/functions or native binary")
    actual_settings = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                       "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                       "default_device": str(torch.get_default_device())}
    if receipt.get("deterministic_settings") != actual_settings:
        raise ValueError("Actual deterministic/thread settings differ from root-bound engineering runtime")
    pins = json.loads((ROOT / "PYG_CODE_FETCH_RECEIPT.json").read_text())["requests"]
    if len(pins) != len(functions):
        raise AssertionError("Five normalization/loop/undirected/dataset source pins required")
    for function, relative in zip(functions, ("nn__conv__gcn_conv.py", "utils__convert.py", "utils__loop.py",
                                            "utils__undirected.py", "datasets__heterophilous_graph_dataset.py")):
        installed = Path(inspect.getfile(function))
        pinned = ROOT / "pyg_source_pins" / relative
        if installed.read_bytes() != pinned.read_bytes():
            raise RuntimeError("Imported PyG function file differs from the saved commit pin")
    if (torch.get_default_dtype() != torch.float32 or torch.cuda.is_initialized() or
            actual_settings["default_device"] not in ("cpu", "cpu:0")):
        raise RuntimeError("Isolated explicit float32 CPU engineering fixture required")
    return {"versions": actual_versions, "files_bound": len(rows), "PyG2_3_parity_qualified": False,
            "complete_dependency_closure_qualified": False,
            "file_binding_scope": "declared files plus mandatory package/PyG/selected optimizer-functional-scatter sources and Torch-NumPy-SciPy extensions; linked/full transitive closure not certified"}


def tensor_hash(tensor):
    """Fixture encoding: shape/dtype JSON plus contiguous CPU uint8 bytes."""
    import torch
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError("Explicit CPU strided fixture encoding required")
    header = json.dumps((str(tensor.dtype), tuple(tensor.shape)), separators=(",", ":")).encode()
    body = bytes(tensor.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
    return sha256(header + b"\0" + body).hexdigest()


def fixture_views(K=2, nodes=96):
    """Fixed irregular simple reciprocal graph; last node isolated.

    Removal units are synthetic engineering inputs, not the prospective 10%
    sampler, chosen study masks or an Amazon data/label/recipe qualification.
    """
    import torch
    from native_source.native_preprocess import mono_base
    if nodes != 96:
        raise ValueError("Predetermined supported synthetic role fixture only")
    x = torch.sin(torch.arange(nodes * 3, dtype=torch.float32, device="cpu").reshape(nodes, 3) / 7)
    x = x + .2 * torch.cos(torch.arange(nodes * 3, dtype=torch.float32, device="cpu").reshape(nodes, 3) / 13)
    units = {(i, i + 1) for i in range(nodes - 2)}
    units.update((i, i + 2) for i in range(0, nodes - 3, 3))
    units.update((i, i + 7) for i in range(4, nodes - 8, 11))
    native_records = [(u, v) for u, v in sorted(units)] + [(v, u) for u, v in sorted(units)] + [(0, 0), (7, 7)]
    native_edges = torch.tensor(native_records, dtype=torch.long, device="cpu").T.contiguous()
    labels = tuple((i, i % 2) for i in range(nodes))
    roles = LabelRoles(tuple(range(64)), tuple(range(64, 96)), labels, 2)
    neighbors = {n: tuple(sorted(v if u == n else u for u, v in units if n in (u, v))) for n in range(nodes)}
    fit_keys = raw_group_keys(roles, "fit", neighbors, adjacency_semantics_receipt="synthetic_simple_nonloop_neighbors")
    control_keys = raw_group_keys(roles, "control", neighbors, adjacency_semantics_receipt="synthetic_simple_nonloop_neighbors")
    fit, control = build_fit_plan(roles, fit_keys).plan, build_control_plan(roles, control_keys).plan
    removals = {"native": (), "same_removal": ((0, 2), (6, 8)), "other_removal": ((1, 2), (7, 8))}
    assert all(set(r) <= units for r in removals.values())
    assert all(dict(labels)[u] == dict(labels)[v] for u, v in removals["same_removal"])
    assert all(dict(labels)[u] != dict(labels)[v] for u, v in removals["other_removal"])
    role_hash = sha256(json.dumps((roles.fit, roles.control, labels), separators=(",", ":")).encode()).hexdigest()
    edge_views, banks = {}, []
    for view, removed in removals.items():
        records = [(u, v) for u, v in native_records if u == v or tuple(sorted((u, v))) not in removed]
        edges = torch.tensor(records, dtype=torch.long, device="cpu").T.contiguous()
        assert len(native_records) - len(records) == 2 * len(removed)
        identity = CacheIdentity("synthetic_fixture_release", tensor_hash(x), tensor_hash(native_edges), "none",
                                 sha256(json.dumps(removed).encode()).hexdigest(), role_hash, view, K, "mono",
                                 sha256((ROOT / "pyg_source_pins/nn__conv__gcn_conv.py").read_bytes()).hexdigest(),
                                 "explicit_root_bound_float32_CPU_fixture")
        bank = build_uncached_mono_tokens(x, edges, None, identity)
        # Independent unweighted normalization oracle: drop pre-existing loops,
        # add exactly one unit loop per complete node, destination degree, native
        # row/source and column/destination orientation; reciprocal edges remain.
        nonloops = [(u, v) for u, v in records if u != v]
        normalized = nonloops + [(n, n) for n in range(nodes)]
        degree = [1 + sum(v == n for _, v in nonloops) for n in range(nodes)]
        inverse = torch.tensor(degree, dtype=torch.float32, device="cpu").pow(-.5)
        expected = torch.zeros((nodes, nodes), dtype=torch.float32, device="cpu")
        for u, v in normalized:
            expected[u, v] = inverse[u] * inverse[v]
        coalesced = bank.operator.coalesce()
        assert coalesced._nnz() == len(normalized)
        assert torch.equal(coalesced.to_dense(), expected)
        assert torch.equal(expected[nodes - 1], torch.nn.functional.one_hot(torch.tensor(nodes - 1), nodes).float())
        source = mono_base(K, x, edges, None)
        expected_token = x
        for order, token in enumerate(bank.tokens):
            assert torch.equal(token, source[order])
            assert torch.allclose(token, expected_token, atol=1e-6, rtol=1e-5)
            expected_token = expected @ expected_token
        edge_views[view] = edges
        banks.append((view, bank))
    complete = CompleteViewBank(tuple(banks), nodes, 3, "synthetic_producer_encoding_and_readonly_callback")
    assert len({bank.identity.fingerprint for _, bank in banks}) == 3
    for view in ("same_removal", "other_removal"):
        assert not torch.equal(dict(banks)[view].operator.to_dense(), dict(banks)["native"].operator.to_dense())
    return x, native_edges, edge_views, complete, roles, fit, control


def input_hook(x, native_edges, edge_views, complete):
    """Separate synthetic input audit: frozen descriptors do not freeze tensors."""
    def tensors():
        result = [x, native_edges] + list(edge_views.values())
        for _, bank in complete.banks:
            result += list(bank.tokens) + [bank.operator._indices(), bank.operator._values()]
        return tuple(result)

    def capture():
        return tuple(t.detach().clone() for t in tensors())

    def restore(image):
        import torch
        current = tensors()
        if len(current) != len(image):
            raise RuntimeError("Synthetic immutable input registration changed")
        with torch.no_grad():
            for t, value in zip(current, image):
                t.copy_(value)
    return ExternalStateHook("synthetic_complete_view_inputs", "synthetic_input_audit_not_actual_data_custody", capture, restore)


def native_view_identity():
    import torch
    x, edges, edge_views, complete, _, _, _ = fixture_views()
    original = complete.fingerprint
    banks = dict(complete.banks)
    changed = replace(banks["same_removal"].identity, removed_units_sha256="different_synthetic_removed_units")
    changed_bank = replace(banks["same_removal"], identity=changed)
    different = replace(complete, banks=tuple((v, changed_bank if v == "same_removal" else b) for v, b in complete.banks))
    assert different.fingerprint != original  # identity invalidation only; producer must rebuild corresponding contents
    bad = replace(banks["other_removal"], identity=replace(banks["other_removal"].identity, features_sha256="wrong_X"))
    try:
        replace(complete, banks=tuple((v, bad if v == "other_removal" else b) for v, b in complete.banks))
    except ValueError:
        pass
    else:
        raise AssertionError("Cross-view feature-identity mismatch accepted")
    hook = input_hook(x, edges, edge_views, complete)
    saved = hook.capture()
    with torch.no_grad():
        banks["same_removal"].tokens[1][0, 0].add_(1)
    assert not all(torch.equal(a, b) for a, b in zip(saved, hook.capture()))
    hook.restore(saved)
    assert all(torch.equal(a, b) for a, b in zip(saved, hook.capture()))
