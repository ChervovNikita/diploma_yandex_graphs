"""Fixed-CSR sparse execution with explicit first-order adjoints.

No E-by-M-by-D edge-feature tensor is constructed. Graph storage is fixed,
has no trainable values, and must include separately built adjoint CSR.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from .contract import SCHEMA, confined_output, fixture_shape, sha256


class OperationLedger:
    """Count actual sparse work, including every backward and recomputation.

    Wall timings are deliberately absent. Python counters do not measure
    asynchronous CUDA work. An adopted engineering gate must synchronize and
    charge all dense operations, optimizer, pooling and graph preparation too.
    """

    def __init__(self):
        self.sparse = []
        self.dense = []
        self._scope = ["ordinary"]

    @contextmanager
    def scope(self, name):
        self._scope.append(str(name))
        try:
            yield
        finally:
            self._scope.pop()

    def record(self, operator, direction, tensor):
        self.sparse.append({
            "operator": operator.name, "direction": direction,
            "scope": self._scope[-1], "nnz": operator.nnz,
            "channels": int(tensor.shape[1]),
            "edge_channels": operator.nnz * int(tensor.shape[1]),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
        })

    def record_dense(self, operation, tensor, **extra):
        self.dense.append({"operation": operation, "scope": self._scope[-1],
                           "shape": list(tensor.shape), **extra})

    def summary(self):
        totals = {}
        for row in self.sparse:
            key = row["direction"] + ":" + row["scope"]
            totals[key] = totals.get(key, 0) + row["edge_channels"]
        return {"sparse_calls": len(self.sparse), "edge_channels_by_scope": totals,
                "dense_calls": len(self.dense), "wall_latency_measured": False}

    def clear(self):
        self.sparse.clear()
        self.dense.clear()


class _FixedCSRMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, operator):
        ctx.operator = operator
        operator.ledger.record(operator, "forward", features)
        # The fixed sparse tensors are held by operator. No input state or
        # edge-expanded tensor is needed for d(A X)/dX with fixed A.
        return torch.sparse.mm(operator.forward_csr, features)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        operator = ctx.operator
        operator.ledger.record(operator, "backward", upstream)
        return torch.sparse.mm(operator.adjoint_csr, upstream), None


class FixedCSR:
    def __init__(self, name, forward_csr, adjoint_csr, ledger=None):
        if forward_csr.layout != torch.sparse_csr or adjoint_csr.layout != torch.sparse_csr:
            raise ValueError("Forward and explicit adjoint must both be CSR")
        if forward_csr.ndim != 2 or forward_csr.shape[0] != forward_csr.shape[1]:
            raise ValueError("Expected a square incoming-node mean operator")
        if adjoint_csr.shape != forward_csr.shape:
            raise ValueError("Adjoint shape mismatch")
        if forward_csr.dtype != adjoint_csr.dtype or forward_csr.device != adjoint_csr.device:
            raise ValueError("Forward and adjoint dtype/device mismatch")
        if forward_csr.requires_grad or adjoint_csr.requires_grad:
            raise ValueError("Graph gradients are outside this fixed-operator hypothesis")
        fixture_shape(forward_csr.shape[0], forward_csr._nnz())
        self.name = str(name)
        self.forward_csr, self.adjoint_csr = forward_csr, adjoint_csr
        self.nnz = int(forward_csr._nnz())
        self.nodes = int(forward_csr.shape[0])
        self.ledger = ledger if ledger is not None else OperationLedger()

    def __call__(self, features):
        if features.ndim != 2 or features.shape[0] != self.nodes:
            raise ValueError("CSR input must be [N, channels]")
        if features.dtype != self.forward_csr.dtype or features.device != self.forward_csr.device:
            raise ValueError("CSR input dtype/device must match the frozen operator")
        return _FixedCSRMM.apply(features, self)

    def payload_bytes(self):
        tensors = []
        for matrix in (self.forward_csr, self.adjoint_csr):
            tensors += [matrix.crow_indices(), matrix.col_indices(), matrix.values()]
        return sum(t.numel() * t.element_size() for t in tensors)


class CSRFamily:
    def __init__(self, operators, fingerprint, ledger, metadata=None):
        if "full" not in operators or not all(f"color{c}" in operators for c in range(4)):
            raise ValueError("Full graph and all four colors must be frozen")
        self.operators = dict(operators)
        self.fingerprint = fingerprint
        self.ledger = ledger
        self.metadata = {} if metadata is None else dict(metadata)
        nodes = operators["full"].nodes
        if any(op.nodes != nodes or op.ledger is not ledger for op in operators.values()):
            raise ValueError("All operators must share nodes and the operation ledger")
        if sum(operators[f"color{c}"].nnz for c in range(4)) != operators["full"].nnz:
            raise ValueError("Canonical CSR colors do not partition the full supports")

    @property
    def full(self):
        return self.operators["full"]

    def color(self, color):
        if color not in range(4):
            raise ValueError("Color must be 0,1,2,3")
        return self.operators[f"color{color}"]

    def naive(self, features, color, path):
        if color not in range(4):
            raise ValueError("Color must be 0,1,2,3")
        if path == "cached_half":
            key = f"half{color}"
            if key not in self.operators:
                raise ValueError("Half unions were not prepared; no runtime conversion")
            return self.operators[key](features) * 2.0
        if path == "lazy_two_colors":
            self.ledger.record_dense("naive_two_color_sum", features)
            return (self.color(color)(features) + self.color((color + 1) % 4)(features)) * 2.0
        raise ValueError("Choose a declared naive sparse path")

    def storage_report(self):
        return {"per_operator_payload_bytes": {
            name: operator.payload_bytes() for name, operator in self.operators.items()
        }, "total_payload_bytes": sum(op.payload_bytes() for op in self.operators.values()),
                "allocator_reserved_peak_measured": False}


def load_fixture_family(manifest_path, *, device="cpu", dtype=torch.float32,
                        include_half_unions=False):
    """Read an audited synthetic manifest. No real dataset loader is supplied."""
    import numpy as np

    manifest_path = confined_output(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("purpose") != "synthetic_fixture_only":
        raise ValueError("Only this packet's synthetic fixture manifests are accepted")
    fixture_shape(manifest["nodes"], manifest["raw_edges"])
    arrays = {}
    for name, record in manifest["arrays"].items():
        path = (manifest_path.parent / record["file"]).resolve()
        if path.parent != manifest_path.parent or sha256(path) != record["sha256"]:
            raise ValueError("Fixture array confinement/hash check failed")
        # Avoid a Torch tensor that aliases a read-only memmap. H2D/copies must
        # be timed separately by any future complete-graph gate.
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError("Fixture array metadata mismatch")
        arrays[name] = value
    ledger = OperationLedger()
    operators = {}
    names = ["full"] + [f"color{c}" for c in range(4)]
    if include_half_unions:
        names += [f"half{c}" for c in range(4)]
    for name in names:
        matrices = []
        for direction in ("forward", "adjoint"):
            prefix = f"{name}_{direction}_"
            matrices.append(torch.sparse_csr_tensor(
                torch.as_tensor(arrays[prefix + "rowptr"], dtype=torch.int64, device=device),
                torch.as_tensor(arrays[prefix + "columns"], dtype=torch.int64, device=device),
                torch.as_tensor(arrays[prefix + "weights"], dtype=dtype, device=device),
                size=(manifest["nodes"], manifest["nodes"]), device=device, dtype=dtype,
            ))
        operators[name] = FixedCSR(name, *matrices, ledger)
    return CSRFamily(operators, manifest["P_edge_only_fingerprint"], ledger, manifest)


if __name__ == "__main__":
    from .contract import refuse_real_pilot
    refuse_real_pilot()
