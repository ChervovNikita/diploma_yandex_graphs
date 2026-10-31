"""Complete canonical CSR preparation from the audited training-only export."""

from __future__ import annotations

import hashlib
import json
import resource
import time

import numpy as np

from .contract import SCHEMA, CPU_SECONDS, DISK_BYTES, output_path, sha256
from propagation_cost_impl_v1.csr_prepare import color_offset, subset_csr, adjoint_csr, structural_bytes


def peak_rss_bytes():
    import sys
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def _fingerprint(n, degree, rowptr, columns, weights):
    digest = hashlib.sha256(f"incoming-mean-raw-degree-v1|{n}|".encode())
    for name, values, dtype in (("degree", degree, "<i8"), ("rowptr", rowptr, "<i8"),
                                ("columns", columns, "<i8"), ("weights", weights, "<f4")):
        array = np.asarray(values, dtype=dtype, order="C")
        digest.update(name.encode() + b"|" + str(array.size).encode() + b"|")
        view = memoryview(array).cast("B")
        for start in range(0, view.nbytes, 8 * 1024 * 1024):
            digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def prepare_complete(export, data_binding, output, budget):
    output = output_path(output)
    if output.exists():
        raise FileExistsError("Prepared complete graph must have a new exclusive directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    edge = export["edge_index"].numpy()
    sources, receivers = edge[0], edge[1]
    n, e = int(export["x"].shape[0]), int(edge.shape[1])
    if e <= 0:
        raise ValueError("Adopted complete provider graph cannot have an empty edge tensor")
    budget.check()
    degree = np.bincount(receivers, minlength=n).astype(np.int64, copy=False)
    order = np.lexsort((sources, receivers))
    columns, rows = sources[order], receivers[order]
    del order
    first = np.r_[True, (columns[1:] != columns[:-1]) | (rows[1:] != rows[:-1])]
    positions = np.flatnonzero(first)
    del first
    multiplicity = np.diff(np.r_[positions, e])
    canonical_columns, canonical_rows = columns[positions], rows[positions]
    del columns, rows, positions
    weights = (multiplicity / degree[canonical_rows]).astype(np.float32)
    rowptr = np.r_[0, np.cumsum(np.bincount(canonical_rows, minlength=n), dtype=np.int64)]
    del canonical_rows
    fingerprint = _fingerprint(n, degree, rowptr, canonical_columns, weights)
    canonical_seconds = time.monotonic() - started
    graph = {"nodes": n, "rowptr": rowptr, "columns": canonical_columns,
             "weights": weights, "degree": degree, "fingerprint": fingerprint}
    offsets = np.zeros(n, dtype=np.uint8)
    for row in np.flatnonzero(np.diff(rowptr)):
        offsets[row] = color_offset(fingerprint, row)
    colors = np.empty(canonical_columns.size, dtype=np.uint8)
    for start in range(0, colors.size, 1_000_000):
        budget.check()
        stop = min(start + 1_000_000, colors.size)
        pos = np.arange(start, stop, dtype=np.int64)
        recv = np.searchsorted(rowptr, pos, side="right") - 1
        colors[start:stop] = (pos - rowptr[recv] + offsets[recv]) % 4
    color_assignment_seconds = time.monotonic() - started - canonical_seconds
    storage = structural_bytes(n, canonical_columns.size, True)
    # Acquisition and preparation together consume the shared extra-disk cap.
    # Phase admission supplies acquisition bytes from its complete receipt.
    if storage["total_payload_bytes"] > DISK_BYTES:
        raise RuntimeError("Projected complete CSR payload alone exceeds disk cap")
    records, measured_bytes = {}, 0

    def save(name, array):
        nonlocal measured_bytes
        budget.check()
        path = output_path(output / (name + ".npy"))
        np.save(path, array, allow_pickle=False)
        measured_bytes += path.stat().st_size
        if measured_bytes > DISK_BYTES:
            raise RuntimeError("Prepared CSR disk cap exceeded")
        records[name] = {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
                         "bytes": path.stat().st_size, "payload_bytes": int(array.nbytes), "sha256": sha256(path)}

    for name, array in (("degree", degree), ("multiplicity", multiplicity),
                        ("color_offsets", offsets), ("entry_colors", colors)):
        save(name, array)
    operators, seconds = {}, {}
    for name in ["full"] + [f"color{c}" for c in range(4)] + [f"half{c}" for c in range(4)]:
        step = time.monotonic()
        budget.check()
        if name == "full":
            arrays = rowptr, canonical_columns, weights
        elif name.startswith("color"):
            arrays = subset_csr(graph, colors, {int(name[-1])})
        else:
            c = int(name[-1])
            arrays = subset_csr(graph, colors, {c, (c + 1) % 4})
        operators[name] = int(arrays[1].size)
        for suffix, array in zip(("rowptr", "columns", "weights"), arrays):
            save(f"{name}_forward_{suffix}", array)
        transposed = adjoint_csr(*arrays, n)
        for suffix, array in zip(("rowptr", "columns", "weights"), transposed):
            save(f"{name}_adjoint_{suffix}", array)
        del arrays, transposed
        seconds[name] = time.monotonic() - step
    budget.check()
    manifest = {
        "schema": SCHEMA, "kind": "complete_graph_prepared_CSR",
        "task": data_binding["task"], "nodes": n, "raw_edges": e,
        "canonical_edges": int(canonical_columns.size), "operator_nnz": operators,
        "P_edge_only_fingerprint": fingerprint, "data_binding": data_binding,
        "arrays": records, "complete_graph": True, "edge_changes": False,
        "implicit_loops": False, "sampled_degree_normalization": False,
        "canonicalization": "raw degree then coalesced original weights then disjoint colors",
        "array_storage": storage, "actual_prepared_file_bytes": measured_bytes,
        "preparation_wall_seconds": time.monotonic() - started,
        "preparation_peak_process_RSS_bytes": peak_rss_bytes(),
        "canonicalization_wall_seconds": canonical_seconds,
        "color_assignment_wall_seconds": color_assignment_seconds,
        "per_operator_prepare_hash_save_wall_seconds": seconds,
        "preparation_cpu_budget_includes_acquisition_prior": True,
        "heldout_target_values_loaded": False, "predictive_scores": None,
        "full_data_scientific_admission": False,
    }
    path = output_path(output / "PREPARED_CSR.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def audit_prepared(path, phase, data_binding):
    """Hash every prepared array once; later loaders reject size/mtime drift."""
    path = output_path(path)
    if sha256(path) != phase["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest differs from explicit phase binding")
    metadata = json.loads(path.read_text())
    if (metadata.get("schema") != SCHEMA or metadata.get("kind") != "complete_graph_prepared_CSR"
            or metadata.get("task") != phase["task"] or metadata.get("complete_graph") is not True
            or metadata["data_binding"] != data_binding):
        raise ValueError("Complete graph preparation/input identity differs")
    for row in metadata["arrays"].values():
        file = output_path(path.parent / row["file"])
        if file.parent != path.parent or file.stat().st_size != row["bytes"] or sha256(file) != row["sha256"]:
            raise ValueError("Prepared array hash/confinement failed")
        row["verified_mtime_ns"] = file.stat().st_mtime_ns
    return metadata, path.parent


def load_family(metadata, directory, arm, naive_path, device="cuda:0"):
    import torch
    from .complete_models import CompleteCSR, NoAdjointCSR, CompleteFamily, OperationLedger
    ledger = OperationLedger()
    names = (["full"] if arm in ("single_base", "exact_member") else
             ["full"] + [f"color{c}" for c in range(4)] if arm in ("candidate", "detached_current_member_mean") else
             [f"half{c}" for c in range(4)] if naive_path == "cached_half" else
             [f"color{c}" for c in range(4)])
    operators = {}
    for name in names:
        matrices = []
        directions = ("forward",) if arm == "detached_current_member_mean" and name == "full" else ("forward", "adjoint")
        for direction in directions:
            arrays = []
            for suffix in ("rowptr", "columns", "weights"):
                row = metadata["arrays"][f"{name}_{direction}_{suffix}"]
                file = output_path(directory / row["file"])
                stat = file.stat()
                if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["verified_mtime_ns"]:
                    raise ValueError("Prepared graph changed after full hash audit")
                array = np.load(file, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                    raise ValueError("Prepared array structure differs")
                # GPU copy prevents aliasing read-only host memmap pages.
                arrays.append(torch.tensor(array, device=device))
                del array
            matrices.append(torch.sparse_csr_tensor(*arrays, size=(metadata["nodes"], metadata["nodes"]),
                                                    device=device, dtype=torch.float32))
        operators[name] = (NoAdjointCSR(name, matrices[0], ledger, metadata["nodes"]) if len(matrices) == 1
                           else CompleteCSR(name, *matrices, ledger, metadata["nodes"]))
    return CompleteFamily(operators, metadata, ledger, device)
