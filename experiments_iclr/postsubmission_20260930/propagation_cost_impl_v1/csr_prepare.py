"""Fixed incoming mean and color CSR preparation, fixture inputs only.

No provider, dataset path, label, split, checkpoint or remote interface exists.
The same array kernels are prospective engineering source, not measured
complete-graph feasibility. Sorting/coalescing workspace must be profiled.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .contract import SCHEMA, confined_output, fixture_shape, sha256


def _csr_fingerprint(nodes, degree, rowptr, columns, weights):
    """Edge-only fingerprint; no features, labels or predictions enter it."""
    digest = hashlib.sha256(f"incoming-mean-raw-degree-v1|{nodes}|".encode())
    for name, array, dtype in (
        ("degree", degree, "<i8"), ("rowptr", rowptr, "<i8"),
        ("columns", columns, "<i8"), ("weights", weights, "<f4"),
    ):
        canonical = np.asarray(array, dtype=dtype, order="C")
        digest.update(name.encode() + b"|" + str(canonical.size).encode() + b"|")
        digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def color_offset(fingerprint, receiver):
    payload = f"{fingerprint}|{int(receiver)}|20260930".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 4


def canonical_mean(source, receiver, nodes):
    """Coalesce raw duplicates AFTER computing original incoming degrees.

    Each coalesced weight is raw multiplicity/original degree. Existing self
    edges are retained. No implicit loops or sampled-degree normalization.
    """
    source, receiver = np.asarray(source), np.asarray(receiver)
    if source.dtype.kind not in "iu" or receiver.dtype.kind not in "iu":
        raise ValueError("Original edge IDs must be integer arrays; no silent truncation")
    fixture_shape(nodes, source.size)
    if source.ndim != 1 or receiver.shape != source.shape:
        raise ValueError("One-dimensional source/receiver arrays must match")
    if source.size and (source.min() < 0 or source.max() >= nodes
                        or receiver.min() < 0 or receiver.max() >= nodes):
        raise ValueError("Edge node ID outside the supplied node order")
    source = source.astype(np.int64, copy=False)
    receiver = receiver.astype(np.int64, copy=False)
    degree = np.bincount(receiver, minlength=nodes).astype(np.int64)
    order = np.lexsort((source, receiver))
    sorted_sources, sorted_receivers = source[order], receiver[order]
    if source.size:
        first = np.r_[True, (sorted_sources[1:] != sorted_sources[:-1])
                      | (sorted_receivers[1:] != sorted_receivers[:-1])]
        positions = np.flatnonzero(first)
        multiplicity = np.diff(np.r_[positions, source.size])
        columns = sorted_sources[positions]
        rows = sorted_receivers[positions]
        weights = (multiplicity / degree[rows]).astype(np.float32)
    else:
        columns = rows = np.empty(0, dtype=np.int64)
        multiplicity = np.empty(0, dtype=np.int64)
        weights = np.empty(0, dtype=np.float32)
    counts = np.bincount(rows, minlength=nodes)
    rowptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    fingerprint = _csr_fingerprint(nodes, degree, rowptr, columns, weights)
    return {
        "nodes": int(nodes), "raw_edges": int(source.size), "degree": degree,
        "rowptr": rowptr, "columns": columns, "weights": weights,
        "multiplicity": multiplicity, "fingerprint": fingerprint,
    }


def assign_colors(graph, chunk_entries=1_000_000):
    """Color canonical CSR entries, never raw duplicate copies separately."""
    rowptr, fingerprint = graph["rowptr"], graph["fingerprint"]
    offsets = np.zeros(graph["nodes"], dtype=np.uint8)
    for row in np.flatnonzero(np.diff(rowptr)):
        offsets[row] = color_offset(fingerprint, row)
    colors = np.empty(graph["columns"].size, dtype=np.uint8)
    for start in range(0, colors.size, chunk_entries):
        stop = min(start + chunk_entries, colors.size)
        positions = np.arange(start, stop, dtype=np.int64)
        rows = np.searchsorted(rowptr, positions, side="right") - 1
        colors[start:stop] = (positions - rowptr[rows] + offsets[rows]) % 4
    return offsets, colors


def subset_csr(graph, colors, keep):
    positions = np.flatnonzero(np.isin(colors, list(keep)))
    rows = np.searchsorted(graph["rowptr"], positions, side="right") - 1
    counts = np.bincount(rows, minlength=graph["nodes"])
    return (
        np.r_[0, np.cumsum(counts, dtype=np.int64)],
        graph["columns"][positions], graph["weights"][positions],
    )


def adjoint_csr(rowptr, columns, weights, nodes):
    """Prebuild CSR for the transpose once; no runtime edge-feature gather."""
    rows = np.repeat(np.arange(nodes, dtype=np.int64), np.diff(rowptr))
    order = np.argsort(columns, kind="stable")
    counts = np.bincount(columns, minlength=nodes)
    return (np.r_[0, np.cumsum(counts, dtype=np.int64)],
            rows[order], weights[order])


def structural_bytes(nodes, canonical_edges, cache_half_unions):
    """Exact array payload for all stored CSR+adjoints; excludes .npy headers.

    int64 rowptr/column; float32 values. Full+all colors cost 4E entries;
    additionally all four consecutive half unions cost another 4E entries.
    Includes original degrees, canonical multiplicities, color IDs and offsets.
    It excludes sort workspace, Python objects, GPU allocator and node states.
    """
    csr_count = 18 if cache_half_unions else 10
    retained_entries = canonical_edges * (8 if cache_half_unions else 4)
    csr_payload = retained_entries * 12 + csr_count * (nodes + 1) * 8
    metadata_payload = nodes * 9 + canonical_edges * 9
    return {"csr_payload_bytes": csr_payload,
            "metadata_payload_bytes": metadata_payload,
            "total_payload_bytes": csr_payload + metadata_payload,
            "sort_workspace_bytes": None, "measured_peak_bytes": None}


def prepare_fixture(source, receiver, nodes, output, *, cache_half_unions=False):
    """Write only a bounded hand-constructed fixture inside this packet."""
    fixture_shape(nodes, len(source))
    output = confined_output(output)
    if output.exists():
        raise FileExistsError("Do not overwrite a prepared fixture")
    started = time.monotonic()
    graph = canonical_mean(source, receiver, nodes)
    offsets, colors = assign_colors(graph)
    output.mkdir(parents=True)
    records = {}

    def save(name, array):
        path = confined_output(output / (name + ".npy"))
        np.save(path, array, allow_pickle=False)
        records[name] = {"file": path.name, "shape": list(array.shape),
                         "dtype": str(array.dtype), "payload_bytes": array.nbytes,
                         "file_bytes": path.stat().st_size, "sha256": sha256(path)}

    for name in ("degree", "multiplicity"):
        save(name, graph[name])
    save("color_offsets", offsets)
    save("entry_colors", colors)
    operator_names = ["full"] + [f"color{c}" for c in range(4)]
    if cache_half_unions:
        operator_names += [f"half{c}" for c in range(4)]
    nnz = {}
    for name in operator_names:
        if name == "full":
            arrays = (graph["rowptr"], graph["columns"], graph["weights"])
        elif name.startswith("color"):
            arrays = subset_csr(graph, colors, {int(name[-1])})
        else:
            color = int(name[-1])
            arrays = subset_csr(graph, colors, {color, (color + 1) % 4})
        nnz[name] = int(arrays[1].size)
        for direction, values in (
            ("forward", arrays), ("adjoint", adjoint_csr(*arrays, nodes)),
        ):
            for suffix, array in zip(("rowptr", "columns", "weights"), values):
                save(f"{name}_{direction}_{suffix}", array)
    manifest = {
        "schema": SCHEMA, "purpose": "synthetic_fixture_only",
        "nodes": int(nodes), "raw_edges": graph["raw_edges"],
        "canonical_edges": int(graph["columns"].size),
        "P_edge_only_fingerprint": graph["fingerprint"],
        "canonicalization": "raw degree then sorted/coalesced original weights then colors",
        "existing_self_edges_preserved": True, "implicit_loops": False,
        "sampled_degree_normalization": False, "cache_half_unions": cache_half_unions,
        "operator_nnz": nnz, "arrays": records,
        "storage": structural_bytes(nodes, graph["columns"].size, cache_half_unions),
        "elapsed_preparation_wall_seconds": time.monotonic() - started,
        "actual_complete_dataset_feasibility": False,
    }
    path = confined_output(output / "FIXTURE_CSR_MANIFEST.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    from .contract import refuse_real_pilot
    refuse_real_pilot()
