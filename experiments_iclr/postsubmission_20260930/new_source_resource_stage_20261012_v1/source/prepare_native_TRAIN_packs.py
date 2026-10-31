"""CPU-only three-pack geometry preparation from existing official TRAIN packs.

No dataset provider constructor, model, optimizer, CUDA, heldout target, outcome
or new scientific fit. The root reviews and invokes this inside the allocation.
Input receipt request contains two current molecular manifests and the frozen
author-study receipt. Never modifies original data, source or author workdirs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def verify_route():
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Exact authorized allocation hostname required")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.split() != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Exact sole authorized allocation UUID required")


def local(root, relative):
    value = Path(relative)
    path = root / value
    if value.is_absolute() or ".." in value.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Project-relative confined path required")
    return path


def receipt(root, path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve().relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    path = local(root, row["path"])
    if receipt(root, path) != row:
        raise ValueError("Exact existing data/source receipt changed")
    return path


def prepare(root, request, output):
    # Set only this process's normal runtime environment before Torch import.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(2)
    author = json.loads(gzip.decompress(bound(root, request["author_study"]).read_bytes()))
    if author["source"]["git_commit"] != "61e9784ca76edeaa6e259ba0f836099608ff0586":
        raise ValueError("Pinned native OGB author source required")
    if set(request["molecular_manifests"]) != {"ogbg-molhiv", "ogbg-molbace"}:
        raise ValueError("Only both declared prospective molecular datasets")
    if output.exists():
        raise ValueError("Fresh directory required; no input or failed output overwrite")
    output.mkdir(parents=True)
    result = {"schema": "ogb-native-TRAIN-engineering-packs-preparation-v1", "datasets": {},
              "models_created": 0, "optimizer_objects": 0, "CUDA_executions": 0, "scientific_fits": 0,
              "heldout_targets_decoded": False, "scores_emitted": 0, "GO_created": False}
    for dataset in ("ogbg-molhiv", "ogbg-molbace"):
        manifest_row = request["molecular_manifests"][dataset]
        manifest = json.loads(bound(root, manifest_row).read_text())
        expected_count = 32901 if dataset == "ogbg-molhiv" else 1210
        if (manifest["dataset"] != dataset or manifest["target_policy"] != "complete_binary" or manifest.get("complete") is not True
                or manifest.get("scientific_launch_allowed") is not False
                or manifest["role_validation"].get("original_atom_bond_fields") is not True
                or manifest["role_validation"].get("graph_local_edges") is not True
                or manifest["pack_descriptives"]["train"]["graphs"] != expected_count):
            raise ValueError("Existing official TRAIN categorical graph custody changed")
        train_row = manifest["roles"]["train"]
        provider_row = manifest["provider_custody_receipt"]
        provider = json.loads(bound(root, provider_row).read_text())
        official_custody = author["data"]["dataset_custody"][dataset]
        native_raw = official_custody["raw_files"]
        # Compare receipt metadata against existing provider raw-file metadata.
        # Neither full provider constructors nor graph-label CSV are opened.
        provider_raw = {}
        pivot_name = dataset.replace("-", "_")
        for row in provider["cache_files"]:
            parts = Path(row["path"]).parts
            if pivot_name in parts:
                suffix = str(Path(*parts[parts.index(pivot_name) + 1:]))
                provider_raw[suffix] = (row["bytes"], row["sha256"])
        for row in native_raw:
            parts = Path(row["path"]).parts
            suffix = str(Path(*parts[parts.index(pivot_name) + 1:]))
            if provider_raw.get(suffix) != (row["bytes"], row["sha256"]):
                raise ValueError("Native author's bound raw/scaffold receipts differ from prepared TRAIN provider custody")
        raw_train = next(row for row in native_raw if row["path"].endswith("/split/scaffold/train.csv.gz"))
        with gzip.open(bound(root, raw_train), "rt") as stream:
            official_ids = [int(row[0]) for row in csv.reader(stream)]
        if len(official_ids) != expected_count or len(set(official_ids)) != expected_count:
            raise ValueError("Official native TRAIN role count/identity changed")
        pack = torch.load(bound(root, train_row), map_location="cpu", weights_only=True)
        if (set(pack) != {"x", "edge_index", "edge_attr", "node_ptr", "edge_ptr", "ids", "y"}
                or pack["ids"].shape != (expected_count,) or set(pack["ids"].tolist()) != set(official_ids)
                or pack["node_ptr"].shape != (expected_count + 1,) or pack["edge_ptr"].shape != (expected_count + 1,)
                or pack["x"].dtype != torch.long or pack["x"].ndim != 2 or pack["x"].shape[1] != 9
                or pack["edge_index"].dtype != torch.long or pack["edge_index"].ndim != 2 or pack["edge_index"].shape[0] != 2
                or pack["edge_attr"].dtype != torch.long or pack["edge_attr"].shape != (pack["edge_index"].shape[1], 3)
                or pack["y"].shape != (expected_count, 1) or pack["y"].dtype != torch.float32):
            raise ValueError("Existing native-format official TRAIN packed tensors changed")
        if (pack["node_ptr"][0].item() != 0 or pack["edge_ptr"][0].item() != 0
                or pack["node_ptr"][-1].item() != len(pack["x"]) or pack["edge_ptr"][-1].item() != pack["edge_index"].shape[1]):
            raise ValueError("TRAIN packed field endpoints differ")
        node_counts = pack["node_ptr"].diff().tolist()
        edge_counts = pack["edge_ptr"].diff().tolist()
        ids = pack["ids"].tolist(); positions = {identifier: index for index, identifier in enumerate(ids)}
        if min(node_counts) < 1 or min(edge_counts) < 0:
            raise ValueError("Nonempty actual TRAIN graph nodes required")
        top_nodes = sorted(range(expected_count), key=lambda i: (-node_counts[i], ids[i]))[:32]
        top_edges = sorted(range(expected_count), key=lambda i: (-edge_counts[i], ids[i]))[:32]
        tail_count = expected_count % 32
        chosen = {"top_nodes": top_nodes, "top_edges": top_edges, "tail": [positions[i] for i in official_ids[-tail_count:]]}
        source_bindings = {"molecular_manifest": manifest_row, "TRAIN_pack": train_row, "provider_custody": provider_row,
                           "native_author_raw_custody": official_custody["receipt"], "official_TRAIN_ids": raw_train}
        saved = {}
        for name, selected in chosen.items():
            graphs = []
            for index in selected:
                n0, n1 = [int(v) for v in pack["node_ptr"][index:index + 2]]
                e0, e1 = [int(v) for v in pack["edge_ptr"][index:index + 2]]
                edges = pack["edge_index"][:, e0:e1]
                if edges.numel() and (edges.min().item() < 0 or edges.max().item() >= n1-n0):
                    raise ValueError("TRAIN graph's original edge coordinates are not local")
                graphs.append({"graph_id": ids[index], "x": pack["x"][n0:n1].clone(), "edge_index": edges.clone(),
                               "edge_attr": pack["edge_attr"][e0:e1].clone(), "y": pack["y"][index:index + 1].clone()})
            path = output / (dataset + "_" + name + ".pt")
            value = {"schema": "ogb-native-TRAIN-engineering-pack-v1", "dataset": dataset, "role": "TRAIN",
                     "pack_name": name, "source_bindings": source_bindings, "graphs": graphs}
            torch.save(value, path)
            saved[name] = receipt(root, path)
        result["datasets"][dataset] = {"packs": saved, "source_bindings": source_bindings,
                                      "TRAIN_pack_custodian_verified": True,
                                      "graph_id_sets": {name: [ids[index] for index in selected] for name, selected in chosen.items()},
                                      "label_free_choice": True, "shuffle_claim": "fixed legal native TRAIN packs; not source shuffled trajectory",
                                      "coordinate_envelopes": {"maximum_possible_batch32_node_sum": sum(sorted(node_counts, reverse=True)[:32]),
                                                               "maximum_possible_batch32_edge_sum": sum(sorted(edge_counts, reverse=True)[:32]),
                                                               "mixed_batch_memory_dominance_proven": False},
                                      "pack_shapes": {name: {"graphs": len(selected), "nodes": sum(node_counts[index] for index in selected),
                                                             "edges": sum(edge_counts[index] for index in selected)} for name, selected in chosen.items()}}
        del pack
    (output / "PACKS.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify_route()
    root = Path(args.project_root).resolve()
    request = json.loads(local(root, args.request).read_text())
    expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (request.get("schema") != "root-reviewed-native-TRAIN-pack-preparation-v1" or request.get("preparer_sha256") != expected
            or request.get("root_source_review_confirmed") is not True
            or request.get("scientific_GO") is not False or request.get("source_preparation_only") is not True):
        raise ValueError("Root exact-source CPU-only pack preparation receipt required")
    output = local(root, args.output)
    started = time.monotonic()
    try:
        result = prepare(root, request, output)
        print(json.dumps({"datasets": list(result["datasets"]), "CUDA_executions": 0, "scientific_fits": 0, "elapsed_seconds": time.monotonic()-started}))
    except BaseException as error:
        if output.exists():
            (output / "FAILURE.json").write_text(json.dumps({"status": "FAILED_CPU_TRAIN_PACK_PREPARATION_RETAINED", "type": type(error).__name__,
                                                            "models_created": 0, "CUDA_executions": 0, "scientific_fits": 0}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
