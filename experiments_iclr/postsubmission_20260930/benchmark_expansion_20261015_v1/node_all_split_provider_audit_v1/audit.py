"""Materialize native split-provider roles, without constructing a model.

Only the guarded allocation copy may run. Existing processed data and author
function bytes are pinned. No native feature transform, basis generation,
optimizer, task forward or TEST metric is executed.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
from types import SimpleNamespace


def pin(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run():
    started = time.perf_counter()
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    root = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    phase = root / "experiments_iclr/postsubmission_20260930"
    here = Path(__file__).resolve().parent
    assert here == phase / "benchmark_expansion_20261015_v1/node_all_split_provider_audit_v1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    output = here / "attempt01"
    output.mkdir(exist_ok=False)
    inputs = json.loads((here / "INPUTS.json").read_text())
    for row in inputs["source_files"]:
        p = (phase / row["path_relative_to_phase"]).resolve()
        assert p.is_relative_to(root)
        actual = pin(p)
        assert all(actual[k] == row[k] for k in ("bytes", "sha256"))
    import torch
    import numpy as np
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    utils = phase / inputs["source_files"][0]["path_relative_to_phase"]
    # Compile complete unchanged author definitions; omit unrelated imports.
    names = {"index_to_mask", "take_rest", "random_splits"}
    tree = ast.parse(utils.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    scope = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(utils), "exec"), scope)
    groups = {}
    for cell_id, cell in inputs["cells"].items():
        groups.setdefault(cell["dataset"], []).append((cell_id, cell))
    records = []
    for dataset, cells in sorted(groups.items()):
        binding = cells[0][1]
        assert len(cells) == 2
        assert cells[0][1]["official_split_bindings"] == cells[1][1]["official_split_bindings"]
        data_pin = binding["data_role_receipts"]["shared_processed_graph_container"]
        data_path = Path(data_pin["path"]).resolve()
        assert data_path.is_relative_to(root)
        actual = pin(data_path)
        assert all(actual[k] == data_pin[k] for k in ("bytes", "sha256"))
        # Trusted existing provider output; feature arrays are not transformed
        # or used. Keep only y after deserializing the native provider record.
        packet = torch.load(data_path, map_location="cpu", weights_only=False)
        obj = packet[0]
        y = (obj["y"] if isinstance(obj, dict) else obj.y).detach().cpu().reshape(-1).clone()
        del obj, packet
        gc.collect()
        assert y.dtype == torch.long and int(y.min()) == 0
        classes = int(y.max()) + 1
        assert torch.equal(torch.unique(y), torch.arange(classes))
        n = len(y)
        per_class = int(round(.6 * n / classes))
        validation_count = int(round(.2 * n))
        role_cache = {}
        for split_id, split in sorted(binding["official_split_bindings"].items(), key=lambda x: int(x[0])):
            seed = split["source_split_seed"]
            first = scope["random_splits"](SimpleNamespace(y=y, num_nodes=n), classes, per_class, validation_count, seed)
            second = scope["random_splits"](SimpleNamespace(y=y, num_nodes=n), classes, per_class, validation_count, seed)
            masks = [first.train_mask, first.val_mask, first.test_mask]
            assert all(mask.dtype == torch.bool and tuple(mask.shape) == (n,) for mask in masks)
            assert all(torch.equal(getattr(first, name), getattr(second, name)) for name in ("train_mask", "val_mask", "test_mask"))
            assert not any((masks[a] & masks[b]).any().item() for a, b in ((0, 1), (0, 2), (1, 2)))
            assert torch.stack(masks).to(torch.int64).sum(0).eq(1).all().item()
            assert int(first.val_mask.sum()) == validation_count
            train_ids = torch.nonzero(first.train_mask, as_tuple=False).flatten()
            valid_ids = torch.nonzero(first.val_mask, as_tuple=False).flatten()
            assert torch.unique(y[train_ids]).numel() == classes
            role_cache[split_id] = {"TRAIN_ids": train_ids.clone(), "VALID_ids": valid_ids.clone(), "source_split_seed": seed}
            records.append({"dataset": dataset, "cell_ids": [c[0] for c in cells], "split": split_id,
                            "source_split_seed": seed, "nodes": n, "classes": classes,
                            "TRAIN_count": len(train_ids), "VALID_count": len(valid_ids),
                            "TEST_count_only": int(first.test_mask.sum()),
                            "TRAIN_per_class": torch.bincount(y[train_ids], minlength=classes).tolist(),
                            "mask_sha256": {name: hashlib.sha256(getattr(first, name).numpy().tobytes()).hexdigest()
                                             for name in ("train_mask", "val_mask", "test_mask")},
                            "repeat_materialization_bit_exact": True,
                            "original_source_operator_changed": False, "native_data_receipt": actual})
        cache = output / (dataset + "_TRAIN_VALID_roles.pt")
        assert not cache.exists()
        torch.save({"schema": "native-node-TRAIN-VALID-roles-v1", "dataset": dataset,
                    "source_utils": pin(utils), "source_data": actual, "splits": role_cache,
                    "TEST_ids_or_labels_exported": False}, cache)
        del role_cache, y, first, second, masks, train_ids, valid_ids
        gc.collect()
    assert len(records) == 40
    assert not torch.cuda.is_initialized()
    result = {"schema": "all-native-node-split-provider-audit-v1", "status": "passed",
              "unique_dataset_split_roles": 40, "native_setting_split_bindings": 80,
              "datasets": 4, "settings": 8, "records": records,
              "unchanged_source_function_AST": True, "repeat_materializations": 80,
              "mask_roles_disjoint_and_exhaustive": True,
              "native_models_constructed": 0, "task_forwards": 0, "optimizer_steps": 0,
              "TEST_forward_or_metric": False, "TEST_identity_role_count_and_digest_only": True,
              "CUDA_context_initialized": False, "full_campaign_GO": False,
              "elapsed_seconds": time.perf_counter() - started,
              "process_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "pending": ["actual full native session and all-basis/operator qualification",
                          "use source mask generator at runtime; these cache files do not replace native preprocessing",
                          "GPU/training/selected serving/resource qualification"]}
    write(output / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("status", "unique_dataset_split_roles", "native_setting_split_bindings", "elapsed_seconds", "process_peak_RSS_KiB", "native_models_constructed")}))


if __name__ == "__main__":
    run()
