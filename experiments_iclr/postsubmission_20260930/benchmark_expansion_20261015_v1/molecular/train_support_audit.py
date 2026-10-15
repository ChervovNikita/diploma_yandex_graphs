"""TRAIN-only label support for native molecular batches; no predictions.

The hypergeometric calculation describes a uniformly shuffled, full batch from
the observed TRAIN population. It is not an observed epoch, predictive result,
or change to any sampler/objective. No VALID/TEST labels are inspected.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def at_most_one(total, count, batch):
    def probability(k):
        if k > count or batch - k > total - count:
            return 0.0
        return math.comb(count, k) * math.comb(total - count, batch - k) / math.comb(total, batch)
    return probability(0), probability(1)


def run(expansion, output):
    import torch
    torch.set_num_threads(2)
    root = Path(expansion).resolve()
    target = Path(output).resolve()
    if not target.is_relative_to(root) or target.exists():
        raise ValueError("New own-project output required")
    bindings_path = root / "job_binding_preparation_v1/CELL_BINDINGS.json"
    bindings = json.loads(bindings_path.read_text())["cells"]
    start = time.monotonic()
    results = []
    for dataset in ["ogbg-molbace", "ogbg-molhiv"]:
        cell = bindings["ogb-" + dataset + "-gin-drop0.5"]
        authorities = cell["data_role_receipts"]
        paths = {}
        for role in ["shared_processed_graph_container", "TRAIN_indices"]:
            receipt = authorities[role]
            path = Path(receipt["path"]).resolve()
            if not path.is_relative_to(root.parent) or path.stat().st_size != receipt["bytes"] or sha(path) != receipt["sha256"]:
                raise ValueError("TRAIN data authority changed: " + role)
            paths[role] = path
        with gzip.open(paths["TRAIN_indices"], "rt") as stream:
            ids = torch.tensor([int(row[0]) for row in csv.reader(stream)], dtype=torch.long)
        assert ids.numel() == torch.unique(ids).numel()
        payload = torch.load(paths["shared_processed_graph_container"], map_location="cpu")
        data = payload[0]
        # Read only the TRAIN subset of the shared native processed y storage.
        storage = data["y"] if isinstance(data, dict) else data.y
        labels = storage.index_select(0, ids).reshape(-1)
        assert labels.numel() == ids.numel()
        observed = ~torch.isnan(labels)
        assert ((labels[observed] == 0) | (labels[observed] == 1)).all()
        counts = [int((labels == cls).sum()) for cls in (0, 1)]
        missing = int((~observed).sum())
        batch = int(cell["native_fixed_config"]["batch_size"])
        total = len(labels)
        probabilities = [at_most_one(total, count, batch) for count in counts]
        results.append({
            "dataset": dataset, "TRAIN_graphs": total, "class_counts": counts,
            "missing_TRAIN_labels": missing, "native_batch_size": batch,
            "full_batches": total // batch, "tail_graphs": total % batch,
            "random_full_batch": {
                "class_0": {"P_zero": probabilities[0][0], "P_one": probabilities[0][1],
                            "P_at_least_two": 1 - sum(probabilities[0])},
                "class_1": {"P_zero": probabilities[1][0], "P_one": probabilities[1][1],
                            "P_at_least_two": 1 - sum(probabilities[1])},
                "P_both_classes": 1 - probabilities[0][0] - probabilities[1][0] if missing == 0 else None,
            },
            "meaning": "The alignment term needs both classes. Class-conditional variance and distinct same-class donors need at least two distinct graphs of that class. Route copies do not add distinct molecules.",
            "source_receipts": {role: authorities[role] for role in paths},
        })
        del payload, data, storage, labels, observed, ids
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {"schema": "native-molecular-TRAIN-support-audit-v1", "datasets": results,
              "assumptions": "Exact finite-population probabilities for a uniformly shuffled full native batch. Actual per-epoch batch counts, tail handling and model gradient magnitudes are not measured here.",
              "wall_seconds": time.monotonic() - start, "peak_RSS_KiB_linux": usage.ru_maxrss,
              "PID": os.getpid(), "model_constructions": 0, "optimizer_steps": 0,
              "VALID_labels_inspected": False, "TEST_labels_inspected": False,
              "native_batch_or_sampler_changed": False, "binding_sha256": sha(bindings_path),
              "source_sha256": sha(Path(__file__))}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--expansion", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.expansion, args.output)
    print(json.dumps({"output": args.output, "wall_seconds": result["wall_seconds"], "model_constructions": 0}))
