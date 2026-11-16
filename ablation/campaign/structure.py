"""Validation error structure with labels restricted to TRAIN and VALID."""
import csv
import json
from pathlib import Path
import torch
from datasets import load_dataset
from ablation.common.resumable import atomic_write_json

def analyse_structure(job_dirs, out, data_dir="data"):
    config = json.loads((Path(job_dirs[0]) / "config.json").read_text())
    data, train, valid, _, _, _, binary = load_dataset(
        config["dataset"], add_self_loops=True, device=torch.device("cpu"), data_dir=data_dir)
    split = config["split"]
    artifacts = [torch.load(Path(directory) / "predictions.pt", map_location="cpu",
                             weights_only=False) for directory in job_dirs]
    reference = artifacts[0]["roles"]["val"]
    for artifact in artifacts[1:]:
        other = artifact["roles"]["val"]
        if not torch.equal(other["indices"], reference["indices"]) or not torch.equal(other["labels"], reference["labels"]):
            raise ValueError("Structure analysis requires matched validation indices and labels")
    logits = torch.cat([item["roles"]["val"]["logits"] for item in artifacts])
    nodes = artifacts[0]["roles"]["val"]["indices"]
    labels = artifacts[0]["roles"]["val"]["labels"].long()
    predictions = (logits.squeeze(-1) > 0).long() if binary else logits.argmax(-1)
    pooled = (logits.mean(0).squeeze(-1) > 0).long() if binary else logits.mean(0).argmax(-1)
    errors = predictions.ne(labels)
    edge = data.edge_index.cpu()
    source, target = edge
    nonself = source.ne(target)
    source, target = source[nonself], target[nonself]
    degree = torch.bincount(target, minlength=data.x.shape[0]).float()
    known = train[:, split] | valid[:, split]
    known_edges = known[source] & known[target]
    known_source, known_target = source[known_edges], target[known_edges]
    count = torch.bincount(known_target, minlength=len(degree)).float()
    agreement = torch.bincount(known_target, weights=data.y[known_source].eq(data.y[known_target]).float(),
                               minlength=len(degree))
    homophily = agreement / count.clamp_min(1)
    rows = []
    for name, values, available in (
            ("degree", degree[nodes], torch.ones(len(nodes), dtype=torch.bool)),
            ("observed_label_homophily", homophily[nodes], count[nodes] > 0)):
        if not available.any():
            continue
        thresholds = torch.quantile(values[available], torch.tensor([1 / 3, 2 / 3]))
        bins = torch.bucketize(values, thresholds, right=True)
        for bin_id in range(3):
            mask = available & bins.eq(bin_id)
            n = int(mask.sum())
            if n == 0:
                continue
            row = dict(feature=name, bin=bin_id, count=n,
                       lower_threshold=float(thresholds[0]), upper_threshold=float(thresholds[1]),
                       mean_member_accuracy=float((~errors[:, mask]).float().mean()),
                       ensemble_accuracy=float(pooled[mask].eq(labels[mask]).float().mean()),
                       oracle_accuracy=float((~errors[:, mask]).any(0).float().mean()),
                       all_members_wrong=float(errors[:, mask].all(0).float().mean()))
            row["pooling_gain_accuracy"] = row["ensemble_accuracy"] - row["mean_member_accuracy"]
            rows.append(row)
    out = Path(out)
    with (out / "structure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["feature"])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(dataset=config["dataset"], model=config["model"], num_layers=config["num_layers"],
                  split=split, role="val", rows=rows,
                  label_scope="TRAIN and VALID endpoints only; no held-out labels",
                  interpretation="Descriptive error strata, fixed split, tied thresholds may produce empty bins")
    atomic_write_json(result, out / "structure.json")
    return result
