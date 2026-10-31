"""Fresh CPU autograd replay of archived PPI sharing probes, train-only.

Requires the existing torch/PyG project environment. Loads only the frozen
training dataset and its own immutable probe checkpoints. No optimizer
step or validation/test loader exists.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True

import torch

from ppi_probe import (HERE, PHASE, load_train_only, mask_manifest, masked_graphs,
                       node_masks, output_path, probe_model, research_path, sha, sources)
from recurrence_collector import TrainingUnit, collect_record, state_hash
from replay_recurrence import check_stats, replay_and_decide


def numerical_difference(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Replay structures differ")
        return max((numerical_difference(a[key], b[key]) for key in a), default=0.0)
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("Replay array lengths differ")
        return max((numerical_difference(x, y) for x, y in zip(a, b)), default=0.0)
    if not isinstance(a, (float, int)) or not isinstance(b, (float, int)):
        raise ValueError("Non-numeric replay value")
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Nonfinite replay value")
    difference = abs(a - b)
    if difference > 1e-7 * max(1.0, abs(a), abs(b)):
        raise ValueError(f"Fresh autograd replay drift: {a} versus {b}")
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, default=PHASE / "new_tasks" / "data" / "ppi" / "frozen_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = output_path(args.run_directory)
    destination = output_path(args.output)
    if destination.exists():
        raise ValueError("Refusing to overwrite an autograd replay")
    contract = json.loads((run / "contract.json").read_text())
    if contract["source_sha256"] != sources() or contract["data_manifest_sha256"] != sha(args.data_manifest):
        raise ValueError("Fresh source/data identity differs from probe contract")
    recorded_result = json.loads((run / "result.json").read_text())
    if recorded_result["status"] != "completed_train_only_probe":
        raise ValueError("Probe run did not complete")
    if sha(run / "selector_lock.json") != recorded_result["selector_lock_sha256"]:
        raise ValueError("Selector decision bytes changed")
    torch.set_num_threads(1)
    torch.manual_seed(contract["seed"])
    random.seed(contract["seed"])
    device = torch.device("cpu")
    graphs = load_train_only(args.data_manifest)
    masks = node_masks(graphs, contract["label_fraction"])
    if mask_manifest(graphs, masks) != contract["mask_manifest"]:
        raise ValueError("Observed-node mask changed")
    if sha(run / "label_masks.pt") != contract["label_masks_sha256"]:
        raise ValueError("Archived observed-node masks changed")
    mask_archive = torch.load(run / "label_masks.pt", map_location="cpu")
    if (len(mask_archive["masks"]) != len(masks)
            or mask_archive["label_fraction"] != contract["label_fraction"]
            or mask_archive["unit_ids"] != [f"ppi:train:{i:02d}" for i in range(len(graphs))]
            or any(not torch.equal(expected, actual) for expected, actual in zip(masks, mask_archive["masks"]))):
        raise ValueError("Regenerated masks differ from archived masks")
    safe_graphs = masked_graphs(graphs, masks, device)
    units = [TrainingUnit(f"ppi:train:{i:02d}", graph, graph.x, graph.y)
             for i, graph in enumerate(safe_graphs)]
    folds = contract["folds"]
    fresh_records, old_records, comparisons = [], [], []
    started = time.perf_counter()
    for epoch in (0, 10, 25):
        old = json.loads((run / f"probe_epoch{epoch}.json").read_text())
        old_records.append(old)
        checkpath = run / old["probe_checkpoint_filename"]
        if checkpath.parent != run or checkpath.name != f"probe_epoch{epoch}.pt":
            raise ValueError("Unexpected checkpoint reference")
        if sha(checkpath) != old["probe_checkpoint_sha256"]:
            raise ValueError("Archived checkpoint changed")
        checkpoint = torch.load(checkpath, map_location="cpu")
        if (checkpoint["epoch"] != epoch or checkpoint["seed"] != contract["seed"]
                or checkpoint["label_fraction"] != contract["label_fraction"]):
            raise ValueError("Checkpoint identity differs")
        if checkpoint["protocol_sha256"] != contract["protocol_sha256"]:
            raise ValueError("Checkpoint protocol differs")
        model = probe_model(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if state_hash(model) != old["model_state_sha256"]:
            raise ValueError("Loaded model state differs from collection")
        if "optimizer" not in checkpoint or "rng_cpu" not in checkpoint:
            raise ValueError("Required actual optimizer/RNG state not archived")
        fresh = collect_record(model, units, folds["probe_fit_unit_ids"], folds["partitions"],
                               epoch, contract["seed"], old["dataset_identity"])
        for field in ("schema", "dataset_identity", "optimizer_seed", "checkpoint_epoch",
                      "members", "objective", "fold_weighting", "probe_fit_unit_ids",
                      "training_unit_fingerprints", "block_parameters", "collector_source_sha256"):
            if old[field] != fresh[field]:
                raise ValueError(f"Fresh collection provenance differs in {field}")
        if len(old["partitions"]) != len(fresh["partitions"]):
            raise ValueError("Fresh partition count differs")
        maximum = 0.0
        for prior, current in zip(old["partitions"], fresh["partitions"]):
            if prior["partition_id"] != current["partition_id"]:
                raise ValueError("Fresh partition identity differs")
            for side in ("A", "B"):
                for field in ("unit_ids", "unit_count", "observed_entries", "observed_entries_by_unit",
                              "unit_weights", "fold_weighting", "loss_rule"):
                    if prior[side][field] != current[side][field]:
                        raise ValueError(f"Fresh fold provenance differs in {field}")
                maximum = max(maximum, numerical_difference(prior[side]["member_mean_loss"],
                                                            current[side]["member_mean_loss"]))
            if prior["blocks"].keys() != current["blocks"].keys():
                raise ValueError("Fresh block identities differ")
            for key in prior["blocks"]:
                check_stats(prior["blocks"][key], 4)
                check_stats(current["blocks"][key], 4)
                for field in ("C", "N", "raw_gram", "centered_gram", "member_permutation_C"):
                    maximum = max(maximum, numerical_difference(prior["blocks"][key][field],
                                                                current["blocks"][key][field]))
                for field in ("gain_A_to_B", "gain_B_to_A", "symmetric_gain"):
                    maximum = max(maximum, numerical_difference(
                        prior["blocks"][key]["virtual_native_adam_first_step"][field],
                        current["blocks"][key]["virtual_native_adam_first_step"][field]))
        fresh_records.append(fresh)
        comparisons.append({"epoch": epoch, "max_gram_or_stat_absolute_drift": maximum,
                            "checkpoint_sha256": old["probe_checkpoint_sha256"]})
    fresh_decision = replay_and_decide(fresh_records, contract["private_parameter_budget"])
    recorded_decision = replay_and_decide(old_records, contract["private_parameter_budget"])
    old_decision = json.loads((run / "selector_lock.json").read_text())
    if (fresh_decision["private_indices"] != old_decision["private_indices"]
            or recorded_decision["private_indices"] != old_decision["private_indices"]
            or old_decision["policy"] != fresh_decision["policy"]
            or old_decision["ranking_rule"] != fresh_decision["ranking_rule"]):
        raise ValueError("Fresh autograd changes selected private blocks")
    report = {"status": "PASS", "scope": "fresh training-only CPU autograd replay; no optimizer update or predictive evaluation",
              "comparisons": comparisons, "private_indices": fresh_decision["private_indices"],
              "elapsed_seconds": time.perf_counter() - started,
              "validation_loader_accessed": False, "test_loader_accessed": False,
              "contract_sha256": sha(run / "contract.json"), "source_sha256": sources(),
              "replay_source_sha256": sha(Path(__file__)), "torch_version": torch.__version__}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
