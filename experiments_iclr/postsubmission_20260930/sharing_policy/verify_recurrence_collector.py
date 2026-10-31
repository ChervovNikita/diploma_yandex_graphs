"""Synthetic CPU checks for collector/replay; no downloaded data or training."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "protocols"))

import torch
from torch_geometric.data import Data

from baseline_adapters import ProjectorEnsemble, SelectiveProjectorEnsemble, model_options
from recurrence_collector import (TrainingUnit, block_statistics, collect_record, fold_gradients,
                                  loss_sum_and_count, save_record, virtual_native_adam_first_step)
from replay_recurrence import replay_and_decide
from ppi_probe import mask_manifest, masked_graphs, node_masks, train_member_loss


def require_rejection(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Expected invalid collector/replay input to be rejected")


def check_equal_unit_gradient(model, units):
    """Unequal observed sizes: compare accumulated gradients to direct losses."""
    model.eval()
    masked_units = []
    for index, unit in enumerate(units):
        targets = unit.targets.clone()
        targets[1 + index % 3:] = float("nan")
        masked_units.append(TrainingUnit(unit.unit_id, unit.graph, unit.features, targets))
    by_id = {unit.unit_id: unit for unit in masked_units}
    ids = [unit.unit_id for unit in masked_units]
    blocks = model.probe_blocks()
    parameters = tuple(p for block in blocks.values() for p in block)
    collected, metadata = fold_gradients(model, by_id, ids, blocks, "bce")
    largest_error, largest_pooling_difference = 0.0, 0.0
    for member in range(model.members):
        losses_and_counts = [loss_sum_and_count(model.forward_member(u.graph, u.features, member),
                                               u.targets, "bce") for u in masked_units]
        direct_loss = sum(loss / count for loss, count in losses_and_counts) / len(ids)
        pooled_loss = sum(loss for loss, _ in losses_and_counts) / sum(count for _, count in losses_and_counts)
        direct = torch.autograd.grad(direct_loss, parameters, retain_graph=True)
        pooled = torch.autograd.grad(pooled_loss, parameters)
        offset = 0
        for key, block in blocks.items():
            expected = torch.cat([g.detach().reshape(-1).double().cpu() for g in direct[offset:offset + len(block)]])
            pooled_vector = torch.cat([g.detach().reshape(-1).double().cpu() for g in pooled[offset:offset + len(block)]])
            torch.testing.assert_close(collected[key][member], expected, rtol=1e-11, atol=1e-12)
            largest_error = max(largest_error, float((collected[key][member] - expected).abs().max()))
            largest_pooling_difference = max(largest_pooling_difference, float((expected - pooled_vector).abs().max()))
            offset += len(block)
        assert abs(metadata["member_mean_loss"][member] - float(direct_loss.detach())) < 1e-12
    assert len(set(metadata["observed_entries_by_unit"])) > 1
    assert largest_pooling_difference > 1e-10, "Toy must distinguish equal-unit from entry pooling"
    model.train()
    return {"status": "PASS", "observed_entries_by_unit": metadata["observed_entries_by_unit"],
            "max_direct_gradient_absolute_error": largest_error,
            "max_entry_pooled_gradient_difference": largest_pooling_difference,
            "masked_unobserved_targets": "NaN"}


def check_masked_ppi_helpers(units):
    graphs = [u.graph for u in units]
    scarce, full = node_masks(graphs, 0.1), node_masks(graphs, 1.0)
    safe = masked_graphs(graphs, scarce, torch.device("cpu"))
    for graph, original, mask, full_mask in zip(safe, graphs, scarce, full):
        assert bool((~mask | full_mask).all())
        assert bool(torch.isnan(graph.y[~mask]).all())
        torch.testing.assert_close(graph.y[mask], original.y[mask], rtol=0, atol=0)
        logits = torch.linspace(-1, 1, 4 * graph.num_nodes * 121, dtype=torch.float64).reshape(4, graph.num_nodes, 121)
        actual = train_member_loss(logits, graph.y, mask)
        expected = sum(loss_sum_and_count(row, graph.y, "bce")[0] /
                       loss_sum_and_count(row, graph.y, "bce")[1] for row in logits) / 4
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    require_rejection(lambda: masked_graphs(graphs, scarce[:-1], torch.device("cpu")))
    require_rejection(lambda: mask_manifest(graphs, scarce[:-1]))
    return {"status": "PASS", "nested_observed_masks": True,
            "hidden_labels_physically_NaN": True, "mean_member_BCE_matches_direct": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if HERE not in output.parents or output.exists():
        raise SystemExit("Use a fresh output directory inside sharing_policy")
    torch.set_num_threads(1)
    torch.manual_seed(930701)
    units = []
    for i in range(6):
        nodes = 8 + i
        features = torch.randn(nodes, 7, dtype=torch.float64)
        targets = torch.randint(2, (nodes, 121)).double()
        graph = Data(x=features, y=targets, edge_index=torch.randint(nodes, (2, 2 * nodes + 5)))
        units.append(TrainingUnit(f"toy:train:{i:02d}", graph, features, targets))
    fit = [u.unit_id for u in units[:2]]
    q = [u.unit_id for u in units[2:]]
    partitions = [
        {"partition_id": "701", "A": q[:2], "B": q[2:]},
        {"partition_id": "702", "A": q[::2], "B": q[1::2]},
        {"partition_id": "703", "A": [q[0], q[3]], "B": [q[1], q[2]]},
    ]
    original = ProjectorEnsemble(model_options(7, 121, hidden_dim=16, layers=2, dropout=0.4)).double()
    model = SelectiveProjectorEnsemble(original).double()
    # Intentionally use train mode and nonzero dropout. Collection must
    # temporarily use eval mode and restore mode/RNG without draws.
    model.train()
    sentinel = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.full_like(parameter, 0.137)
        sentinel[name] = parameter.grad.clone()
    records = [collect_record(model, units, fit, partitions, epoch, 0,
                              {"kind": "synthetic-preflight-only", "epochs": "mock states"})
               for epoch in (0, 10, 25)]
    assert model.training and all(module.training for module in model.modules())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, sentinel[name], rtol=0, atol=0)
    budget = 3 * records[0]["block_parameters"]["0"]["count"]
    result = replay_and_decide(records, budget)
    assert result["spent_private_parameters"] <= budget
    equal_unit_check = check_equal_unit_gradient(model, units[2:])
    masked_helpers_check = check_masked_ppi_helpers(units)
    positive = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    positive_stats = block_statistics(positive, positive)
    assert positive_stats["C"] == 1.25 and positive_stats["N"] == 0
    adam_check = virtual_native_adam_first_step(positive, positive)
    assert abs(adam_check["symmetric_gain"]) < 1e-10
    assert adam_check["gain_A_to_B"] == adam_check["gain_B_to_A"]
    tampered = copy.deepcopy(records)
    tampered[0]["partitions"][0]["blocks"]["0"]["C"] += 1
    require_rejection(lambda: replay_and_decide(tampered, budget))
    bad_weights = copy.deepcopy(records)
    bad_weights[0]["partitions"][0]["A"]["unit_weights"][0] = 1.0
    require_rejection(lambda: replay_and_decide(bad_weights, budget))
    bad_counts = copy.deepcopy(records)
    bad_counts[0]["partitions"][0]["A"]["observed_entries"] += 1
    require_rejection(lambda: replay_and_decide(bad_counts, budget))
    require_rejection(lambda: replay_and_decide(records[:2], budget))
    bad_partitions = copy.deepcopy(partitions)
    bad_partitions[0]["A"][0] = fit[0]
    require_rejection(lambda: collect_record(model, units, fit, bad_partitions, 0, 0, {}))
    nontraining = list(units)
    unit = units[-1]
    nontraining[-1] = TrainingUnit(unit.unit_id, unit.graph, unit.features, unit.targets, "val")
    require_rejection(lambda: collect_record(model, nontraining, fit, partitions, 0, 0, {}))
    with torch.no_grad():
        for block in (model.model.input_be_block, model.model.output_be_block):
            for value in (block.R, block.S, block.B):
                value[:] = value[0].clone()
    duplicate_records = [collect_record(model, units, fit, partitions, epoch, 0,
                                       {"kind": "synthetic-identical-members"})
                         for epoch in (0, 10, 25)]
    duplicate_result = replay_and_decide(duplicate_records, budget)
    assert duplicate_result["private_indices"] == []
    maximum_duplicate = max(abs(stats[key]) for record in duplicate_records
                            for partition in record["partitions"]
                            for stats in partition["blocks"].values() for key in ("C", "N"))
    assert maximum_duplicate == 0
    output.mkdir(parents=True)
    for record in records:
        save_record(record, output / f"toy_epoch{record['checkpoint_epoch']}.json")
    for record in duplicate_records:
        save_record(record, output / f"identical_epoch{record['checkpoint_epoch']}.json")
    files = [Path(__file__), HERE / "recurrence_collector.py", HERE / "replay_recurrence.py",
             HERE / "ppi_probe.py", HERE / "replay_ppi_probe_gradients.py",
             HERE.parent / "protocols" / "baseline_adapters.py",
             HERE.parent / "source_snapshot" / "models.py"]
    report = {"status": "PASS", "scope": "synthetic CPU implementation check only",
              "torch_version": torch.__version__, "mock_checkpoint_epochs": [0, 10, 25],
              "whole_training_graph_units": len(units), "objective": "BCE with 121 labels",
              "regular_decision": result, "identical_member_decision": duplicate_result,
              "identical_member_max_C_or_N": maximum_duplicate,
              "equal_unit_direct_gradient_check": equal_unit_check,
              "masked_ppi_helper_check": masked_helpers_check,
              "all_positive_native_adam_counterexample": {"status": "PASS", "C": positive_stats["C"],
                                                           "N": positive_stats["N"], "diagnostic": adam_check},
              "invalid_input_controls": ["Gram/stat tamper", "unit weight tamper", "observed count tamper",
                                         "missing checkpoint", "fit/Q overlap", "validation unit"],
              "dropout_mode_rng_and_parameter_grad_preservation": "PASS",
              "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files}}
    with (output / "collector_check.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
