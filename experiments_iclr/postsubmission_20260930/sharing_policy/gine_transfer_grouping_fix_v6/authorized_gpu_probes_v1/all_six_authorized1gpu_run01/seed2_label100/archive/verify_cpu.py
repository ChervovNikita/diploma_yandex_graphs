"""One-batch GINE autograd gate, synthetic unequal graph/target counts.

No dataset/provider/download, GPU, optimizer step or predictive fitting.
The production recipe is counted, while float64 smaller-width graphs
check batched grams against direct per-graph and direct batch gradients.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
from common import HERE, confined, sha, source_hashes, write_json
import torch
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from gine_adapters import MolecularProjectorEnsemble, gine_options, copied_private, parameter_count
from recurrence_collector import TrainingUnit, fold_gradients, state_hash, tensor_hash, block_statistics
from replay_recurrence import replay_and_decide, check_stats
from gine_collector import collect_record, fold_gradients_batched, block_contract, equal_graph_bce
from group_folds import grouped_folds, validate_grouped


def assert_reject(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    raise AssertionError("Guard failed to reject invalid input")


def fixture():
    graphs = []
    for i, nodes in enumerate((3, 8, 5, 9, 4, 7, 2, 10, 6, 11, 12, 13)):
        x = torch.zeros(nodes, 9, dtype=torch.long)
        x[:, 0] = torch.arange(nodes) % 6 + i % 4
        x[:, 1] = (torch.arange(nodes) + i) % 4
        edges = [(j, (j + 1) % nodes) for j in range(nodes)]
        edges += [(target, source) for source, target in edges]
        edge_index = torch.tensor(edges).T.contiguous()
        bonds = torch.zeros(len(edges), 3, dtype=torch.long)
        bonds[:, 0] = torch.arange(len(edges)) % 3
        y = torch.tensor([[float(i % 2), float((i + 1) % 2), float(i % 3 == 0)]], dtype=torch.float64)
        # Unequal1/2/3 observed targets, with physically erased hidden labels.
        y[0, (i % 3) + 1:] = float("nan")
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=bonds, y=y))
    mapping = {f"gate:train:{i}": f"group:{i // 2}" for i in range(len(graphs))}
    folds = grouped_folds(mapping)
    units = [TrainingUnit(f"gate:train:{i}", graph, graph.x, graph.y) for i, graph in enumerate(graphs)]
    return units, folds


def direct_batch_gradients(model, by_id, ids, blocks):
    batch = Batch.from_data_list([by_id[key].graph for key in ids])
    params = tuple(p for block in blocks.values() for p in block)
    rows = {key: [] for key in blocks}
    for member in range(4):
        logits = model.forward_member(batch, batch.x, member)
        # Independent explicit per-graph reduction; do not call tested loss helper.
        means = []
        for graph in range(len(ids)):
            finite = torch.isfinite(batch.y[graph])
            means.append(F.binary_cross_entropy_with_logits(logits[graph, finite], batch.y[graph, finite]))
        gradients = torch.autograd.grad(torch.stack(means).mean(), params)
        offset = 0
        for key, block in blocks.items():
            rows[key].append(torch.cat([g.detach().reshape(-1) for g in gradients[offset:offset + len(block)]]))
            offset += len(block)
    return {key: torch.stack(value) for key, value in rows.items()}


def run():
    torch.set_num_threads(1)
    torch.manual_seed(8217)
    random.seed(8217)
    units, folds = fixture()
    validate_grouped(folds)
    assert_reject(lambda: validate_grouped({**folds, "probe_fit_unit_ids": folds["probe_fit_unit_ids"] + folds["score_unit_ids"][:1]}))
    model = MolecularProjectorEnsemble(gine_options(output_dim=3, hidden_dim=16, num_layers=3,
                                                    dropout=0.1, hidden_dim_multiplier=2), members=4).double()
    model.eval()
    by_id, blocks = {u.unit_id: u for u in units}, model.probe_blocks()
    block_contract(model)
    comparisons = []
    for part in folds["partitions"]:
        both = {}
        for side in ("A", "B"):
            ids = part[side]
            chunked, meta = fold_gradients_batched(model, by_id, ids, blocks, batch_size=3)
            unchunked, _ = fold_gradients_batched(model, by_id, ids, blocks, batch_size=128)
            sequential, oldmeta = fold_gradients(model, by_id, ids, blocks, objective="bce")
            direct = direct_batch_gradients(model, by_id, ids, blocks)
            assert meta == oldmeta or (meta["unit_ids"] == oldmeta["unit_ids"] and
                meta["observed_entries_by_unit"] == oldmeta["observed_entries_by_unit"])
            maximum = 0.0
            for key in blocks:
                for candidate in (unchunked[key], sequential[key], direct[key]):
                    assert torch.allclose(chunked[key], candidate, atol=2e-11, rtol=2e-10)
                    maximum = max(maximum, float((chunked[key] - candidate).abs().max()))
            both[side] = chunked
            comparisons.append({"partition": part["partition_id"], "side": side, "graphs": len(ids),
                                "nodes_by_graph": [by_id[key].graph.num_nodes for key in ids],
                                "finite_targets_by_graph": meta["observed_entries_by_unit"],
                                "maximum_gradient_error": maximum})
        for key in blocks:
            check_stats(block_statistics(both["A"][key], both["B"][key]), 4)
    # NaN erasure and gradients: unequal label-count graph means have distinct loss from entry weighting.
    example = torch.tensor([[[0.1, -0.8, 1.7], [-2.1, 0.3, 2.9]]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1., 0., float("nan")], [0., float("nan"), float("nan")]], dtype=torch.float64)
    loss = equal_graph_bce(example, targets)
    expected = (F.binary_cross_entropy_with_logits(example[0, 0, :2], targets[0, :2]) +
                F.binary_cross_entropy_with_logits(example[0, 1, :1], targets[1, :1])) / 2
    assert torch.allclose(loss, expected, atol=1e-15, rtol=1e-15)
    loss.backward()
    hidden = ~torch.isfinite(targets)
    assert torch.equal(example.grad[0, hidden], torch.zeros_like(example.grad[0, hidden]))
    assert torch.isfinite(example.grad).all()
    assert_reject(lambda: equal_graph_bce(example, torch.full_like(targets, float("nan"))))
    finite = torch.isfinite(targets)
    pooled_entry_loss = F.binary_cross_entropy_with_logits(example[0, finite], targets[finite])
    assert not torch.allclose(loss, pooled_entry_loss)
    # Preserve parameters, optimizer-gradient buffers, all module modes and CPU/Python RNG.
    model.train()
    model.blocks[1].normalization.eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.031
    before = state_hash(model)
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    modes = [module.training for module in model.modules()]
    grad_hashes = [tensor_hash(p.grad) for p in model.parameters()]
    record = collect_record(model, units, folds["probe_fit_unit_ids"], folds["partitions"],
                            0, 0, {"dataset": "synthetic-engineering-only"}, batch_size=3)
    assert before == state_hash(model) and torch.equal(rng, torch.get_rng_state())
    assert python_rng == random.getstate() and modes == [module.training for module in model.modules()]
    assert grad_hashes == [tensor_hash(p.grad) for p in model.parameters()]
    for part in record["partitions"]:
        for stats in part["blocks"].values():
            check_stats(stats, 4)
    records = []
    for epoch in (0, 10, 25):
        # Engineering fixture only: unchanged checkpoint is not a training trajectory.
        value = copy.deepcopy(record)
        value["checkpoint_epoch"] = epoch
        records.append(value)
    decision = replay_and_decide(records, 3 * record["block_parameters"]["0"]["count"])
    assert decision["policy"] == "recurrence-budget-v2"
    assert len(decision["private_indices"]) <= 1
    null_records = copy.deepcopy(records)
    for value in null_records:
        for part in value["partitions"]:
            for key in value["block_parameters"]:
                identical = torch.ones(4, value["block_parameters"][key]["count"], dtype=torch.float64)
                part["blocks"][key] = block_statistics(identical, identical)
    assert replay_and_decide(null_records, 3 * record["block_parameters"]["0"]["count"])["private_indices"] == []
    # Production width128: added storage includes all learned bond embeddings.
    with torch.random.fork_rng(devices=[]):
        production = MolecularProjectorEnsemble(gine_options(), members=4)
        contract = block_contract(production)
        original_count = parameter_count(production)
        cost_checks = []
        production.eval()
        production_batch = Batch.from_data_list([Data(x=u.graph.x, edge_index=u.graph.edge_index,
                                               edge_attr=u.graph.edge_attr) for u in units[:3]])
        original_logits = production(production_batch, production_batch.x)
        for index in range(3):
            private = copied_private(production, [index]).eval()
            actual = parameter_count(private) - original_count
            expected_cost = 3 * contract[str(index)]["count"]
            assert actual == expected_cost
            assert torch.equal(original_logits, private(production_batch, production_batch.x))
            source_bond = production.blocks[index].bond_encoder
            for member in range(4):
                copied_bond = private.private_blocks[str(index)][member].bond_encoder
                for a, b in zip(source_bond.parameters(), copied_bond.parameters()):
                    assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
            cost_checks.append({"private_index": index, "block_parameters": contract[str(index)]["count"],
                                "actual_added_parameters": actual, "private_parameters": parameter_count(private),
                                "all_bond_encoders_copied": True, "initial_forward_identity": True})
    # Native mean-member gradient equals sum of4 copied private gradients, with bonds included.
    model.eval()
    private = copied_private(model, [1]).eval()
    batch = Batch.from_data_list([u.graph for u in units[:4]])
    shared_grad = torch.autograd.grad(equal_graph_bce(model(batch, batch.x), batch.y), tuple(model.blocks[1].parameters()))
    member_params = [tuple(private.private_blocks["1"][m].parameters()) for m in range(4)]
    flat_params = tuple(p for group in member_params for p in group)
    private_grad = torch.autograd.grad(equal_graph_bce(private(batch, batch.x), batch.y), flat_params)
    size = len(member_params[0])
    summed = [sum(private_grad[m * size + j] for m in range(4)) for j in range(size)]
    assert all(torch.allclose(a, b, atol=2e-11, rtol=2e-10) for a, b in zip(shared_grad, summed))
    return {"status": "passed", "scope": "CPU synthetic GINE engineering; no predictive fit",
            "dataset_accessed": False, "optimizer_steps": 0, "CUDA_context_created": False,
            "source_sha256": source_hashes(), "torch_version": torch.__version__,
            "unequal_graph_and_label_count_comparisons": comparisons,
            "production_width128_block_costs": cost_checks, "tied_parameters": original_count,
            "unchanged_selector_called": True, "selector_fixture_is_not_a_fit": True,
            "hidden_NaN_targets_zero_gradient": True, "state_gradients_modes_and_rng_preserved": True,
            "native_shared_gradient_equals_sum_copied_private_gradients": True,
            "real_training_bundle_gate_pending": True}


def real_batch_gate(bundle):
    """Actual frozen MolHIV graph inputs, one batch, zero optimizer steps."""
    from probe import prepare, new_model
    units, _, mask, _ = prepare(confined(bundle), 0.1)
    selected = sorted(units, key=lambda unit: unit.unit_id)[:128]
    if len(selected) != 128:
        raise ValueError("Require a full128 observed-training-graph engineering batch")
    payload = torch.load(confined(bundle), map_location="cpu", weights_only=False)
    official_groups = payload["group_by_official_train_id"]
    mapping = {unit.unit_id: official_groups[str(int(unit.unit_id.split(":")[-1]))] for unit in selected}
    folds = grouped_folds(mapping)
    model = new_model().eval()
    by_id, blocks = {unit.unit_id: unit for unit in selected}, model.probe_blocks()
    before, rng = state_hash(model), torch.get_rng_state().clone()
    comparisons = []
    # All128 complete graphs fit in a single engineering batch; fold chunks use the exact graph mean.
    # MolHIV has one target, so unequal-target-count semantics are covered by the separate synthetic gate.
    for part in folds["partitions"]:
        both = {}
        for side in ("A", "B"):
            ids = part[side]
            actual, metadata = fold_gradients_batched(model, by_id, ids, blocks, batch_size=128)
            direct = direct_batch_gradients(model, by_id, ids, blocks)
            maximum = 0.0
            for key in blocks:
                if not torch.allclose(actual[key], direct[key].double(), atol=1e-7, rtol=2e-4):
                    raise AssertionError("Real MolHIV batch autograd differs from direct graph-mean gradients")
                maximum = max(maximum, float((actual[key] - direct[key]).abs().max()))
            both[side] = actual
            comparisons.append({"partition": part["partition_id"], "side": side,
                                "graphs": len(ids), "maximum_gradient_error": maximum,
                                "finite_targets_by_graph": metadata["observed_entries_by_unit"]})
        for key in blocks:
            check_stats(block_statistics(both["A"][key], both["B"][key]), 4)
    record = collect_record(model, selected, folds["probe_fit_unit_ids"], folds["partitions"],
                            0, 0, {"dataset": "MolHIV-real-training-engineering-only", "bundle_sha256": sha(bundle)}, batch_size=128)
    assert before == state_hash(model) and torch.equal(rng, torch.get_rng_state())
    nodes = [unit.graph.num_nodes for unit in selected]
    assert len(set(nodes)) > 1
    return {"status": "passed", "scope": "one128-graph real training batch, no optimizer/predictive fit",
            "train_bundle_sha256": sha(bundle), "label_mask": mask,
            "batch_selection": "first128 sorted observed official train unit IDs under fixed tenpercent mask",
            "unit_ids": [unit.unit_id for unit in selected], "node_counts": nodes,
            "nodes": sum(nodes), "directed_edges": sum(unit.graph.num_edges for unit in selected),
            "grouped_folds": folds, "direct_autograd_comparisons": comparisons,
            "one_target_per_graph": True, "unequal_target_count_gate": "separate synthetic actual GINE autograd",
            "state_and_rng_unchanged": record["state_and_rng_unchanged"],
            "optimizer_steps": 0, "predictive_metrics": None, "heldout_data_accessed": False,
            "not_a_representative_accuracy_pilot": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-bundle", type=Path)
    args = parser.parse_args()
    report = run()
    if args.train_bundle is not None:
        report["real_training_batch"] = real_batch_gate(args.train_bundle)
        report["real_training_bundle_gate_pending"] = False
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
