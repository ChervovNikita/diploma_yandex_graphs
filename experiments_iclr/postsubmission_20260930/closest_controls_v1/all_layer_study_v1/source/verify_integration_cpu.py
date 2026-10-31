"""Pending synthetic CPU check of new factories and both-readout selection/replay.

No provider construction, optimizer steps, fits, or scientific scores occur.
Engineering fixtures are retained under the new amendment, including failures.
"""
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import amendment_paths as amendment


def require(value, message):
    if not value:
        raise RuntimeError(message)


def expect_error(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError("Expected guard/refusal was missing")


def run(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA devices")
    import torch
    import torch_geometric
    from torch.nn import functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
    import common_readouts as readouts

    expected = json.loads((SOURCE / "closest_port/EXPECTED_RUNTIME.json").read_text())["runtime"]
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
               "torch_geometric": torch_geometric.__version__,
               "SAGEConv_sha256": hashlib.sha256(inspect.getsource(SAGEConv).encode()).hexdigest(),
               "LayerNorm_sha256": hashlib.sha256(inspect.getsource(torch.nn.LayerNorm).encode()).hexdigest()}
    require(runtime == expected, "Repository runtime ABI changed")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    sys.path.insert(0, str(SOURCE / "protocols"))
    sys.path.insert(0, str(SOURCE / "new_tasks"))
    import ppi_study as ppi
    import mnist_pilot as mnist
    fixture_root = amendment.output_inside(args.report.parent / (args.report.stem + "_fixtures"))
    require(not fixture_root.exists(), "Refuse existing synthetic fixture tree")
    fixture_root.mkdir(parents=True)

    # Actual new factories, ordinary forward/backward, and unchanged training loss.
    edges = torch.tensor([[0, 0, 1, 2, 4], [1, 1, 1, 0, 4]])
    graph = SimpleNamespace(edge_index=edges, num_nodes=6)
    features = torch.linspace(-1, 1, 300).reshape(6, 50)
    graph.x = features
    graph.y = torch.arange(726).reshape(6, 121).remainder(2).float()
    with patch.object(ppi, "PPI", side_effect=RuntimeError("Dataset loader forbidden")), \
         patch.object(mnist, "GNNBenchmarkDataset", side_effect=RuntimeError("Dataset loader forbidden")):
        torch.manual_seed(9)
        node_model, mode, definition = ppi.impl.make_model(readouts.ALL_LAYER_ARM, 3)
        node_logits = ppi.impl.forward(node_model, mode, graph, None)
        require(tuple(node_logits.shape) == (4, 6, 121), "PPI comparator ABI differs")
        ppi.impl.mean_member_loss(node_logits, graph.y, torch.ones(6, dtype=torch.bool)).backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in node_model.parameters()), "New PPI factory has missing/nonfinite gradients")
        first = Data(x=torch.linspace(-1, 1, 15).reshape(5, 3), edge_index=edges[:, :5], y=torch.tensor([2]))
        second = Data(x=torch.ones(1, 3), edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.tensor([7]))
        batch = Batch.from_data_list([first, second])
        torch.manual_seed(9)
        graph_model, definition = mnist.make_model(readouts.ALL_LAYER_ARM)
        graph_logits = graph_model(batch, batch.x)
        require(tuple(graph_logits.shape) == (4, 2, 10), "MNIST comparator ABI differs")
        mnist.graph_member_ce(graph_logits, batch.y).backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in graph_model.parameters()), "New MNIST factory has missing/nonfinite gradients")
        require("all_layer_be_sage_adaptation_prob" in ppi.SHARING and "all_layer_be_sage_adaptation_prob" in ppi.ARCHITECTURE, "PPI family amendment missing")

    # Independent direct probability arithmetic agrees, and an effective disagreement fixture.
    z1 = torch.tensor([-12.0, 2.0, 2.0, 2.0]).reshape(4, 1, 1)
    z2 = torch.full((4, 1, 1), 0.3)
    ppi_y = torch.ones(1, 1)
    p = readouts.ppi_metrics([z1], [ppi_y], readouts.ALL_LAYER_ARM)
    require(p["prob_micro_f1"] == 1 and p["raw_micro_f1"] == 0, "PPI pooling disagreement fixture ineffective")
    direct_ppi = F.binary_cross_entropy(z1.double().sigmoid().mean(0), ppi_y.double()).item()
    require(abs(direct_ppi - p["prob_pooled_bce"]) < 1e-12, "PPI probability BCE differs from independent arithmetic")
    z1_mnist = torch.cat((z1, torch.zeros_like(z1)), dim=-1)
    z2_mnist = torch.cat((z2, torch.zeros_like(z2)), dim=-1)
    mnist_y = torch.zeros(1, dtype=torch.long)
    q = readouts.mnist_metrics(z1_mnist, mnist_y, readouts.ALL_LAYER_ARM)
    direct_mnist = -z1_mnist.double().softmax(-1).mean(0)[0, 0].log().item()
    require(q["prob_accuracy"] == 1 and q["raw_accuracy"] == 0 and abs(q["prob_pooled_ce"]-direct_mnist) < 1e-12, "MNIST probability CE or disagreement fixture differs")
    for arm in ["gnnm", readouts.ALL_LAYER_ARM]:
        primary = readouts.primary_pooling(arm)
        value = readouts.ppi_metrics([z1], [ppi_y], arm)
        require(value["micro_f1"] == value[primary + "_micro_f1"], "PPI primary selection alias differs")
        value = readouts.mnist_metrics(z1_mnist, mnist_y, arm)
        require(value["accuracy"] == value[primary + "_accuracy"], "MNIST primary selection alias differs")
    extreme_ppi = readouts.ppi_metrics([torch.full((4, 1, 1), -1200.0)], [ppi_y], "gnnm")
    extreme_mnist = readouts.mnist_metrics(torch.tensor([[[-1200.0, 1200.0]]] * 4), mnist_y, "gnnm")
    require(all(math.isfinite(x) for x in [*extreme_ppi.values(), *extreme_mnist.values()]), "Stable probability log losses became nonfinite")

    # Both readouts select different epochs from one synthetic three-epoch trajectory.
    # A repeated epoch verifies earliest ties. This is neither a fit nor dataset scoring.
    trajectory_reports = {}
    for task in ["ppi", "mnist"]:
        for arm in ["gnnm", readouts.ALL_LAYER_ARM]:
            directory = fixture_root / (task + "_" + arm)
            directory.mkdir()
            evidence_dir = directory / ("validation_epochs" if task == "ppi" else "validation_evidence")
            evidence_dir.mkdir()
            selected = {}
            rows = []
            targets = [ppi_y] if task == "ppi" else mnist_y
            for epoch, logits in enumerate([z1, z2, z2], 1):
                logits = [logits] if task == "ppi" else torch.cat((logits, torch.zeros_like(logits)), dim=-1)
                metrics = readouts.ppi_metrics(logits, targets, arm) if task == "ppi" else readouts.mnist_metrics(logits, targets, arm)
                row = {"epoch": epoch, **metrics}
                rows.append(row)
                if task == "ppi":
                    torch.save(logits, evidence_dir / f"epoch{epoch:03d}.pt")
                    retained = logits
                else:
                    torch.save({"epoch": epoch, "member_logits": logits}, evidence_dir / f"epoch_{epoch:04d}.pt")
                    retained = {"member_logits": logits, "targets": targets, "official_graph_indices": torch.tensor([0])}
                readouts.update_selected(directory, selected, row,
                    {"state_dict": {"fixture_weight": torch.tensor([epoch])}, "epoch": epoch,
                     "validation": metrics, "contract_sha256": "synthetic-no-fit"}, retained, task, arm)
            with (directory / "validation_trace.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            replayed = readouts.verify_retained(directory, task, arm, 3, targets, "synthetic-no-fit")
            require(replayed["raw"]["epoch"] == 2 and replayed["prob"]["epoch"] == 1, "Readout-specific/earliest checkpoint selection differs")
            trajectory_reports[task + ":" + arm] = {pool: row["epoch"] for pool, row in replayed.items()}
            altered = directory.with_name(directory.name + "_tampered")
            shutil.copytree(directory, altered)
            lines = list(csv.DictReader((altered / "validation_trace.csv").open()))
            key = "prob_pooled_bce" if task == "ppi" else "prob_pooled_ce"
            lines[0][key] = str(float(lines[0][key]) + 0.125)
            with (altered / "validation_trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(lines[0]))
                writer.writeheader(); writer.writerows(lines)
            expect_error(lambda: readouts.verify_retained(altered, task, arm, 3, targets, "synthetic-no-fit"))
    expect_error(lambda: amendment.output_inside(ppi.PHASE / "new_tasks/protocols/mnist_dev_v1_runtime.json"))
    expect_error(lambda: amendment.output_inside(amendment.PACKET / "source/forbidden.json"))
    # The frozen ceiling limit requires late AND nontrivial improvement.
    for task in ["ppi", "mnist"]:
        directory = fixture_root / ("ceiling_" + task)
        directory.mkdir()
        loss_key, quality_key = ("pooled_bce", "micro_f1") if task == "ppi" else ("pooled_ce", "accuracy")
        def ceiling_fixture(selected_epoch, fall):
            rows = [{"epoch": epoch, **{pool + "_" + loss_key: 1.0 - (fall if epoch > 10 else 0.0) for pool in readouts.READOUTS},
                     **{pool + "_" + quality_key: 0.7 for pool in readouts.READOUTS}} for epoch in range(1, 21)]
            with (directory / "validation_trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            for pool in readouts.READOUTS:
                (directory / ("selected_" + pool + ".json")).write_text(json.dumps({"epoch": selected_epoch}))
            return readouts.ceiling_diagnostics(directory, task, 20)
        require(all(not value["still_improving_flag"] and value["late_and_raw_sign_sensitivity_flag"] for value in ceiling_fixture(20, 1e-7).values()), "Rounding-size drift incorrectly flags the ceiling")
        require(all(not value["still_improving_flag"] and value["nontrivial_improving_trend"] for value in ceiling_fixture(1, 1e-3).values()), "Early selected checkpoint incorrectly flags the ceiling")
        require(all(value["still_improving_flag"] for value in ceiling_fixture(20, 1e-3).values()), "Late nontrivial improvement failed to flag the ceiling")
    archived_source = fixture_root / "archived_source"
    shutil.copytree(SOURCE, archived_source)
    (archived_source / "runtime_classes").mkdir()
    (archived_source / "runtime_classes/SAGEConv.py.txt").write_text(inspect.getsource(SAGEConv))
    require(amendment.source_hashes(archived_source) == amendment.source_hashes(SOURCE),
            "Archived source binding was changed by retained runtime-class evidence")
    require(not torch.cuda.is_initialized(), "CUDA context was initialized")
    return {"status": "PASS", "scope": "new-factory and stable dual-readout/selection/retained-replay CPU engineering only",
            "source_sha256": amendment.source_hashes(SOURCE), "runtime": runtime,
            "checks": ["new_PPI_MNIST_factory_forward_backward", "independent_probability_arithmetic",
                       "both_model_primary_aliases", "extreme_logits_finite_NLL", "same_trajectory_two_readout_selection_earliest_ties",
                       "all_epoch_metrics_and_both_selected_checkpoints_replay", "tampered_probability_metric_rejected", "old_output_paths_rejected", "late_AND_nontrivial_ceiling_limit", "archived_source_binding_with_runtime_snapshots"],
            "synthetic_selected_epochs": trajectory_reports, "retained_fixture_root": str(fixture_root),
            "dataset_loads": 0, "optimizer_steps": 0, "CUDA_context_created": False,
            "historical_scores_recomputed": False, "performance_evidence": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report = amendment.output_inside(args.report)
    require(not args.report.exists(), "Refuse existing integration gate report")
    try:
        report = run(args)
    except Exception as exc:
        report = {"status": "FAIL", "scope": "synthetic engineering only", "error": type(exc).__name__ + ": " + str(exc),
                  "source_sha256": amendment.source_hashes(SOURCE), "performance_evidence": False}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
