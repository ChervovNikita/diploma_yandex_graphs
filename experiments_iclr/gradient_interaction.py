"""Post-run GNNM shared graph-layer gradient interaction on Roman Empire.

Reads only selected checkpoints and labels indexed by official TRAIN masks.
It does not select checkpoints, evaluate held-out nodes, or train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
EXP = REPO / "experiments_iclr"
ROOTS = {"SAGE": EXP / "results", "GAT": EXP / "gat_failure_pair_results"}
OUTPUT = EXP / "gradient_interaction_results"
DATA = REPO / "data" / "roman_empire.npz"
SPLITS = tuple(range(5))
DEPTH = {"SAGE": {s: 5 for s in SPLITS},
         "GAT": {0: 5, 1: 5, 2: 4, 3: 4, 4: 4}}
MEMBERS = 4
FIELDS = ("backbone", "official_mask", "selected_step", "layers", "train_nodes",
          "shared_parameter_count", "kappa", "sum_gradient_norm",
          "sum_member_gradient_norm", "member_0_norm", "member_1_norm",
          "member_2_norm", "member_3_norm", "cos_01", "cos_02", "cos_03",
          "cos_12", "cos_13", "cos_23")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row_sha256(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed result table is missing: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "model", "variant", "split", "seed", "m",
                    "num_layers", "hidden_dim", "lr", "num_steps",
                    "best_step", "checkpoint"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing required result columns: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed result row: {path}")
    return rows


def check_gat_completion(rows: list[dict[str, str]]) -> dict:
    """Check frozen GAT provenance and completion without reading predictions."""
    root = ROOTS["GAT"]
    marker = root / "run.log"
    if not marker.is_file() or not marker.read_text().strip().endswith("GAT_PAIR_COMPLETE"):
        raise RuntimeError("GAT pair has no final completion marker")
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("Frozen GAT protocol is missing")
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("model") != "GAT" or protocol.get("dataset") != "roman-empire"
            or protocol.get("members") != MEMBERS
            or protocol.get("depth_by_official_mask") !=
            {str(s): depth for s, depth in DEPTH["GAT"].items()}):
        raise ValueError("GAT protocol does not match this diagnostic")
    hashes = protocol.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("GAT protocol lacks source hashes")
    for relative, expected in hashes.items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO.resolve()) or not path.is_file():
            raise ValueError(f"Invalid frozen source path: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Frozen GAT source/data changed: {relative}")
    expected_keys = {(s, variant) for s in SPLITS
                     for variant in ("gnnm", "untied_backbone")}
    keys = [(int(row["split"]), row["variant"]) for row in rows]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("GAT pair table is incomplete or has duplicate/extra rows")
    return {"protocol_path": str(protocol_path.relative_to(REPO)),
            "protocol_sha256": sha256(protocol_path),
            "run_log_sha256": sha256(marker)}


def check_sage_audit(row: dict[str, str], checkpoint: Path) -> dict:
    """Honor the validation-only split-0 pilot/fresh adoption decision."""
    path = ROOTS["SAGE"] / "gnnm_split0_verification_audit.json"
    if not path.is_file():
        raise FileNotFoundError("SAGE split-0 selection audit is missing")
    audit = json.loads(path.read_text())
    adopted = audit.get("adopted")
    if adopted not in ("pilot", "fresh"):
        raise ValueError("SAGE split-0 audit has no valid adoption decision")
    if row_sha256(row) != audit[f"{adopted}_row_sha256"]:
        raise ValueError("SAGE split-0 selected row differs from its audit")
    if sha256(checkpoint) != audit[f"{adopted}_checkpoint_sha256"]:
        raise ValueError("SAGE split-0 checkpoint differs from its audit")
    if sha256(EXP / "projector_controls.py") != audit["runner_sha256"]:
        raise ValueError("SAGE training source differs from its audit")
    return {"audit_path": str(path.relative_to(REPO)),
            "audit_sha256": sha256(path), "adopted": adopted}


def selected_rows(backbone: str) -> tuple[list[tuple[dict[str, str], Path]], dict]:
    root = ROOTS[backbone]
    table = root / "projector_controls.csv"
    rows = read_csv(table)
    provenance = {"result_table": str(table.relative_to(REPO)),
                  "result_table_sha256": sha256(table)}
    if backbone == "GAT":
        provenance.update(check_gat_completion(rows))
    chosen: dict[int, tuple[dict[str, str], Path]] = {}
    for row in rows:
        if (row["dataset"], row["model"], row["variant"]) != ("roman-empire", backbone, "gnnm"):
            continue
        split = int(row["split"])
        if split not in SPLITS or split in chosen:
            raise ValueError(f"Duplicate or unexpected {backbone} GNNM mask {split}")
        expected = (split, DEPTH[backbone][split], 512, 3e-5, MEMBERS, 5000)
        actual = (int(row["seed"]), int(row["num_layers"]),
                  int(row["hidden_dim"]), float(row["lr"]),
                  int(row["m"]), int(row["num_steps"]))
        if (actual[:3] != expected[:3]
                or not math.isclose(actual[3], expected[3], rel_tol=0, abs_tol=1e-12)
                or actual[4:] != expected[4:]):
            raise ValueError(f"Mixed {backbone} GNNM protocol at mask {split}")
        step = int(row["best_step"])
        if not 1 <= step <= 5000 or (step != 1 and step % 10):
            raise ValueError(f"Invalid validation-selected step for {backbone} mask {split}")
        stem = f"roman-empire_{backbone}_gnnm_split{split}.pt"
        allowed = {root / "checkpoints" / stem}
        if backbone == "SAGE" and split == 0:
            allowed.add(EXP / "gnnm_split0_verification" / "checkpoints" / stem)
        recorded = Path(row["checkpoint"])
        if recorded.is_absolute() or ".." in recorded.parts:
            raise ValueError(f"Unsafe checkpoint path for {backbone} mask {split}")
        checkpoint = (REPO / recorded).resolve()
        if checkpoint not in {p.resolve() for p in allowed} or not checkpoint.is_file():
            raise FileNotFoundError(f"Selected checkpoint missing or unexpected: {recorded}")
        if checkpoint.stat().st_size == 0:
            raise ValueError(f"Empty checkpoint: {checkpoint}")
        chosen[split] = row, checkpoint
    if set(chosen) != set(SPLITS):
        raise ValueError(f"Incomplete {backbone} GNNM masks: {sorted(set(SPLITS)-set(chosen))}")
    if backbone == "SAGE":
        provenance.update(check_sage_audit(*chosen[0]))
    return [chosen[s] for s in SPLITS], provenance


def train_graph_and_labels() -> tuple[object, list[torch.Tensor], list[torch.Tensor]]:
    """Reproduce runner graph; retain only indexed TRAIN labels."""
    import dgl
    from torch_geometric.utils import from_dgl

    with np.load(DATA, allow_pickle=False) as data:
        x = np.asarray(data["node_features"], dtype=np.float32)
        edges = np.asarray(data["edges"], dtype=np.int64)
        masks = np.asarray(data["train_masks"], dtype=np.bool_)
        labels = np.asarray(data["node_labels"], dtype=np.int64)
    n = x.shape[0]
    if masks.ndim != 2:
        raise ValueError("Expected two-dimensional official train masks")
    if masks.shape[0] < masks.shape[1]:
        masks = masks.T
    if masks.shape[0] != n or masks.shape[1] < 5 or labels.shape != (n,):
        raise ValueError("Roman Empire training data shape changed")
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=n)
    graph = dgl.to_bidirected(graph)
    graph = dgl.add_self_loop(graph)
    graph = from_dgl(graph).to("cpu")
    graph.x = torch.from_numpy(x.copy())
    train_indices = []
    train_labels = []
    for split in SPLITS:
        index = torch.from_numpy(np.flatnonzero(masks[:, split]).astype(np.int64))
        if index.numel() == 0:
            raise ValueError(f"Empty official train mask {split}")
        y = torch.from_numpy(labels[index.numpy()].copy())
        train_indices.append(index)
        train_labels.append(y)
    return graph, train_indices, train_labels


def interaction(vectors: list[torch.Tensor]) -> dict[str, float]:
    if len(vectors) != MEMBERS or any(v.ndim != 1 for v in vectors):
        raise ValueError("Expected four flat member gradients")
    width = vectors[0].numel()
    if width == 0 or any(v.numel() != width or not bool(torch.isfinite(v).all())
                         for v in vectors):
        raise ValueError("Empty, mismatched, or nonfinite gradients")
    norms = [float(torch.linalg.vector_norm(v)) for v in vectors]
    if any(norm <= 0 for norm in norms):
        raise ValueError("Zero member gradient makes cosine undefined")
    summed = torch.stack(vectors).sum(dim=0)
    sum_norm = float(torch.linalg.vector_norm(summed))
    denom = sum(norms)
    kappa = sum_norm / denom
    if not math.isfinite(kappa) or kappa > 1 + 1e-10:
        raise AssertionError("Gradient interaction violates triangle inequality")
    result = {"kappa": kappa, "sum_gradient_norm": sum_norm,
              "sum_member_gradient_norm": denom}
    for member, norm in enumerate(norms):
        result[f"member_{member}_norm"] = norm
    for i, j in combinations(range(MEMBERS), 2):
        result[f"cos_{i}{j}"] = float(torch.dot(vectors[i], vectors[j]) /
                                        (norms[i] * norms[j]))
    return result


def analyze_checkpoint(backbone: str, split: int, row: dict[str, str],
                       checkpoint: Path, graph: object, index: torch.Tensor,
                       labels: torch.Tensor) -> dict[str, float | int | str]:
    from experiments_iclr.projector_controls import make_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid state dictionary: {checkpoint}")
    output_weight = state.get("output_be_block.W.weight")
    input_weight = state.get("input_be_block.W.weight")
    input_factors = state.get("input_be_block.R")
    if (not isinstance(output_weight, torch.Tensor) or output_weight.ndim != 2
            or not isinstance(input_weight, torch.Tensor)
            or tuple(input_weight.shape) != (512, graph.x.shape[1])
            or not isinstance(input_factors, torch.Tensor)
            or tuple(input_factors.shape) != (MEMBERS, graph.x.shape[1])):
        raise ValueError(f"Checkpoint architecture differs from GNNM: {checkpoint}")
    output_dim = int(output_weight.shape[0])
    if output_dim != 18 or labels.min().item() < 0 or labels.max().item() >= output_dim:
        raise ValueError(f"Class count or TRAIN labels changed: {checkpoint}")
    args = SimpleNamespace(model=backbone, num_layers=DEPTH[backbone][split],
                           hidden_dim=512, m=MEMBERS)
    model = make_model(args, "gnnm", graph.x.shape[1], output_dim,
                       torch.device("cpu"))
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    if any(module.training for module in model.modules()):
        raise AssertionError("Model did not enter deterministic eval mode")
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if name.startswith("residual_modules.") and parameter.requires_grad]
    if not named or any("input_be_block" in name or "output_be_block" in name
                        for name, _ in named):
        raise ValueError("Shared graph-layer parameter set is invalid")
    parameters = [parameter for _, parameter in named]
    vectors = []
    for member in range(MEMBERS):
        model.zero_grad(set_to_none=True)
        logits = model(graph, graph.x, tabm_seed=member)
        loss = F.cross_entropy(logits.index_select(0, index), labels)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
        vectors.append(torch.cat([g.detach().reshape(-1).to(dtype=torch.float64)
                                  for g in gradients]))
        del logits, loss, gradients
    result = {"backbone": backbone, "official_mask": split,
              "selected_step": int(row["best_step"]),
              "layers": DEPTH[backbone][split],
              "train_nodes": int(index.numel()),
              "shared_parameter_count": sum(p.numel() for p in parameters)}
    result.update(interaction(vectors))
    return result


def self_test() -> None:
    torch.manual_seed(7)
    weight = torch.tensor([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]],
                          dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1.0, -0.5, 0.2], [0.3, 1.2, -0.7],
                      [-0.4, 0.1, 0.8]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 0])
    factors = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
                            [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
                           dtype=torch.float64)
    losses = [F.cross_entropy((x * factor) @ weight, labels)
              for factor in factors]
    separate = [torch.autograd.grad(loss, weight, retain_graph=True)[0].flatten()
                for loss in losses]
    joint = torch.autograd.grad(sum(losses) / MEMBERS, weight)[0].flatten()
    torch.testing.assert_close(joint, torch.stack(separate).mean(0),
                               rtol=1e-12, atol=1e-12)
    result = interaction([g.detach() for g in separate])
    assert 0 <= result["kappa"] <= 1
    opposite = [torch.tensor(v, dtype=torch.float64) for v in
                ([1, 0], [-1, 0], [0, 1], [0, -1])]
    check = interaction(opposite)
    assert check["kappa"] == 0 and check["cos_01"] == -1
    # Exercise the actual SAGE and GAT shared stacks on a tiny CPU graph.
    from experiments_iclr.projector_controls import make_model
    graph = SimpleNamespace(
        x=torch.tensor([[1.0, 0.5, -0.2], [-0.3, 0.7, 0.4],
                        [0.2, -0.6, 0.9], [0.8, 0.1, -0.5]]),
        edge_index=torch.tensor([[0, 1, 2, 3, 0, 2, 1, 3],
                                 [1, 2, 3, 0, 0, 2, 1, 3]]),
    )
    tiny_labels = torch.tensor([0, 1, 2, 1])
    for backbone in ("SAGE", "GAT"):
        args = SimpleNamespace(model=backbone, num_layers=2, hidden_dim=8, m=MEMBERS)
        model = make_model(args, "gnnm", 3, 3, torch.device("cpu")).eval()
        shared = [p for name, p in model.named_parameters()
                  if name.startswith("residual_modules.")]
        member_losses = [F.cross_entropy(model(graph, graph.x, tabm_seed=m), tiny_labels)
                         for m in range(MEMBERS)]
        separate = [torch.cat([g.flatten() for g in
                    torch.autograd.grad(loss, shared, retain_graph=True)])
                    for loss in member_losses]
        joint = torch.cat([g.flatten() for g in torch.autograd.grad(
            sum(member_losses) / MEMBERS, shared)])
        torch.testing.assert_close(joint, torch.stack(separate).mean(0),
                                   rtol=1e-5, atol=1e-6)
        interaction([g.to(torch.float64) for g in separate])
    print(json.dumps({"self_test": "passed",
                      "accumulation": "mean member gradient equals joint mean-loss gradient",
                      "toy_kappa": result["kappa"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="CPU synthetic check only; no study files read or written")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    selected = {}
    provenance = {}
    for backbone in ("SAGE", "GAT"):
        selected[backbone], provenance[backbone] = selected_rows(backbone)
    graph, indices, labels = train_graph_and_labels()
    rows = []
    checkpoint_provenance = []
    for backbone in ("SAGE", "GAT"):
        for split, (row, checkpoint) in enumerate(selected[backbone]):
            print(f"GRADIENT {backbone} official mask {split}", flush=True)
            rows.append(analyze_checkpoint(backbone, split, row, checkpoint,
                                           graph, indices[split], labels[split]))
            checkpoint_provenance.append({
                "backbone": backbone, "official_mask": split,
                "checkpoint": str(checkpoint.relative_to(REPO)),
                "checkpoint_sha256": sha256(checkpoint),
                "selected_row_sha256": row_sha256(row),
                "train_index_sha256": hashlib.sha256(indices[split].numpy().tobytes()).hexdigest(),
                "train_label_sha256": hashlib.sha256(labels[split].numpy().tobytes()).hexdigest(),
            })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "gradient_interaction.csv"
    audit_path = OUTPUT / "gradient_interaction_provenance.json"
    csv_temp = csv_path.with_suffix(".csv.tmp")
    audit_temp = audit_path.with_suffix(".json.tmp")
    with csv_temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema": 1,
        "purpose": "Post hoc selected-checkpoint shared graph-layer gradient interaction",
        "definition": "g_m = gradient of full TRAIN cross-entropy for member m with respect to all trainable residual_modules parameters, at the same selected eval-mode checkpoint; kappa = norm(sum_m g_m) / sum_m norm(g_m)",
        "no_validation_or_test_labels_used": True,
        "dataset": str(DATA.relative_to(REPO)),
        "dataset_sha256": sha256(DATA),
        "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in
                          (Path(__file__).resolve(), EXP / "projector_controls.py",
                           REPO / "models.py", REPO / "ablation" / "models_ablation.py")},
        "input_tables": provenance,
        "checkpoints": checkpoint_provenance,
        "eval_mode": True,
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "output_csv_sha256": sha256(csv_temp),
    }
    with audit_temp.open("w") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(csv_temp, csv_path)
    os.replace(audit_temp, audit_path)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path.relative_to(REPO)),
                      "provenance": str(audit_path.relative_to(REPO))}, sort_keys=True))


if __name__ == "__main__":
    main()
