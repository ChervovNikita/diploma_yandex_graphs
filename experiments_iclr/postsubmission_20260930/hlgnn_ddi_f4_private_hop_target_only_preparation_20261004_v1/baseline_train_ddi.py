"""Prospective native HL-GNN DDI training; release is disabled in this packet.

No OGB dataset loader or checkpoint loader is used. A later qualified artifact
must contain only TRAIN graph/labels and the fixed VALID candidate split.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


PIN = "0855b0de74a8f0586b8cc203e9ba4dbbb57243f4"
AUTHOR_RECIPE = {
    "data_name": "ogbl-ddi",
    "encoder": "HLGNN",
    "predictor": "MLP",
    "optimizer": "Adam",
    "loss_func": "WeightedHingeAUC",
    "neg_sampler": "global",
    "gnn_num_layers": 15,
    "mlp_num_layers": 2,
    "emb_hidden_channels": 512,
    "gnn_hidden_channels": 512,
    "mlp_hidden_channels": 512,
    "dropout": 0.3,
    "grad_clip_norm": 2.0,
    "batch_size": 65536,
    "lr": 0.001,
    "num_neg": 3,
    "epochs": 500,
    "eval_steps": 5,
    "use_lr_decay": False,
    "use_node_feats": False,
    "train_node_emb": True,
    "pretrain_emb": None,
    "use_valedges_as_input": False,
    "random_walk_augment": False,
    "alpha": 0.5,
    "init": "KI",
}


def read_release_config(path):
    config = json.loads(path.read_text())
    if config["release_enabled"] is not True:
        raise SystemExit("Release disabled: source preparation only; no training was started.")
    if config["author_commit"] != PIN or config["recipe"] != AUTHOR_RECIPE:
        raise ValueError("The pinned author DDI recipe must remain exact.")
    if config["selection_metric"] != "Hits@20":
        raise ValueError("This adapter selects the DDI VALID Hits@20 checkpoint.")
    seeds = config["seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Declare distinct nonnegative integer seeds.")
    qualification = config["qualification"]
    if not all(qualification[k] is True for k in ("train_valid_contract", "runtime", "full_budget")):
        raise ValueError("TRAIN/VALID, runtime and full-budget qualification are required.")
    if not qualification["report"]:
        raise ValueError("Declare the later qualification report.")
    contract = config["artifact_contract"]
    if contract["schema"] != "hlgnn-ddi-train-valid-v1" or type(contract["train_weight_present"]) is not bool:
        raise ValueError("Declare the TRAIN/VALID schema and actual TRAIN weight presence.")
    if not contract["path"] or not contract["sha256"]:
        raise ValueError("Declare the qualified TRAIN/VALID artifact and its SHA256.")
    artifact_path = Path(contract["path"]).expanduser().resolve()
    with artifact_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != contract["sha256"]:
        raise ValueError("TRAIN/VALID artifact SHA256 differs from the qualified contract.")
    return config, artifact_path


def load_train_valid(artifact_path, contract, torch, Data, ToSparseTensor):
    # weights_only allows tensors/primitives; this is a dataset artifact, never a model state.
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    required = {"schema", "num_nodes", "graph_edge_index", "graph_edge_weight", "train", "valid"}
    if set(payload) != required or payload["schema"] != contract["schema"]:
        raise ValueError("Artifact must contain exactly the declared TRAIN/VALID keys.")
    num_nodes = payload["num_nodes"]
    if type(num_nodes) is not int or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")
    train, valid = payload["train"], payload["valid"]
    train_keys = {"edge", "weight"} if contract["train_weight_present"] else {"edge"}
    if set(train) != train_keys or set(valid) != {"edge", "edge_neg"}:
        raise ValueError("TRAIN/VALID fields disagree with the declared loss branch.")

    def edges(value, name, transpose=False):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU int64 tensor.")
        if value.ndim != 2 or value.shape[0 if transpose else 1] != 2 or value.numel() == 0:
            raise ValueError(f"{name} has an invalid edge shape.")
        if value.min().item() < 0 or value.max().item() >= num_nodes:
            raise ValueError(f"{name} contains an out-of-range node.")

    edges(payload["graph_edge_index"], "graph_edge_index", transpose=True)
    edges(train["edge"], "train.edge")
    edges(valid["edge"], "valid.edge")
    edges(valid["edge_neg"], "valid.edge_neg")
    if valid["edge_neg"].shape[0] < 100:
        raise ValueError("Native Hits@20/50/100 needs at least 100 VALID negatives.")
    for value, count, name in (
        (payload["graph_edge_weight"], payload["graph_edge_index"].shape[1], "graph_edge_weight"),
        (train.get("weight"), train["edge"].shape[0], "train.weight"),
    ):
        if value is not None and (
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.ndim != 1 or value.shape[0] != count or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{name} must be an aligned finite CPU tensor or None.")
    if contract["train_weight_present"] and train["weight"] is None:
        raise ValueError("Declared TRAIN weights cannot be None or synthesized.")

    # The later contract owns equality of this graph to the native TRAIN graph.
    # Retain the author's ToSparseTensor conversion and edge_index reconstruction.
    data = Data(edge_index=payload["graph_edge_index"],
                edge_weight=payload["graph_edge_weight"], num_nodes=num_nodes)
    if data.edge_weight is not None:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)
    split_edge = {"train": train, "valid": valid}
    return data, split_edge, num_nodes


def run_seed(seed, config, data, split_edge, num_nodes, device, output_root, torch, np, BaseModel, Evaluator):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    recipe = config["recipe"]
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)

    # A fresh object also constructs a fresh Adam optimizer. The native reset
    # does not reset HLGNN.lin1, so object reuse cannot provide independent seeds.
    model = BaseModel(
        lr=recipe["lr"], dropout=recipe["dropout"], grad_clip_norm=recipe["grad_clip_norm"],
        gnn_num_layers=recipe["gnn_num_layers"], mlp_num_layers=recipe["mlp_num_layers"],
        emb_hidden_channels=recipe["emb_hidden_channels"], gnn_hidden_channels=recipe["gnn_hidden_channels"],
        mlp_hidden_channels=recipe["mlp_hidden_channels"], num_nodes=num_nodes, num_node_feats=0,
        gnn_encoder_name=recipe["encoder"], predictor_name=recipe["predictor"],
        loss_func=recipe["loss_func"], optimizer_name=recipe["optimizer"], device=device,
        use_node_feats=recipe["use_node_feats"], train_node_emb=recipe["train_node_emb"],
        pretrain_emb=recipe["pretrain_emb"], alpha=recipe["alpha"], init=recipe["init"],
    )
    model.param_init()
    evaluator = Evaluator(name=recipe["data_name"])
    actual_loss = "WeightedHingeAUC" if "weight" in split_edge["train"] else "AUC"
    metadata = {
        "status": "in_progress", "seed": seed, "author_commit": PIN,
        "actual_loss_branch": actual_loss, "config": config,
        "selection_metric": "VALID Hits@20", "completed_epochs": 0,
    }
    (seed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_metric, best_epoch = -math.inf, None
    with (seed_dir / "valid_history.jsonl").open("x") as history:
        for epoch in range(1, recipe["epochs"] + 1):
            loss = model.train(data, split_edge, batch_size=recipe["batch_size"],
                               neg_sampler_name=recipe["neg_sampler"], num_neg=recipe["num_neg"])
            if epoch % recipe["eval_steps"] == 0:
                results = model.validate(data, split_edge, recipe["batch_size"], evaluator)
                results = {key: float(value) for key, value in results.items()}
                metric = results["Hits@20"]
                if not all(math.isfinite(value) for value in results.values()):
                    raise RuntimeError("Nonfinite VALID metric; run is incomplete.")
                record = {"epoch": epoch, "train_loss": float(loss), "valid": results}
                history.write(json.dumps(record) + "\n")
                history.flush()
                # Strict improvement keeps the earliest epoch on an exact tie.
                if metric > best_metric:
                    best_metric, best_epoch = metric, epoch
                    torch.save({
                        "encoder": model.encoder.state_dict(), "predictor": model.predictor.state_dict(),
                        "embedding": model.emb.state_dict(), "seed": seed, "epoch": epoch,
                        "valid_hits20": metric, "actual_loss_branch": actual_loss,
                        "author_commit": PIN, "recipe": recipe,
                        "artifact_sha256": config["artifact_contract"]["sha256"],
                    }, seed_dir / "valid_best.pt")
                print(json.dumps({"seed": seed, **record}), flush=True)
    metadata.update(status="complete", completed_epochs=recipe["epochs"],
                    best_epoch=best_epoch, best_valid_hits20=best_metric)
    (seed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"seed": seed, "best_epoch": best_epoch, "best_valid_hits20": best_metric,
            "actual_loss_branch": actual_loss, "completed_epochs": recipe["epochs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    config, artifact_path = read_release_config(args.config.resolve())

    # Deliberately behind the disabled release gate. Source checks import none of these.
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToSparseTensor
    from ogb.linkproppred import Evaluator
    from native.model import BaseModel

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Declared CUDA runtime is unavailable; qualify the runtime first.")
    data, split_edge, num_nodes = load_train_valid(artifact_path, config["artifact_contract"], torch, Data, ToSparseTensor)
    data = data.to(device)
    output_root = Path(config["output_dir"]).expanduser()
    if not output_root.is_absolute():
        output_root = args.config.resolve().parent / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in config["seeds"]:
        summaries.append(run_seed(seed, config, data, split_edge, num_nodes, device,
                                  output_root, torch, np, BaseModel, Evaluator))
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
