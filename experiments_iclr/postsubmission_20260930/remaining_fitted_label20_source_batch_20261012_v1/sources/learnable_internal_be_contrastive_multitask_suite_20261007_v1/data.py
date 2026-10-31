"""Narrow TRAIN/VALID projections. No OGB/WikiCS raw dataset hydrator.

Root must prepare official role projections separately and bind their custody.
The source loader never opens TEST tensors or a provider containing all labels.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def bound(phase, row):
    phase = Path(phase).resolve()
    path = (phase / row["path"]).resolve(strict=True)
    if not path.is_relative_to(phase) or not path.is_file() or sha(path) != row["sha256"]:
        raise ValueError("Phase custody/hash mismatch")
    return path


def require_keys(tensors, keys):
    if not isinstance(tensors, dict) or set(tensors) != set(keys):
        raise ValueError("Exact role projection keys; extra held-out fields forbidden")
    if not all(isinstance(x, torch.Tensor) for x in tensors.values()):
        raise TypeError("Tensor-only weights_only projection required")


def load_projection(phase, binding, task, with_valid):
    manifest = json.loads(bound(phase, binding).read_text())
    if manifest.get("schema") != "internal-be-official-role-projection-v1" or manifest.get("task") != task:
        raise ValueError("Exact named data role authority required")
    if manifest.get("official_split_preserved") is not True or manifest.get("TEST_values_in_payload") is not False:
        raise ValueError("Official split + no TEST deserialization required")
    if set(manifest["payloads"]) != {"train", "valid"}:
        raise ValueError("TRAIN and VALID must be separate payload authorities")
    train = torch.load(bound(phase, manifest["payloads"]["train"]), map_location="cpu", weights_only=True)
    valid = (torch.load(bound(phase, manifest["payloads"]["valid"]), map_location="cpu", weights_only=True)
             if with_valid else None)
    check_projection(task, train, valid, manifest)
    return train, valid, manifest


def check_projection(task, train, valid, authority):
    if task == "wikics":
        require_keys(train, ("x", "edge_index", "ids", "y"))
        if train["x"].dtype != torch.float32 or train["x"].shape != (11701, 300):
            raise ValueError("Complete official WikiCS features")
        if len(train["ids"]) != authority["train_count"] or authority["split_index"] != 0:
            raise ValueError("Prospective pilot official split0")
        if valid is not None:
            require_keys(valid, ("ids", "y"))
            if len(valid["ids"]) != authority["valid_count"] or torch.isin(train["ids"], valid["ids"]).any():
                raise ValueError("Disjoint complete official TRAIN/VALID labels")
    elif task == "collab":
        require_keys(train, ("x", "positive"))
        if train["x"].shape != (235868, 128) or train["x"].dtype != torch.float32:
            raise ValueError("Complete official collab feature population")
        if len(train["positive"]) != authority["train_count"]:
            raise ValueError("Complete TRAIN records, no year/subgraph reduction")
        if valid is not None:
            require_keys(valid, ("positive", "negative"))
            if len(valid["positive"]) != authority["valid_count"] or len(valid["negative"]) != authority["valid_negative_count"]:
                raise ValueError("Complete official VALID shared negative population")
            if torch.isin(pair_ids(valid["positive"], len(train["x"])), pair_ids(train["positive"], len(train["x"]))).any():
                raise ValueError("VALID positives leak into support")
        for payload in (train, valid):
            if payload is not None:
                for key in ("positive", "negative"):
                    if key in payload:
                        q = payload[key]
                        if q.dtype != torch.long or q.ndim != 2 or q.shape[1] != 2 or q.min() < 0 or q.max() >= len(train["x"]):
                            raise ValueError("Canonical integer edge pairs required")
    elif task == "molhiv":
        if authority["split_kind"] != "official_scaffold" or authority["scaffold_roles_disjoint"] is not True:
            raise ValueError("Official scaffold custody required")
        for role, payload in (("train", train), ("valid", valid)):
            if payload is None:
                continue
            require_keys(payload, ("x", "edge_index", "edge_attr", "node_ptr", "edge_ptr", "ids", "y"))
            if len(payload["y"]) != authority[role + "_count"]:
                raise ValueError("All official graphs/labels required")
            n = len(payload["y"])
            if payload["node_ptr"].shape != (n+1,) or payload["edge_ptr"].shape != (n+1,):
                raise ValueError("Exact concatenated graph boundaries")
            if payload["x"].dtype != torch.long or payload["x"].shape[1] != 9 or payload["edge_attr"].dtype != torch.long or payload["edge_attr"].shape[1] != 3:
                raise ValueError("Original OGB atom/bond fields")
            if int(payload["node_ptr"][0]) != 0 or int(payload["node_ptr"][-1]) != len(payload["x"]) or int(payload["edge_ptr"][0]) != 0 or int(payload["edge_ptr"][-1]) != payload["edge_index"].shape[1]:
                raise ValueError("Complete molecular boundary coverage")
            if (payload["node_ptr"].diff() <= 0).any() or (payload["edge_ptr"].diff() < 0).any():
                raise ValueError("Invalid molecule boundaries")
        if valid is not None and torch.isin(train["ids"], valid["ids"]).any():
            raise ValueError("Repeated TRAIN/VALID molecule IDs")
    else:
        raise ValueError(task)
    for payload in (train, valid):
        if payload is None:
            continue
        if "y" in payload and (not torch.isfinite(payload["y"]).all() or len(torch.unique(payload["ids"])) != len(payload["ids"])):
            raise ValueError("Finite labels and unique complete role IDs required")


def pair_ids(pairs, nodes):
    a, b = torch.minimum(pairs[:, 0], pairs[:, 1]), torch.maximum(pairs[:, 0], pairs[:, 1])
    return a * nodes + b


def support_graph(positives, nodes, masked_queries=None):
    # Remove ALL duplicate records of every selected positive target, BOTH ways.
    # This is deliberately stricter than author's record-only minibatch removal.
    if masked_queries is not None:
        keep = ~torch.isin(pair_ids(positives, nodes), pair_ids(masked_queries, nodes))
        positives = positives[keep]
    edges = torch.cat((positives.t(), positives.flip(1).t()), 1)
    adjacency = SparseTensor(row=edges[0], col=edges[1], sparse_sizes=(nodes, nodes)).coalesce()
    if masked_queries is not None:
        row, col, _ = adjacency.coo()
        if torch.isin(pair_ids(torch.stack((row, col), 1), nodes), pair_ids(masked_queries, nodes)).any():
            raise ValueError("Positive target retained in message/NCN support graph")
    return adjacency


def molecular_batch(payload, positions, device):
    graphs = []
    for i in positions.tolist():
        ns, ne = map(int, payload["node_ptr"][i:i+2])
        es, ee = map(int, payload["edge_ptr"][i:i+2])
        edge = payload["edge_index"][:, es:ee]
        # Stored edges are graph-local; validity checked per graph, no repair.
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= ne-ns):
            raise ValueError("Molecular graph-local edge indexing contract")
        graphs.append(Data(x=payload["x"][ns:ne], edge_index=edge,
                           edge_attr=payload["edge_attr"][es:ee]))
    return {"graph": Batch.from_data_list(graphs).to(device)}, payload["y"][positions].flatten().to(device)


def batches(task, train, recipe, epoch, seed, device):
    gen = torch.Generator().manual_seed(seed + 19709 + 1000*epoch)
    if task == "wikics":
        yield {"x": train["x"].to(device), "edge_index": train["edge_index"].to(device),
               "ids": train["ids"].to(device)}, train["y"].to(device)
    elif task == "molhiv":
        order = torch.randperm(len(train["y"]), generator=gen)
        for ids in order.split(recipe["batch_size"]):
            yield molecular_batch(train, ids, device)
    elif task == "collab":
        positives = train["positive"].to(device)
        raw_edges = torch.cat((positives.t(), positives.flip(1).t()), 1)
        # TRAIN-only negative authority; no future-positive rejection via held labels.
        negatives = negative_sampling(raw_edges, num_nodes=len(train["x"]), num_neg_samples=len(positives)).t()
        if len(negatives) < len(positives):
            raise ValueError("Incomplete negative population, no replacement rescue")
        order = torch.randperm(len(positives), generator=gen)
        for ids in order.split(recipe["batch_size"]):
            # Include complete tail; changed from native drop-last, disclosed.
            p = positives[ids.to(device)]
            q = torch.cat((p, negatives[ids.to(device)]), 0)
            yield {"x": train["x"].to(device), "adj": support_graph(positives, len(train["x"]), p), "query": q}, torch.cat((torch.ones(len(p)), torch.zeros(len(p)))).to(device)
    else:
        raise ValueError(task)


def valid_batches(task, train, valid, recipe, device):
    if task == "wikics":
        yield {"x": train["x"].to(device), "edge_index": train["edge_index"].to(device), "ids": valid["ids"].to(device)}, valid["y"]
    elif task == "molhiv":
        for ids in torch.arange(len(valid["y"])).split(recipe["batch_size"]):
            yield molecular_batch(valid, ids, device)
    elif task == "collab":
        q = torch.cat((valid["positive"], valid["negative"]), 0)
        adj = support_graph(train["positive"].to(device), len(train["x"]))
        for ids in torch.arange(len(q)).split(recipe["eval_batch_size"]):
            yield {"x": train["x"].to(device), "adj": adj, "query": q[ids].to(device)}, ids
