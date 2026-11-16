"""Official OGB split order and NCN's unweighted adjacency construction."""
import hashlib
from dataclasses import dataclass

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor


@dataclass
class Bundle:
    x: torch.Tensor
    edge_index: torch.Tensor
    split: dict
    adj: SparseTensor
    full_adj: SparseTensor | None
    identity: str
    test_edge_order_sha256: str | None = None

    @property
    def num_nodes(self):
        return self.x.shape[0]


def tensor_identity(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_bundle(data_dir, device, release_test=False):
    dataset = PygLinkPropPredDataset(name="ogbl-collab", root=str(data_dir))
    graph = dataset[0]
    split = dataset.get_edge_split()
    # Dataset storage contains the published splits. Only train/valid tensors
    # are consumed during training and smoke evaluation.
    used = {
        "train": {"edge": split["train"]["edge"]},
        "valid": {"edge": split["valid"]["edge"], "edge_neg": split["valid"]["edge_neg"]},
    }
    identity = tensor_identity([
        ("x", graph.x), ("edge_index", graph.edge_index),
        *[(f"{part}.{name}", value) for part, fields in used.items() for name, value in fields.items()],
    ])
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.shape[0], x.shape[0]))
    adj = adj.to_symmetric().coalesce()
    full_adj = None
    test_edge_order_sha256 = None
    if release_test:
        used["test"] = {"edge": split["test"]["edge"], "edge_neg": split["test"]["edge_neg"]}
        test_edge_order_sha256 = tensor_identity(
            [(f"test.{name}", value) for name, value in used["test"].items()])
        full_edges = torch.cat([edge_index, split["valid"]["edge"].t().to(device)], dim=-1)
        full_adj = SparseTensor.from_edge_index(full_edges, sparse_sizes=adj.sizes()).coalesce().to_symmetric()
    used = {part: {name: value.to(device) for name, value in fields.items()}
            for part, fields in used.items()}
    return Bundle(x, edge_index, used, adj, full_adj, identity, test_edge_order_sha256)


def masked_adjacency(bundle, permutation):
    positive = bundle.split["train"]["edge"].t()
    mask = torch.ones(positive.shape[1], dtype=torch.bool, device=positive.device)
    mask[permutation] = False
    return SparseTensor.from_edge_index(
        positive[:, mask], sparse_sizes=(bundle.num_nodes, bundle.num_nodes)).to_symmetric()
