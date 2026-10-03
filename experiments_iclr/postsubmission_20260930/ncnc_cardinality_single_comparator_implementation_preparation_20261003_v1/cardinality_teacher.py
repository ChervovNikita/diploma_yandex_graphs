"""Label-only complete TRAIN membership oracle, outside all model inputs.

SOURCE ONLY. Project creation requires the exact authenticated TRAIN digest;
fabricated qualification creation is separately typed and cannot be used for
project epochs. No VALID/TEST split, target label, or synthetic-mask flag is
accepted by the label lookup interface.
"""
from dataclasses import dataclass
from hashlib import sha256
import torch
from cardinality_graph import Graph, membership
from cardinality_density import require

AUTHENTICATED_TRAIN_RECORDS_SHA256 = "7bfe6ac3bc99d6620590cd9ddd30fd87ed28a307e4a03f605fb4f32a2f5272bd"


def tensor_sha(value):
    # Exact empty-safe V3 shape/dtype and contiguous byte digest convention.
    array = value.detach().cpu().contiguous().numpy()
    digest = sha256(str((array.shape, array.dtype)).encode())
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainObservationTeacher:
    nodes: int
    keys: torch.Tensor
    train_records_sha256: str
    scope: str

    @classmethod
    def from_authenticated_train(cls, pairs, nodes, *, expected_train_records_sha256):
        require(pairs.shape == (1179052, 2) and nodes == 235868, "Complete project TRAIN snapshot required")
        digest = tensor_sha(pairs)
        require(digest == expected_train_records_sha256 == AUTHENTICATED_TRAIN_RECORDS_SHA256, "Authenticated TRAIN record digest differs")
        graph = Graph.from_pairs(pairs, nodes)
        return cls(nodes, graph.row * nodes + graph.col, digest, "complete_authenticated_TRAIN")

    @classmethod
    def from_fabricated_train(cls, pairs, nodes):
        # Only later source-bound fabricated qualification may use this scope.
        require(len(pairs) < 1179052 and nodes < 235868, "Fabricated fixture cannot impersonate project snapshot")
        graph = Graph.from_pairs(pairs, nodes)
        return cls(nodes, graph.row * nodes + graph.col, tensor_sha(pairs), "fabricated_TRAIN_fixture")

    def labels(self, counterpart_pairs):
        require(self.scope in ("complete_authenticated_TRAIN", "fabricated_TRAIN_fixture"), "Forbidden split teacher")
        require(counterpart_pairs.dtype == torch.long and counterpart_pairs.ndim == 2 and counterpart_pairs.shape[1] == 2,
                "Teacher requires already enumerated counterpart endpoints")
        if counterpart_pairs.numel():
            require(int(counterpart_pairs.min()) >= 0 and int(counterpart_pairs.max()) < self.nodes, "Teacher counterpart out of range")
        # keys are symmetric complete TRAIN, counterpart identity canonical.
        return membership(self.keys, counterpart_pairs[:, 0] * self.nodes + counterpart_pairs[:, 1]).to(torch.float64).detach()

    def receipt(self):
        return {"nodes": self.nodes, "observed_directed_unique_entries": len(self.keys), "keys_sha256": tensor_sha(self.keys),
                "TRAIN_records_sha256": self.train_records_sha256, "scope": self.scope,
                "source_zero": "unobserved_in_complete_TRAIN_not_verified_latent_nonlink", "model_input": False}


def sealed_support_digest(queries, support, labels=None):
    coordinates = torch.cat((support.rows[:, None], queries[support.rows], support.nodes[:, None], support.counterpart_pairs), 1)
    if labels is not None:
        require(labels.shape == support.rows.shape, "Teacher cannot change support coordinates")
        coordinates = torch.cat((coordinates, labels.long()[:, None]), 1)
    return tensor_sha(coordinates)
