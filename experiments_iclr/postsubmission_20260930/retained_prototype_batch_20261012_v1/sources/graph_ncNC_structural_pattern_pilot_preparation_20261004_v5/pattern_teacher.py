"""Full TRAIN observation-membership labels; never a predictor input."""
from dataclasses import dataclass
import torch
from graph_ops import Graph, membership
from pilot_common import require
from pilot_data import tensor_sha


@dataclass(frozen=True)
class ObservationTeacher:
    nodes: int
    keys: torch.Tensor

    @classmethod
    def from_train(cls, pairs, nodes):
        full = Graph.from_pairs(pairs, nodes)
        return cls(nodes, full.row * nodes + full.col)

    def labels(self, queries, neighbors):
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        rows = torch.cat((lq, rq))
        counterparts = torch.cat((queries[lq, 1] * self.nodes + ln,
                                  queries[rq, 0] * self.nodes + rn))
        labels = membership(self.keys, counterparts).to(torch.float32)
        require(rows.shape == labels.shape and len(queries) == neighbors.queries, "Residual label alignment differs")
        return rows, labels

    def support_digest(self, queries, neighbors, labels):
        """Bind ordered query, side, candidate and missing-link coordinates."""
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        rows = torch.cat((lq, rq))
        side = torch.cat((torch.zeros_like(lq), torch.ones_like(rq)))
        candidates = torch.cat((ln, rn))
        counterparts = torch.cat((queries[lq, 1], queries[rq, 0]))
        require(len(rows) == len(labels), "Support digest label alignment differs")
        coordinates = torch.stack((rows, side, queries[rows, 0], queries[rows, 1],
                                   candidates, counterparts, labels.long()), dim=1)
        return tensor_sha(coordinates)

    def receipt(self):
        return {"nodes": self.nodes, "directed_unique_observed_entries": len(self.keys),
                "keys_sha256": tensor_sha(self.keys), "teacher": "complete_TRAIN_observation_membership",
                "source_zero_semantics": "unobserved_in_TRAIN_not_verified_latent_nonlink",
                "teacher_used_as_predictor_input": False}
