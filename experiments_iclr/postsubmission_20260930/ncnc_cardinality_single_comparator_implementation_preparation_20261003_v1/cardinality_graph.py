"""Portable unweighted NCNC query-graph operations; no dataset loader.

Written for this attributed qualification prototype. Uses PyTorch, not copied
third-party implementation. Native collab has no predictor adjacency dropout,
neighbor sampling or edge weights. Unsupported variants are rejected.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Graph:
    nodes: int
    row: torch.Tensor
    col: torch.Tensor
    rowptr: torch.Tensor

    @classmethod
    def from_pairs(cls, pairs, nodes):
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
            raise ValueError("Expected integer [records,2] TRAIN pairs")
        if nodes <= 0 or nodes * nodes >= 2**63:
            raise ValueError("Unsupported node count")
        if pairs.numel() and (pairs.min() < 0 or pairs.max() >= nodes):
            raise ValueError("Pair out of range")
        directed = torch.cat((pairs, pairs.flip(1)), dim=0)
        keys = torch.unique(directed[:, 0] * nodes + directed[:, 1], sorted=True)
        row, col = torch.div(keys, nodes, rounding_mode="floor"), keys % nodes
        count = torch.bincount(row, minlength=nodes)
        ptr = torch.cat((count.new_zeros(1), count.cumsum(0)))
        return cls(nodes, row, col, ptr)

    @classmethod
    def mask_train_batch(cls, train_pairs, selected_records, nodes):
        # Mask records before symmetrization. Remaining duplicate records can
        # retain the same edge, exactly as the native supervision-list mask.
        keep = torch.ones(len(train_pairs), dtype=torch.bool, device=train_pairs.device)
        keep[selected_records] = False
        return cls.from_pairs(train_pairs[keep], nodes)

    def adjacency_for_encoder(self, training, probability):
        row, col = self.row, self.col
        value = torch.ones(len(row), device=row.device, dtype=torch.float32)
        if training and probability > 0:
            # Native DropAdj independently drops directed coalesced entries;
            # do not restore symmetry afterward.
            mask = torch.rand_like(col, dtype=torch.float32) > probability
            row, col, value = row[mask], col[mask], value[mask] / (1 - probability)
        # GCNConv's SparseTensor route replaces the diagonal with ones.
        off = row != col
        diag = torch.arange(self.nodes, device=row.device)
        return (torch.cat((row[off], diag)), torch.cat((col[off], diag)),
                torch.cat((value[off], torch.ones(self.nodes, device=row.device))))

    def rows_for_queries(self, endpoints):
        counts = self.rowptr[endpoints + 1] - self.rowptr[endpoints]
        query = torch.repeat_interleave(torch.arange(len(endpoints), device=endpoints.device), counts)
        total = int(counts.sum().item())  # one measured synchronization, no cap
        prefix = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        offset = torch.arange(total, device=endpoints.device) - torch.repeat_interleave(prefix[:-1], counts)
        indices = torch.repeat_interleave(self.rowptr[endpoints], counts) + offset
        return query, self.col[indices]


@dataclass(frozen=True)
class QueryNeighbors:
    common: tuple
    left: tuple
    right: tuple
    queries: int


def membership(sorted_keys, candidates):
    if len(sorted_keys) == 0:
        return torch.zeros_like(candidates, dtype=torch.bool)
    index = torch.searchsorted(sorted_keys, candidates)
    safe = index.clamp(max=len(sorted_keys) - 1)
    return (index < len(sorted_keys)) & (sorted_keys[safe] == candidates)


def enumerate_neighbors(graph, queries):
    if queries.ndim != 2 or queries.shape[1] != 2 or queries.dtype != torch.long:
        raise ValueError("Expected [queries,2] integer endpoints")
    lq, ln = graph.rows_for_queries(queries[:, 0])
    rq, rn = graph.rows_for_queries(queries[:, 1])
    lk, rk = lq * graph.nodes + ln, rq * graph.nodes + rn
    lm, rm = membership(rk, lk), membership(lk, rk)
    return QueryNeighbors((lq[lm], ln[lm]), (lq[~lm], ln[~lm]),
                          (rq[~rm], rn[~rm]), len(queries))


def feature_sum(features, row_node, queries, weight=None):
    row, node = row_node
    result = features.new_zeros((queries, features.shape[1]))
    values = features[node]
    if weight is not None:
        if weight.shape != (len(row),):
            raise ValueError("Candidate weight shape mismatch")
        values = values * weight[:, None]
    return result.index_add(0, row, values)
