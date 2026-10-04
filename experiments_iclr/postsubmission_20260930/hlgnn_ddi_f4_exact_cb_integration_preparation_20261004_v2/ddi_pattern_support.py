"""Complete TRAIN membership and exact residual supports for the DDI adaptation.

No dataset loader, count feature, sampling inside a support, or serving hook.
Unlike NCNC's record mask, this new auxiliary mask removes edge identities.
"""
from dataclasses import dataclass
import hashlib
import torch
from torch_sparse import SparseTensor


def tensor_sha(value):
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def owned_seed(seed, epoch, batch, purpose):
    text = f'HLGNN-DDI-CB-v1/{seed}/{epoch}/{batch}/{purpose}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63 - 1)


def select_strata(batch_records, num_neg, seed, epoch, batch, per_stratum=32):
    """Uniform positions without replacement, using an owned CPU generator.

    Selection is stratified by positive/negative origin and independent of
    predictions, candidate support sizes, teacher counts, and fitted state.
    """
    generator = torch.Generator(device='cpu')
    generator.manual_seed(owned_seed(seed, epoch, batch, 'query-selection'))
    positive = torch.randperm(batch_records, generator=generator)[:min(per_stratum, batch_records)]
    negative = torch.randperm(batch_records * num_neg, generator=generator)[:min(per_stratum, batch_records * num_neg)]
    return positive, negative


def membership(sorted_keys, keys):
    if len(sorted_keys) == 0:
        return torch.zeros_like(keys, dtype=torch.bool)
    index = torch.searchsorted(sorted_keys, keys)
    safe = index.clamp(max=len(sorted_keys) - 1)
    return (index < len(sorted_keys)) & (sorted_keys[safe] == keys)


@dataclass(frozen=True)
class ResidualSupports:
    left: tuple
    right: tuple
    queries: int


@dataclass(frozen=True)
class VisibleTrain:
    nodes: int
    keys: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    rowptr: torch.Tensor
    adj_t: SparseTensor

    def neighbors(self, endpoints):
        counts = self.rowptr[endpoints + 1] - self.rowptr[endpoints]
        rows = torch.repeat_interleave(torch.arange(len(endpoints), device=endpoints.device), counts)
        prefix = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        total = int(counts.sum().item())
        offset = torch.arange(total, device=endpoints.device) - torch.repeat_interleave(prefix[:-1], counts)
        indices = torch.repeat_interleave(self.rowptr[endpoints], counts) + offset
        return rows, self.col[indices]

    def supports(self, queries):
        left_rows, left_nodes = self.neighbors(queries[:, 0])
        right_rows, right_nodes = self.neighbors(queries[:, 1])
        # Exclude both query endpoints even for diagonal/retained-edge cases.
        left_keep = (left_nodes != queries[left_rows, 0]) & (left_nodes != queries[left_rows, 1])
        right_keep = (right_nodes != queries[right_rows, 0]) & (right_nodes != queries[right_rows, 1])
        left_rows, left_nodes = left_rows[left_keep], left_nodes[left_keep]
        right_rows, right_nodes = right_rows[right_keep], right_nodes[right_keep]
        left_keys = left_rows * self.nodes + left_nodes
        right_keys = right_rows * self.nodes + right_nodes
        left_only = ~membership(right_keys, left_keys)
        right_only = ~membership(left_keys, right_keys)
        return ResidualSupports((left_rows[left_only], left_nodes[left_only]),
                                (right_rows[right_only], right_nodes[right_only]), len(queries))


@dataclass(frozen=True)
class FullTrainTeacher:
    nodes: int
    keys: torch.Tensor

    @classmethod
    def from_train(cls, train_pairs, nodes, native_adj_t):
        directed = torch.cat((train_pairs, train_pairs.flip(1)), dim=0)
        keys = torch.unique(directed[:, 0] * nodes + directed[:, 1], sorted=True)
        row, col, value = native_adj_t.coo()
        native_keys = torch.unique(row * nodes + col, sorted=True)
        if value is not None or not torch.equal(keys, native_keys):
            raise ValueError('Only the qualified unweighted complete native TRAIN graph is supported.')
        return cls(nodes, keys)

    def visible(self, batch_positive_pairs, queries):
        # Every native batch positive identity is removed in both directions;
        # duplicate records cannot leak an auxiliary hidden edge back in.
        hidden = torch.cat((batch_positive_pairs, queries), dim=0)
        hidden = torch.cat((hidden, hidden.flip(1)), dim=0)
        hidden_keys = torch.unique(hidden[:, 0] * self.nodes + hidden[:, 1], sorted=True)
        keys = self.keys[~membership(hidden_keys, self.keys)]
        row, col = torch.div(keys, self.nodes, rounding_mode='floor'), keys % self.nodes
        counts = torch.bincount(row, minlength=self.nodes)
        rowptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        # TRAIN is undirected. This is the same Boolean graph for support and encoder.
        adj_t = SparseTensor(row=row, col=col, value=None,
                             sparse_sizes=(self.nodes, self.nodes), is_sorted=True)
        return VisibleTrain(self.nodes, keys, row, col, rowptr, adj_t), hidden_keys

    def labels(self, queries, supports):
        left_rows, left_nodes = supports.left
        right_rows, right_nodes = supports.right
        coordinates = torch.cat((queries[left_rows, 1] * self.nodes + left_nodes,
                                 queries[right_rows, 0] * self.nodes + right_nodes))
        return membership(self.keys, coordinates).to(torch.float32).detach()

    def support_sha(self, queries, supports, bits):
        left_rows, left_nodes = supports.left
        right_rows, right_nodes = supports.right
        rows = torch.cat((left_rows, right_rows))
        side = torch.cat((torch.zeros_like(left_rows), torch.ones_like(right_rows)))
        candidates = torch.cat((left_nodes, right_nodes))
        counterpart = torch.cat((queries[left_rows, 1], queries[right_rows, 0]))
        return tensor_sha(torch.stack((rows, side, queries[rows, 0], queries[rows, 1],
                                       candidates, counterpart, bits.long()), dim=1))
