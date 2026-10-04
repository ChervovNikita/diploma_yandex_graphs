"""Attributed contextual autoregressive conditional-single auxiliary prototype."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from conditional_loss import require, TrainPatternLabels

WIDTH = 64
SLOT_FEATURES = 1 + 3 * WIDTH
QUERY_FEATURES = 3 * WIDTH + 2
PREFIX_FEATURES = 2 * WIDTH
PROGRESS_FEATURES = 8
HEAD_INPUT = SLOT_FEATURES + QUERY_FEATURES + PREFIX_FEATURES + PROGRESS_FEATURES
HEAD_PARAMETERS = HEAD_INPUT * WIDTH + WIDTH + WIDTH + 1


@dataclass(frozen=True)
class SingleVisibleContext:
    slot: torch.Tensor
    query: torch.Tensor
    candidate: torch.Tensor
    rows_left: torch.Tensor
    rows_right: torch.Tensor
    queries: int


def _csr_layout(rows_left, rows_right, queries):
    """Tensor CSR counts/starts; no query list or per-query scalar extraction."""
    require(type(queries) is int and queries > 0, 'Positive query count required')
    result = []
    for rows in (rows_left, rows_right):
        require(rows.ndim == 1 and rows.dtype == torch.long, 'Integer slot/query rows required')
        require(not len(rows) or bool(((rows >= 0) & (rows < queries)).all())
                and bool((rows[1:] >= rows[:-1]).all()), 'Native query-grouped order required')
        counts = torch.bincount(rows, minlength=queries)
        starts = counts.cumsum(0) - counts
        result.append((counts, starts))
    return result


def visible_context(h, queries, neighbors, unary_left, unary_right):
    """Only visible width64 features/unary scores; no teacher/count arguments.

    Native residual slot order and original endpoint orientation are preserved.
    h belongs to one native predictor. This helper has no graph/teacher loader.
    """
    require(h.ndim == 2 and h.shape[1] == WIDTH and queries.ndim == 2 and queries.shape[1] == 2
            and queries.dtype == torch.long and len(queries) == neighbors.queries, 'Native single/context geometry differs')
    require(bool(((queries >= 0) & (queries < len(h))).all()), 'Query endpoint ID outside visible node range')
    left_rows, left_nodes = neighbors.left
    right_rows, right_nodes = neighbors.right
    require(unary_left.shape == left_rows.shape and unary_right.shape == right_rows.shape, 'Single unary slots differ')
    require(h.device == queries.device == left_rows.device == right_rows.device == unary_left.device == unary_right.device
            and h.dtype == unary_left.dtype == unary_right.dtype, 'Single context dtype/device differs')
    layout = _csr_layout(left_rows, right_rows, len(queries))
    slots, means, candidates = [], [], []
    for side, (rows, nodes, unary) in enumerate(((left_rows,left_nodes,unary_left),(right_rows,right_nodes,unary_right))):
        require(nodes.shape == rows.shape and nodes.dtype == torch.long and nodes.device == rows.device
                and (not len(nodes) or (int(nodes.min()) >= 0 and int(nodes.max()) < len(h))), 'Visible residual candidate IDs invalid')
        require(not len(nodes) or bool((nodes[1:][rows[1:]==rows[:-1]] > nodes[:-1][rows[1:]==rows[:-1]]).all()),
                'Canonical native unique CSR candidate order required')
        own, other = queries[rows, side], queries[rows, 1 - side]
        n, starts = layout[side]
        centered = unary - unary[starts[rows]]
        slots.append(torch.cat((centered[:, None], h[own], h[other], h[nodes]), dim=1))
        candidates.append(h[nodes])
        mean = h.new_zeros((len(queries),WIDTH)).index_add(0, rows, h[nodes]) / n.clamp(min=1)[:, None]
        means.append((mean, torch.log1p(n.to(h.dtype))))
    q = torch.cat((h[queries[:,0]] * h[queries[:,1]], means[0][0], means[1][0],
                   means[0][1][:,None], means[1][1][:,None]), dim=1)
    return SingleVisibleContext(torch.cat(slots), q, torch.cat(candidates), left_rows, right_rows, len(queries))


def _selected_prefix(rows, candidate, bits, queries, counts, starts):
    """Exact teacher counts and differentiable exclusive segmented feature sums.

    Integer global cumsums use exact start subtraction. Floating cumsums are
    local to each query, grouped by visible support length without padding.
    No floating prefix subtracts a global total from unrelated queries.
    Group order is inverted before any original slot is assembled/evaluated.
    """
    integer_bits = bits.to(torch.int64)
    teacher = integer_bits.new_zeros(queries).index_add(0, rows, integer_bits)
    cumulative_counts = torch.cat((integer_bits.new_zeros(1), integer_bits.cumsum(0)))
    selected = cumulative_counts[:-1] - cumulative_counts[starts[rows]]
    if not len(rows):
        return teacher, selected, candidate[:0], candidate.new_zeros((queries, WIDTH))
    nonempty = torch.nonzero(counts > 0, as_tuple=False).flatten()
    query_order = nonempty[torch.argsort(counts[nonempty])]
    lengths, group_counts = torch.unique_consecutive(counts[query_order], return_counts=True)
    prefix_parts, addresses_parts, totals_parts = [], [], []
    cursor = 0
    # One dispatch per distinct support size, never per query or slot. Metadata
    # transfers/sort/gathers, cumulative values and gradients remain charged.
    for length, group in zip(lengths.tolist(), group_counts.tolist()):
        ids = query_order[cursor:cursor + group]
        addresses = starts[ids, None] + torch.arange(length, device=rows.device)[None, :]
        weighted = candidate[addresses] * bits[addresses, None]
        cumulative = weighted.cumsum(dim=1)
        exclusive = torch.cat((candidate.new_zeros((group, 1, WIDTH)), cumulative[:, :-1]), dim=1)
        prefix_parts.append(exclusive.reshape(-1, WIDTH))
        addresses_parts.append(addresses.reshape(-1))
        totals_parts.append(cumulative[:, -1])
        cursor += group
    addresses = torch.cat(addresses_parts)
    prefix = candidate.new_zeros((len(rows), WIDTH)).index_copy(0, addresses, torch.cat(prefix_parts))
    complete = candidate.new_zeros((queries, WIDTH)).index_copy(0, query_order, torch.cat(totals_parts))
    return teacher, selected, prefix, complete


def teacher_forced_features(context, labels):
    """All past-only slot features, forced masks and tensor support counts.

    Counts are derived only from detached TRAIN bits in this training helper.
    Left slots see no right selected identity; right slots see all left selected
    identities and only the preceding right identities. Current/future bits
    affect features only through the already declared side count budgets.
    """
    require(isinstance(context, SingleVisibleContext) and isinstance(labels, TrainPatternLabels), 'Auxiliary context/teacher types required')
    slots = len(context.rows_left) + len(context.rows_right)
    require(context.slot.shape == (slots,SLOT_FEATURES) and context.query.shape == (context.queries,QUERY_FEATURES)
            and context.candidate.shape == (slots,WIDTH), 'Frozen single feature dimensions differ')
    for tensor in (context.slot,context.query,context.candidate):
        require(tensor.dtype in (torch.float32,torch.float64) and tensor.dtype == context.slot.dtype
                and tensor.device == context.slot.device and bool(torch.isfinite(tensor).all()), 'Finite aligned visible context required')
    require(context.rows_left.device == context.rows_right.device == context.slot.device, 'Context row device differs')
    labels.check(context.slot.t(), slots)
    layout = _csr_layout(context.rows_left, context.rows_right, context.queries)
    split = len(context.rows_left)
    rows_sides = (context.rows_left, context.rows_right)
    bits_sides = (labels.values[:split], labels.values[split:])
    candidate_sides = (context.candidate[:split], context.candidate[split:])
    summaries = [_selected_prefix(rows, candidate, bits, context.queries, *side_layout)
                 for rows, candidate, bits, side_layout in zip(rows_sides, candidate_sides, bits_sides, layout)]
    features_parts, forced_parts = [], []
    for side, rows in enumerate(rows_sides):
        n = [pair[0][rows] for pair in layout]
        k = [summary[0][rows] for summary in summaries]
        selected_here = summaries[side][1]
        position = torch.arange(len(rows), device=rows.device) - layout[side][1][rows]
        zero_counts = torch.zeros_like(selected_here)
        selected = (selected_here, zero_counts) if side == 0 else (k[0], selected_here)
        remaining = (k[0] - selected_here, k[1]) if side == 0 else (zero_counts, k[1] - selected_here)
        unseen = (n[0] - position, n[1]) if side == 0 else (zero_counts, n[1] - position)
        prefix_here = summaries[side][2] / selected_here.clamp(min=1).to(context.slot.dtype)[:, None]
        empty_prefix = context.candidate.new_zeros((len(rows), WIDTH))
        left_complete = summaries[0][3][rows] / k[0].clamp(min=1).to(context.slot.dtype)[:, None]
        prefix = torch.cat((prefix_here, empty_prefix) if side == 0 else (left_complete, prefix_here), dim=1)
        denominators = [count.clamp(min=1).to(context.slot.dtype) for count in n]
        progress = torch.stack((context.slot.new_full((len(rows),), float(side == 0)),
                                context.slot.new_full((len(rows),), float(side == 1)),
                                unseen[0] / denominators[0], unseen[1] / denominators[1],
                                remaining[0] / denominators[0], remaining[1] / denominators[1],
                                selected[0] / denominators[0], selected[1] / denominators[1]), dim=1)
        base = 0 if side == 0 else split
        features_parts.append(torch.cat((context.slot[base:base + len(rows)], context.query[rows], prefix, progress), dim=1))
        active_remaining, active_unseen = remaining[side], unseen[side]
        require(bool(((active_remaining >= 0) & (active_remaining <= active_unseen)).all()), 'Infeasible teacher prefix')
        forced = (active_remaining == 0) | (active_remaining == active_unseen)
        expected_bit = (active_remaining == active_unseen).to(context.slot.dtype)
        require(bool((~forced | (bits_sides[side] == expected_bit)).all()), 'Forced bit/count mismatch')
        forced_parts.append(forced)
    return torch.cat(features_parts), torch.cat(forced_parts), [pair[0] for pair in layout]


class ConditionalSingleHead(nn.Module):
    """Width64 MLP, canonical left then right count-constrained teacher forcing.

    This head is auxiliary-only, not an ensemble or target-serving route.
    Every slot evaluates the head, including forced/extreme-count choices.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(HEAD_INPUT, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 1))
        require(sum(p.numel() for p in self.parameters()) == HEAD_PARAMETERS, 'Head parameter algebra differs')

    def training_nll(self, context, labels):
        features, forced, counts = teacher_forced_features(context, labels)
        # The unchanged no-dropout head evaluates all real rows once, including
        # forced rows; there is also one zero-row call for all-empty geometry.
        scores = self.network(features).squeeze(-1)
        require(bool(torch.isfinite(scores).all()), 'Nonfinite contextual auxiliary logit')
        signed = torch.where(labels.values.bool(), scores, -scores)
        terms = torch.where(forced, scores * 0, -F.logsigmoid(signed))
        split = len(context.rows_left)
        total = (torch.segment_reduce(terms[:split], 'sum', lengths=counts[0])
                 + torch.segment_reduce(terms[split:], 'sum', lengths=counts[1]))
        zero = sum(p.reshape(-1)[:0].sum() for p in self.parameters()) + context.slot[:0].sum()
        return (total + zero) / (counts[0] + counts[1]).clamp(min=1).to(context.slot.dtype)
