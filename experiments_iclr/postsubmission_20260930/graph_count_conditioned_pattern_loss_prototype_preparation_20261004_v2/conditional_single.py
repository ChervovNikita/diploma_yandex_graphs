"""Attributed contextual autoregressive conditional-single auxiliary prototype."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from conditional_loss import require, ragged_layout, TrainPatternLabels

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
    _, offsets = ragged_layout(left_rows,right_rows,len(queries))
    slots, means, candidates = [], [], []
    for side, (rows, nodes, unary) in enumerate(((left_rows,left_nodes,unary_left),(right_rows,right_nodes,unary_right))):
        require(nodes.shape == rows.shape and nodes.dtype == torch.long and nodes.device == rows.device
                and (not len(nodes) or (int(nodes.min()) >= 0 and int(nodes.max()) < len(h))), 'Visible residual candidate IDs invalid')
        require(not len(nodes) or bool((nodes[1:][rows[1:]==rows[:-1]] > nodes[:-1][rows[1:]==rows[:-1]]).all()),
                'Canonical native unique CSR candidate order required')
        own, other = queries[rows, side], queries[rows, 1 - side]
        centered = [unary[start:end]-unary[start] for start,end in zip(offsets[side][:-1],offsets[side][1:]) if end>start]
        centered = torch.cat(centered) if centered else unary[:0]
        slots.append(torch.cat((centered[:, None], h[own], h[other], h[nodes]), dim=1))
        candidates.append(h[nodes])
        n = torch.bincount(rows, minlength=len(queries))
        mean = h.new_zeros((len(queries),WIDTH)).index_add(0, rows, h[nodes]) / n.clamp(min=1)[:, None]
        means.append((mean, torch.log1p(n.to(h.dtype))))
    q = torch.cat((h[queries[:,0]] * h[queries[:,1]], means[0][0], means[1][0],
                   means[0][1][:,None], means[1][1][:,None]), dim=1)
    return SingleVisibleContext(torch.cat(slots), q, torch.cat(candidates), left_rows, right_rows, len(queries))


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
        require(isinstance(context, SingleVisibleContext) and isinstance(labels, TrainPatternLabels), 'Auxiliary context/teacher types required')
        slots = len(context.rows_left) + len(context.rows_right)
        require(context.slot.shape == (slots,SLOT_FEATURES) and context.query.shape == (context.queries,QUERY_FEATURES)
                and context.candidate.shape == (slots,WIDTH), 'Frozen single feature dimensions differ')
        for tensor in (context.slot,context.query,context.candidate):
            require(tensor.dtype in (torch.float32,torch.float64) and tensor.dtype == context.slot.dtype
                    and tensor.device == context.slot.device and bool(torch.isfinite(tensor).all()), 'Finite aligned visible context required')
        require(context.rows_left.device == context.rows_right.device == context.slot.device, 'Context row device differs')
        labels.check(context.slot.t(), slots)
        counts, offsets = ragged_layout(context.rows_left,context.rows_right,context.queries)
        values = []
        for query in range(context.queries):
            teacher_counts = []
            for side in range(2):
                start,end = offsets[side][query:query+2]
                base = 0 if side == 0 else len(context.rows_left)
                teacher_counts.append(int(labels.values[base+start:base+end].sum(dtype=torch.int64).item()))
            remaining = list(teacher_counts)
            selected = [0,0]
            selected_features = [context.slot.new_zeros(WIDTH),context.slot.new_zeros(WIDTH)]
            terms = []
            for side in range(2):
                start,end = offsets[side][query:query+2]
                base = 0 if side == 0 else len(context.rows_left)
                for index in range(start,end):
                    unseen = [counts[0][query],counts[1][query]]
                    unseen[side] = end-index
                    if side == 1:
                        unseen[0] = 0
                    progress = context.slot.new_tensor([
                        float(side==0),float(side==1),
                        unseen[0]/max(counts[0][query],1),unseen[1]/max(counts[1][query],1),
                        remaining[0]/max(counts[0][query],1),remaining[1]/max(counts[1][query],1),
                        selected[0]/max(counts[0][query],1),selected[1]/max(counts[1][query],1)])
                    prefix = torch.cat([selected_features[s]/max(selected[s],1) for s in range(2)])
                    features = torch.cat((context.slot[base+index],context.query[query],prefix,progress))
                    score = self.network(features).squeeze(0)
                    require(bool(torch.isfinite(score)), 'Nonfinite contextual auxiliary logit')
                    bit = int(labels.values[base+index].item())
                    require(0 <= remaining[side] <= unseen[side], 'Infeasible teacher prefix')
                    if remaining[side] == 0 or remaining[side] == unseen[side]:
                        require(bit == int(remaining[side] == unseen[side]), 'Forced bit/count mismatch')
                        terms.append(score * 0)
                    else:
                        terms.append(-F.logsigmoid(score if bit else -score))
                    remaining[side] -= bit
                    selected[side] += bit
                    selected_features[side] = selected_features[side] + bit * context.candidate[base+index]
            require(remaining == [0,0], 'Teacher budget not exhausted')
            # Empty context still exposes connected exact zero head gradients.
            zero = sum(p.reshape(-1)[:0].sum() for p in self.parameters()) + context.slot[:0].sum()
            total = torch.stack(terms).sum() if terms else zero
            values.append((total + zero) / max(counts[0][query]+counts[1][query],1))
        return torch.stack(values)
