"""Attributed conditional-Bernoulli TRAIN auxiliary; no predictor/serving code."""
from dataclasses import dataclass
from math import log
import torch


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_logits(eta):
    require(eta.ndim == 2 and eta.shape[0] > 0 and eta.dtype in (torch.float32, torch.float64),
            'Expected finite float32/64 [members,slots] logits')
    require(bool(torch.isfinite(eta).all()), 'Nonfinite logits')


def _log_esp_reachable(eta, degree):
    # Never form logaddexp(-inf,-inf): every stored state is reachable and finite.
    # This rolling list uses O(degree) forward tensors, but autograd retains the
    # recurrence graph. It is not an O(degree) training-memory claim.
    table = [eta[:, :0].sum(dim=1)]
    for slot in range(eta.shape[1]):
        previous = table
        table = [previous[0]]
        for count in range(1, min(slot + 1, degree) + 1):
            choose = eta[:, slot] + previous[count - 1]
            table.append(choose if count == len(previous) else torch.logaddexp(previous[count], choose))
    return table[degree]


def log_elementary_symmetric(eta, count):
    """log e_count(exp eta), exact finite-support algebra, differentiable logits."""
    finite_logits(eta)
    require(type(count) is int and 0 <= count <= eta.shape[1], 'Count outside support')
    slots = eta.shape[1]
    if count == 0:
        return eta[:, :0].sum(dim=1)
    if count == slots:
        return eta.sum(dim=1)
    if count > slots - count:
        return eta.sum(dim=1) + _log_esp_reachable(-eta, slots - count)
    return _log_esp_reachable(eta, count)


def conditional_member_nll(eta, bits):
    """One fixed-count law/member; counts are labels derived from TRAIN bits only."""
    finite_logits(eta)
    require(bits.ndim == 1 and len(bits) == eta.shape[1] and bits.device == eta.device
            and bits.dtype == eta.dtype and not bits.requires_grad
            and bool(((bits == 0) | (bits == 1)).all()), 'Invalid auxiliary observation bits')
    count = int(bits.sum(dtype=torch.int64).item())
    slots = eta.shape[1]
    if count in (0, slots):
        # One feasible pattern. Preserve a connected graph and exact zero grads.
        return eta[:, :0].sum(dim=1)
    # A differentiable common anchor removes arbitrary side offsets. Complement
    # the likelihood itself to avoid subtracting two large total-logit sums.
    centered = eta - eta[:, :1]
    if count > slots - count:
        centered, bits, count = -centered, 1 - bits, slots - count
    return _log_esp_reachable(centered, count) - (centered * bits[None, :]).sum(dim=1)


@dataclass(frozen=True)
class TrainPatternLabels:
    values: torch.Tensor
    authority: str

    def check(self, logits, slots):
        require(self.authority == 'complete_TRAIN_observation_membership', 'TRAIN-only teacher authority required')
        require(self.values.shape == (slots,) and self.values.device == logits.device
                and self.values.dtype == logits.dtype and not self.values.requires_grad
                and bool(((self.values == 0) | (self.values == 1)).all()), 'Aligned detached TRAIN bits required')


def ragged_layout(rows_left, rows_right, queries):
    """Existing native left-then-right CSR order; never reorder slots or sample."""
    require(type(queries) is int and queries > 0, 'Positive query count required')
    counts, offsets = [], []
    for rows in (rows_left, rows_right):
        require(rows.ndim == 1 and rows.dtype == torch.long, 'Integer slot/query rows required')
        require(not len(rows) or (int(rows.min()) >= 0 and int(rows.max()) < queries
                and bool((rows[1:] >= rows[:-1]).all())), 'Native query-grouped order required')
        n = torch.bincount(rows, minlength=queries).tolist()
        ptr = [0]
        for value in n:
            ptr.append(ptr[-1] + value)
        counts.append(n)
        offsets.append(ptr)
    return counts, offsets


def _group_log_esp_reachable(centered, degree):
    """Exact reachable ESP cells for [members,group_queries,slots].

    One Python iteration per slot in this group, not per query or count cell.
    Rolling forward storage is not an autograd training-memory bound.
    """
    zero = centered[..., :0].sum(dim=-1, keepdim=True)
    table = zero
    for slot in range(centered.shape[-1]):
        add = centered[..., slot:slot + 1]
        common = torch.logaddexp(table[..., 1:], add + table[..., :-1])
        if slot < degree:
            # The newly reachable last cell has no exclusion contribution.
            table = torch.cat((zero, common, add + table[..., -1:]), dim=-1)
        else:
            table = torch.cat((zero, common), dim=-1)
    return table[..., degree]


def _grouped_side_nll(eta, rows, bits, queries):
    """Exact side laws grouped by (n,min(k,n-k)); no Python query loop.

    Groups contain only their real slots. A batch-size padding axis is absent.
    All count/layout operations are detached integer or teacher metadata.
    """
    require(rows.ndim == 1 and rows.dtype == torch.long
            and (not len(rows) or (bool(((rows >= 0) & (rows < queries)).all())
                 and bool((rows[1:] >= rows[:-1]).all()))),
            'Native query-grouped rows within query range required')
    support = torch.bincount(rows, minlength=queries)
    teacher = torch.zeros(queries, dtype=torch.long, device=rows.device).index_add(0, rows, bits.to(torch.long))
    reduced = torch.minimum(teacher, support - teacher)
    keys, inverse = torch.unique(torch.stack((support, reduced), dim=1), dim=0,
                                 sorted=True, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    group_sizes = torch.bincount(inverse, minlength=len(keys))
    # Two group-level metadata transfers replace per-query count.item() calls.
    descriptors, sizes = keys.tolist(), group_sizes.tolist()
    offsets = torch.cat((support.new_zeros(1), support.cumsum(0)))
    pieces, cursor = [], 0
    zero = eta[:, :0].sum(dim=1, keepdim=True)
    for (slots, degree), size in zip(descriptors, sizes):
        query_ids = order[cursor:cursor + size]
        cursor += size
        if degree == 0:
            # Includes empty, all-zero and all-one teacher patterns. Never
            # sum extreme finite logits merely to multiply the sum by zero.
            pieces.append(zero.expand(-1, size))
            continue
        indices = offsets[query_ids, None] + torch.arange(slots, device=rows.device)[None, :]
        logits = eta[:, indices]
        observed = bits[indices]
        centered = logits - logits[..., :1]
        complement = teacher[query_ids] > support[query_ids] - teacher[query_ids]
        centered = torch.where(complement[None, :, None], -centered, centered)
        observed = torch.where(complement[:, None], 1 - observed, observed)
        require(bool(torch.isfinite(centered).all()), 'Finite-logit centering exceeded working precision')
        if degree == 1:
            # k=1 or its exact k=n-1 complement is a categorical law.
            selected = observed.argmax(dim=-1)
            chosen = centered.gather(-1, selected[None, :, None].expand(eta.shape[0], -1, 1)).squeeze(-1)
            loss = torch.logsumexp(centered, dim=-1) - chosen
        else:
            loss = _group_log_esp_reachable(centered, degree) - (centered * observed[None, :, :]).sum(dim=-1)
        require(bool(torch.isfinite(loss).all()), 'Conditional grouped loss exceeded working precision')
        pieces.append(loss)
    require(cursor == queries, 'Grouped query coverage differs')
    grouped = torch.cat(pieces, dim=1)
    restore = torch.empty_like(order)
    restore[order] = torch.arange(queries, device=rows.device)
    return grouped[:, restore], support, teacher, reduced


def training_pattern_losses(eta_left, eta_right, rows_left, rows_right, labels, queries):
    """Exact J_K/shared, J_K_sep/independent sides and W_K on ragged queries.

    This groups only auxiliary ESP arithmetic. The upstream full native neural
    schedule, ordered slots, all-query mean and coefficient1 remain mandatory.
    Mathematical laws are unchanged; floating operation ordering is reviewed
    and tested prospectively under the existing fixed CPU oracle tolerances.
    """
    finite_logits(eta_left)
    finite_logits(eta_right)
    require(type(queries) is int and queries > 0, 'Positive query count required')
    require(eta_left.shape[0] == eta_right.shape[0] and eta_left.dtype == eta_right.dtype
            and eta_left.device == eta_right.device == rows_left.device == rows_right.device,
            'Aligned side/member/device schema required')
    require(eta_left.shape[1] == len(rows_left) and eta_right.shape[1] == len(rows_right), 'Slot rows differ')
    labels.check(eta_left, len(rows_left) + len(rows_right))
    left, nl, kl, rl = _grouped_side_nll(eta_left, rows_left, labels.values[:len(rows_left)], queries)
    right, nr, kr, rr = _grouped_side_nll(eta_right, rows_right, labels.values[len(rows_left):], queries)
    members = eta_left.shape[0]
    total = left + right
    denom = (nl + nr).clamp(min=1)
    zero = eta_left[:, :0].sum() + eta_right[:, :0].sum()
    lm = torch.where(rl > 0, -torch.logsumexp(-left, dim=0) + log(members), zero)
    rm = torch.where(rr > 0, -torch.logsumexp(-right, dim=0) + log(members), zero)
    shared = torch.where((rl > 0) | (rr > 0), -torch.logsumexp(-total, dim=0) + log(members), zero)
    result = {'J_K': shared / denom, 'J_K_sep': (lm + rm) / denom,
              'W_K': total.mean(dim=0) / denom, 'member_nll': total,
              'rho': torch.softmax(-total, dim=0)}
    result.update(support_counts=[nl.tolist(), nr.tolist()], teacher_counts=[kl.tolist(), kr.tolist()],
                  denominator=denom.tolist(), teacher_counts_for_serving=False)
    return result
