"""Disabled exact-CB support-bucket hypothesis; no fit or serving integration.

Import only in a separately source-reviewed, physically supervised, root-released
fabricated CPU/native qualification. This module does not replace a declared core.
"""
from math import log
import torch
from conditional_loss import require, finite_logits, _group_log_esp_reachable


def _support_bucket_plan(descriptors):
    """Detached unique (n,r) metadata -> exact-r adaptive support buckets.

    All constant sides share one connected-zero group. Nonconstant supports use
    next-power-of-two length buckets. For r>=2 merge only if the padded width is
    smaller than the sum of distinct real lengths: recurrence Python slot loops
    strictly decrease. Otherwise retain the exact direct (n,r) groups. For r1
    merge distinct lengths within one bucket; there is no recurrence slot loop.
    This fixed engineering rule is not a measured runtime optimization.
    """
    buckets, constant = {}, []
    for source_group, (slots, degree) in enumerate(descriptors):
        require(type(slots) is int and type(degree) is int and 0 <= 2 * degree <= slots,
                'Invalid exact support/reduced-degree descriptor')
        if degree == 0:
            constant.append(source_group)
        else:
            width = 1 << (slots - 1).bit_length()
            buckets.setdefault((degree, width), []).append(source_group)
    plan = []
    if constant:
        plan.append((0, 0, constant))
    for (degree, width), source_groups in sorted(buckets.items()):
        lengths = [descriptors[source_group][0] for source_group in source_groups]
        merge = len(source_groups) > 1 and (degree == 1 or width < sum(lengths))
        if merge:
            plan.append((degree, width, source_groups))
        else:
            plan.extend((degree, descriptors[source_group][0], [source_group])
                        for source_group in source_groups)
    assignment = [-1] * len(descriptors)
    for group, (_, _, source_groups) in enumerate(plan):
        for source_group in source_groups:
            require(assignment[source_group] == -1, 'Support group assigned twice')
            assignment[source_group] = group
    require(all(group >= 0 for group in assignment), 'Support group omitted')
    return plan, assignment


def _masked_log_esp_reachable(centered, real, degree):
    """Tail padding has zero ESP weight; every retained state stays reachable.

    Caller guarantees real[q,i] iff i<n_q and n_q>=2*degree. Thus all first
    degree slots are real and every count 0..degree is finite before padding.
    Padding then only forms logaddexp(finite,-inf), never (-inf,-inf). Use
    finite centered logits separately for the observed-score dot product.
    """
    weighted = centered.masked_fill(~real[None, :, :], -float('inf'))
    return _group_log_esp_reachable(weighted, degree)


def _bucketed_side_nll(eta, rows, bits, queries):
    require(rows.ndim == 1 and rows.dtype == torch.long
            and (not len(rows) or (bool(((rows >= 0) & (rows < queries)).all())
                 and bool((rows[1:] >= rows[:-1]).all()))),
            'Native query-grouped rows within query range required')
    support = torch.bincount(rows, minlength=queries)
    teacher = torch.zeros(queries, dtype=torch.long, device=rows.device).index_add(0, rows, bits.to(torch.long))
    reduced = torch.minimum(teacher, support - teacher)
    keys, direct_inverse = torch.unique(torch.stack((support, reduced), dim=1), dim=0,
                                        sorted=True, return_inverse=True)
    # Transfer unique integer group descriptors, never per-query .item counts.
    plan, assignment = _support_bucket_plan(keys.tolist())
    codebook = torch.tensor(assignment, dtype=torch.long, device=rows.device)
    inverse = codebook[direct_inverse]
    order = torch.argsort(inverse, stable=True)
    sizes = torch.bincount(inverse, minlength=len(plan)).tolist()
    offsets = torch.cat((support.new_zeros(1), support.cumsum(0)))
    pieces, cursor = [], 0
    zero = eta[:, :0].sum(dim=1, keepdim=True)
    for (degree, width, source_groups), size in zip(plan, sizes):
        query_ids = order[cursor:cursor + size]
        cursor += size
        if degree == 0:
            pieces.append(zero.expand(-1, size))
            continue
        positions = torch.arange(width, device=rows.device)[None, :]
        real = positions < support[query_ids, None]
        raw = offsets[query_ids, None] + positions
        # A padded gather repeats its own first real slot: no out-of-range or
        # cross-query address. Every active query has n>=2r>=2 real slots.
        indices = torch.where(real, raw, offsets[query_ids, None])
        logits = eta[:, indices]
        observed = bits[indices]
        centered = logits - logits[..., :1]
        complement = teacher[query_ids] > support[query_ids] - teacher[query_ids]
        centered = torch.where(complement[None, :, None], -centered, centered)
        observed = torch.where(complement[:, None], 1 - observed, observed)
        observed = observed.masked_fill(~real, 0)
        require(bool(torch.isfinite(centered).all()), 'Finite-logit centering exceeded working precision')
        if degree == 1:
            selected = observed.argmax(dim=-1)
            chosen = centered.gather(-1, selected[None, :, None].expand(eta.shape[0], -1, 1)).squeeze(-1)
            weighted = centered.masked_fill(~real[None, :, :], -float('inf'))
            loss = torch.logsumexp(weighted, dim=-1) - chosen
        else:
            # Unmerged groups preserve the inherited exact vector recurrence.
            normalizer = (_masked_log_esp_reachable(centered, real, degree)
                          if len(source_groups) > 1 else _group_log_esp_reachable(centered, degree))
            loss = normalizer - (centered * observed[None, :, :]).sum(dim=-1)
        require(bool(torch.isfinite(loss).all()), 'Conditional bucketed loss exceeded working precision')
        pieces.append(loss)
    require(cursor == queries, 'Bucketed query coverage differs')
    grouped = torch.cat(pieces, dim=1)
    restore = torch.empty_like(order)
    restore[order] = torch.arange(queries, device=rows.device)
    return grouped[:, restore], support, teacher, reduced


def bucketed_training_pattern_losses(eta_left, eta_right, rows_left, rows_right, labels, queries):
    """Separate hypothesis API with the declared core's arguments/result schema.

    Exact laws/slots/query denominator/teacher/serving constraints are preserved.
    Every native neural call remains mandatory even when auxiliary degree is0.
    """
    finite_logits(eta_left)
    finite_logits(eta_right)
    require(type(queries) is int and queries > 0, 'Positive query count required')
    require(eta_left.shape[0] == eta_right.shape[0] and eta_left.dtype == eta_right.dtype
            and eta_left.device == eta_right.device == rows_left.device == rows_right.device,
            'Aligned side/member/device schema required')
    require(eta_left.shape[1] == len(rows_left) and eta_right.shape[1] == len(rows_right), 'Slot rows differ')
    labels.check(eta_left, len(rows_left) + len(rows_right))
    left, nl, kl, rl = _bucketed_side_nll(eta_left, rows_left, labels.values[:len(rows_left)], queries)
    right, nr, kr, rr = _bucketed_side_nll(eta_right, rows_right, labels.values[len(rows_left):], queries)
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
