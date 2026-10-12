"""Stateless nonlocal TRAIN-label retrieval on existing captured native H.

The caller supplies torch and its existing model outputs, role-restricted TRAIN
anchors, common query draw, optimizer, RNG and selector. No imports of scientific
libraries, trainable parameters, model owner, cache, graph edit or lifecycle.
This source is prospective and has not been numerically qualified.
"""


def _rows(torch, rows, n, device, name):
    if rows.dtype != torch.long or rows.ndim != 1 or rows.device != device:
        raise ValueError(name + ' must be a one-dimensional long tensor on the output device')
    if rows.numel() == 0 or rows.min().item() < 0 or rows.max().item() >= n:
        raise ValueError(name + ' must contain nonempty valid full-graph row positions')
    if torch.unique(rows).numel() != rows.numel():
        raise ValueError(name + ' must contain unique row positions')


def common_half_query(torch, anchor_rows, anchor_labels, classes, *, generator):
    """Fixed approximately-half-per-class draw using a caller-owned generator.

    For odd counts, ascending class IDs receive the extra query slot until
    floor(A/2) slots are filled. Every class needs two TRAIN anchors. Nothing is
    drawn from native/member RNG streams or from a heldout role.
    """
    if classes < 2 or anchor_labels.dtype != torch.long:
        raise ValueError('Multiclass TRAIN labels are required')
    if anchor_rows.ndim != 1 or anchor_labels.shape != anchor_rows.shape:
        raise ValueError('Aligned role-restricted TRAIN row/label arrays required')
    if torch.unique(anchor_rows).numel() != anchor_rows.numel():
        raise ValueError('TRAIN rows must be unique')
    if anchor_labels.min().item() < 0 or anchor_labels.max().item() >= classes:
        raise ValueError('TRAIN label outside declared class domain')
    counts = [int(anchor_labels.eq(c).sum().item()) for c in range(classes)]
    if min(counts) < 2:
        raise ValueError('Every class must occur in both TRAIN query and support sets')
    quotas = [count // 2 for count in counts]
    extra = anchor_rows.numel() // 2 - sum(quotas)
    for c, count in enumerate(counts):
        if extra and count % 2:
            quotas[c] += 1
            extra -= 1
    pieces = []
    for c, quota in enumerate(quotas):
        rows = anchor_rows[anchor_labels.eq(c)]
        order = torch.randperm(rows.numel(), generator=generator,
                               device=generator.device).to(rows.device)
        pieces.append(rows[order[:quota]])
    return torch.cat(pieces)


def label_memory(torch, logits, hidden, anchor_rows, anchor_labels, query_rows,
                 *, mode, stop_attention_gradient=False, chunk_rows=512):
    """Return native/retrieval/mixed log probabilities for aligned query rows.

    `logits[M,N,C]` and `hidden[M,N,D]` must come from the same current label-free
    full native forward. In fit mode query_rows is the ONE common Q for every
    route; all Q labels are removed from every support. Serve mode permits only
    queries outside TRAIN. Label values are immutable one-hots; retrieval never
    modifies or re-encodes hidden. Fixed cosine temperature=.1 and mixture=.5.
    """
    if mode not in ('fit', 'serve') or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError('Explicit fit/serve role and positive chunk size required')
    if logits.ndim != 3 or hidden.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
        raise ValueError('Aligned full native logits[M,N,C] and hidden[M,N,D] required')
    if logits.shape[0] not in (1, 4) or logits.shape[2] < 2 or hidden.shape[2] < 1:
        raise ValueError('Native M1/M4 multiclass outputs required')
    if logits.device != hidden.device or logits.dtype != torch.float32 or hidden.dtype != torch.float32:
        raise ValueError('Existing full float32 logits/hidden convention required')
    m, n, classes = logits.shape
    _rows(torch, anchor_rows, n, logits.device, 'TRAIN anchors')
    _rows(torch, query_rows, n, logits.device, 'queries')
    if anchor_labels.shape != anchor_rows.shape or anchor_labels.dtype != torch.long or anchor_labels.device != logits.device:
        raise ValueError('Only aligned role-restricted TRAIN labels may supply values')
    if anchor_labels.min().item() < 0 or anchor_labels.max().item() >= classes:
        raise ValueError('TRAIN label outside declared class domain')
    query_in_train = torch.isin(query_rows, anchor_rows)
    if mode == 'fit':
        if not query_in_train.all().item() or query_rows.numel() != anchor_rows.numel() // 2:
            raise ValueError('Fit requires the common half-TRAIN Q')
        visible = ~torch.isin(anchor_rows, query_rows)
        support_rows, support_labels = anchor_rows[visible], anchor_labels[visible]
        if any(not support_labels.eq(c).any().item() or
               not anchor_labels[torch.isin(anchor_rows, query_rows)].eq(c).any().item()
               for c in range(classes)):
            raise ValueError('Every class must be represented in both Q and support')
    else:
        if query_in_train.any().item():
            raise ValueError('Serving queries must be disjoint from TRAIN')
        support_rows, support_labels = anchor_rows, anchor_labels
    if torch.isin(support_rows, query_rows).any().item():
        raise ValueError('All query labels must be absent from every route support')
    normalized = torch.nn.functional.normalize(hidden, p=2, dim=-1, eps=1e-12)
    support = normalized[:, support_rows, :]
    retrieval = []
    for start in range(0, query_rows.numel(), chunk_rows):
        rows = query_rows[start:start + chunk_rows]
        scores = (normalized[:, rows, :] @ support.transpose(-2, -1)) / 0.1
        log_attention = scores.log_softmax(-1)
        if stop_attention_gradient:
            log_attention = log_attention.detach()
        class_mass = []
        for c in range(classes):
            belongs = support_labels.eq(c)
            if belongs.any().item():
                class_mass.append(torch.logsumexp(log_attention[..., belongs], dim=-1))
            else:
                class_mass.append(log_attention.new_full((m, rows.numel()), float('-inf')))
        retrieval.append(torch.stack(class_mass, dim=-1))
    log_retrieval = torch.cat(retrieval, dim=1)
    log_native = logits[:, query_rows, :].log_softmax(-1)
    # logaddexp preserves zero retrieval mass for absent support classes while
    # the retained finite native probability keeps the served mixture finite.
    log_mixed = torch.logaddexp(log_native, log_retrieval) - 0.6931471805599453
    return {'native_log_probs': log_native, 'retrieval_log_probs': log_retrieval,
            'mixed_log_probs': log_mixed, 'support_rows': support_rows}
