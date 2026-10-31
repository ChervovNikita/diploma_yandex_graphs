"""Stateless full-neighborhood projected quartiles; caller owns Torch and graph.

Live native preclassifier hidden states, learned directions and residual maps.
No label, degree cap, sampled neighborhood, detached state, RNG or parameter
construction. Degree buckets retain all incoming nonself record multiplicity.
"""
def project(torch, hidden, directions):
    """H[N,D] with K directions, or H[M,N,D] with one/common route direction."""
    if hidden.ndim not in (2, 3) or directions.ndim != 2:
        raise ValueError('Live native hidden states and [directions,D] parameter')
    if hidden.shape[-1] != directions.shape[-1] or directions.shape[0] < 1:
        raise ValueError('Projection directions must span the native hidden width')
    if hidden.device != directions.device or hidden.dtype != directions.dtype:
        raise ValueError('Hidden states and directions must share dtype and device')
    unit = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)
    if hidden.ndim == 2:
        return unit @ hidden.transpose(0, 1)
    if directions.shape[0] not in (1, hidden.shape[0]):
        raise ValueError('One common or one private direction per live route')
    return (hidden * unit[:, None, :]).sum(-1)


def describe(torch, projected, degree_buckets, *, kind='quartiles'):
    """Return centered quartiles+degree or fixed population moments+degree.

    Each bucket is (positive_degree, recipient_ids[N_d], sources[N_d,d]).
    Caller topology preparation binds complete, nonoverlapping positive-degree
    recipients in native node order. Empty recipient rows remain all zero.
    Stable sorting retains the factual source order when projected values tie.
    """
    if kind not in ('quartiles', 'moments'):
        raise ValueError('Explicit quartiles or fixed mean/std/min/max moments')
    if projected.ndim != 2:
        raise ValueError('Projected full-node scalar fields [directions,N]')
    fields, recipients = [], []
    for degree, target, source in degree_buckets:
        if (type(degree) is not int or degree < 1 or target.ndim != 1
                or source.shape != (len(target), degree)
                or target.dtype != torch.long or source.dtype != torch.long
                or target.device != projected.device or source.device != projected.device):
            raise ValueError('Complete positive-degree factual record bucket')
        values = projected.index_select(1, source.reshape(-1)).reshape(
            projected.shape[0], len(target), degree)
        ordered = values.sort(dim=-1, stable=True).values
        mean = values.mean(-1)
        if kind == 'quartiles':
            quantiles = []
            for numerator, denominator in ((1, 4), (1, 2), (3, 4)):
                position = (degree - 1) * numerator
                lower, remainder = divmod(position, denominator)
                upper = lower if remainder == 0 else lower + 1
                fraction = remainder / denominator
                quantiles.append((1 - fraction) * ordered[..., lower] + fraction * ordered[..., upper])
            feature = torch.stack(quantiles, dim=-1) - mean[..., None]
        else:
            std = ((values - mean[..., None]).square().mean(-1) + 1e-8).sqrt()
            feature = torch.stack((mean, std, ordered[..., 0], ordered[..., -1]), dim=-1)
        degree_field = torch.full_like(mean[..., None], degree).log1p()
        fields.append(torch.cat((feature, degree_field), dim=-1))
        recipients.append(target)
    result = projected.new_zeros((projected.shape[0], projected.shape[1], 4 if kind == 'quartiles' else 5))
    if fields:
        result = result.index_copy(1, torch.cat(recipients), torch.cat(fields, dim=1))
    return result


def correct(native_logits, descriptor, score_maps):
    """Private bias-free 4×C or 5×C maps after each native route; gradients live."""
    if (native_logits.ndim != 3 or descriptor.ndim != 3 or descriptor.shape[-1] not in (4, 5)
            or descriptor.shape[:2] != native_logits.shape[:2]
            or score_maps.shape != (native_logits.shape[0], descriptor.shape[-1], native_logits.shape[-1])):
        raise ValueError('Aligned native scores, fixed descriptors and private score maps')
    return native_logits + descriptor @ score_maps
