"""Stateless endpoint-conditioned VALUE operation; no native installer.

The caller supplies Torch and two registered 2*c -> d*d builders. Reuse the
existing FactorLinear and member_context for shared W/b and private r/s.
Call before native scalar normalization/attention and native aggregation.
Inputs are live gathered x_j/x_i from the same native propagation stage;
2-D GCN/SAGE tensors or 3-D [edge, head, channel] GAT tensors are supported.
This helper owns no parameters, RNG, graph, state, optimizer or diagnostics.
"""
from math import sqrt


def transport_values(torch, source_values, target_values, source_builder,
                     target_builder, *, stalk_dim, self_edge, conditioning):
    """Apply F_tgt.T @ F_src within each head, preserving native self messages.

    conditioning='endpoint': [flatten(x_j), flatten(x_i)].
    conditioning='node': [flatten(x_i), silu(flatten(x_i))], with exactly the
    same builder sizes and trainable parameter opportunity. Neither builder
    is constructed or modified here. self_edge must refer to the actual
    propagated edges after native self-loop insertion/removal.
    """
    if conditioning not in ('endpoint', 'node'):
        raise ValueError('Explicit endpoint or matched node conditioning')
    if type(stalk_dim) is not int or stalk_dim < 1:
        raise ValueError('Positive integer stalk dimension')
    if source_values.ndim not in (2, 3) or target_values.shape != source_values.shape:
        raise ValueError('Aligned native source/target VALUE tensors')
    if (not source_values.is_floating_point()
            or target_values.dtype != source_values.dtype
            or target_values.device != source_values.device):
        raise ValueError('Same floating dtype and device for live endpoints')
    edges = source_values.shape[0]
    if (self_edge.shape != (edges,) or self_edge.dtype != torch.bool
            or self_edge.device != source_values.device):
        raise ValueError('Boolean self-edge mask in propagated edge order')
    heads = source_values.shape[1] if source_values.ndim == 3 else 1
    channels = source_values.shape[-1]
    if heads < 1 or channels < 1 or channels % stalk_dim:
        raise ValueError('Each native head width must divide into stalk blocks')
    width = heads * channels
    source = source_values.reshape(edges, width)
    target = target_values.reshape(edges, width)
    context = torch.cat((source, target), dim=-1) if conditioning == 'endpoint' else \
        torch.cat((target, torch.nn.functional.silu(target)), dim=-1)
    src_raw = source_builder(context)
    tgt_raw = target_builder(context)
    expected = (edges, stalk_dim * stalk_dim)
    if src_raw.shape != expected or tgt_raw.shape != expected:
        raise ValueError('Two caller-owned 2*c -> d*d builders required')
    if any(value.dtype != source_values.dtype or value.device != source_values.device
           for value in (src_raw, tgt_raw)):
        raise ValueError('Builders must preserve VALUE dtype and device')
    identity = torch.eye(stalk_dim, dtype=source_values.dtype,
                         device=source_values.device)
    src = identity + src_raw.tanh().reshape(edges, stalk_dim, stalk_dim) / sqrt(stalk_dim)
    tgt = identity + tgt_raw.tanh().reshape(edges, stalk_dim, stalk_dim) / sqrt(stalk_dim)
    transport = tgt.transpose(-2, -1) @ src
    blocks = source_values.reshape(edges, heads, stalk_dim, channels // stalk_dim)
    moved = (transport[:, None, :, :] @ blocks).reshape_as(source_values)
    mask = self_edge.reshape((edges,) + (1,) * (source_values.ndim - 1))
    return torch.where(mask, source_values, moved)
