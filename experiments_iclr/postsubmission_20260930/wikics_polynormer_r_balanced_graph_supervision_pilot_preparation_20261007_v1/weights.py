"""Fixed label-free graph/IID TRAIN supervision fields from protocol v2."""
def make_weights(torch, edge_index, train_ids, nodes, base_seed):
    generator = torch.Generator(device='cpu').manual_seed(700001 + base_seed)
    white = torch.randn(nodes, 2, generator=generator, dtype=torch.float64)
    edge = edge_index.cpu(); row, col = edge
    degree = torch.bincount(row, minlength=nodes).double()
    if bool((degree <= 0).any()): raise ValueError('Prepared graph must contain one loop per node')
    value = degree[row].rsqrt() * degree[col].rsqrt()
    operator = torch.sparse_coo_tensor(edge, value, (nodes, nodes)).coalesce()
    current = white; graph = white.clone()
    for _ in range(3):
        current = torch.sparse.mm(operator, current); graph += current
    raw = torch.cat((graph / 4, white), dim=1)
    centered = raw - raw[train_ids].mean(0)
    rms = centered[train_ids].square().mean(0).sqrt()
    if not bool(torch.isfinite(centered).all() and torch.isfinite(rms).all()) or bool((rms <= 0).any()):
        raise ValueError('Zero/nonfinite field; no redraw')
    normalized = centered / rms
    common_bound = max(1., float(normalized.abs().max()))
    fields = normalized / common_bound
    def routes(offset):
        pair = fields[:, offset:offset+2]
        contrasts = torch.stack((pair[:, 0], -pair[:, 0], pair[:, 1], -pair[:, 1]), 0)
        return (1 + .5 * contrasts).float()
    graph_weights, iid_weights = routes(0), routes(2)
    diagnostics = {'white_seed': 700001 + base_seed, 'common_full_node_bound': common_bound,
        'raw_TRAIN_RMS': rms.tolist(), 'final_TRAIN_RMS': fields[train_ids].square().mean(0).sqrt().tolist(),
        'final_TRAIN_means': fields[train_ids].mean(0).tolist(),
        'graph_weight_TRAIN_means': graph_weights[:, train_ids].mean(1).tolist(),
        'iid_weight_TRAIN_means': iid_weights[:, train_ids].mean(1).tolist(),
        'maximum_graph_member_mean_drift': float((graph_weights.mean(0)-1).abs().max()),
        'maximum_iid_member_mean_drift': float((iid_weights.mean(0)-1).abs().max()),
        'label_values_used': False, 'normalization': 'sum(weight*CE)/N_TRAIN; no sum-weight denominator',
        'balance_scope': 'Real-arithmetic pointwise route mean1/TRAIN mean1; realized FP32 drift recorded.'}
    if bool((graph_weights < .5).any() or (graph_weights > 1.5).any() or (iid_weights < .5).any() or (iid_weights > 1.5).any()):
        raise ValueError('Fixed weight range failed')
    return graph_weights, iid_weights, diagnostics
