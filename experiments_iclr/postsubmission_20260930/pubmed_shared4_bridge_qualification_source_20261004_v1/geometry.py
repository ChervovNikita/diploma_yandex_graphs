"""New Pubmed encoder/decoder geometry parity, fresh engineering units only."""
import sys
from common import require


def copy_unit_decoder(native_decoder, facade, prototype, torch):
    """Engineering identity limit only; never called by the resource factory."""
    target = facade.decoder
    with torch.no_grad():
        for name in ('xlin', 'xcnlin', 'xijlin', 'lin', 'ptlin'):
            for index, operation in getattr(target, name).ops.items():
                source = getattr(native_decoder, name)[int(index)]
                if isinstance(operation, prototype.FactorLinear):
                    operation.weight.copy_(source.weight)
                    operation.bias.copy_(source.bias)
                    operation.r.fill_(1)
                    operation.s.fill_(1)
                else:
                    operation.weight.copy_(source.weight[None].expand_as(operation.weight))
                    operation.bias.copy_(source.bias[None].expand_as(operation.bias))
        target.beta.copy_(native_decoder.beta.expand_as(target.beta))
        for name in ('scale', 'offset', 'alpha', 'pt'):
            getattr(target, name).copy_(getattr(native_decoder, name))


def gradient_projection(unit, prototype, native, torch):
    encoder, facade = unit[:2]
    result = {'encoder': {name: value.grad for name, value in encoder.named_parameters()}, 'predictor': {}}
    core = facade.decoder
    for name in ('xlin', 'xcnlin', 'xijlin', 'lin'):
        for index, operation in getattr(core, name).ops.items():
            for field in ('weight', 'bias'):
                value = getattr(operation, field).grad
                require(value is not None and bool(torch.isfinite(value).all()), 'Missing/nonfinite bridge native-projected gradient')
                result['predictor'][name + '.' + index + '.' + field] = value if isinstance(operation, prototype.FactorLinear) else value.sum(0)
    result['predictor']['beta'] = core.beta.grad.sum().reshape(1)
    for name, value in facade.named_parameters():
        if 'ptlin.' in name:
            require(value.grad is None, 'Fixed-pt unused native parameters acquired gradient')
        else:
            require(value.grad is not None and bool(torch.isfinite(value.grad).all()), 'Missing/nonfinite active bridge gradient: ' + name)
    return native.clone(result, torch)


def run_geometry(native, bodies, bridge, prototype, graph_ops, make, x, train, valid, pool, torch, np, progress, compare, exact, reports):
    progress.update(phase='new_geometry_parity')
    reference = bodies.candidate_factory('NCNC', 0, x, train)
    progress.add(native_reference_factory_calls=1)
    reference_initial = native.capture(reference, torch, np)
    private = make(0, 'private')
    pooled = make(0, 'pooled_after_clamp')
    for label, unit in (('private', private), ('pooled_after_clamp', pooled)):
        exact(reference_initial['encoder'], native.capture(unit, torch, np)['encoder'], label + '/exact_native_encoder_initialization', reports)
        copy_unit_decoder(reference[1], unit[1], prototype, torch)
        unit[0].eval()
        unit[1].eval()
    reference[0].eval()
    reference[1].eval()
    adj = reference[4].adj_t
    queries = torch.cat((train[:16], valid[:16])).to(bodies.device)
    negative = pool[:32, 0].to(bodies.device)
    # Actual task geometry, no full initialized VALID serve or metric calculation.
    projected = reference[0].xemb(reference[4].x)
    base = reference[0].convs[0](projected, adj)
    h = reference[0](reference[4].x, adj)
    compare(h, base * reference[0].jkparams.reshape(1, 1), 'active_one_layer_native_JK_output', reports)
    derivative = torch.autograd.grad(h.square().sum(), reference[0].jkparams)[0]
    expected = (2 * reference[0].jkparams * base.square().sum()).reshape_as(derivative)
    compare(derivative, expected, 'active_one_layer_native_JK_gradient', reports)
    require(bool((derivative != 0).any()), 'JK fixture fails to expose its active scalar gradient')

    def pass_loss(unit, is_native):
        encoder, predictor = unit[:2]
        unit[2].zero_grad(set_to_none=True)
        h = encoder(unit[4].x, unit[4].adj_t)
        positive = predictor.multidomainforward(h, unit[4].adj_t, queries.t(), cndropprobs=[])
        negatives = predictor.multidomainforward(h, unit[4].adj_t, negative.t(), cndropprobs=[])
        loss = -torch.nn.functional.logsigmoid(positive).mean() - torch.nn.functional.logsigmoid(-negatives).mean()
        loss.backward()
        progress.add(gradient_backward_calls=1)
        if is_native:
            gradient = native.clone({'encoder': {name: value.grad for name, value in encoder.named_parameters()},
                'predictor': {name: value.grad for name, value in predictor.named_parameters() if not name.startswith('ptlin.')}}, torch)
        else:
            gradient = gradient_projection(unit, prototype, native, torch)
        for name, value in encoder.named_parameters():
            require(value.grad is not None and bool(torch.isfinite(value.grad).all()), 'Missing/nonfinite native encoder gradient: ' + name)
        return native.clone((positive, negatives, loss), torch), gradient

    reference_values, reference_gradients = pass_loss(reference, True)
    for label, unit in (('private', private), ('pooled_after_clamp', pooled)):
        values, gradients = pass_loss(unit, False)
        for side in (0, 1):
            compare(values[side], reference_values[side].expand(-1, 4), label + '/all_unit_member_native_logits_' + str(side), reports)
            compare(unit[1](unit[0](unit[4].x, adj), adj, (queries if side == 0 else negative).t()),
                    reference_values[side], label + '/raw_mean_native_serving_' + str(side), reports)
        compare(values[2], reference_values[2], label + '/native_logsigmoid_loss', reports)
        compare(gradients, reference_gradients, label + '/active_encoder_and_native_projected_decoder_gradients', reports)

    # New facade conversion and masking check with duplicates, a self-loop,
    # asymmetric candidate neighborhoods, and an entirely empty neighborhood.
    pairs = torch.tensor([[0,1],[0,1],[0,2],[1,2],[1,3],[2,4],[4,5],[6,7],[8,8]], dtype=torch.long, device=bodies.device)
    indices = torch.tensor([0], dtype=torch.long, device=bodies.device)
    retained = pairs[1:]
    sparse = bodies.SparseTensor.from_edge_index(retained.t(), sparse_sizes=(19717,19717)).to_symmetric().coalesce()
    graph = private[1].graph(sparse)
    expected_graph = graph_ops.Graph.mask_train_batch(pairs, indices, 19717)
    exact((graph.row, graph.col, graph.rowptr), (expected_graph.row, expected_graph.col, expected_graph.rowptr), 'native_adjacency_record_mask_and_duplicate_support', reports)
    fabricated = torch.tensor([[0,1],[0,3],[9,10],[8,8],[3,4]], dtype=torch.long, device=bodies.device)
    neighbors = graph_ops.enumerate_neighbors(graph, fabricated)
    module = sys.modules[bodies.GCN.__module__]
    cn, left, right = module.adjoverlap(sparse, sparse, fabricated.t(), False, calresadj=True, cnsampledeg=-1, ressampledeg=-1)
    for name, actual, expected_sparse in (('common',neighbors.common,cn),('left',neighbors.left,left),('right',neighbors.right,right)):
        row, col, _ = expected_sparse.coo()
        expected_codes = torch.sort(row * 19717 + col).values
        actual_codes = torch.sort(actual[0] * 19717 + actual[1]).values
        exact(actual_codes, expected_codes, 'native_overlap_support_' + name, reports)
    require(not bool(((neighbors.common[0] == 2) | (neighbors.left[0] == 2)).any()) and not bool((neighbors.right[0] == 2).any()), 'Empty-neighborhood fixture is not empty')
    for label, unit in (('private',private),('pooled_after_clamp',pooled)):
        h = unit[0](unit[4].x, sparse)
        compare(unit[1](h, sparse, fabricated.t()), reference[1](reference[0](reference[4].x, sparse), sparse, fabricated.t()), label + '/asymmetric_empty_native_completion_logits', reports)
    return {'status':'PASS','native_encoder':'puregcn with input Linear and active one-layer JK',
            'actual_feature_geometry':[19717,500],'actual_hidden':256,'native_reference_factories':1,
            'bridge_unit_factor_engineering_factories':2,'loss_backward_calls':3,'JK_autograd_calls':1,
            'optimizer_updates':0,'full_initialized_VALID_serves':0,'donor_states_exported':False,
            'gradient_mapping':'Shared dense/bias gradients direct; private LN/beta gradients sum across identical unit-factor members. r/s gradients finite/present; prior analytic factor oracle reused.',
            'native_Adam_parameterization_identity_claim':False}
