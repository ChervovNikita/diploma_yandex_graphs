"""Exact four NCN interaction credits on caller-authenticated live source rows."""
METHODS = ('interaction_geometry_input', 'interaction_responsibility_private_pool',
           'interaction_input_meanJ', 'interaction_input_conditionalJ')
QUERY_SCOPE = 'ogbl-collab/official-time/source-node-ID/undirected/relation-0/v1'


def build(torch, *, method, link, responsibility, sampler, aggregation,
          aligned, raw_logits, native_losses, native_pool, gate, seed,
          actual_route_rows):
    if method not in METHODS or len(actual_route_rows) != 4:
        raise ValueError('Explicit recipe and four actual native route captures required')
    groups = aligned['groups']
    logits = torch.stack(raw_logits)
    reconstructed = link.reduce_native(torch, link.native_record_losses(torch, logits, groups, 'NCN'), groups, 'NCN')
    if not torch.allclose(reconstructed.detach(), torch.stack(native_losses).detach(), atol=1e-7, rtol=1e-6):
        raise ValueError('Captured raw queries do not recover all unchanged native source risks')
    F = torch.stack(native_losses).mean()
    if method == 'interaction_geometry_input':
        value = link.build_objectives(torch, family='NCN', logits=logits,
            native_member_losses=native_losses, native_pool_loss=native_pool, seed=seed,
            contrast='endpoint', responsibility='none', aux_credit='private', coefficient=.05,
            temperature=.2, anchor_limit=1024, negative_limit=16, role='TRAIN', **aligned)
        return {**value, 'logged_objective': value['private_objective'], 'scalar_update': False,
            'interaction_diagnostics': {'native_source_reduction_parity': True,
                'credit': 'sharedF;privateF+.05A_endpoint', 'contrast': value['diagnostics']}}
    if method == 'interaction_responsibility_private_pool':
        signatures, batch = [], None
        for rows in actual_route_rows:
            keys = [key for row in rows for key in row['keys']]
            labels = [group for row in rows for group in row['groups']]
            current = sampler.make_query_batch([key[:2] for key in keys], [key[2] for key in keys],
                scope=QUERY_SCOPE, directed=False)
            signatures.append(sampler.authenticate(current, labels, role='TRAIN'))
            if batch is None:
                batch = current
        terms = responsibility.build_objectives(torch, batch=batch, groups=groups, family='NCN', logits=logits,
            native_member_losses=native_losses, route_query_signatures=signatures, seed=seed,
            mode='endpoint', floor=.25, role='TRAIN')
        private = .5*terms['weighted_private_task_loss']+.5*native_pool
        return {**terms, 'shared_objective': terms['native_own_loss'], 'private_objective': private,
            'logged_objective': private, 'scalar_update': False,
            'interaction_diagnostics': {'native_source_reduction_parity': True,
                'credit': 'sharedF;private.5W_endpoint_v2+.5L_mean',
                'responsibility': terms['diagnostics'], 'actual_route_v2_query_signatures': signatures}}
    if gate is None:
        raise ValueError('Actual declared mean/conditional readout required')
    def native_loss(value):
        labels = torch.as_tensor(groups, dtype=torch.long, device=value.device)
        return -torch.nn.functional.logsigmoid(value[labels == 1]).mean() \
               -torch.nn.functional.logsigmoid(-value[labels == 0]).mean()
    joint = aggregation.native_objective(torch, native_member_losses=native_losses, raw_logits=raw_logits,
        aggregator=gate, served_native_loss=native_loss, native_regularizer=F.new_zeros(()))
    return {'shared_objective': joint['J'], 'private_objective': joint['J'], 'logged_objective': joint['J'],
        'native_own_loss': joint['F'], 'actual_served_loss': joint['L'], 'scalar_update': True,
        'interaction_diagnostics': {'native_source_reduction_parity': True,
            'credit': 'all-block.5F+.5L_actual', 'aggregation_mode': gate.metadata()['mode'],
            'mean_route_weights': joint['weights'].detach().mean(1).cpu().tolist()}}
