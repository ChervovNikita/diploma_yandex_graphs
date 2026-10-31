"""Unexecuted synchronized mask round for four separate TABM1 parameter bodies.

No constructors, numerical imports, data reads, checkpoint writes or run loop.
The external owner supplies persistent per-body native RNG scopes, own selectors,
stopping flags and selected-state storage. V1's native route wrapper is reused.
"""


def coupled_I4_round(models, optimizers, active, graph, x, train_nodes, train_targets,
                     *, torch, functional, mask_sampler, native_route_forward,
                     body_rng_scope):
    """One sign-paired graph-input draw, then own updates for active bodies only.

    models: four separately constructed live native TABM1 objects, with no shared
    parameters/storage. Finished bodies retain their LAST acquisition states here;
    selected checkpoints stay separate and must not overwrite companion states.
    optimizers: one independent optimizer per actual body.
    active: four frozen booleans for this round, set by the external own-selector,
    300-stale/5000-update state. An inactive body is never restarted or updated.
    mask_sampler: persistent V1 activation_sign_paired sampler on its own stream.
    body_rng_scope(i): existing owner context preserving body's native FFN stream;
    it must not reseed at every round or consume the controlled sampler stream.
    native_route_forward: exact V1 helper; no new slow-map implementation.
    """
    if not (len(models) == len(optimizers) == len(active) == 4):
        raise ValueError("Exactly four native bodies, optimizers and activity flags")
    if len({id(m) for m in models}) != 4 or len({id(o) for o in optimizers}) != 4:
        raise ValueError("Four actual independent objects are required")
    if any(type(flag) is not bool for flag in active):
        raise ValueError("Activity is an explicit per-body boolean")
    owned = [set(id(p) for p in model.parameters()) for model in models]
    if len(set.union(*owned)) != sum(map(len, owned)):
        raise ValueError("Independent bodies cannot share Parameter objects")
    storages = [
        {(p.device.type, p.device.index, p.untyped_storage().data_ptr())
         for p in model.parameters() if p.numel()}
        for model in models
    ]
    if len(set.union(*storages)) != sum(map(len, storages)):
        raise ValueError("Separate bodies cannot alias parameter storage")
    for model, optimizer in zip(models, optimizers):
        if model.input_be_block.tabm_inits != 1:
            raise ValueError("Each body must be an unchanged native TABM1")
        if (model.model_name != 'SAGE' or len(model.residual_modules) != 2
                or model.dropout.p != models[0].dropout.p
                or type(model.residual_modules[0].normalization).__name__ != 'LayerNorm'):
            raise ValueError("Use the common bound two-block SAGE/LayerNorm law")
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        if optimized != trainable:
            raise ValueError("Each optimizer must own only its complete native body")
    if not any(active):
        return {'active': list(active), 'losses': [None] * 4, 'prefix_forwards': 0,
                'inactive_companion_prefix_forwards': 0, 'own_updates': 0}

    # No random operation or running-statistic update in the fixed LayerNorm
    # prefix. Every current body participates, including stopped companions.
    with torch.no_grad():
        prefixes = []
        for model in models:
            h = model.input_be_block(x, tabm_seed=0)
            h = model.act(h)
            h = model.residual_modules[0].normalization(h)
            prefixes.append(h)
        values = torch.stack(prefixes, dim=0)
        masks = mask_sampler(values, p=models[0].dropout.p, training=True,
                             site='unshared_first_normalized_sage_input')

    losses = [None] * 4
    for i, (model, optimizer, is_active) in enumerate(zip(models, optimizers, active)):
        if not is_active:
            continue
        with body_rng_scope(i):
            model.train()
            optimizer.zero_grad()
            out = native_route_forward(
                model, graph, x, 0, placement='first_normalized_sage_input',
                torch=torch, multiplier=masks[i])
            # Full own CE: neither /4 scaling nor a joint/shared optimizer.
            loss = functional.cross_entropy(out[train_nodes], train_targets.long())
            loss.backward()
            optimizer.step()
            losses[i] = loss.item()
            del out, loss
    return {'active': list(active), 'losses': losses, 'prefix_forwards': 4,
            'inactive_companion_prefix_forwards': sum(not flag for flag in active),
            'own_updates': sum(active), 'TRAIN_suffix_routes': sum(active),
            'native_FFN_dropout_calls': 4 * sum(active)}
