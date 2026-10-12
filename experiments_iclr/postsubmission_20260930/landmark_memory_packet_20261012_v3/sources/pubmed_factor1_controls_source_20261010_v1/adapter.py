"""Only change: all23 native maps gain one unit R/S row before any Adam step."""
REFERENCE_FACTORY=None


def fresh_single(condition,seed,tensor,device='cuda:0'):
    if REFERENCE_FACTORY is None:raise RuntimeError('Exact reviewed reference surface must bind the factory')
    if condition not in ('single_native','single_mean4_dropout'):
        raise ValueError('Only explicit ordinary/four-loss internal aliases are allowed')
    import torch
    session=REFERENCE_FACTORY(condition,seed,tensor,device=device)
    body=session.bodies[0];original=dict(body.named_parameters())
    if session.counters['updates']!=0 or any(optimizer.state for optimizer in session.optimizers):
        raise ValueError('Factor installation must precede all numerical optimizer work')
    sites=session.factors.install_factors(body,1);body.to(session.device)
    named=dict(body.named_parameters())
    private=[p for n,p in named.items() if n.endswith(('.r','.s'))]
    if sites!=23 or any(named[n] is not p for n,p in original.items()):
        raise ValueError('All original native slow parameter objects must survive all23 factor sites')
    if len(private)!=46 or any(p.shape[0]!=1 or not torch.equal(p.detach(),torch.ones_like(p)) for p in private):
        raise ValueError('Exactly one unit R/S row per native affine map')
    if sum(p.numel() for p in private)!=13943 or sum(p.numel() for p in body.parameters())!=2083818:
        raise ValueError('Native2069875 plus13943 redundant factor coordinates required')
    # The reference factory's original Adam is empty and unused. Rebuild its
    # identical name-based grouping to include new coordinates; charge both.
    session.optimizers=[session._optimizer(body)]
    optimized=[p for g in session.optimizers[0].param_groups for p in g['params']]
    if len(optimized)!=len(named) or {id(p) for p in optimized}!={id(p) for p in body.parameters()}:
        raise ValueError('One deduplicated original Adam owns every slow and factor coordinate')
    for (name,p),group in zip(body.named_parameters(),session.optimizers[0].param_groups):
        if len(group['params'])!=1 or group['params'][0] is not p or group['lr']!=(.0005 if 'attnmodule' in name else .005) or group['weight_decay']!=(1e-8 if 'attnmodule' in name else .001) or group['eps']!=1e-8 or group['betas']!=(.9,.999):
            raise ValueError('Unchanged native parameter-name Adam grouping and defaults required')
    if session.decoder is not None or session.spec['members']!=1 or session.spec['masked']:
        raise ValueError('One factual predictor, no masking/decoder/CORE')
    return session
