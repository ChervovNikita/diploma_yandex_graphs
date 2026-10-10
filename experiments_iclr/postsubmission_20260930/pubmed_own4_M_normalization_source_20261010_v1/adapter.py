"""Native own4 construction; only all23 private R/S Adam decay/epsilon change."""
ENABLED=False
CONDITION='shared4_own_M_normalized'
M=4


def fresh(method,seed,tensor,device='cuda:0',*,numerical_admitted=False):
    if numerical_admitted is not True or ENABLED is not False:
        raise RuntimeError('Disabled disk source; exact root numerical admission required')
    import torch
    session=method.Session('shared4_own',seed,tensor['x'],tensor['edge_index'],tensor['train_ids'],tensor['train_y'],device=device)
    body=session.bodies[0];maps=[m for m in body.modules() if isinstance(m,session.factors.FactorLinear)]
    private={id(p) for m in maps for p in (m.r,m.s)};named=list(body.named_parameters());optimizer=session.optimizers[0]
    if len(maps)!=23 or len(private)!=46 or len(session.bodies)!=1 or len(session.optimizers)!=1 or session.decoder is not None or session.spec['masked'] or session.spec['members']!=M or optimizer.state:
        raise ValueError('One fresh native factual own4,23 factor sites and empty Adam required')
    if any(m.r.shape[0]!=M or m.s.shape[0]!=M or not torch.equal(p.detach(),torch.ones_like(p)) for m in maps for p in (m.r,m.s)):
        raise ValueError('Original all4 unit factor rows required')
    if sum(p.numel() for _,p in named)!=2125647 or sum(p.numel() for _,p in named if id(p) in private)!=55772:
        raise ValueError('Unchanged complete native/private coordinate counts required')
    if len(optimizer.param_groups)!=len(named):raise ValueError('Original one-parameter name groups required')
    rows=[]
    for index,((name,p),group) in enumerate(zip(named,optimizer.param_groups)):
        attention='attnmodule' in name
        if group['params']!=[p] or group['lr']!=(.0005 if attention else .005) or group['weight_decay']!=(1e-8 if attention else .001) or group['eps']!=1e-8 or group['betas']!=(.9,.999):
            raise ValueError('Exact original native Adam group required')
        before={k:v for k,v in group.items() if k!='params'}
        is_private=id(p) in private
        if is_private:
            group['weight_decay']/=M;group['eps']/=M
        after={k:v for k,v in group.items() if k!='params'}
        expected=dict(before)
        if is_private:expected.update(weight_decay=before['weight_decay']/M,eps=before['eps']/M)
        if after!=expected:raise ValueError('Only private decay and epsilon may change')
        rows.append(dict(index=index,name=name,private_R_S=is_private,coordinates=p.numel(),native=before,normalized=after))
    if sum(r['private_R_S'] for r in rows)!=46 or sum(not r['private_R_S'] for r in rows)!=52:
        raise ValueError('Exactly46 private/52 native parameter groups required')
    session.name=CONDITION
    session._M_normalization_inventory=dict(M=M,groups=rows,changed_private_groups=46,unchanged_native_groups=52,
        changed_fields=['weight_decay','eps'],original_learning_rates=True,original_unit_start=True,
        original_native_mean_own_train_step=True,optimizer_rebuilt=False,native_shared_groups_unchanged=True)
    verify(session)
    return session


def verify(session):
    rows=session._M_normalization_inventory['groups']
    named=list(session.bodies[0].named_parameters())
    if session.name!=CONDITION or len(named)!=len(rows) or len(session.optimizers[0].param_groups)!=len(rows):
        raise ValueError('Exact normalized own4 identity required')
    for (name,p),row,group in zip(named,rows,session.optimizers[0].param_groups):
        if name!=row['name'] or group['params']!=[p] or {k:v for k,v in group.items() if k!='params'}!=row['normalized']:
            raise ValueError('Native/shared or normalized/private optimizer group changed')
    return session._M_normalization_inventory


def work(session,updates,evaluations=0):
    verify(session)
    return dict(updates=updates,factual_forwards=M*(updates+evaluations),masked_forwards=0,
        backwards=M*updates,optimizer_steps=updates,serving_forwards=M*evaluations,preprocessing_banks=1)
