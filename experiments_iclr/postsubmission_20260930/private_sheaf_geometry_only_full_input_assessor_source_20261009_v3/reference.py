"""Full native M1 references, one tape/body at a time, CPU gradient accumulation."""
import gc
import time
from support import require


def make_m1(torch, adapter, factory, factor_paths, expected_native_objects, expected_native_count):
    model = factory()
    require(type(model) is adapter.load_native_general_class(), 'Fresh pinned original M1 native class')
    require(len(list(model.parameters())) == expected_native_objects and sum(value.numel() for value in model.parameters()) == expected_native_count,
            'Complete original M1 learned parameter opportunity')
    for path in factor_paths:
        original = adapter._module_at(model, path)
        require(type(original) is torch.nn.Linear and original.bias is None and original.in_features == 2*model.hidden_dim
                and original.out_features == model.d**2, 'Same original bias-free ordered incidence affine')
        adapter._replace(model, path, adapter.SharedBELinear(original))
    return model


def topology_tensors(torch, model, q):
    for owner_name, owner in (('model',model), ('builder',model.laplacian_builder)):
        for name in q.TOPOLOGY_NAMES:
            value = getattr(owner,name,None)
            if isinstance(value,torch.Tensor): yield owner_name+':'+name,value


def topology_snapshot(torch, model, q, bank_topology):
    original_references, original_copies = bank_topology
    references, copies = {}, {}
    values = dict(topology_tensors(torch, model, q))
    require(set(values) == {name[2:] for name in original_references if name.startswith('0:')}, 'Complete M1 original topology attribute schema')
    for name,value in values.items():
        reference_key = original_references['0:'+name][0]
        require(torch.equal(value.detach().cpu(), original_copies[reference_key]), 'M1 static topology exactly equals bank topology')
        references[name] = (id(value),value._version)
        if id(value) not in copies: copies[id(value)] = value.detach().cpu().clone()
    return references,copies


def verify_topology(torch, model, q, snapshot):
    references,copies = snapshot
    values = dict(topology_tensors(torch,model,q))
    require(set(values)==set(references) and all((id(value),value._version)==references[name]
            and torch.equal(value.detach().cpu(),copies[id(value)]) for name,value in values.items()), 'M1 topology identity/version/values unchanged')
    return dict(identity_version_values_unchanged=True, equivalent_original_topology=True,
                CPU_snapshot_bytes=sum(value.numel()*value.element_size() for value in copies.values()))


def ownership(torch, model, q, private_names, expected_names, role_meta, data, gradients=False):
    parameters = dict(model.named_parameters())
    require(set(parameters)==expected_names and len(parameters)==len({id(value) for value in parameters.values()}), 'M1 complete unique parameter schema')
    fast = [parameters[name] for name in sorted(private_names)]
    slow = [value for name,value in parameters.items() if name not in private_names]
    fast_keys = [q.storage_key(value) for value in fast]
    require(len(set(fast_keys))==len(fast_keys) and not set(fast_keys).intersection(q.storage_key(value) for value in slow), 'M1 disjoint private/shared storage')
    require(model.edge_index is model.laplacian_builder.edge_index, 'M1 native model/builder same immutable graph')
    if gradients:
        require(all(value.grad is not None and torch.isfinite(value.grad).all().item() for value in parameters.values()), 'M1 own scaled NLL has finite gradients for every native/private parameter')
        gradient_keys = [q.storage_key(value.grad) for value in parameters.values()]
        require(len(set(gradient_keys))==len(gradient_keys), 'M1 private/native gradient storage disjoint')
        shape = (role_meta['support_counts']['canonical_undirected'],model.d,model.d)
        learners,caches,rows = [],[],[]
        for layer,learner in enumerate(model.sheaf_learners):
            cache = learner.L
            require(isinstance(cache,torch.Tensor) and tuple(cache.shape)==shape and cache.device==data['x'].device
                    and cache.dtype==data['x'].dtype and not cache.requires_grad and cache.grad_fn is None
                    and torch.isfinite(cache).all().item(), 'M1 complete native detached transport cache')
            learners.append(id(learner)); caches.append(q.storage_key(cache))
            rows.append(dict(layer=layer,shape=list(cache.shape),dtype=str(cache.dtype),device=str(cache.device),bytes=cache.untyped_storage().nbytes()))
        require(len(set(learners))==len(set(caches))==model.layers and not set(caches).intersection(q.storage_key(value) for value in parameters.values()),
                'M1 distinct layer learners/cache storage, no parameter alias')
    else: rows=[]
    return dict(complete_native_M1=True,private_storage_disjoint=True,gradient_storage_checked=gradients,
                native_builders=1,private_factor_objects=len(fast),parameter_objects=len(parameters),caches=rows)


def accumulate(np, torch, adapter, factory, helpers, old, q, data, role_meta, initial_state, factor_paths,
               expected_native_objects, expected_native_count, original_topology, forward_rng, backward_end_rng,
               streamed_gradients, counters, records, persist, phase_memory):
    """Never retain a GPU body, tape, gradient or operator from another M1."""
    private_names = {path+suffix for path in factor_paths for suffix in ('.r','.s')}
    member0 = {name[len('members.0.'):]:value for name,value in initial_state.items() if name.startswith('members.0.')}
    expected_names = {name[len('members.0.'): ] for name in streamed_gradients if name.startswith('members.0.')}
    accumulator = {name:torch.zeros_like(value,device='cpu') for name,value in streamed_gradients.items()}
    counts = {name:0 for name in accumulator}
    model = None
    try:
        for member in range(4):
            require(model is None,'One live differential M1 body at a time')
            tick = time.perf_counter()
            row = dict(member=member,status='started',fresh_original_constructor=True,live_M1_bodies=1,
                       all_TRAIN_rows=True,full_graph=True,one_live_autograd_graph=True)
            records.append(row); persist()
            model = make_m1(torch,adapter,factory,factor_paths,expected_native_objects,expected_native_count)
            prefix = 'members.'+str(member)+'.'
            own_state = {name[len(prefix):]:value for name,value in initial_state.items() if name.startswith(prefix)}
            require(set(own_state)==set(member0),'Matching complete bank member state schema')
            model.load_state_dict(own_state,strict=True)
            require(old.exact(torch,old.cpu_tree(torch,model.state_dict()),own_state),'M1 exact original learned/private factor/buffer values')
            snapshot = topology_snapshot(torch,model,q,original_topology)
            row.update(parameter_buffer_values_exact=True,initial_ownership=ownership(torch,model,q,private_names,expected_names,role_meta,data))
            helpers.restore_rng(np,torch,forward_rng[member])
            require(old.exact(torch,helpers.capture_rng(np,torch),forward_rng[member]),'Exact corresponding actual-bank forward RNG')
            model.train(); model.zero_grad(set_to_none=True)
            versions = [(value,value._version) for value in model.parameters()]
            counters['reference_train_forward_attempts'] += 1
            persist()
            logp = model(data['x'])
            require(logp.shape==(data['x'].shape[0],2) and torch.isfinite(logp).all().item(),'Finite complete differential M1 TRAIN output')
            counters['reference_train_forwards_completed'] += 1
            nll = torch.nn.functional.nll_loss(logp[data['train_index']],data['train_y'])
            require(torch.isfinite(nll).item(),'Finite differential own all-TRAIN NLL')
            row['own_TRAIN_NLL'] = float(nll.item())
            counters['reference_backward_attempts'] += 1
            persist()
            (nll/4).backward()
            counters['reference_backwards_completed'] += 1
            del logp,nll
            require(all(value._version==version for value,version in versions),'M1 backward uses exact unchanged old parameters')
            del versions
            row.update(after_backward_ownership=ownership(torch,model,q,private_names,expected_names,role_meta,data,gradients=True),
                       topology=verify_topology(torch,model,q,snapshot),
                       matching_backward_end_RNG_exact=old.exact(torch,helpers.capture_rng(np,torch),backward_end_rng[member]))
            require(row['matching_backward_end_RNG_exact'],'Differential M1 complete forward/backward consumes same owned RNG')
            contributions={}
            for name,value in model.named_parameters():
                target = ('members.'+str(member)+'.' if name in private_names else 'members.0.')+name
                gradient = value.grad.detach().cpu().clone()
                require(target in accumulator and gradient.shape==accumulator[target].shape and gradient.dtype==accumulator[target].dtype,
                        'Exact common-slow/matching-private gradient mapping')
                accumulator[target].add_(gradient)
                counts[target] += 1
                contributions[target]=float(gradient.double().norm().item())
            # Loop locals must release parameter/gradient references before destroying the whole body.
            del value,gradient
            row.update(status='complete',CPU_accumulation=True,scaled_own_NLL_gradient=0.25,contribution_L2_norms=contributions,
                       complete_body_seconds_before_cleanup=time.perf_counter()-tick)
            helpers.synchronize(torch,data['x'].device)
            phase_memory('reference_M1_'+str(member))
            model = None
            gc.collect(); torch.cuda.empty_cache()
            row.update(complete_body_seconds=time.perf_counter()-tick,released_before_next_M1=True)
            persist()
        require(all(counts[name]==(1 if group_name(name,private_names) else 4) for name in counts),
                'Four own likelihood contributions per common slow entry; one per matching private entry')
        require(all(torch.isfinite(value).all().item() for value in accumulator.values()),'Finite complete CPU gradient accumulation')
        return accumulator,counts
    finally:
        model = None
        gc.collect(); torch.cuda.empty_cache()


def group_name(name, private_names):
    return name.split('.',2)[2] in private_names


def apply_accumulated_update(torch, model, opt, initial_state, accumulator, old, counters):
    require(old.exact(torch,old.cpu_tree(torch,model.state_dict()),initial_state),'Reference Adam starts at same exact initial bank values')
    opt.zero_grad(set_to_none=True)
    parameters = dict(model.named_parameters())
    require(set(parameters)==set(accumulator),'Exact deduplicated accumulated gradient/parameter schema')
    for name,value in parameters.items():
        require(value.shape==accumulator[name].shape and value.dtype==accumulator[name].dtype,'Matching complete reference gradient')
        value.grad = accumulator[name].to(device=value.device).clone()
    require(all(value.grad is not None and torch.isfinite(value.grad).all().item() for value in parameters.values()),'Finite transferred CPU sum for tape-free reference Adam')
    model.assert_ownership()
    counters['reference_Adam_attempts'] += 1
    opt.step()
    counters['reference_Adam_steps_completed'] += 1
