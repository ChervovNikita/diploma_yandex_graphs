"""Finite exact-factory CMCL/Adam/CORE resource proxies. No dispatch on prepare."""
from contextlib import ExitStack,contextmanager
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import statistics
import sys
import time
from types import SimpleNamespace
from unittest import mock

CAP_HELPER='practical_resource_qualifier_source_20261012_v2/runtime/qa/practical_zero_update_resources.py'
CAP_SHA='effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'
PHYSICS_HELPER_SHA='3546f03fe24de6d188320e830349659975d8091e69af70dae5889518aa0171b2'
PACKET='massive_private_source_packets_20261012_v3'
FAMILIES=('private_cmcl','private_adam','masked_core')
MOLBACE_OWNER={'pid':737641,'start_ticks':6058883970,'pgid':737641}
MOLBACE_END='practical_modal_resource_minimized_molbace_20261012_v1/OWNER_END.json'
CMCL={'own_floor':(4,4),'private_cmcl':(8,8),'all_block_cmcl':(8,4),'vanilla_cmcl':(8,4),'private_uniform':(4,8),'private_constant_credit':(4,8)}
CORE={'shared4_own':(1,4,4),'shared4_masked_ce':(1,8,4),'shared4_core':(2,8,4),'shared4_core_shuffled':(2,8,4),
      'shared4_core_rewired':(2,8,4),'single_native':(1,1,1),'independent4_native':(4,4,4),'single_four_view_core':(2,5,5),'untied4_shared_decoder_core':(5,8,4)}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def local(root,relative):
    value=Path(relative)
    if value.is_absolute() or '..' in value.parts:raise ValueError('Project-relative source only')
    path=root/value
    if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Source/input escapes project')
    return path
def receipt(root,path):
    raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
    path=local(root,row['path'])
    if receipt(root,path)!=row:raise ValueError('Exact source/input bytes changed')
    return path
def module(root,row,name):
    path=bound(root,row);spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def role_paths(value):
    if isinstance(value,str):return {value}
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value):return {value['path']}
        return set().union(*(role_paths(v) for v in value.values())) if value else set()
    return set()


def closed_molbace_custody(root,binding=None):
    path=local(root,MOLBACE_END)
    if binding is not None:bound(root,binding)
    value=json.loads(path.read_text())
    if (value.get('status')!='CLOSED' or any(value.get('owner',{}).get(k)!=v for k,v in MOLBACE_OWNER.items())
            or len(value.get('closed',[]))!=7 or value.get('unexecuted_case_ids')!=[]):
        raise ValueError('Exact whole seven-case MolBACE queue closure required; no overlap')
    process=Path('/proc')/str(MOLBACE_OWNER['pid'])/'stat'
    if process.exists():
        fields=process.read_text().rsplit(')',1)[1].split()
        if int(fields[19])==MOLBACE_OWNER['start_ticks'] and fields[0]!='Z':raise ValueError('Exact MolBACE guardian still live; no overlap')
    return {'binding':receipt(root,path),'scope':'administrative closed seven-case queue only; no scientific outcome or case metric opened','overlap':False}


def operator_law(config,multiplicity=1):
    if config['architecture']=='private_cmcl':
        forwards,vjps=CMCL[config['cmcl_condition']];owners=1;backwards=0
    elif config['architecture']=='masked_core':
        owners,forwards,backwards=CORE[config['core_condition']];vjps=0
    else:owners=1;forwards=backwards=config['members'];vjps=0
    if owners!=config['source_optimizer_count']:raise ValueError('Exact declared optimizer owner count differs')
    return {'native_forwards':multiplicity*forwards,'autograd_grad_VJP_calls':multiplicity*vjps,
        'Tensor_backward_calls':multiplicity*backwards,'intercepted_optimizer_attempts':multiplicity*owners}


def prepare_manifest(root,helper,cap,physics_helper,start,watcher):
    cases=[];packets=[]
    for family in FAMILIES:
        directory=root/PACKET/family;entry=json.loads((directory/'MASTER_INPUT_ENTRY.json').read_text());plan=json.loads(bound(root,entry['plan']).read_text())
        if plan['scientific_launch_enabled'] is not False:raise ValueError('Disabled exact packet only')
        packets.append({'family':family,'entry':receipt(root,directory/'MASTER_INPUT_ENTRY.json'),'plan':entry['plan'],'plan_hash':digest(plan)})
        for spec in plan['studies']:
            path=local(root,spec['path']);study=json.loads(gzip.decompress(path.read_bytes()))
            if digest(study)!=spec['spec_hash'] or study['dataset'] not in ('wikics','amazon-photo'):raise ValueError('Exact admitted new provider/source study required')
            custody_rows=[r for r in study['source']['files'] if r['path'].endswith('/'+study['dataset']+'/CUSTODY.json') and '/prepared_data/' in r['path']]
            if len(custody_rows)!=1:raise ValueError('Exact provider metadata custody required')
            custody=json.loads(bound(root,custody_rows[0]).read_text());counts={r['split_id']:r['role_counts']['train'] for r in custody['splits']};mask=min(counts,key=lambda k:(-counts[k],k))
            forbidden=set().union(*(role_paths(study['data']['roles'][r]) for r in ('select','cal','evaluate')))
            for r in study['runs']:forbidden.update(role_paths(r['config'].get('endpoint_inputs',{})))
            runs={r['run_id']:r for r in study['runs']};fits=[r for r in study['runs'] if r['kind']=='model_fit']
            def add(search,cohort,multiplicity=1,bank=None):
                recipe=copy.deepcopy(search);recipe['split']=mask;recipe['config']['node_inputs']=copy.deepcopy(cohort['config']['node_inputs'])
                if 'identity' in recipe['config']:recipe['config']['identity']=copy.deepcopy(cohort['config']['identity'])
                c=recipe['config'];factory=local(root,recipe['adapter'].partition(':')[0]);pins={}
                for field in ('cmcl_source','core_source'):
                    for name,pin in c.get(field,{}).items():pins[field+'/'+name]=pin
                if c.get('private_adam_source'):pins['private_adam_source']=c['private_adam_source']
                row={'schema':'engineering-only-resource-case-v1','case_id':'','family':family+'_exact_factory_zero_update','packet_family':family,
                    'condition':c.get('cmcl_condition',c.get('control_condition',c.get('core_condition'))),'source_spec_or_source_recipe_binding':receipt(root,path),'source_spec_hash':spec['spec_hash'],
                    'source_search_run_id':search['run_id'],'source_input_cohort_run_id':cohort['run_id'],'run':recipe,'prototype_bodies':multiplicity,'independent_bank_method':bank,
                    'builder_code_binding':helper,'source_adapter_binding':receipt(root,factory),'source_operator_pins':pins,'runtime_package':str(factory.parent.parent.relative_to(root)),
                    'GRAPH_TRAIN_bindings':{k:c['node_inputs'][k] for k in ('graph','train')},'auxiliary_GRAPH_binding':c.get('auxiliary_graph'),
                    'provider_custody':custody_rows[0],'maximum_TRAIN_rows':counts[mask],'all_prepared_masks':sorted(counts),'representative_mask':mask,
                    'prohibited_original_role_paths':sorted(forbidden),'role_authority':'immutable GRAPH/TRAIN and separately frozen auxiliary GRAPH only; original SELECT/CAL/evaluate bytes denied even for hashes; one unused public node synthetic SELECT',
                    'probe_unit':'one cold plus three warm complete exact source callbacks at fixed random source state; all optimizer and parameter/moment actions denied',
                    'actual_parameter_and_optimizer_ownership':'all actual source optimizers; manual zero Adam/AdamW moments; complete nonoverlapping parameter owner roster, per-owner attempt counts and hashes',
                    'source_operator_counts':operator_law(c,multiplicity),'random_or_paid_ancestry_status':'fresh source unit factors/native Xavier, untrained/unselected; no registry or paid ancestor',
                    'state_equality_proofs':'exact parameters, registered buffers, unregistered PolyAttn bias, manual moments, fixed source input tensors; owned/global RNG/counters/modes restored after every cycle',
                    'intrinsic_interception_contract':'CMCL final optimizer.step only; CORE final all-owner loop only; private Adam exact CORE own train_step; .grad accumulation/division remains live; protected parameter/moment storage writes and .data replacement blocked',
                    'allocator_cap_and_external_usage':'reviewed normal child Torch min(fresh_free-4GiB,floor(.4*total)); external PID memory observed before factory and at completion',
                    'owner_PID_start_PGID':'saved before numerical import; exact reviewed Physics owner custody before/after','timeout_and_terminal_cost':{'maximum_seconds':600,'terminal_failure_cost_retained':True},
                    'no_retry':True,'no_score':True,'qualification_limits':'random source derivative/storage and disposable serializer proxy only; no optimizer kernel timing, trained state, SELECT/CAL/evaluate, scientific fit or full-study ETA',
                    'historical_CORE_stage2_activated':False,'new_comparison_policy':c.get('new_comparison_policy'),'source_anchor':c['learning_rate']==.005 and c['dropout']==.5,
                    'source_frontier_proxy':c['dropout']==.7,'scientific_parent_receipt':None}
                row['case_id']=digest(row)[:24];cases.append(row)
            for selector in study['runs']:
                if selector['kind']!='selection':continue
                members=[runs[i] for point in selector['config']['configurations'] for i in point['run_ids']]
                cohort=min((r for r in fits if r['split']==mask and r['config'].get('selected_config_from')==selector['run_id']),key=lambda r:(r['body'],r['seed']))
                for dropout in (.5,.7):
                    search=min((r for r in members if r['config']['learning_rate']==.005 and r['config']['dropout']==dropout),key=lambda r:(r['body'],r['seed']))
                    add(search,cohort)
            banks=[r for r in study['runs'] if r['kind']=='ensemble_assembly' and r['split']==mask and len(r['config']['fit_parents'])==4]
            for method in sorted({r['method'] for r in banks}):
                bank=min((r for r in banks if r['method']==method),key=lambda r:r['seed']);cohort=runs[bank['config']['fit_parents'][0]]
                search=min((r for r in fits if r['config']['stage']=='search' and r['method']==cohort['method'] and r['config']['learning_rate']==.005 and r['config']['dropout']==.7),key=lambda r:(r['body'],r['seed']))
                add(search,cohort,4,method)
    return {'schema':'private-source-exact-factory-zero-update-manifest-v1','helper':helper,'cap_helper':cap,'Physics_custody_helper':physics_helper,
        'Physics_owner_start':start,'Physics_original_watcher':watcher,'packets':packets,'cases':cases,'execution_authorized':False,'scientific_launch_enabled':False,
        'actual_optimizer_updates_allowed':0,'case_timeout_seconds':600,'warm_iterations':3,'no_retry':True,'no_score':True,'CUDA_executions':0,
        'MolBACE_queue_overlap_allowed':False,'separate_root_source_review_and_case_authorization_required':True,'historical_CORE_stage2_activated':False}


class ProtectedTensorCommit(RuntimeError):pass


class TensorCommitDenial:
    """Additional source-reviewed boundary beyond concrete/functional step denial."""
    OPERATIONS=('add_','sub_','mul_','div_','addcmul_','addcdiv_','lerp_','copy_','zero_','fill_','set_','resize_','resize_as_',
        'index_copy_','index_put_','scatter_','scatter_add_','masked_fill_','masked_scatter_','__setitem__')
    def __init__(self,torch,parameters,moments):
        self.t=torch;self.objects={id(v) for v in [*parameters,*moments]};self.storages={(str(v.device),v.untyped_storage().data_ptr()) for v in [*parameters,*moments]};self.stack=ExitStack();self.denied=0
    def protected(self,value):return id(value) in self.objects or (str(value.device),value.untyped_storage().data_ptr()) in self.storages
    def __enter__(self):
        def wrapper(original):
            def checked(value,*a,**kw):
                if self.protected(value):self.denied+=1;raise ProtectedTensorCommit('Protected parameter/manual-moment storage commit denied')
                return original(value,*a,**kw)
            return checked
        for name in self.OPERATIONS:
            if hasattr(self.t.Tensor,name):self.stack.enter_context(mock.patch.object(self.t.Tensor,name,wrapper(getattr(self.t.Tensor,name))))
        original=self.t.Tensor.__setattr__
        def setter(value,name,data):
            if name=='data' and self.protected(value):self.denied+=1;raise ProtectedTensorCommit('Parameter/manual-moment .data replacement denied')
            return original(value,name,data)
        self.stack.enter_context(mock.patch.object(self.t.Tensor,'__setattr__',setter));return self
    def __exit__(self,*a):self.stack.__exit__(*a)


def tree_digest(torch,value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=repr):h.update(repr(key).encode());visit(v[key])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for item in v:visit(item)
        elif v is None or type(v) in (str,int,float,bool):h.update(repr(v).encode())
        else:raise TypeError('Source engineering state must remain plain tensor/container tree')
    visit(value);return h.hexdigest()


def source_fixed_tensors(adapter):
    s=getattr(adapter,'source_session',None)
    if s is None:return list(adapter.model.named_buffers())
    if hasattr(s,'inputs'):
        return [('token'+str(i),v) for i,v in enumerate(s.inputs.factual.list_mat)]+[(r+'/'+k,getattr(getattr(s.inputs,r),k)) for r in ('train','validation') for k in ('ids','targets')]
    return [('x',s.x),('edges',s.edges),('train_ids',s.train_ids),('train_y',s.train_y),*[(str(i)+'/token'+str(j),v) for i,view in enumerate([s.factual,*s.masked]) for j,v in enumerate(view.list_mat)],
        *[('anchor'+str(i),v) for i,v in enumerate(s.anchor_ids)],('select_ids',s.select_ids),('select_y',s.select_y)]


def measure_source(torch,h,adapters,denial,output,warm_iterations=3):
    if not 1<=warm_iterations<=3:raise ValueError('One to three warm cycles only')
    all_optimizers=[o for a in adapters for o in h.optimizers(a)]
    model=torch.nn.ModuleList([a.model for a in adapters]);roster=SimpleNamespace(model=model,optimizer=SimpleNamespace(optimizers=all_optimizers))
    owned=[p for o in all_optimizers for g in o.param_groups for p in g['params']]
    if len(owned)!=len({id(p) for p in owned}) or {id(p) for p in owned}!={id(p) for p in model.parameters()}:raise ValueError('Every source parameter needs exactly one real optimizer owner')
    for a in adapters:
        for name in ('validate','observe_validation','evaluate','export'):
            if hasattr(a,name):setattr(a,name,lambda *x,**k:(_ for _ in ()).throw(h.ForbiddenRoleAccess('Selectors/metrics/endpoints prohibited')))
    h.allocate_adam_moments(torch,roster)
    moments=[v for _,v in h.optimizer_tensors(roster)]
    p0=h.tensor_digest(torch,model.named_parameters());m0=h.tensor_digest(torch,h.optimizer_tensors(roster));b0=h.tensor_digest(torch,model.named_buffers())
    fixed=[(str(i)+'/'+name,v) for i,a in enumerate(adapters) for name,v in source_fixed_tensors(a)];fixed0=h.tensor_digest(torch,fixed)
    versions=[int(v._version) for _,v in fixed]
    states=[copy.deepcopy(a.capture_state()) for a in adapters];state0=tree_digest(torch,states);modes=[(m,m.training) for m in model.modules()]
    source_costs=[copy.deepcopy(a.source_session.costs.records) if hasattr(getattr(a,'source_session',None),'costs') else None for a in adapters]
    attempts={id(o):0 for o in all_optimizers};counts={};times=[];observations=[];gradient_bytes=0
    bodies=[]
    for a in adapters:
        s=getattr(a,'source_session',None);bodies+=list(s.bodies) if s is not None and hasattr(s,'bodies') else [s.model] if s is not None else [a.model.body]
    def synchronized():
        if next(model.parameters()).device.type=='cuda':torch.cuda.synchronize()
    with ExitStack() as stack:
        for cls in set(v for v in vars(torch.optim).values() if isinstance(v,type) and issubclass(v,torch.optim.Optimizer) and v is not torch.optim.Optimizer):
            old=cls.step
            def counted(optimizer,*a,_old=old,**kw):
                if id(optimizer) not in attempts:raise ValueError('Unexpected optimizer owner reached source commit')
                attempts[id(optimizer)]+=1;counts['intercepted_optimizer_attempts']+=1;return _old(optimizer,*a,**kw)
            stack.enter_context(mock.patch.object(cls,'step',counted))
        old_grad=torch.autograd.grad;old_backward=torch.Tensor.backward
        def grad(*a,**kw):counts['autograd_grad_VJP_calls']+=1;return old_grad(*a,**kw)
        def backward(value,*a,**kw):counts['Tensor_backward_calls']+=1;return old_backward(value,*a,**kw)
        stack.enter_context(mock.patch.object(torch.autograd,'grad',grad));stack.enter_context(mock.patch.object(torch.Tensor,'backward',backward))
        for body in bodies:stack.callback(body.register_forward_hook(lambda *a:counts.__setitem__('native_forwards',counts['native_forwards']+1)).remove)
        with TensorCommitDenial(torch,list(model.parameters()),moments) as tensor_guard:
            for cycle in range(1+warm_iterations):
                counts=dict(native_forwards=0,autograd_grad_VJP_calls=0,Tensor_backward_calls=0,intercepted_optimizer_attempts=0)
                synchronized();tick=time.perf_counter()
                for a in adapters:
                    ignored=a.train_step(cycle+1);del ignored
                synchronized();times.append(time.perf_counter()-tick);observations.append(dict(counts))
                gradient_bytes=max(gradient_bytes,sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None))
                if p0!=h.tensor_digest(torch,model.named_parameters()) or m0!=h.tensor_digest(torch,h.optimizer_tensors(roster)):raise ValueError('Source parameter/moment equality failed')
                for a,state,costs in zip(adapters,states,source_costs):
                    a.restore_state(copy.deepcopy(state))
                    if costs is not None:a.source_session.costs.records=copy.deepcopy(costs)
                for m,mode in modes:m.training=mode
                if state0!=tree_digest(torch,[a.capture_state() for a in adapters]):raise ValueError('Owned source streams/counters/global RNG were not restored')
        if tensor_guard.denied or denial.denied_functional_calls:raise ValueError('A source intrinsic parameter/moment or functional optimizer action was attempted')
    p1=h.tensor_digest(torch,model.named_parameters());m1=h.tensor_digest(torch,h.optimizer_tensors(roster));b1=h.tensor_digest(torch,model.named_buffers());fixed1=h.tensor_digest(torch,fixed)
    if (p0,m0,b0,fixed0)!=(p1,m1,b1,fixed1) or versions!=[int(v._version) for _,v in fixed]:raise ValueError('Source parameter, buffer, moment or fixed input state changed')
    expected={k:sum(operator_law(a.provenance['config'])[k] for a in adapters) for k in observations[0]}
    if any(row!=expected for row in observations) or any(v!=1+warm_iterations for v in attempts.values()):raise ValueError('Actual source VJP/backward/forward/all-owner attempts differ')
    tick=time.perf_counter();payload={'engineering_random_untrained_unselected':True,'scientific_parent_receipt':None,
        'parameters':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'shadow_moments':[(k,v.detach().cpu().clone()) for k,v in h.optimizer_tensors(roster)],
        'source_owned_state':states,'unregistered_bias_and_runtime':[a.capture_state()['external'] for a in adapters]}
    clone_seconds=time.perf_counter()-tick;path=output/'DISPOSABLE_RANDOM_ENGINEERING_STATE.pt';tick=time.perf_counter();torch.save(payload,path);saved=time.perf_counter()-tick;serialized=h.receipt(output,path);path.unlink();del payload
    owners=[];names={id(p):name for name,p in model.named_parameters()}
    for index,o in enumerate(all_optimizers):
        parameters=[p for g in o.param_groups for p in g['params']];owner_moments=[(str(j)+'/'+k,v) for j,state in enumerate(o.state.values()) for k,v in sorted(state.items()) if torch.is_tensor(v)]
        owners.append({'index':index,'optimizer_type':type(o).__name__,'parameter_objects':len(parameters),'coordinates':sum(p.numel() for p in parameters),
            'parameter_roster':[[names[id(p)] for p in g['params']] for g in o.param_groups],'group_hyperparameters':[{k:v for k,v in g.items() if k!='params'} for g in o.param_groups],
            'parameter_sha256':h.tensor_digest(torch,[(names[id(p)],p) for p in parameters]),'manual_moment_sha256':h.tensor_digest(torch,owner_moments),'denied_step_calls':attempts[id(o)]})
    return {'schema':'private-source-zero-update-resource-result-v1','scope':'RANDOM_SOURCE_FULL_CALLBACK_MEMORY_GRADIENT_TIME_PROXY_ONLY','optimizer_objects':len(all_optimizers),'owners':owners,
        'parameters_sha256_before':p0,'parameters_sha256_after':p1,'moments_sha256_before':m0,'moments_sha256_after':m1,'buffers_sha256_before':b0,'buffers_sha256_after':b1,
        'immutable_GRAPH_TRAIN_stub_sha256_before':fixed0,'immutable_GRAPH_TRAIN_stub_sha256_after':fixed1,'source_owned_state_sha256_before':state0,'source_owned_state_sha256_after':tree_digest(torch,[a.capture_state() for a in adapters]),
        'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'gradient_bytes':gradient_bytes,'manual_moment_bytes':sum(v.numel()*v.element_size() for v in moments),
        'cold_source_cycle_seconds':times[0],'warm_source_cycle_seconds':times[1:],'warm_median_seconds':statistics.median(times[1:]),'actual_source_operator_counts_by_cycle':observations,
        'denied_optimizer_step_calls':denial.denied_step_calls,'denied_tensor_commit_calls':tensor_guard.denied,'actual_optimizer_kernel_calls':0,'actual_optimizer_updates':0,'zero_update_equality_proof_complete':True,
        'disposable_serialization':{**serialized,'clone_seconds':clone_seconds,'save_seconds':saved,'removed':True,'never_selected_checkpoint':True},
        'scores_emitted':0,'accuracy_calls':0,'scientific_fits':0,'historical_CORE_stage2_activated':False,'optimizer_kernel_timing':None,'SELECT_CAL_evaluate_resource_qualified':False,'all_fit_ETA':None}


def engineering_adapters(torch,h,root,case,study,output,guard,device):
    clone,run,stub=h.engineering_clone(torch,root,study,case,output);guard.allowed.add(stub.resolve())
    if 'identity' in run['config']:run['config']['identity']['VALID_custody_sha256']=h.receipt(root,stub)['sha256']
    clone['runs']=[run]
    class NoScientificRegistry:
        def __getattr__(self,name):raise h.ForbiddenRoleAccess('No registry, paid parent or scientific claim API in resource proxy')
    context={'registry':NoScientificRegistry(),'output_root':output,'paid_parent_checkpoints':{}}
    sys.path.insert(0,str(root/case['runtime_package']))
    from run_study import load_factory
    factory=load_factory(root,run['adapter']);adapters=[]
    for body in range(case['prototype_bodies']):
        current=copy.deepcopy(run);current['body']=body
        adapters.append(factory(torch=torch,project_root=root,study=clone,run=current,device=device,execution_context=context))
    return adapters,stub


def execute_case(root,manifest,case,output):
    h=module(root,manifest['cap_helper'],'reviewed_private_denial_cap');physics=module(root,manifest['Physics_custody_helper'],'reviewed_private_Physics_custody')
    physics0=physics.physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher'])
    molbace=closed_molbace_custody(root,manifest['MolBACE_closed_queue']['binding'])
    path=bound(root,case['source_spec_or_source_recipe_binding']);study=json.loads(gzip.decompress(path.read_bytes()))
    if digest(study)!=case['source_spec_hash']:raise ValueError('Exact source study changed')
    bound(root,case['source_adapter_binding'])
    for row in case['source_operator_pins'].values():bound(root,row)
    import torch
    torch.set_num_threads(2);cap,usage0,usage1=h.configure_cuda_allocator(torch,output);torch.cuda.reset_peak_memory_stats(0)
    allowed={root/v for v in role_paths(case['GRAPH_TRAIN_bindings'])};allowed.update(root/v for v in role_paths(case['auxiliary_GRAPH_binding']))
    guard=h.RoleIO(torch,root,allowed,{root/v for v in case['prohibited_original_role_paths']},output)
    with h.OptimizerDenial(torch) as denial,guard:
        tick=time.perf_counter();adapters,stub=engineering_adapters(torch,h,root,case,study,output,guard,'cuda:0');torch.cuda.synchronize();constructor=time.perf_counter()-tick
        if denial.denied_step_calls or denial.denied_functional_calls:raise ValueError('Factory reached premature optimizer action')
        result=measure_source(torch,h,adapters,denial,output,manifest['warm_iterations'])
        result.update(constructor_seconds=constructor,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,heldout_original_file_accesses=guard.denied,tensor_load_paths=guard.tensor_load_paths,
            synthetic_unused_SELECT_interface=h.receipt(root,stub),original_spec_unchanged=True)
    result.update(case_id=case['case_id'],allocator_cap=cap,GPU_usage_before_init=usage0,GPU_usage_after_init=usage1,GPU_usage_at_completion=h.GPU_process_usage(),
        Physics_custody_before=physics0,Physics_custody_after=physics.physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher']),
        MolBACE_queue_before=molbace,MolBACE_queue_after=closed_molbace_custody(root,manifest['MolBACE_closed_queue']['binding']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','output','physics-helper'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('prepare-manifest','execute-case'),default='prepare-manifest');parser.add_argument('--manifest');parser.add_argument('--case-id');parser.add_argument('--resource-review')
    parser.add_argument('--reviewed-helper-sha256');parser.add_argument('--reviewed-manifest-sha256');parser.add_argument('--case-timeout-seconds',type=int,default=600)
    args=parser.parse_args()
    from prepare_node_data import verify_allocation
    verify_allocation();root=Path(args.project_root).resolve();output=local(root,args.output);output.mkdir(parents=True,exist_ok=False)
    helper=receipt(root,Path(__file__).resolve());cap=receipt(root,local(root,CAP_HELPER));physics_helper=receipt(root,local(root,args.physics_helper))
    if cap['sha256']!=CAP_SHA or physics_helper['sha256']!=PHYSICS_HELPER_SHA:raise ValueError('Reviewed cap/denial/Physics helper changed')
    physics=module(root,physics_helper,'reviewed_private_Physics_custody');start=receipt(root,local(root,physics.PHYSICS_DIR+'/OWNER_START.json'));watcher=receipt(root,root/'private_GRU_acquisition_root_20261011_v1/monitor.py')
    if args.mode=='prepare-manifest':
        custody=physics.physics_custody(root,start,watcher);value=prepare_manifest(root,helper,cap,physics_helper,start,watcher);value['Physics_ownership_at_preparation']=custody
        value['MolBACE_closed_queue']=closed_molbace_custody(root)
        (output/'MANIFEST.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(canonical({'cases':len(value['cases']),'CUDA_executions':0,'numerical_imports':0,'science':0,'GO':False}));return
    h=module(root,cap,'reviewed_private_denial_cap');owner=h.own_identity();tick=time.monotonic()
    (output/'OWNER_START.json').write_text(json.dumps({'owner':owner,'case_id':args.case_id,'helper':helper},indent=2)+'\n')
    try:
        if not all((args.manifest,args.case_id,args.resource_review)) or args.reviewed_helper_sha256!=helper['sha256'] or not 1<=args.case_timeout_seconds<=600:raise ValueError('Separate exact source/manifest and one-case resource approval required')
        manifest_path=local(root,args.manifest);manifest_binding=receipt(root,manifest_path)
        if manifest_binding['sha256']!=args.reviewed_manifest_sha256:raise ValueError('Manifest source review hash differs')
        review=json.loads(local(root,args.resource_review).read_text())
        if review!={'schema':'private-source-one-case-resource-review-v1','approved':True,'CUDA_engineering_execution_authorized':True,'helper':helper,'manifest':manifest_binding,'case_id':args.case_id,'scientific_fits_authorized':False,'MolBACE_queue_overlap_allowed':False}:
            raise ValueError('Exact separately reviewed engineering-only one-case authorization required')
        manifest=json.loads(manifest_path.read_text())
        if manifest['helper']!=helper or manifest['cap_helper']!=cap or manifest['Physics_custody_helper']!=physics_helper:raise ValueError('Reviewed helper dependencies changed')
        candidates=[v for v in manifest['cases'] if v['case_id']==args.case_id]
        if len(candidates)!=1 or digest({**candidates[0],'case_id':''})[:24]!=args.case_id:raise ValueError('One exact finite content-hashed case required')
        def expired(*a):raise h.ResourceTimeout('Owned source resource case deadline/termination')
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(args.case_timeout_seconds)
        result=execute_case(root,manifest,candidates[0],output);result.update(status='RANDOM_SOURCE_ZERO_UPDATE_PROXY_COMPLETE',owner=owner,elapsed_wall_seconds=time.monotonic()-tick,no_retry=True)
        (output/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    except BaseException as error:
        result=h.terminal_failure(error,owner,tick);(output/'TERMINAL_FAILURE.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');raise SystemExit(2)
    finally:signal.alarm(0)


if __name__=='__main__':main()
