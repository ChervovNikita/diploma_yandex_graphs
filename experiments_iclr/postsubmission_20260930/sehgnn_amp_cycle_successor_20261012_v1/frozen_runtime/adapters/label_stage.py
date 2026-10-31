"""Paid-native capture + exact source label maps; imports are initially stdlib."""
import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import types

try:
    from .classifier import ClassifierAdapter
    from .native_polynormer_qk import bound,receipt
    from ..spec import fingerprint
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from adapters.native_polynormer_qk import bound,receipt
    from spec import fingerprint

ARMS={
 'staged_posterior_c4':('posterior','C4','core',4,4,4,1),
 'staged_posterior_s_joint4head':('posterior','S_joint4head','multihead_single4',1,4,1,1),
 'staged_posterior_u4_sharedb':('posterior','U4_sharedB','shared_backbone_untied_correctors4',4,4,4,4),
 'staged_posterior_s_one_path':('posterior','S_one_path','one_path',1,1,1,1),
 'query_value_gate_c4_gate':('gate','C4_gate','core',4,4,4,1),
 'query_value_gate_s_joint4head_gate':('gate','S_joint4head_gate','multihead_single4',1,4,1,1),
 'query_value_gate_u4_sameb_full_untied_gate':('gate','U4_sameB_full_untied_gate','shared_backbone_untied_correctors4',4,4,4,4),
 'query_value_gate_s_one_path_gate':('gate','S_one_path_gate','one_path',1,1,1,1),
 'query_value_gate_c4_gate_identity_erased':('gate','C4_gate_identity_erased','core',4,4,4,1)}
SOURCE_PINS={
 'p0_core':('label_only_private_corrector_core_source_20261008_v1/core.py','41e0165829d091e73ec08a7520f5a830cb4fb37006a27d26684cb2ad2b8d675d'),
 'p0_controls':('label_only_same_operation_controls_source_20261008_v1/controls.py','82fb6545878e6b2170cf2896c1c108d185c9d5c4b3ebfef965e6b23aabf8089a'),
 'posterior':('label_only_staged_posterior_full_WikiCS_source_20261008_v1/posterior.py','3d4d36cbd648dcda59e71d6fe17ef31e383f802130166627f3f356b934a521ab'),
 'p0_stage':('label_only_staged_posterior_full_WikiCS_source_20261008_v1/stage.py','dffcb1d0d6ae321ff3bebca0358ae2491c6adc93b087080d9effbb0b58644d02'),
 'gate_core':('query_conditioned_value_gate_full15_callable_source_20261009_v1/core.py','760d8a676c45239e21ee3b34ad23ace1fac1f38b8901051271306d982a8a55c8'),
 'gate_controls':('query_conditioned_value_gate_full15_callable_source_20261009_v1/controls.py','7d2408176e9dc44899b3a67c7b0eb87c6b82789b41caf7bdbe7cad5dfe6c3727'),
 'gate_hook':('query_conditioned_value_gate_full15_callable_source_20261009_v1/query_value_gate_hook.py','46e70016760aefbca612f261d87e418ce55abe1921cee9c3e9ed7700d95de7b8'),
 'gate_stage':('query_conditioned_value_gate_full15_callable_source_20261009_v1/stage.py','e96b91bda524b251e880d6f71b4cb92d7bcf3c09a51caf451db7ebe431d97a0a'),
 'capture':('label_only_private_corrector_native_single_integration_source_20261008_v1/capture.py','b7ec8a84f6e3ce8736c9b9c2631947c73d8411d2daa719085647a422ec7975fe')}


def source_module(root,row,tag,classes,*,posterior_only=False):
    path=bound(root,row);tree=ast.parse(path.read_bytes());changes=[]
    if posterior_only:
        wanted={'require','nonlinear_single','anchor_reach','bounded'}
        tree.body=[n for n in tree.body if isinstance(n,ast.Import) or isinstance(n,ast.FunctionDef) and n.name in wanted]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('nonlinear_single','bounded'):
                for n in ast.walk(node):
                    if isinstance(n,ast.Constant) and type(n.value) is int and n.value==10:n.value=classes;changes.append(node.name+':declared_C')
    elif tag=='gate_hook':
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CLASSES' for t in node.targets):node.value=ast.Constant(classes);changes.append('CLASSES:declared_C')
    elif tag in ('gate_core','gate_controls'):
        # Change only the old factory's class-domain guard, not operator algebra.
        for node in ast.walk(tree):
            if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='classes':
                for n in node.comparators:
                    if isinstance(n,ast.Constant) and n.value==10:n.value=classes;changes.append('factory_class_guard:declared_C')
    name='paid_native_label_'+tag+'_'+row['sha256'][:16]+'_C'+str(classes)
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(ast.fix_missing_locations(tree),str(path)+':explicit_class_provider_binding','exec'),m.__dict__)
    return m,{'source_sha256':row['sha256'],'provider_changes':changes,'bound_AST_sha256':hashlib.sha256(ast.dump(tree).encode()).hexdigest()}


def source_bank(torch,root,sources,arm,classes,edges,train_ids,y,seed,device,context,native_id,lr):
    family,alias,mode,members,heads,backs,owners=ARMS[arm]
    core_tag='gate_core' if family=='gate' else 'p0_core';controls_tag='gate_controls' if family=='gate' else 'p0_controls'
    core,core_binding=source_module(root,sources[core_tag],core_tag,classes)
    controls,control_binding=source_module(root,sources[controls_tag],controls_tag,classes)
    if family=='gate':
        gates,gate_binding=source_module(root,sources['gate_hook'],'gate_hook',classes)
        core._value_gate_helpers=lambda:gates
    else:gate_binding=None
    controls._load_core=lambda ignored:core
    posterior,posterior_binding=source_module(root,sources['posterior'],'posterior',classes,posterior_only=True)
    policy={'native_local_restore':'each_own_native_local_selector','corrector_local_restore':'same_own_native_selected_epoch',
            'end_local_streams':'live_no_rewind','final_selector':'mean_probability_family' if mode=='shared_backbone_untied_correctors4' else 'each_own_corrected_predictor',
            'role':'VALID','metric_name':'accuracy'}
    policy['policy_sha256']=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    args={'nodes':len(context['node_order']),'feature_width':512,'classes':classes,'edge_index':edges,'train_ids':train_ids,'train_labels':y,
          'mask_seed':seed+1900001,'device':str(device),'later_execution_authorized':True}
    if mode=='core':
        bank=core.make_core(**args,initializer_seed=seed,public_root=root/'portable_internal_be_public_interface_20261007_v2',
                            **({'erase_value_class_identity':alias.endswith('identity_erased')} if family=='gate' else {}))
        opts=[bank.optimizer]
    else:
        bank=controls.make_shared_controls(**args,mode=mode,initializer_seeds=tuple(seed+1009*h for h in range(heads)),
                                          context_identity=context['hashes'],backbone_ids=(native_id,),selection_policy=policy,core_root=root)
        if mode=='multihead_single4':bank=posterior.nonlinear_single(bank,torch=torch,core_module=core,controls_module=controls,initializer_seed=seed,device=device)
        opts=bank.optimizers
        class OptimizerFacade(controls.OptimizerBank):
            @property
            def param_groups(self):return [g for o in self.optimizers for g in o.param_groups]
        bank.optimizer=OptimizerFacade(opts,mode)
        policy.update(native_local_restore='completed_native_acquisition_only',corrector_local_restore='none_in_staged_label_training',
                      final_selector='coherent_complete_bank_bounded_mixture_strict_first_SELECT',role='SELECT')
        policy.pop('policy_sha256');policy['policy_sha256']=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
        bank.selection_policy=policy
    for o in opts:
        for group in o.param_groups:group['lr']=lr
    if len(opts)!=owners or mode=='shared_backbone_untied_correctors4' and bank.backward_loss_scale!=.25:raise ValueError('Exact source ownership/CE scaling differs')
    return bank,posterior,{'core':core_binding,'controls':control_binding,'posterior':posterior_binding,'gate':gate_binding},core


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    c=run['config'];root=Path(project_root).resolve();arm=c['paid_label_arm'];device=torch.device(device)
    if arm not in ARMS or c['label_cycles']!=5000 or c['label_learning_rate'] not in (.0005,.001,.002):raise ValueError('Declared exact paid label stage required')
    if not execution_context:raise ValueError('Paid native/capture context required; no free historical teacher')
    synthetic=execution_context.get('engineering_synthetic') is True
    if synthetic:
        if study['stage']!='engineering':raise ValueError('Synthetic ancestry is prohibited in scientific studies')
        cache=execution_context['synthetic_cache'];cache_receipt={'scope':'synthetic_CPU_no_scientific_ancestor'}
    else:
        parent=c['lifecycle']['parent_fit_ids'][0]
        if parent not in execution_context['paid_parent_checkpoints']:raise ValueError('Actual paid own-selected native parent required')
        row=execution_context['registry'].run(study['study_id'],c['capture_parent'])
        if row['state']!='complete' or row['contract']['kind']!='selection' or row['contract']['config'].get('native_capture') is not True:
            raise ValueError('Closed source-bound native capture required')
        cache_receipt=row['result']['cache'];cache_path=bound(root,cache_receipt);cache=torch.load(cache_path,map_location='cpu',weights_only=True)
        authority=execution_context['paid_parent_checkpoints'][parent]['receipt']
        if cache['native_fit_id']!=parent or cache['selected_checkpoint']!=authority or cache['schema']!='paid-native-active-head-capture-v1':
            raise ValueError('Cache differs from paid native selection/custody')
    sources=c['label_sources']
    if set(sources)!=set(SOURCE_PINS) or any(sources[k]['sha256']!=pin[1] for k,pin in SOURCE_PINS.items()):raise ValueError('Complete original source pins required')
    for row in sources.values():bound(root,row)
    payloads={k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in c['node_inputs'].items()}
    graph,train,select=(payloads[k] for k in ('graph','train','select'))
    H=cache['H'].to(device);base=cache['native_logits'].to(device);zero=torch.zeros_like(base)
    N=len(graph['x']);C=c['classes'];ids=train['ids'].to(device);y=train['y'].to(device);edges=graph['edge_index']
    if torch.isin(ids,select['ids'].to(device)).any():raise ValueError('TRAIN and SELECT overlap')
    if H.shape!=(N,512) or base.shape!=(N,C) or H.requires_grad or base.requires_grad or not torch.isfinite(H).all() or not torch.isfinite(base).all():
        raise ValueError('Full detached own-native capture on current provider required')
    def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    hashes={'node_order_sha256':digest(torch.arange(N)),'observed_edges_sha256':digest(edges),'TRAIN_ids_sha256':digest(ids),
            'TRAIN_labels_sha256':digest(y),'context_policy_sha256':fingerprint({'TRAIN_only':True,'Q':'uniform half; entire Q excluded','scale':'(A-1)/(A-Q) once','incoming':'nonself records unchanged'})}
    seed=run['seed']+1009*run['body'];native_id='paid-native-'+str(cache.get('native_fit_id','CPU_SYNTHETIC'))
    bank,posterior,binding,core=source_bank(torch,root,sources,arm,C,edges,ids,y,seed,device,{'node_order':range(N),'hashes':hashes},native_id,c['label_learning_rate'])
    mode=ARMS[arm][2];members=ARMS[arm][3];reach=posterior.anchor_reach(torch,bank)
    nontrain=torch.arange(N,device=device)[~torch.isin(torch.arange(N,device=device),ids)]
    native_meta={'backbone_id':native_id,'source_parameter_epoch':cache['native_selected_step'],
                 'native_logical_steps_before_update':cache['native_logical_steps'],'global_mode':cache['native_global']}
    fixed={name:digest(v) for name,v in bank.named_buffers()};fixed_capture=(digest(H),digest(base))
    def logits_fn(model,training,step):
        pred=(model.serve(H,zero,nontrain) if mode=='core' else model.serve(H,zero,nontrain,native_capture=native_meta))
        bounded=posterior.bounded(torch,pred['member_logits'],base,reach,nontrain)
        rows=base.log_softmax(-1)[None].expand(members,-1,-1).clone();rows[:,nontrain]=bounded['member_log_probabilities']
        return rows
    def custom_train(adapter,step):
        if step!=bank.steps+1:raise ValueError('Sequential complete label cycles required')
        mask=bank.draw_common_query_mask()
        result=(bank.train_corrector_step(H,zero,mask) if mode=='core' else bank.train_corrector_step(H,zero,mask,native_capture=native_meta))
        return {'objective':result['own_corrected_CE_mean'],'ce':result['own_corrected_CE_mean'],'regularization':0.,
                'operation_counts':{'optimizer_updates':ARMS[arm][6],'model_forwards':ARMS[arm][4],'route_predictions':members,
                                    'backward_calls':ARMS[arm][5],'supervised_examples':bank.query_count,'native_forwards':0,'native_optimizer_updates':0}}
    def capture():
        pending=bank._pending_mask
        return {'mask_rng':bank.mask_generator.get_state().clone(),'steps':bank.steps,'failed':bank._failed_update,'counters':copy.deepcopy(bank.counters),
                'pending_mask':None if pending is None else {k:v for k,v in dataclasses.asdict(pending).items() if k!='owner_id'},
                'last_native_capture':copy.deepcopy(getattr(bank,'last_native_capture',None)),
                'erasure':getattr(bank,'erase_value_class_identity',None),'fixed_erasure':getattr(bank,'_fixed_value_erasure_policy',None),
                'source_selected':copy.deepcopy(getattr(bank,'_selected',None)),'source_selected_versions':copy.deepcopy(getattr(bank,'_selected_versions',None)),
                'factor_members':{n:m.member for n,m in bank.named_modules() if hasattr(m,'r') and hasattr(m,'s')},
                'fixed_buffers':fixed,'fixed_capture':fixed_capture}
    def restore(state):
        if {n:digest(v) for n,v in bank.named_buffers()}!=state['fixed_buffers'] or (digest(H),digest(base))!=tuple(state['fixed_capture']):
            raise ValueError('Fixed native/GRAPH/TRAIN capture changed')
        bank._record_role_versions();bank.mask_generator.set_state(state['mask_rng']);bank.steps=state['steps'];bank._failed_update=state['failed'];bank.counters=copy.deepcopy(state['counters'])
        bank._pending_mask=None if state['pending_mask'] is None else core.CommonQueryMask(owner_id=id(bank),**state['pending_mask'])
        if hasattr(bank,'last_native_capture'):bank.last_native_capture=copy.deepcopy(state['last_native_capture'])
        if state['erasure'] is not None:
            bank.erase_value_class_identity=state['erasure'];bank._fixed_value_erasure_policy=state['fixed_erasure']
        if state['source_selected'] is not None:
            bank._selected=copy.deepcopy(state['source_selected']);bank._selected_versions=copy.deepcopy(state['source_selected_versions'])
        modules=dict(bank.named_modules())
        for name,member in state['factor_members'].items():modules[name].member=member
    adapter=ClassifierAdapter(torch,torch.nn.functional,bank,bank.optimizer,logits_fn=logits_fn,train_ids=ids,train_y=y,
        select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda m:H.new_zeros(()),fusion='mean_probabilities',
        selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5001,'save_every':1},custom_train=custom_train,
        external_capture=capture,external_restore=restore,provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,
            'actual_model_seed':seed,'paid_native_capture':cache_receipt,'paid_native_fit_id':cache.get('native_fit_id','CPU_SYNTHETIC'),
            'source_bindings':binding,'synthetic_engineering_only':synthetic,'native_updates_in_label_stage':0,'native_acquisition_is_paid':True})
    adapter.bank=bank;adapter.H=H;adapter.base=base
    return adapter
