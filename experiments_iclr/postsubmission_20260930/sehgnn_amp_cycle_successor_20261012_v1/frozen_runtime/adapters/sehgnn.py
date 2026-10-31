"""Complete native typed source fits, explicit Bernoulli selection and state."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

try:
    from ..source_utils import bound,load,native_runtime
    from .classifier import ClassifierAdapter
    from .sehgnn_source_modules import modules,bind_constructor
    from .sehgnn_native import BasicSession
    from ..spec import fingerprint
except ImportError:
    from source_utils import bound,load,native_runtime
    from adapters.classifier import ClassifierAdapter
    from adapters.sehgnn_source_modules import modules,bind_constructor
    from adapters.sehgnn_native import BasicSession
    from spec import fingerprint

SUPPLY=('shared_own_only','native_pool_credit','source_view_supervision','uncoupled_source_contrast','COMMON_cycle','assigned_source_supply')
JOINT=('BE_P','BE_PS','ADD_P','ADD_PS','ADD_P_copied_dictionary','ADD_PS_neutral_all_source_average',
       'independent_ADD_P','independent_ADD_PS','native_single_P','rank4_single_P','rank1_single_P')
BASIC=('native_single','factorized_native_single','original_gnnm')


class SourceCosts:
    """Numerical operation timings go into the engine trace; no outside IO."""
    def __init__(self,rt):self.rt=rt;self.rows=[]
    def measure(self,scope,gpu=False):
        from contextlib import contextmanager
        import time
        @contextmanager
        def manager():
            start=time.monotonic()
            if gpu and self.rt['device'].type=='cuda':self.rt['torch'].cuda.synchronize(self.rt['device'])
            row={'scope':scope}
            try:yield row;row['status']='complete'
            except BaseException as e:row.update(status='failed',error=str(e));raise
            finally:
                if gpu and self.rt['device'].type=='cuda':self.rt['torch'].cuda.synchronize(self.rt['device'])
                row['seconds']=time.monotonic()-start;self.rows.append(row)
        return manager()


class ProspectiveConfig:
    """Current source contract; no historical scientific admission is inherited."""
    def __init__(self,run,sources,body_seed,condition,context_binding,role_binding,engineering):
        self.condition=condition;self.base_seed=body_seed;self.assignments=('actor','director','keyword',None);self.source_corrections=True
        self.member_rng_seeds=tuple(1000*body_seed+m+1 for m in range(4))
        self.adapter_seeds=tuple(body_seed+200001+1009*m for m in range(4));self.body_seeds=(body_seed,)
        if condition.startswith('independent_'):self.body_seeds=tuple(body_seed+1009*m for m in range(4))
        if condition in ('native_single_P','rank4_single_P','rank1_single_P'):self.member_rng_seeds=self.member_rng_seeds[:1]
        self.study_binding=fingerprint({'run_id':run['run_id'],'config':run['config'],'sources':sources,'independent_source_contract':True})
        self.role_binding=role_binding;self.full_view_binding=context_binding
        # Original source validators use this field as an identity. It denotes
        # the new source contract here and is explicitly not resource evidence.
        self.native_qualification_binding=self.study_binding
        self.source_seal_sha256='source-byte-identity-set-by-caller'
        self.engineering=engineering;self.run=run;self.sources=sources
    def require_enabled(self):
        if self.run['config'].get('sehgnn_epochs')!=1000 or self.condition not in (*SUPPLY,*JOINT):raise ValueError('Exact independent1000complete-epoch native source contract required')
        if self.run['config'].get('historical_state_inherited') is not False:raise ValueError('Historical fitted state/authorization cannot supply a new source fit')


class AdamFacade:
    def __init__(self,optimizers):self.optimizers=tuple(optimizers)
    @property
    def param_groups(self):return [g for o in self.optimizers for g in o.param_groups]
    def state_dict(self):return {'source_owner_states':[o.state_dict() for o in self.optimizers]}
    def load_state_dict(self,state):
        if len(state['source_owner_states'])!=len(self.optimizers):raise ValueError('Source owner roster changed')
        for o,s in zip(self.optimizers,state['source_owner_states']):o.load_state_dict(s)
    def zero_grad(self,**kwargs):
        for o in self.optimizers:o.zero_grad(**kwargs)
    def step(self,*args,**kwargs):
        for o in self.optimizers:o.step(*args,**kwargs)


class BernoulliAdapter(ClassifierAdapter):
    def probabilities(self,logits):
        if logits.ndim!=3 or logits.shape[-1]!=5 or not self.t.isfinite(logits).all().item():raise ValueError('Finite complete native five-label logits required')
        return logits.sigmoid().mean(0) if self.fusion=='mean_probabilities' else logits.mean(0).sigmoid()
    def validate(self,step):
        self.model.eval()
        with self.t.no_grad():
            z=self.row_logits_fn(self.select_ids);truth=self.select_y
            if self.fusion=='mean_logits':loss=self.F.binary_cross_entropy_with_logits(z.mean(0),truth)
            else:
                observed=self.t.where(truth.bool()[None],self.F.logsigmoid(z),self.F.logsigmoid(-z));loss=-(self.t.logsumexp(observed,dim=0)-math.log(len(z))).mean()
            value=float(loss)
            if not math.isfinite(value):raise ValueError('Finite complete SELECT marginal BCE required')
            return {'objective':value,'ce':value,'regularization':0.,'pooled_BCE':value,'selection_metric':value,
                    'examples':int(len(truth)),'label_events':int(truth.numel()),'fusion':self.fusion,
                    'operation_counts':{'optimizer_updates':0,'model_forwards':len(z),'backward_calls':0,'route_predictions':len(z),'supervised_examples':0}}


def context(torch,rt,root,c,seed,synthetic=None):
    if synthetic is None:
        inputs=c['typed_inputs'];train=torch.load(bound(root,inputs['train']),map_location='cpu',weights_only=True);select=torch.load(bound(root,inputs['select']),map_location='cpu',weights_only=True)
        feature={k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in inputs['features'].items()}
        label={k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in inputs['labels'].items()}
    else:train,select,feature,label=(synthetic[k] for k in ('train','select','features','labels'))
    ids,valid=train['ids'],select['ids'];N=feature['full']['M'].shape[0]
    if torch.isin(ids,valid).any() or train['y'].shape!=(len(ids),5) or select['y'].shape!=(len(valid),5):raise ValueError('Disjoint complete five-label TRAIN/SELECT roles required')
    if any(not torch.isfinite(v['y']).all() or not ((v['y']==0)|(v['y']==1)).all() for v in (train,select)):raise ValueError('Only hard known Bernoulli labels may enter source context')
    targets=torch.full((N,5),float('nan'));targets[ids]=train['y'];targets[valid]=select['y'];placed=targets.to(rt['device'])
    data=SimpleNamespace(train_ids=tuple(ids.tolist()),valid_ids=tuple(valid.tolist()),counts=(N,2393,6124,7971),class_dim=5,
        train_labels=tuple(map(tuple,train['y'].tolist())),valid_labels=tuple(map(tuple,select['y'].tolist())))
    train_loader=torch.utils.data.DataLoader(ids,batch_size=10000,shuffle=True,drop_last=False)
    known=torch.cat((ids,valid));full=SimpleNamespace(data=data,seed=seed,feats=feature['full'],label_feats=label['full'],
        data_size={k:v.shape[-1] for k,v in feature['full'].items()},targets=targets,targets_cuda=placed,train_index=ids,valid_index=valid,
        train_loader=train_loader,train_count=len(ids),valid_count=len(valid),eval_batches=[(known,{k:v[known] for k,v in feature['full'].items()},{k:v[known] for k,v in label['full'].items()},None)])
    for name in ('full','actor','director','keyword'):
        if sorted(feature[name])!=rt['protocol']['feature_paths'] or sorted(label[name])!=rt['protocol']['label_paths']:raise ValueError('All original37 feature/label namespaces required')
        if any(v.shape!=feature['full'][k].shape for k,v in feature[name].items()) or any(v.shape!=(N,5) for v in label[name].values()):raise ValueError('Complete source-view geometry required')
        if not torch.equal(feature[name]['M'],feature['full']['M']):raise ValueError('Source views must retain movie-own features')
    return full,feature,label,train,select


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    c=run['config'];root=Path(project_root).resolve();arm=c['sehgnn_arm'];sources=c['sehgnn_sources'];engineering=bool(execution_context and execution_context.get('engineering_synthetic'))
    if c.get('sehgnn_epochs')!=1000 or arm not in (*SUPPLY,*JOINT,*BASIC) or c.get('historical_state_inherited') is not False:raise ValueError('Exact prospective1000complete-epoch native SeHGNN source arm required')
    if engineering:
        if study['stage']!='engineering' or str(device)!='cpu':raise ValueError('Only declared synthetic CPU source qualification')
    else:
        expected='complete' if execution_context and execution_context.get('endpoint_reconstruction') else 'running'
        if not execution_context or execution_context['registry'].run(study['study_id'],run['run_id'])['state']!=expected:
            raise ValueError('Registered current fit/selected-endpoint context required; no direct fit or historical admission')
    if c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.3,.5,.7):raise ValueError('Frozen finite native source domain required')
    rt,engine=native_runtime(root,sources,device);costs=SourceCosts(rt);constructor=bind_constructor(rt,engine,c['dropout']);mods=modules(root,sources,c['learning_rate'],engineering)
    seed=run['seed']+1009*run['body'];rt['native'].set_random_seed(seed)
    ctx,features,labels,train,select=context(torch,rt,root,c,seed,execution_context.get('fixture') if engineering else None)
    context_binding=fingerprint({'inputs':c.get('typed_inputs',{}),'engineering':engineering,'role':run['split']});role_binding=fingerprint({'role_inputs':c.get('typed_inputs',{}),'mask':run['split']})
    cfg=ProspectiveConfig(run,sources,seed,arm,context_binding,role_binding,engineering)
    old_bank=json.loads(bound(root,sources['metadata']['bank_sources']).read_text());old_joint=json.loads(bound(root,sources['metadata']['joint_sources']).read_text())
    views={}
    for family in ('actor','director','keyword'):
        metadata={'construction_complete':True,'full_view_binding':context_binding,'role_binding':role_binding,'native_qualification_binding':cfg.study_binding,
            'source_seal_sha256':old_bank['view_seal_sha256'],'synthetic_engineering_fixture':engineering,'resource_qualification_claimed':False}
        views[family]=mods['views'].NativeFamilyView(family,fingerprint({'current_inputs':context_binding,'source':family}),metadata,features[family],labels[family],ctx.data_size,
            ctx.data,seed,ctx.targets,ctx.targets_cuda,ctx.train_index,ctx.valid_index,ctx.train_count,ctx.valid_count)
    if arm in BASIC:
        session=BasicSession(rt,engine,ctx,costs,mods,sources,root,arm,seed,c['learning_rate']);model=session.model;optimizers=(session.optimizer,);policy=None
    elif arm in SUPPLY:
        cfg.source_seal_sha256=old_bank['bank_seal_sha256'] if 'bank_seal_sha256' in old_bank else old_bank.get('source_seal_sha256','SOURCE_BYTES_BOUND_BY_NEW_PACKET')
        rt['native'].set_random_seed(seed);prototype=engine.make_model(rt,ctx,costs);scalar=torch.cuda.amp.GradScaler(enabled=rt['device'].type=='cuda')
        session=mods['bank'].BankSession(rt,engine,mods['factor'],mods['helper'],ctx,views,prototype,scalar,costs,cfg)
        policy=mods['policy'].ScheduledSession(session,arm,{'current_study_source_contract':cfg.study_binding});model=session.bank;optimizers=(session.optimizer,)
    else:
        # A rank-one single is the identical original independent-P learner,
        # exposed separately so a genuine separately selected I4 can be built.
        if arm=='rank1_single_P':mods['joint'].SPECS={**mods['joint'].SPECS,'rank1_single_P':('single','add',1,'none',False)}
        for v in views.values():v.metadata['source_seal_sha256']=old_joint['source_view_seal_sha256']
        injected={'engine':engine,'adapter':mods['factor'],'metadata':mods['bank'],'helper':mods['helper'],
                  'role_loader':load(root,sources['modules']['role_loader'],'data_loader'),'source_views':mods['views'],'diagnostics':load(root,sources['modules']['diagnostics'],'diagnostics')}
        session=mods['joint'].Session(rt,injected,ctx,views,costs,cfg);policy=None;model=session.group;optimizers=tuple(session.optimizers)
    models=tuple(session.models if hasattr(session,'models') else session.bank.members)
    optimizer=optimizers[0] if len(optimizers)==1 else AdamFacade(optimizers)
    fusion='mean_logits' if arm=='original_gnnm' else 'mean_probabilities'
    def row_logits_fn(rows):
        caller=engine.capture_rng(rt['numpy'],torch);rows=rows.detach().cpu();outputs=[]
        try:
            for m,net in enumerate(models):
                with mods['bank'].scratch_native_state(rt,engine,net,'eval',session.member_rng[m]):
                    feats={k:v[rows].to(rt['device']) for k,v in ctx.feats.items()};labs={k:v[rows].to(rt['device']) for k,v in ctx.label_feats.items()}
                    outputs.append(net(rows.to(rt['device']),feats,labs,None).float())
            return torch.stack(outputs)
        finally:engine.restore_rng(rt['numpy'],torch,caller)
    def logits_fn(unused,training,step):
        if training:raise ValueError('Source owns its complete native training cycle')
        return row_logits_fn(torch.arange(ctx.feats['M'].shape[0]))
    def counter_state():
        return {'counters':copy.deepcopy(session.counters),'member_rng':engine.cpu_tree(torch,session.member_rng),
            'master_rng':engine.capture_rng(rt['numpy'],torch),'scalar':engine.cpu_tree(torch,session.scalar.state_dict()) if hasattr(session,'scalar') else None,
            'body_counters':copy.deepcopy(getattr(session,'body_counters',None)),'active':copy.deepcopy(getattr(session,'active',None)),
            'policy':None if policy is None else {k:copy.deepcopy(getattr(policy,k)) for k in ('completed_own_epochs','opportunities','scheduled_slots','accepted','rejected','zero','exposures','actual_dose','logs')}}
    def restore(state):
        session.counters=copy.deepcopy(state['counters']);session.member_rng=engine.cpu_tree(torch,state['member_rng']);engine.restore_rng(rt['numpy'],torch,state['master_rng'])
        if state['scalar'] is not None:session.scalar.load_state_dict(state['scalar'])
        if state['body_counters'] is not None:session.body_counters=copy.deepcopy(state['body_counters'])
        if state['active'] is not None:session.active=copy.deepcopy(state['active'])
        if policy is not None:
            for k,v in state['policy'].items():setattr(policy,k,copy.deepcopy(v))
        if hasattr(session,'verify_optimizer'):session.verify_optimizer()
        if hasattr(session,'verify_optimizers'):session.verify_optimizers()
    def train_cycle(adapter,step):
        before=copy.deepcopy(session.counters);attempts=[0]
        # Observe requests separately from actual source optimizer.step calls.
        # The original bank/native GradScaler law and source FP32 joint algebra
        # remain unchanged; every temporary wrapper delegates and is restored.
        if hasattr(session,'scalar'):
            old_scalar_step=session.scalar.step
            def request(*args,**kwargs):attempts[0]+=1;return old_scalar_step(*args,**kwargs)
            session.scalar.step=request;restores=((session.scalar,'step',old_scalar_step),)
        else:
            restores=[]
            for opt in optimizers:
                original=opt.step
                def request(*args,_original=original,**kwargs):attempts[0]+=1;return _original(*args,**kwargs)
                opt.step=request;restores.append((opt,'step',original))
        try:own=policy.train_epoch() if policy else session.train_epoch()
        finally:
            for obj,name,original in restores:setattr(obj,name,original)
        correction=None
        extra={'source_credit_leaf_reverse_calls':0,'source_private_model_VJP_calls':0,'source_private_trial_attempts':0}
        if policy:
            h=mods['helper'];old_make,old_accumulate,old_trial=h.make_credit_plan,h.accumulate_replay,h.try_private_step
            def measured_make(*args,**kwargs):
                plan=old_make(*args,**kwargs);extra['source_credit_leaf_reverse_calls']+=len(plan.cotangents);return plan
            def measured_accumulate(plan,key,scores):
                count=sum(key in credit for credit in plan.cotangents.values()) if plan.assignments[key.member] is not None else 0
                result=old_accumulate(plan,key,scores);extra['source_private_model_VJP_calls']+=count;return result
            def measured_trial(*args,**kwargs):
                extra['source_private_trial_attempts']+=1;return old_trial(*args,**kwargs)
            h.make_credit_plan,h.accumulate_replay,h.try_private_step=measured_make,measured_accumulate,measured_trial
            try:correction=policy.correct(own['own_displacements'])
            finally:h.make_credit_plan,h.accumulate_replay,h.try_private_step=old_make,old_accumulate,old_trial
        after=session.counters;delta={k:int(after[k]-before.get(k,0)) for k in after if type(after[k]) is int}
        commits=delta['actual_Adam_steps'];skips=attempts[0]-commits
        if attempts[0]!=len(optimizers) or skips<0 or not hasattr(session,'scalar') and skips:raise ValueError('Complete source owner attempt/commit accounting differs')
        amp_enabled=int(hasattr(session,'scalar') and session.scalar.is_enabled())
        if skips and not amp_enabled:raise ValueError('Only enabled native GradScaler may skip an own optimizer map')
        fwd=sum(delta.get(k,0) for k in ('own_member_forwards','source_reference_forwards','source_replay_forwards','source_trial_forwards','reference_forwards','gradient_replays'))
        backward=delta.get('own_backwards',0)+delta.get('P_backwards',0)+delta.get('S_private_VJPs',0)+extra['source_private_model_VJP_calls']
        # The source helper's private VJPs/leaf-credit reverses are additional
        # work; they are prospectively typed and audited in controller costs.
        objective=float(own['mean_native_own_BCE'] if 'mean_native_own_BCE' in own else own['P_reference_complete_mean'])
        return {'objective':objective,'ce':objective,'regularization':0.,
            'operation_counts':{'optimizer_updates':commits,'source_optimizer_attempts':attempts[0],'source_AMP_skipped_optimizer_maps':skips,
                'source_native_complete_cycles':1,'source_native_AMP_enabled':amp_enabled,'model_forwards':fwd,'backward_calls':backward,
                'route_predictions':len(models),'supervised_examples':ctx.train_count,'accepted_private_corrections':int(correction is not None and correction.get('status')=='accepted')},
            'source_counters':{**delta,**extra},'source_correction':correction,'cost_events':copy.deepcopy(costs.rows[-8:])}
    adapter=BernoulliAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=train['ids'].to(device),train_y=train['y'].to(device),
        select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda unused:train['y'].new_zeros(()).to(device),fusion=fusion,
        selector={'metric':'pooled_BCE','direction':'min','ties':'first','patience_steps':1001,'save_every':1},custom_train=train_cycle,
        external_capture=counter_state,external_restore=restore,provenance={'run_id':run['run_id'],'config':c,'data':study['data'],'source':study['source'],
            'actual_model_seed':seed,'constructor_binding':constructor,'operator_provider_bindings':mods['source_bindings'],
            'original_control_binding':getattr(session,'source_binding',{}),'synthetic_engineering_only':engineering,
            'legacy_qualification_identity_fields_are_new_source_contract_not_resource_proof':True,'historical_state_or_authorization_reused':False})
    adapter.session=session;adapter.policy=policy;adapter.source_rt=rt;adapter.source_context=ctx;adapter.source_modules=mods;adapter.row_logits_fn=row_logits_fn;return adapter
