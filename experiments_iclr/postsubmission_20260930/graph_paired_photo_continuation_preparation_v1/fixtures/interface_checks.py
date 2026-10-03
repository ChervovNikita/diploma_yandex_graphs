"""Synthetic actual-interface checks; no numeric originals, Torch, GPU or remote calls."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

sys.dont_write_bytecode = True
packet = Path(__file__).resolve().parents[1]
frozen = json.loads((packet/'FROZEN_STUDY.json').read_text())
root = Path(frozen['canonical_research_root'])
source_path = packet/'prototype/continue_photo.py'
source = source_path.read_bytes()
tree = ast.parse(source)
checks = []
digest = lambda data: hashlib.sha256(data).hexdigest()


def check(name, condition):
    assert condition, name
    checks.append(dict(name=name, passed=True))


def rejects(name, callback):
    try:
        callback()
    except (ValueError, KeyError):
        check(name, True)
    else:
        check(name, False)


def local(record):
    return packet.parent/Path(record['path']).relative_to(root)


def metadata_custody(when):
    rows = [r for r in frozen['original_records'] if Path(r['path']).suffix not in ('.npy','.npz','.pt','.pth')]
    for row in rows:
        data = local(row).read_bytes()
        check(when+':metadata:'+str(Path(row['path']).relative_to(root)),
              digest(data)==row['sha256'] and len(data)==row['bytes'])
    return len(rows)


before = metadata_custody('before')
spec = importlib.util.spec_from_file_location('prepared_Photo_subject', source_path)
subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)
pinned = subject.load('prepared_Photo_pinned_driver', local(frozen['driver_source']))
integration_path = next(local(r) for r in frozen['original_records'] if r['path'].endswith('/prototype/graph_init_training_adapter.py'))
integration = subject.load('prepared_Photo_pinned_integration', integration_path)
shared = subject.load('prepared_Photo_shared', local(frozen['shared_source']))
check('actual_subject_and_pinned_interfaces_import_without_Torch', 'torch' not in sys.modules)
subject.study_guard(frozen)
check('exact_Photo_three_blocks_twelve_fits', len(frozen['cells'])==3 and frozen['fit_slots']==12)
check('seventeen_numeric_descriptors_metadata_only', sum(Path(r['path']).suffix in ('.npy','.npz','.pt','.pth') for r in frozen['original_records'])==17)
check('TRAIN_only_original_label_custody', all(r['kind']=='TRAIN_labels' for r in frozen['original_records'] if '/labels/' in r['path']))
for key,value in [('mode','fixed_two_graph'),('selector','last epoch'),('continuation',dict(global_cap=950,patience=250,midpoint=450)),('nodes',2223),('training_loss','pooled CE')]:
    rejects('changed_policy_refused:'+key, lambda key=key,value=value:subject.study_guard(dict(frozen,**{key:value})))
with patch.object(subject,'verify',return_value=Path('/synthetic/not_opened')) as mocked:
    for path,kind,admit in [('/synthetic/labels/final.npz','validation_labels',True),('/synthetic/labels/seed17_split0_validation.npz','validation_labels',False),('/synthetic/test00.npy','role_ID',False),('/synthetic/unknown.pt','unknown',False)]:
        rejects('scope_before_open:'+Path(path).name, lambda path=path,kind=kind,admit=admit:subject.verify_inputs([dict(path=path,kind=kind,sha256='0'*64)],admit))
    check('scope_rejections_opened_nothing',mocked.call_count==0)
    subject.verify_inputs([dict(path='/synthetic/labels/seed17_split0_validation.npz',kind='validation_labels',sha256='0'*64)],True)
    check('admitted_validation_reaches_verifier',mocked.call_count==1)

release = dict(graph='Photo',execution_authorized=True,run_name='synthetic',prepared_manifest_sha256='synthetic-manifest',
    mode_donor_freeze_sha256=frozen['mode_donor_freeze']['sha256'],expected_gpu_uuid='GPU-synthetic-fixed',
    Photo_resource_evidence=dict(required_free_bytes=1000,planned_wall_seconds=20,wall_budget_seconds=60),
    Squirrel_gate_assertion=dict(complete=True,all3_blocks_complete=True,all12_arm_terminals=True,
        all12_selected_fits=True,development_trigger_satisfied=True,is_quality_evidence=False))
with patch.object(subject,'verify',return_value=local(frozen['mode_donor_freeze'])):
    admitted = subject.admission_guard(frozen,release,'synthetic-manifest','synthetic')
    check('actual_admission_binds_six_donors_and_root_gate', len(admitted['six_donors'])==6 and admitted['gate_is_not_quality_evidence'])
    for key,value in [('graph','Squirrel'),('execution_authorized',False),('prepared_manifest_sha256','changed'),('mode_donor_freeze_sha256','changed'),('run_name','changed'),('expected_gpu_uuid','automatic')]:
        rejects('root_release_mismatch:'+key, lambda key=key,value=value:subject.admission_guard(frozen,dict(release,**{key:value}),'synthetic-manifest','synthetic'))
    for key in release['Squirrel_gate_assertion']:
        gate=dict(release['Squirrel_gate_assertion']);gate[key]=not gate[key]
        rejects('full_root_gate_required:'+key,lambda gate=gate:subject.admission_guard(frozen,dict(release,Squirrel_gate_assertion=gate),'synthetic-manifest','synthetic'))
    bad=dict(release['Photo_resource_evidence'],required_free_bytes=0)
    rejects('nonpositive_root_resource_refused',lambda:subject.admission_guard(frozen,dict(release,Photo_resource_evidence=bad),'synthetic-manifest','synthetic'))
    subject.admission_guard(frozen,dict(release,execution_authorized=False),'synthetic-manifest','synthetic',execute=False)
    check('root_can_admit_resource_only_preflight',True)


def blocks(values):
    return [dict(seed=seed,source_split_index=split,status='completed',fits={arm:dict(status='selected',selection=dict(primary_validation_nll=values[arm])) for arm in subject.ARMS}) for seed,split in subject.PAIRS]


complete=blocks(dict(common_only=2,train_remasked=1.8,full_node=1,full_node_permuted=1.6))
comparison=subject.followup_comparison(complete)
check('actual_all12_followup_comparison',comparison['all12_selected_fits'] and comparison['full_minus_control_NLL']['common_only']==-1)
check('followup_never_creates_another_gate_or_final_release','Photo_trigger' not in comparison and comparison['final_labels_closed'] and not comparison['independent_heldout_confirmation'])
for terminal in ('pending','fit_failed','resource_deferred'):
    partial=copy.deepcopy(complete);partial[1]['fits']['train_remasked']['status']=terminal
    got=subject.followup_comparison(partial)
    check('no_successful_subset:'+terminal,got['status']=='unevaluable' and not got['successful_subset_scored'] and 'validation_macro_NLL' not in got and got['all12_arm_terminals']==(terminal!='pending'))
rejects('missing_block_refused',lambda:subject.followup_comparison(complete[:2]))
rejects('nonfinite_selected_NLL_refused',lambda:subject.followup_comparison(blocks(dict(common_only=2,train_remasked=1.8,full_node=float('inf'),full_node_permuted=1.6))))
check('ties_are_reportable_followup_values',subject.followup_comparison(blocks(dict.fromkeys(subject.ARMS,1)))['full_minus_control_NLL']==dict.fromkeys(subject.CONTROLS,0))

# A small pure-Python tensor/Adam double exercises the actual named transport code.
# It represents synthetic values only; it is not a Torch or gradient qualification.
class Tensor:
    __hash__ = object.__hash__
    def __init__(self,values,shape=None,dtype='float32',device='cpu',requires_grad=True):
        self.values=tuple(values);self.shape=tuple(shape if shape is not None else (len(self.values),));self.dtype=dtype;self.device=device;self.requires_grad=requires_grad
    def detach(self): return self
    def cpu(self): return self.to('cpu')
    def clone(self): return Tensor(self.values,self.shape,self.dtype,self.device,self.requires_grad)
    def to(self,device=None,dtype=None): return Tensor(self.values,self.shape,dtype or self.dtype,device or self.device,self.requires_grad)
    def expand(self,k,last):
        assert len(self.shape)==1 and last==-1
        return Tensor(self.values*k,(k,self.shape[0]),self.dtype,self.device)
    def expand_as(self,other): return self.expand(other.shape[0],-1)
    def __truediv__(self,scale): return Tensor([v/scale for v in self.values],self.shape,self.dtype,self.device)
    def __ge__(self,value): return Tensor([v>=value for v in self.values],self.shape,'bool')
    def __lt__(self,value): return Tensor([v<value for v in self.values],self.shape,'bool')
    def __and__(self,other): return Tensor([a and b for a,b in zip(self.values,other.values)],self.shape,'bool')
    def all(self): return all(self.values)
    def copy(self): return self.clone()
    def __eq__(self,other): return isinstance(other,Tensor) and same(self,other) and self.dtype==other.dtype and self.device==other.device


class Adam:
    def __init__(self,groups): self.param_groups=groups;self.state={}


class Model:
    def __init__(self,rows,members=1): self.rows=rows;self.members=members;self.specification=dict(backbone='polynormer_r')
    def named_parameters(self,remove_duplicate=True):
        seen=set()
        for name,p in self.rows:
            if not remove_duplicate or id(p) not in seen:
                yield name,p
                seen.add(id(p))
    def parameters(self): return [p for _,p in self.named_parameters()]


def same(a,b): return a.shape==b.shape and a.values==b.values
fake_torch=ModuleType('torch');fake_torch.is_tensor=lambda t:isinstance(t,Tensor);fake_torch.equal=same
fake_torch.optim=SimpleNamespace(Adam=Adam);fake_torch.from_numpy=lambda t:t.clone()

with patch.dict(sys.modules,{'torch':fake_torch}):
    native_rows=[];raw_rows=[]
    for boundary,prefix in integration.BOUNDARIES['polynormer_r'].items():
        w=Tensor([2,3],(1,2));b=Tensor([4,5]);R=Tensor([1]*8,(4,2));S=Tensor([1]*8,(4,2))
        native_rows += [('models.0.'+boundary+'.weight',w),('models.0.'+boundary+'.bias',b)]
        raw_rows += [(prefix+'.weight',w.clone()),(prefix+'.B',b.expand(4,-1)),(prefix+'.R',R),(prefix+'.S',S)]
    body=Tensor([7]);native_rows += [('models.0.body.weight',body),('models.0.tied.weight',body)]
    body_copy=body.clone();raw_rows += [('core.body.weight',body_copy),('core.tied.weight',body_copy)]
    native=Model(native_rows);raw=Model(raw_rows,4)
    native_parameters=dict(native.named_parameters())
    names=list(native_parameters)
    groups=[dict(params=[native_parameters[n] for n in names[:4]],lr=.001,eps=1e-8,weight_decay=.02,capturable=False),dict(params=[native_parameters[n] for n in names[4:]],lr=.002,eps=2e-8,weight_decay=.04,capturable=False)]
    opt=Adam(groups)
    for n,p in native_parameters.items():opt.state[p]=dict(step=Tensor([9],(),device='cpu'),exp_avg=Tensor([8]*len(p.values),p.shape),exp_avg_sq=Tensor([32]*len(p.values),p.shape),max_exp_avg_sq=Tensor([48]*len(p.values),p.shape))
    snapshot=integration.named_optimizer_snapshot(native,opt)
    transported,report=integration.transport_optimizer(native,snapshot,raw)
    check('actual_transport_identity',report['operation']==frozen['transport'] and report['members']==4)
    for name,p in raw.named_parameters():
        state=transported.state[p]
        if name.endswith(('.R','.S')):
            check('new_factor_empty_state:'+name,state=={})
        else:
            bias=name.endswith('.B')
            check('named_moment_scale:'+name,state['exp_avg'].values==tuple([2 if bias else 8]*len(p.values)) and state['exp_avg_sq'].values==tuple([2 if bias else 32]*len(p.values)) and state['max_exp_avg_sq'].values==tuple([3 if bias else 48]*len(p.values)))
            check('CPU_Adam_step_custody:'+name,state['step'].values==(9,) and state['step'].device=='cpu')
    check('inactive_local_head_retained',all(name in dict(raw.named_parameters()) for name in ('local_head.weight','local_head.B')) and report['all_native_inactive_state_retained'])
    check('shared_alias_preserved',dict(raw.named_parameters(False))['core.body.weight'] is dict(raw.named_parameters(False))['core.tied.weight'])
    raw_parameters=dict(raw.named_parameters())
    check('only_bias_options_scaled',report['private_bias_eps_scale']==.25 and report['private_bias_coupled_decay_scale']==.25 and all(row['only_B_options_split'] for row in report['native_group_custody']))
    for g in transported.param_groups:
        if any(p is raw_parameters['stem.B'] for p in g['params']):check('actual_bias_group_eps_decay_scale',g['eps']==2.5e-9 and g['weight_decay']==.005)
        if any(p is raw_parameters['stem.R'] for p in g['params']):check('actual_factor_options_inherited',g['lr']==.001 and g['eps']==1e-8 and g['weight_decay']==.02)
    saved=integration.named_optimizer_snapshot(raw,transported)
    restored=integration.restore_named_optimizer(raw,saved)
    check('actual_named_snapshot_restore',integration.named_optimizer_snapshot(raw,restored)==saved)
    split=Model([(name,p.clone() if name=='core.tied.weight' else p) for name,p in raw.rows],4)
    rejects('split_shared_alias_refused',lambda:integration.transport_optimizer(native,snapshot,split))
    bad=copy.deepcopy(snapshot);bad['groups'][0]['options']['decoupled_weight_decay']=True
    rejects('decoupled_Adam_refused',lambda:integration.transport_optimizer(native,bad,raw))

# The real pinned compact-label loader receives the kindless context descriptor.
with tempfile.TemporaryDirectory() as directory:
    synthetic=Path(directory)/'descriptor.txt';synthetic.write_text('synthetic compact-pack boundary\n')
    descriptor=dict(path=str(synthetic),sha256=digest(synthetic.read_bytes()),bytes=synthetic.stat().st_size)
    check('actual_pinned_descriptor_verified',pinned.verified(descriptor)==synthetic.resolve())
    rejects('kind_bearing_loader_descriptor_refused',lambda:pinned.verified(dict(descriptor,kind='validation_labels')))
    class Pack(dict):
        files=['nodes','labels']
        def __enter__(self):return self
        def __exit__(self,*args):return False
    nodes=Tensor([1,3,5],dtype='int64');labels=Tensor([0,2,7],dtype='int64')
    np_double=SimpleNamespace(int64='int64',array_equal=same,load=lambda path,allow_pickle:Pack(nodes=nodes,labels=labels))
    rt=SimpleNamespace(np=np_double,torch=fake_torch,device='cpu')
    loaded=pinned.load_labels(rt,descriptor,nodes,8)
    check('actual_pinned_load_labels_synthetic_pack',same(loaded.nodes,nodes) and same(loaded.labels,labels))
    rejects('actual_label_role_mismatch_refused',lambda:pinned.load_labels(rt,descriptor,Tensor([1,3,6],dtype='int64'),8))
    check('all_cells_kindless_validation_descriptor',all(c['context']['source_labels']['validation']=={k:v for k,v in c['validation_labels'].items() if k!='kind'} for c in frozen['cells']))

# Exercise the unchanged continuation function, not a replica of its selector.
class State:
    def __init__(self):self.epoch=0
    def state_dict(self):return dict(epoch=self.epoch)
    def load_state_dict(self,state):self.epoch=state['epoch']
    def eval(self):return self

for case in ('epoch0_best','earliest_strict_tie','late_best'):
    model=State();trace=[];saved=[];updates=[];evaluations=[]
    def value(epoch):
        if case=='epoch0_best':return 1 if epoch==0 else 2
        if case=='earliest_strict_tie':return 1 if epoch in (2,3) else 2
        return .5 if epoch==950 else 2
    def evaluate(model,graph,val,**kwargs):
        assert kwargs=={'raw':True};evaluations.append(model.epoch)
        return ('synthetic logits',model.epoch),value(model.epoch)
    def update(model,*args,**kwargs):
        assert kwargs=={'raw':True};model.epoch+=1;updates.append(model.epoch);return .3,2
    with patch.object(integration,'evaluate',side_effect=evaluate),patch.object(integration,'train_update',side_effect=update),patch.object(integration,'cpu_copy',side_effect=copy.deepcopy),patch.object(integration,'named_optimizer_snapshot',return_value={'synthetic':True}):
        best,selection=integration.continuation(model,None,SimpleNamespace(teacher_backbone='polynormer_r'),None,None,trace.append,lambda name,logits:saved.append((name,logits)))
    expected=0 if case=='epoch0_best' else 2 if case=='earliest_strict_tie' else 950
    check('actual_continuation_full950_no_patience:'+case,len(updates)==950 and selection['continuation_updates_completed']==selection['update_cap']==950 and selection['patience'] is None and selection['global_only'])
    check('actual_earliest_strict_selector:'+case,best['continuation_epoch']==expected and selection['selected_actual_update']==expected+250 and model.epoch==expected)
    check('actual_epoch0_every_update_and_replay:'+case,evaluations==list(range(951))+[expected] and trace[0]['initializer_checkpoint_eligible'])
    check('actual_Photo_midpoint450:'+case,selection['native_midpoint_continuation_epoch']==450 and saved[0]==('native_midpoint',('synthetic logits',450)) and saved[1]==('selected',('synthetic logits',expected)))

# Shared resource sentinels retain swallowed allocator failures.
counts=dict(calls_started=0,calls_completed=0,calls_failed=0);failures=[];events=[]
def fail(theta):raise MemoryError('synthetic allocator')
try:shared.observed_forward(fail,None,counts,lambda name,row:events.append((name,row)),failures)
except MemoryError:pass
check('actual_observed_allocator_custody',counts==dict(calls_started=1,calls_completed=0,calls_failed=1) and len(failures)==1)
check('actual_paired_swallowed_allocator_defers',shared.paired_return_status({'synthetic':1},failures)=='resource_deferred')
check('actual_joint_failure_stays_terminal',shared.paired_return_status(None,[])=='joint_failure')
wrapped=RuntimeError('outer');wrapped.__cause__=MemoryError('inner')
check('actual_resource_cause_chain_retained',shared.resource_failure_in_chain(wrapped)['chain_link']=='root.__cause__')
check('actual_resource_plan_insufficient_memory',shared.resource_decision(999,release['Photo_resource_evidence'],True)['status']=='resource_deferred')
check('actual_resource_plan_wrong_UUID',shared.resource_decision(1000,release['Photo_resource_evidence'],False)['status']=='resource_deferred')
check('actual_resource_plan_pass',shared.resource_decision(1000,release['Photo_resource_evidence'],True)['status']=='resource_preflight_passed')

# Exercise run_block orchestration with synthetic model/runtime interfaces.
# The doubles below qualify ordering and custody, not graph numerics or AD.
class Signal:
    device='cpu'
    def __init__(self,shape,dtype='float32',value='synthetic'):
        self.shape=tuple(shape);self.dtype=dtype;self.value=value;self.requires_grad=True
    def detach(self):return self
    def clone(self):return Signal(self.shape,self.dtype,self.value)
    def cpu(self):return self
    def numel(self):return math.prod(self.shape)
    def is_floating_point(self):return self.dtype=='float32'
    def __getitem__(self,key):
        assert key is None
        return Signal((1,)+self.shape,self.dtype,self.value)
    def repeat(self,*sizes):return Signal(tuple(a*b for a,b in zip(self.shape,sizes)),self.dtype,self.value)
    def sort(self):return SimpleNamespace(values=self)
    def tolist(self):return list(range(self.shape[0]))


class Sparse:
    dtype='float32';device='cpu';shape=(7650,7650)
    def indices(self):return Signal((2,2),'int64','edges')
    def values(self):return Signal((2,),value='edge_values')
    def coalesce(self):return self


class Permutation(Signal):
    def __getitem__(self,key):return Signal((2,2),'int64','edges')


class Raw:
    def __init__(self,k):self.members=k;self.state={'warm':Signal((1,))}
    def eval(self):return self
    def state_dict(self):return self.state
    def load_state_dict(self,state):self.state=state
    def named_parameters(self):return [('stem.R',Signal((1,))),('stem.S',Signal((1,)))]
    def __call__(self,*args):return Signal((self.members,7650,8))


for scenario in ('success','fourth_install_fails','fit_fails','swallowed_allocator'):
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);events=[];saved_states={};opened=[];rng=[];differences=[];fits=[]
        native=Raw(1);checkpoint=dict(specification={'synthetic':17},global_stage=True,model=native.state_dict(),optimizer={'synthetic':True},rng={'post_warm':'same'})
        graph=SimpleNamespace(preprocessing={'identity':frozen['cells'][0]['preprocessing_identity']},classes=8)
        train=SimpleNamespace(nodes=Signal((3,),'int64'),labels=Signal((3,),'int64'))
        active={'paired':False}
        def tensor_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.value==b.value
        def closure(theta):
            if scenario=='swallowed_allocator' and active['paired']:raise MemoryError('synthetic paired allocation')
            return Signal((7650,8))
        def tensor_save(state,path):
            saved_states[str(Path(path).resolve())]=state;Path(path).write_text('synthetic checkpoint, no numeric payload\n')
        def tensor_load(path,**kwargs):return saved_states.get(str(Path(path).resolve()),checkpoint)
        def difference(a,b,*args):
            differences.append((a,b));return dict(passed=not(scenario=='fourth_install_fails' and len(differences)==8))
        def continued(model,opt,graph,train,validation,trace,save):
            fits.append(model)
            if scenario=='fit_fails' and len(fits)==2:raise RuntimeError('synthetic fit failure')
            save('selected',Signal((4,7650,8)))
            return dict(state=model.state_dict()),dict(update_cap=950,continuation_updates_completed=950,patience=None,global_only=True,native_midpoint_continuation_epoch=450,native_midpoint_saved=True,primary_validation_nll=1)
        class Ledger:
            def __init__(self,out):self.costs=[];self.out=out
            def measured(self,rt,name,callback):
                self.costs.append({'operation':name});events.append(name);return callback()
            def sink(self,name,row):events.append(name)
        def source_inputs(rt,ctx,ledger,validation):
            assert validation is False and set(ctx['source_labels'])=={'train'}
            return graph,Signal((2,2),'int64'),train,None
        def validation_loader(rt,record,ids,classes):
            assert 'kind' not in record and len(differences)==8
            events.append('synthetic_validation_open');return train
        def save_logits(rt,out,name,logits):(out/(name+'_member_logits.npy')).write_text('synthetic logits boundary only\n')
        driver_double=SimpleNamespace(Ledger=Ledger,source_inputs=source_inputs,load_labels=validation_loader,
            descriptor=pinned.descriptor,write_json=pinned.write_json,append_trace=lambda *args:None,save_logits=save_logits)
        torch_double=SimpleNamespace(float32='float32',int64='int64',load=tensor_load,save=tensor_save,
            no_grad=contextlib.nullcontext,isfinite=lambda x:SimpleNamespace(all=lambda:True),equal=tensor_equal,
            arange=lambda n,**kwargs:Signal((n,),'int64'),sparse_coo_tensor=lambda *args,**kwargs:Sparse(),
            stack=lambda values:Signal((len(values),7650,8)))
        def permutation(S,seed):return Sparse(),Permutation((7650,),'int64')
        integration_double=SimpleNamespace(TRANSPORT=frozen['transport'],restore_native=lambda *args:(native,None),
            optimizer_equivalence_audit=lambda *args:{'passed':True},clone_boundary=lambda n,b,k:Raw(k),
            identity_logits_audit=lambda *args:{'passed':True},raw_arguments=lambda *args:('features','edges'),
            difference=difference,transport_optimizer=lambda *args:(None,{'operation':frozen['transport']}),
            rng_restore=lambda state:rng.append(state),rng_snapshot=lambda:checkpoint['rng'],cpu_copy=lambda state:state,
            named_optimizer_snapshot=lambda *args:{'synthetic':True},restore_named_optimizer=lambda *args:None,continuation=continued)
        method_double=SimpleNamespace(bind_common_model=lambda *args:(Signal((1024,)),closure,{'synthetic':True}),
            symmetric_normalized_adjacency=lambda *args:Sparse(),permute_topology_nodes=permutation,
            install_factor_slices=lambda *args:None)
        rt=SimpleNamespace(torch=torch_double,np=SimpleNamespace(load=lambda *args,**kwargs:Signal((3,),'int64')),
            device='cpu',adapter=SimpleNamespace(specification=lambda *args:{'synthetic':17}),boundary=None,
            integration=integration_double,method=method_double)
        def qualified(observed,theta,*args):
            for _ in range(16):observed(theta)
            return dict(passed=True)
        def initialized(observed,theta,*args,**kwargs):
            assert kwargs=={'homogeneous_full_node_outputs':True}
            active['paired']=True
            try:observed(theta)
            except MemoryError:pass
            active['paired']=False
            return {arm:[theta]*4 for arm in subject.ARMS},dict(vjp_forwards=1,jvp_calls=0,line_search_forward_calls=0)
        paired=SimpleNamespace(initialize_paired_four_arms=initialized)
        with patch.object(subject,'verify',side_effect=lambda row:Path(row['path'])):
            got=subject.run_block(driver_double,rt,shared,paired,SimpleNamespace(qualify_gradient_interface=qualified),
                frozen['cells'][0],frozen,p/'block',opened)
        check('actual_block_all4_terminals:'+scenario,set(got['fits'])==set(subject.ARMS) and all(f['status']!='pending' for f in got['fits'].values()))
        if scenario in ('success','fit_fails'):
            check('actual_block_all4_installs_before_validation:'+scenario,len(got['initializations'])==4 and opened==[frozen['cells'][0]['validation_labels']] and events.index('synthetic_validation_open')>max(i for i,event in enumerate(events) if event=='installed_closure_references'))
            check('actual_block_same_post_warm_RNG:'+scenario,len(rng)==8 and all(state==checkpoint['rng'] for state in rng))
            check('actual_block_donor_unchanged:'+scenario,got['donor_unchanged'])
            check('actual_block_attempts_remaining_fits:'+scenario,len(fits)==4)
            check('actual_block_fits_terminal_status:'+scenario,got['status']==('completed' if scenario=='success' else 'failed_fits'))
        else:
            check('actual_block_validation_stays_closed:'+scenario,not opened and not got['validation_labels_loaded'] and not fits)
            check('actual_block_prerequisite_failure_terminal:'+scenario,got['status']==('qualification_failed' if scenario=='fourth_install_fails' else 'resource_deferred') and all(not f['attempted'] for f in got['fits'].values()))
        if scenario=='fourth_install_fails':check('actual_failed_fourth_install_has_only_three_initialized',len(got['initializations'])==3)
        if scenario=='swallowed_allocator':check('actual_block_retains_swallowed_allocator',got['observed_resource_failures'][0]['exception']['reason']=='MemoryError')

# Main early exits use only synthetic metadata and mocked resource probes.
for status in ('resource_deferred','resource_preflight_passed'):
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);fake=p/'source.txt';fake.write_text('synthetic source custody\n')
        row=dict(path=str(fake),sha256=digest(fake.read_bytes()),kind='source_metadata')
        f=dict(frozen,original_records=[row],shared_source=row)
        (p/'FROZEN_STUDY.json').write_text(json.dumps(f))
        manifest=json.dumps(dict(payload=[])).encode();(p/'MANIFEST.json').write_bytes(manifest)
        (p/'SEAL.json').write_text(json.dumps(dict(manifest_sha256=digest(manifest))))
        admission=p/'release.json';admission.write_text(json.dumps(release))
        mock_shared=SimpleNamespace(resource_preflight=lambda bound,evidence:dict(status=status))
        with patch.object(subject,'PACKET',p),patch.object(subject,'load',return_value=mock_shared),patch.object(subject,'admission_guard',return_value=admitted),contextlib.redirect_stdout(io.StringIO()):
            code=subject.main(['--run-name','synthetic','--preflight-only','--admission',str(admission)])
        got=json.loads((p/'runs/synthetic/FOLLOWUP.json').read_text())
        check('actual_main_early_exit_no_native:'+status,code==0 and not got['blocks'] and got['validation_blocks_loaded']==0)
        check('actual_main_preservation:'+status,got['originals_before']==got['originals_after'])
        check('actual_main_scope:'+status,got['final_labels_closed'] and not got['Squirrel_outcomes_loaded'])
        check('actual_main_no_execution_marker:'+status,not (p/'runs/synthetic/PRE_EXECUTION_ADMISSION.json').exists())

calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
text=source.decode()
check('four_installations_before_validation_source',text.index("require(set(result['initializations']) == set(ARMS)")<text.index("verify_inputs([cell['validation_labels']], allow_validation=True)"))
check('actual_imported_paired_base_before_after_custody',any(r['path'].endswith('/graph_full_node_cotangent_paired_alpha_v1/base/graph_band_route_initializer.py') for r in frozen['original_records']) and text.index("result['originals_before'] = verify_inputs(frozen['original_records'])")<text.index("paired = load('paired_Photo_sealed_helper'")<text.index("result['originals_after'] = verify_inputs(frozen['original_records'])"))
check('active_transitive_initializer_bound',any(r['path'].endswith('/round17_graph_init_driver_integration_v3_precision/prototype/graph_band_route_initializer.py') for r in frozen['original_records']))
check('modern_transitive_sources_bound',all(any(r['path'].endswith('/modern_backbone_teacher_amendment_v3/prototype/'+name+'.py') for r in frozen['original_records']) for name in ('modern_teacher_adapter','backbone_boundary_adapter','native_polynormer','native_polyformer','native_polyformer_outer','native_polyformer_preprocess')))
check('no_phase_registry_or_outcome_entrypoints',not any(isinstance(n.func,ast.Attribute) and n.func.attr in {'context_guard','modern_certificate_guard','register','phase_run','compare','report','warm_native'} for n in calls))
check('no_direct_remote_or_environment_mutation',all(token not in text for token in ('ssh ','scp ','rsync ','os.environ')))
check('no_restart_guard',"'*/PRE_EXECUTION_ADMISSION.json'" in text)
after=metadata_custody('after')
check('no_real_Torch_imported','torch' not in sys.modules)
result=dict(schema='paired-Photo-synthetic-actual-interface-checks-v1',passed=True,checks=checks,
    source_sha256=digest(source),nonnumeric_originals_unchanged=after,numeric_originals_opened=False,
    author_native_remote_GPU_execution=False,actual_Torch_gradient_or_native_qualification=False,
    validation_or_final_label_bytes_opened=False,synthetic_interfaces=['pinned descriptor verifier','pinned compact label loader','pinned named Adam snapshot/transport/restore','pinned Photo950 continuation loop','shared resource sentinel','full12 comparison','root gate/resource admission','main early exits','run_block all-arm installation/admission/RNG/failure orchestration'])
(packet/'FIXTURE_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),nonnumeric_originals_unchanged=after,numeric_originals_opened=False,actual_Torch_gradient_or_native_qualification=False)))
