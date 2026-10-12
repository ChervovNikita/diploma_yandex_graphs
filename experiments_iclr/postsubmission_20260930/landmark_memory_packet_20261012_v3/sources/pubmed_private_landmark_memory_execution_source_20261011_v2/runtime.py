"""Finite owner/source/data/state custody; numerical imports deferred to admission."""
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
NOMINATION_SHA='84d44d954104d0059db53f2393ecd1a99944784975d0b63dff6e366cb816e0fb'
SPEC_SHA='60ec176d581f693633c537be7b82af63d6795af51ec01ede716d3918d7193bf5'
SEEDS=(9101,9203,9307)
ARMS=('B1','B4','T16','S64','FS64','LM4','I4[S64]','I4[FS64]')

def require(ok,message):
    if not ok:raise ValueError(message)

def route():
    require(socket.gethostname()=='anogena-2-0','Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID],'Excluded wrong sole GPU')
    require(Path.cwd()==REPO,'Exact allocation repository required')

def receipt(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def checked(item):
    path=Path(item['path'])
    if not path.is_absolute():path=PHASE/path
    path=path.resolve();require(path.is_relative_to(REPO),'Bound project file required')
    got=receipt(path);require(got['sha256']==item['sha256'] and ('bytes' not in item or got['bytes']==item['bytes']),'Bound file differs: '+str(path));return path

def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def owner_closed(item):
    value=json.loads(checked(item).read_text())
    require(value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and not value['timed_out'] and value['direct_child_wait'] and value['child_pid_absent'] and value['owned_cuda_pid_absent'],'Successful directly waited/reaped finite owner')
    return value

def admit(config,stage):
    require(config.get('execution_enabled') is True and config.get('stage')==stage and stage in ('prepare','qualify','acquire'),'Exact root enabled stage')
    require(config.get('seeds')==list(SEEDS) and config.get('arms')==list(ARMS),'Complete frozen seed/arm roster')
    require(config.get('finite_external_owner') is True and os.environ.get('PILOT_OWNER_PID')==str(os.getppid()),'Direct finite owner required')
    decision=json.loads(checked(config['root_decision']).read_text());require(stage in decision.get('admitted_stages',[]),'Current root stage admission required')
    policy=json.loads(checked(config['engineering_policy']).read_text());require(policy['nomination_manifest_sha256']==NOMINATION_SHA and config['containment_tolerance']==policy['containment_tolerance'] and policy['state_optimizer_tensor_and_global_private_RNG_restoration']=='Exact','Prospectively root-bound containment tolerance and exact state policy')
    limits=config['limits'];require(set(limits)=={'wall_seconds','cuda_bytes','rss_bytes'} and all(type(v) is int and v>0 for v in limits.values()) and limits['cuda_bytes']<=32*1024**3 and limits['rss_bytes']<=16*1024**3,'Positive finite frozen resource envelope')
    manifest=json.loads(checked(config['source_manifest']).read_text())
    for rel,fixed in manifest['files'].items():
        got=receipt(HERE/rel);require(got['bytes']==fixed['bytes'] and got['sha256']==fixed['sha256'],'Sealed source differs: '+rel)
    require(receipt(HERE/'nomination/MANIFEST.json')['sha256']==NOMINATION_SHA and receipt(HERE/'nomination/FIXED_PROSPECTIVE_NOMINATION.json')['sha256']==SPEC_SHA,'Fixed nomination authority')
    binding=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for item in binding['source_dependencies']:checked(item)
    for item in binding['upstream_source_manifests']:
        path=checked(item);upstream=json.loads(path.read_text())
        for row in upstream['files']:
            checked(dict(path=str(path.parent/row['path']),bytes=row['bytes'],sha256=row['sha256']))
    checked(binding['server_data_and_runtime']['runtime']['python'])
    return dict(stage=stage,source_manifest=config['source_manifest'],root_decision=config['root_decision'],nomination_manifest_sha256=NOMINATION_SHA,specification_sha256=SPEC_SHA,bindings=receipt(HERE/'SOURCE_BINDINGS.json'))

@contextlib.contextmanager
def source_surface():
    """Exact existing factory/session/optimizer; only new caller's SELECT role."""
    reference=PHASE/'combination_pubmed_strong_reference_source_20261010_v2'
    factor_path=PHASE/'pubmed_factor1_controls_source_20261010_v1/adapter.py'
    names=('source','admission','reference_plan','adapter','metrics','native','plan')
    saved={name:sys.modules.pop(name,None) for name in names};old_path=list(sys.path)
    try:
        sys.path.insert(0,str(reference))
        engine=load('_LM4_exact_native_fit_engine',reference/'train.py')
        base=load('_LM4_exact_native_factory',reference/'adapter.py')
        factor=load('_LM4_exact_factor1_factory',factor_path);factor.REFERENCE_FACTORY=base.fresh_single
        yield engine,factor
    finally:
        sys.path[:]=old_path
        for name,previous in saved.items():
            if previous is None:sys.modules.pop(name,None)
            else:sys.modules[name]=previous

class Runtime:
    def __init__(self,config,prepared=True):
        self.config=config;self.started=time.monotonic();self.binding=json.loads((HERE/'SOURCE_BINDINGS.json').read_text());data=self.binding['server_data_and_runtime']
        source_custody=json.loads(checked(data['split_custody']).read_text())
        spec=dict(frozen_providers=data['frozen_providers'],train_bundle=data['train_bundle'],_train_custody=source_custody)
        with source_surface() as (engine,factory):
            self.np,self.t,self.providers,self.tensor,self.source_roles=engine.runtime_and_data(spec,valid=False)
        require(self.providers==config['expected_runtime_versions']==data['frozen_providers'],'Exact source/runtime providers')
        self.F=self.t.nn.functional;self.device=self.t.device('cuda:0')
        self.helper=load('_LM4_exact_state_helpers',HERE/'state_helpers.py');self.core=load('_LM4_fixed_core',HERE/'core.py')
        self.fixed=tuple(self.tensor.values());self.versions=tuple(v._version for v in self.fixed);self.caches=[];self.cache_records=[];self.geometries={}
        self.train_custody=data['split_custody'];self.graph=dict(nodes=19717,features=500,classes=3,ordered_edges=88648,original_array_fingerprints=source_custody['array_fingerprints'],native_preprocessing='exact monomial K2',TEST_closed=True)
        if prepared:
            ended=owner_closed(config['prepare_owner_end']);require(ended['stage']=='prepare' and ended['source_manifest']==config['source_manifest'] and ended['produced_artifact']==config['prepared_custody'],'Prepared direct owner/source/produced custody linkage');self.prepared=json.loads(checked(config['prepared_custody']).read_text())
            require(self.prepared['complete'] and self.prepared['nomination_manifest_sha256']==NOMINATION_SHA and self.prepared['train_custody']==self.train_custody and self.prepared['source_manifest']==config['source_manifest'],'Complete identical projected roles and canonical geometry custody')
            capsule=self.t.load(checked(self.prepared['geometry_capsule']),map_location='cpu',weights_only=True);digest=capsule.pop('fingerprint')
            require(self.fp(capsule)==digest==self.prepared['geometry_capsule_fingerprint'] and set(capsule['blocks'])=={str(s) for s in SEEDS},'Whole canonical geometry capsule')
            require(capsule['graph']==self.graph and capsule['nomination_manifest_sha256']==NOMINATION_SHA,'Geometry source graph/nomination identity')
            for seed in SEEDS:
                cpu=capsule['blocks'][str(seed)];require(self.fp(cpu)==self.prepared['geometry_payload_fingerprints'][str(seed)],'Canonical landmark/BFS payload')
                anchors,bias=cpu['anchors'].to(self.device),cpu['bias'].to(self.device)
                self.geometries[seed]=dict(anchors=anchors,bias=bias)
                values=tuple(v for v in cpu.values() if self.t.is_tensor(v))+(anchors,bias)
                self.fixed+=values;self.versions+=tuple(v._version for v in values)
        self.guard()
    def fp(self,value):return self.helper.fingerprint(self.t,value)
    def rng(self):return self.helper.rng(self.t,self.np,self.device)
    def restore_rng(self,value):self.helper.restore_rng(self.t,self.np,self.device,value)
    def guard(self):
        self.t.cuda.synchronize(self.device);limits=self.config['limits']
        require(time.monotonic()-self.started<limits['wall_seconds'] and self.t.cuda.max_memory_allocated(self.device)<=limits['cuda_bytes'] and self.t.cuda.max_memory_reserved(self.device)<=limits['cuda_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limits['rss_bytes'],'Finite whole stage wall/RSS/CUDA cap exceeded')
        require(self.versions==tuple(v._version for v in self.fixed),'Fixed public input/geometry/cache/role tensor mutated')
    def retain_fixed(self,values):
        values=tuple(values);self.fixed+=values;self.versions+=tuple(v._version for v in values)
    def roles(self,select=True,export=False):
        result={name.lower()+'_ids':v[0] for name,v in self.source_roles.items()}
        result.update({name.lower()+'_y':v[1] for name,v in self.source_roles.items()})
        if select:
            with self.np.load(checked(self.prepared['role_files']['select']),allow_pickle=False) as archive:
                require(set(archive.files)=={'ids','y'},'SELECT-only target arrays');ids,y=archive['ids'].copy(),archive['y'].copy()
            require(ids.dtype==y.dtype==self.np.int64 and ids.shape==y.shape==(1313,) and self.np.bincount(y,minlength=3).tolist()==[273,515,525] and len(self.np.unique(ids))==1313,'Frozen full SELECT role')
            result.update(select_ids=self.t.from_numpy(ids).to(self.device),select_y=self.t.from_numpy(y).to(self.device));self.retain_fixed([result['select_ids'],result['select_y']])
        if export:
            for name,count in (('cal',1313),('dev',1316)):
                ids=self.np.load(checked(self.prepared['role_files'][name+'_ids']),allow_pickle=False)
                require(ids.dtype==self.np.int64 and ids.shape==(count,) and len(self.np.unique(ids))==count,'Full label-free export membership');result[name+'_ids_numpy']=ids
        return result
    def fresh_source(self,seed,body):
        require(seed in SEEDS and body in range(4),'Fresh declared source body');before=self.fp(self.rng());tick=time.monotonic()
        with source_surface() as (engine,factory):session=factory.fresh_single('single_native',seed+1009*body,self.tensor,device='cuda:0')
        require(self.fp(self.rng())==before,'Source constructor/factor/preprocessing changed outside RNG')
        require(sum(p.numel() for p in session.bodies[0].parameters())==2083818 and len(session.optimizers)==1 and session.decoder is None,'Complete exact source factorized M1')
        self.FactorLinear=session.factors.FactorLinear
        self.retain_fixed([session.x,session.edges,session.train_ids,session.train_y,*session.factual.list_mat])
        self.guard();return session,dict(seed=seed,body=body,initialization_seed=seed+1009*body,dropout_seed=seed+1009*body+300001,source_parameters=2083818,preprocessing_seconds=session.preparation_seconds,construction_and_preprocessing_seconds=time.monotonic()-tick,outside_RNG_unchanged=True)
    def cache_source(self,session,seed,body,checkpoint):
        native=session.bodies[0];modes=self.modes(native);captured=[];tick=time.monotonic();before=self.fp(self.rng())
        handle=native.lin3.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        try:
            native.eval()
            with self.t.no_grad(),session.factors.member_context(native,0):source=native(session.factual)
        finally:
            handle.remove()
            for name,module in native.named_modules():module.training=modes[name]
        require(len(captured)==1 and captured[0].shape==(19717,256) and source.shape==(19717,3),'Complete exact source pre-lin3 capture')
        H=captured[0].detach().clone();U=H/self.t.linalg.vector_norm(H,ord=2,dim=1,keepdim=True).clamp_min(1)
        with self.t.no_grad(),session.factors.member_context(native,0):replay=native.lin3(H)
        require(self.t.equal(source.max(1)[1],replay.max(1)[1]) and self.t.equal(source[self.source_roles['TRAIN'][0]].max(1)[1].eq(self.source_roles['TRAIN'][1]),replay[self.source_roles['TRAIN'][0]].max(1)[1].eq(self.source_roles['TRAIN'][1])),'Source/cache-head full predictions/TRAIN correct masks containment')
        tolerance=self.config['containment_tolerance'];require(set(tolerance)=={'atol','rtol'} and all(type(v) in (float,int) and v>=0 for v in tolerance.values()),'Root pre-bound engineering tolerance')
        require(self.t.allclose(source,replay,**tolerance) and self.fp(self.rng())==before,'Bound float containment and outside RNG custody')
        cache=dict(H=H,U=U);self.caches.append(cache);self.retain_fixed([H,U])
        record=dict(seed=seed,body=body,checkpoint=checkpoint,H_fingerprint=self.fp(H),U_fingerprint=self.fp(U),H_bytes=H.numel()*H.element_size(),U_bytes=U.numel()*U.element_size(),source_full_predictions_fingerprint=self.fp(source.max(1)[1]),cache_full_predictions_fingerprint=self.fp(replay.max(1)[1]),full_prediction_and_TRAIN_correct_masks_equal=True,float_tolerance=tolerance,source_logits_max_absolute_delta=float((source-replay).abs().max().item()),cache_and_containment_seconds=time.monotonic()-tick)
        self.cache_records.append(record);self.guard();return cache,source.detach(),record
    def construct(self,arm,seed,ancestors,caches):
        before=self.fp(self.rng());core=self.core.build_arm(self.t,self.F,arm,seed,ancestors,caches,self.geometries[seed],FactorLinear=self.FactorLinear)
        for model in core['models']:model.to(self.device)
        require(self.fp(self.rng())==before,'Isolated memory construction changed outside RNG')
        audit=self.core.inspect_parameter_ownership(core);optimizers=self.core.make_optimizers(self.t,core)
        return core,optimizers,audit
    def fresh_stream(self,seed):
        state=self.np.random.RandomState(seed).get_state()
        return dict(python=random.Random(seed).getstate(),numpy=(state[0],state[1].tolist(),state[2],state[3],state[4]),cpu=self.t.Generator(device='cpu').manual_seed(seed).get_state(),cuda=self.t.Generator(device=self.device).manual_seed(seed).get_state())
    def streams(self,seed,body,views):return [self.fresh_stream(seed+491003061+1000003*body+1009*v) for v in range(views)]
    @contextlib.contextmanager
    def scope(self,streams,view):
        outside=self.rng();self.restore_rng(streams[view])
        try:yield
        finally:streams[view]=self.rng();self.restore_rng(outside)
    def modes(self,model):return {name:module.training for name,module in model.named_modules()}
    def external(self,model):
        return {name:self.fp(getattr(model,name)) for name in ('fixed_H','fixed_U','anchors','distance_bias') if hasattr(model,name)}
    def state(self,model,optimizer,streams,meta):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),module_modes=self.modes(model),global_rng=self.rng(),route_rng=streams,external_input_fingerprints=self.external(model),meta=meta)
    def save(self,path,model,optimizer,streams,meta):
        state=self.state(model,optimizer,streams,meta);digest=self.fp(state);state['fingerprint']=digest;self.t.save(state,path);return digest
    def restore(self,path,model,optimizer,streams,meta):
        state=self.t.load(path,map_location='cpu',weights_only=True);digest=state.pop('fingerprint')
        require(state['meta']==meta and self.fp(state)==digest and state['external_input_fingerprints']==self.external(model),'Selected state and immutable external input custody')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);modules=dict(model.named_modules())
        require(set(modules)==set(state['module_modes']),'Every module mode saved')
        for name,module in modules.items():module.training=state['module_modes'][name]
        streams.clear();streams.update(state['route_rng']) if isinstance(streams,dict) else streams.extend(state['route_rng'])
        self.restore_rng(state['global_rng']);optimizer.zero_grad(set_to_none=True)
        require(self.fp(self.state(model,optimizer,streams,meta))==digest,'Exact model/Adam/all modes/global/private RNG restoration')
        self.guard();return digest
    def costs(self):
        return dict(seconds=time.monotonic()-self.started,peak_CUDA_allocated_bytes=self.t.cuda.max_memory_allocated(self.device),peak_CUDA_reserved_bytes=self.t.cuda.max_memory_reserved(self.device),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
