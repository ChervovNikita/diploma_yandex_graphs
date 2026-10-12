"""Bound private-GRU runtime; stdlib at import, numerics only after admission."""
import contextlib
import hashlib
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
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCKS=(10103,10427,10949)
ARMS=('B4','G1','GF1','GM4','GN4','GI16','GFI16')
CORE_SHA='780d1d14d04c699bbbcf17509eafa4a678d396510101a71d4e783a41d124e5ae'
SPEC_SHA='daa1bfc411f4f6a2a8e5511dad4f45d3ec47435dfd67ef2c42fc8dc7a998c7e2'
P1_SHA='3a277cc1bf5026cf906b580bc3828f9407e78b5400c65435f3084fb302cb75ac'
TOUR_DECISION_SHA='5fe15193fe2bde924f7b1fc3d47986623f297f3e32a6743453fa42345c781bab'
TOUR_KEYS=('tokens','reverse_index','starts','occurrences','offsets','generator_before_roots','generator_after_neighbors')

def require(ok,message):
    if not ok:raise ValueError(message)

def route():
    require(socket.gethostname()=='anogena-2-0','Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID],'Excluded wrong sole GPU')
    require(Path.cwd()==REPO,'Authorized allocation repository required')

def receipt(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def checked(item):
    require(receipt(item['path'])==item,'Bound file differs: '+item['path']);return Path(item['path'])

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def owner_closed(item):
    value=json.loads(checked(item).read_text())
    require(value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and not value['timed_out'] and value['direct_child_wait'] and value['child_pid_absent'] and value['owned_cuda_pid_absent'],'Successful finite owner closure required')
    return value

def load(name):
    path=HERE/'sealed_core'
    if name not in ('core','gru_backend'):path=path/'deps'
    spec=importlib.util.spec_from_file_location('fixed_private_gru_'+name,path/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def admit(config,stage):
    require(config.get('blocks')==list(BLOCKS) and config.get('arms')==list(ARMS),'Frozen v2 block/arm binding')
    require(stage=='acquire' and config.get('stage')==stage and config.get('execution_enabled') is True,'Only root-enabled complete acquisition')
    require(os.environ.get('PILOT_OWNER_PID')==str(os.getppid()) and config.get('finite_external_owner') is True,'Direct finite owner required')
    decision=json.loads(checked(config['root_decision']).read_text())
    require(stage in decision.get('admitted_stages',[]),'Current root stage admission required')
    limits=config['limits']
    require(set(limits)=={'wall_seconds','cuda_bytes','rss_bytes'} and all(type(v) is int and v>0 for v in limits.values()),'Positive finite caps required')
    manifest=json.loads(checked(config['source_manifest']).read_text())
    for rel,fixed in manifest['files'].items():
        got=receipt(HERE/rel);require(got['bytes']==fixed['bytes'] and got['sha256']==fixed['sha256'],'Sealed source differs: '+rel)
    for rel,sha in (('sealed_core/MANIFEST.json',CORE_SHA),('sealed_core/FIXED_PROSPECTIVE_NOMINATION.json',SPEC_SHA),('P1_TOUR_CUSTODY.json',P1_SHA),('TOUR_RECONSTRUCTION_DECISION.json',TOUR_DECISION_SHA)):
        require(receipt(HERE/rel)['sha256']==sha,'Fixed scientific authority differs: '+rel)
    core_manifest=json.loads((HERE/'sealed_core/MANIFEST.json').read_text())
    for rel,fixed in core_manifest['files'].items():
        got=receipt(HERE/'sealed_core'/rel);require(got['bytes']==fixed['bytes'] and got['sha256']==fixed['sha256'],'Original core bytes differ: '+rel)
    binding=json.loads((HERE/'BINDING.json').read_text())
    for key in ('derivative','derivative_owner'):
        field='derivative_owner_end' if key=='derivative_owner' else key
        got=config[field];require(got['bytes']==binding[key]['bytes'] and got['sha256']==binding[key]['sha256'],'Actual root derivative evidence required')
        checked(got)
    owner_closed(config['derivative_owner_end'])
    derivative=json.loads(checked(config['derivative']).read_text())
    require(derivative['complete'] and derivative['fixtures']==1 and derivative['core_manifest_sha256']==CORE_SHA and derivative['CUDAfloat64_vs_CPUfloat64'] and derivative['nonzero_readout'] and derivative['model_optimizer_updates']==0 and derivative['backend_operations']['aten::_cudnn_rnn_backward']==4,'Passed sole backend derivative fixture required')
    return dict(stage=stage,root_decision=config['root_decision'],source_manifest=config['source_manifest'],binding=receipt(HERE/'BINDING.json'),core_manifest_sha256=CORE_SHA,specification_sha256=SPEC_SHA)

class Runtime:
    def __init__(self,config,output):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np,self.t,self.F=np,torch,F;self.config=config;self.started=time.monotonic()
        require(torch.get_default_dtype()==torch.float32,'Source float32 default')
        require(config.get('expected_runtime_versions')=={'numpy':np.__version__,'torch':torch.__version__}=={'numpy':'1.26.4','torch':'2.1.2+cu118'},'Exact qualified allocation runtime')
        self.device=torch.device('cuda:0');torch.cuda.set_device(self.device);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        self.core,self.backend,self.native_core,self.author,self.helper,self.tour_helper=(load(n) for n in ('core','gru_backend','native_core','author_model','state_helpers','tours'))
        self.backend.require_runtime(torch);torch.cuda.reset_peak_memory_stats(self.device)
        self.p1_binding=json.loads((HERE/'P1_TOUR_CUSTODY.json').read_text())
        self.tour_decision=json.loads((HERE/'TOUR_RECONSTRUCTION_DECISION.json').read_text())
        require(self.tour_decision['decision']=='ADMIT_EXPLICIT_SAME_CONTENT_TOUR_RECONSTRUCTION_IN_ONE_FULL_INPUT_CHECK' and self.tour_decision['requires_all_12_original_payload_and_RNG_receipts'] and self.tour_decision['requires_persisted_CPU_tour_and_packing_capsule'],'Explicit reconstruction decision')
        require(self.tour_decision['original_tour_bindings']==self.p1_binding['persistent_tours']['expected_payload_bindings'],'Original all12 tour authorities agree')
        require(config['custody']==self.p1_binding['roles_and_graph']['role_custody'],'Identical original role custody')
        self.custody=json.loads(checked(config['custody']).read_text())
        require(self.custody['complete'] and self.custody['blocks']==list(BLOCKS) and self.custody['role_rng']=='RandomState(block+5000011+223000019)','Complete source role custody')
        require(self.custody['graph_sha256']==self.tour_decision['graph'] and self.custody['feature_sha256']==self.tour_decision['features'],'Original input identity')
        require(self.custody['files']['graph.npz']==self.p1_binding['roles_and_graph']['graph_file_receipt'],'Original graph container identity')
        with np.load(checked(self.custody['files']['graph.npz']),allow_pickle=False) as a:
            require(set(a.files)=={'x','edge_index'},'Graph contains no targets');x,e=a['x'].copy(),a['edge_index'].copy()
        require(x.shape==(34493,8415) and x.dtype==np.float32 and np.isfinite(x).all() and np.logical_or(x==0,x==1).all(),'Full raw binary X')
        require(e.shape==(2,530417) and e.dtype==np.int64 and e.min()>=0 and e.max()<34493 and int((e[0]==e[1]).sum())==34493,'Full native support')
        self.features=torch.from_numpy(x).to_sparse_coo().coalesce().to(self.device)
        self.adjacency=torch.sparse_coo_tensor(torch.from_numpy(e).flip(0),torch.ones(530417),(34493,34493)).coalesce().to(self.device)
        require(self.adjacency._nnz()==530417,'Native support changed')
        self.degrees=torch.bincount(self.adjacency.indices()[0],minlength=34493).to(torch.float32).unsqueeze(1)
        require(self.degrees.min().item()>=1,'Positive native degrees')
        parent=list(range(34493))
        def find(v):
            while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
            return v
        for u,v in zip(e[0].tolist(),e[1].tolist()):
            ru,rv=find(u),find(v)
            if ru!=rv:parent[rv]=ru
        components=len({find(v) for v in range(34493)})
        require(config.get('expected_components')==components==self.tour_decision['components']==1,'Root-bound original component count')
        self.graph=dict(nodes=34493,features=8415,classes=5,edges=530417,self_edges=34493,components=components,feature_nnz=self.features._nnz(),feature_sha256=self.custody['feature_sha256'],graph_sha256=self.custody['graph_sha256'],incidence_constructed=False)
        before=self.fp(self.rng())
        qualified=json.loads(checked(config['qualification']).read_text())
        require(config['tour_capsule']==qualified['canonical_CPU_capsule'],'Only actual qualified canonical capsule')
        capsule=self.t.load(checked(config['tour_capsule']),map_location='cpu',weights_only=True);digest=capsule.pop('fingerprint')
        require(digest==qualified['canonical_CPU_capsule_fingerprint'] and self.fp(capsule)==digest,'Canonical full CPU capsule fingerprint')
        require(capsule['schema']=='canonical-private-GRU-CPU-tours-and-packing-v1' and capsule['core_manifest_sha256']==CORE_SHA and capsule['graph']==self.graph,'Canonical full graph/core identity')
        capsule_blocks=capsule['blocks'];require(set(capsule_blocks)=={str(b) for b in BLOCKS},'Whole all12 persisted capsule')
        self.tours={};self.tour_bindings={};self.packing_bindings={};tick=time.monotonic()
        for block in BLOCKS:
            cpu=capsule_blocks[str(block)]['tours'];packed=capsule_blocks[str(block)]['packed']
            require(len(cpu)==len(packed)==4,'Four persisted tours and packing per block')
            require(all(v.device.type=='cpu' for group in (cpu,packed) for t in group for v in t.values() if self.t.is_tensor(v)),'Persisted capsule tensors remain CPU')
            binding=self.tour_binding(cpu);packing=[self.fp(p) for p in packed]
            require(binding==self.tour_decision['original_tour_bindings'][str(block)]==capsule['tour_bindings'][str(block)]==qualified['tour_bindings'][str(block)],'Exact original and qualified tour/RNG payload hashes')
            require(packing==capsule['packing_bindings'][str(block)]==qualified['packing_bindings'][str(block)],'Exact qualified packing payload hashes; no regeneration')
            self.tour_bindings[str(block)]=binding;self.packing_bindings[str(block)]=packing
            self.tours[block]=tuple(self.backend.to_device(p,self.device) for p in packed)
        require(self.fp(self.rng())==before,'Loading persisted tours/packing altered global RNG')
        self.capsule_receipt=config['tour_capsule'];self.capsule_fingerprint=digest;self.packing_seconds=time.monotonic()-tick
        self.fixed=(self.features,self.adjacency,self.degrees)+tuple(v for b in BLOCKS for group in (capsule_blocks[str(b)]['tours'],capsule_blocks[str(b)]['packed'],self.tours[b]) for t in group for v in t.values() if torch.is_tensor(v))
        self.versions=tuple(a._version for a in self.fixed);self.caches=[];self.cache_fingerprints=[];self.ancestor_records=[]
        del x,e,capsule,capsule_blocks;self.guard()
    def tensor(self,a):return self.t.from_numpy(a).to(self.device)
    def fp(self,v):return self.helper.fingerprint(self.t,v)
    def rng(self):return self.helper.rng(self.t,self.np,self.device)
    def restore_rng(self,v):self.helper.restore_rng(self.t,self.np,self.device,v)
    def tour_binding(self,tours):
        return [dict(seed=t['seed'],view=t['view'],components=t['components'],length=t['length'],sampling_law=t['sampling_law'],payload_sha256={k:self.fp(t[k]) for k in TOUR_KEYS}) for t in tours]
    def guard(self):
        self.t.cuda.synchronize(self.device);limits=self.config['limits']
        require(time.monotonic()-self.started<limits['wall_seconds'] and self.t.cuda.max_memory_allocated(self.device)<=limits['cuda_bytes'] and self.t.cuda.max_memory_reserved(self.device)<=limits['cuda_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limits['rss_bytes'],'Finite full-input wall/RSS/CUDA cap exceeded')
        require(self.versions==tuple(a._version for a in self.fixed),'Fixed input/tour/packing/cache mutated')
    def roles(self,block):
        result={};raw={}
        for name,count,perclass in (('train',100,20),('select',150,30)):
            item=self.custody['files'][f'block_{block}/{name}.npz']
            require(item==self.p1_binding['roles_and_graph']['TRAIN_SELECT_file_receipts'][f'block_{block}/{name}.npz'],'Original TRAIN/SELECT receipt')
            with self.np.load(checked(item),allow_pickle=False) as a:
                require(set(a.files)=={'ids','y'},'Role-only labels');ids,y=a['ids'].copy(),a['y'].copy()
            require(ids.shape==y.shape==(count,) and ids.dtype==y.dtype==self.np.int64 and len(self.np.unique(ids))==count and ids.min()>=0 and ids.max()<34493 and y.min()>=0 and y.max()<5 and (self.np.bincount(y,minlength=5)==perclass).all(),'Exact stratified source role')
            raw[name]=ids;result[name+'_ids']=self.tensor(ids);result[name+'_y']=self.tensor(y)
        require(not self.np.intersect1d(raw['train'],raw['select']).size,'Disjoint TRAIN/SELECT')
        for name,count in (('cal',150),('dev',34093)):
            ids=self.np.load(checked(self.custody['files'][f'block_{block}/{name}_ids.npy']),allow_pickle=False)
            require(ids.dtype==self.np.int64 and ids.shape==(count,) and len(self.np.unique(ids))==count and ids.min()>=0 and ids.max()<34493,'Label-free selected export memberships')
            result[name+'_ids_numpy']=ids
        all_ids=self.np.concatenate([raw['train'],raw['select'],result['cal_ids_numpy'],result['dev_ids_numpy']])
        require(self.np.array_equal(self.np.sort(all_ids),self.np.arange(34493,dtype=self.np.int64)),'Whole disjoint source role partition')
        return result
    def fresh_stream(self,seed):
        state=self.np.random.RandomState(seed).get_state()
        return dict(python=random.Random(seed).getstate(),numpy=(state[0],state[1].tolist(),state[2],state[3],state[4]),cpu=self.t.Generator(device='cpu').manual_seed(seed).get_state(),cuda=self.t.Generator(device=self.device).manual_seed(seed).get_state())
    def streams(self,block,body):return [self.fresh_stream(block+459000151+1000003*body+1009*view) for view in range(4)]
    def stage1_streams(self,block,body):
        require(body in (1,2,3),'Only new ordinary P1 dropout streams')
        return [self.fresh_stream(block+459000111+1000003*body)]
    @contextlib.contextmanager
    def scope(self,streams,index):
        outside=self.rng();self.restore_rng(streams[index])
        try:yield
        finally:streams[index]=self.rng();self.restore_rng(outside)
    def inherited_P1(self,block):
        restore_tick=time.monotonic()
        record=next(v for v in self.p1_binding['P1_body0_checkpoints'] if v['block']==block)
        owner_closed(record['original_acquisition_owner']);checked(record['original_source_manifest'])
        state=self.t.load(checked(record['checkpoint']),map_location='cpu',weights_only=True);digest=state.pop('fingerprint')
        require(set(state)=={'model','optimizer','model_training','global_rng','route_rng','meta'} and self.fp(state)==digest,'Original selected full-state digest')
        meta=state['meta']
        require(meta['block']==block and meta['arm']=='P1' and meta['body']==0 and meta['selected_step']==record['selected_step'] and meta['role_custody']==self.config['custody'] and meta['source_manifest']==record['original_source_manifest'],'Inherited own stage1 selector/source/role identity')
        require(meta['graph']['feature_sha256']==self.graph['feature_sha256'] and meta['graph']['graph_sha256']==self.graph['graph_sha256'],'Inherited input identity')
        outside=self.rng();native=self.native_core.build_arm(self.t,self.F,'P1',block);model=native['models'][0].to(self.device)
        self.native_core.inspect_parameter_ownership(native);optimizer=self.author.optimizer_class(self.t)(model.parameters())
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);model.train(state['model_training']);self.restore_rng(state['global_rng'])
        actual=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_training=model.training,global_rng=self.rng(),route_rng=state['route_rng'],meta=meta)
        require(self.fp(actual)==digest and optimizer.state['tf1_global']['updates']==record['selected_step'],'Exact original P1 model/Adam/mode/RNG restore')
        self.restore_rng(outside);model.eval();restore_seconds=time.monotonic()-restore_tick;tick=time.monotonic()
        with self.t.no_grad():hidden=model.bodies[0].first(self.features,self.adjacency,self.degrees,0).detach()
        require(hidden.shape==(34493,32) and not hidden.requires_grad and self.t.isfinite(hidden).all().item(),'Full eval clip/ReLU H without input dropout')
        self.caches.append(hidden);self.cache_fingerprints.append(self.fp(hidden));self.fixed+=tuple([hidden]);self.versions+=tuple([hidden._version])
        self.ancestor_records.append(dict(block=block,body=0,checkpoint=record['checkpoint'],original_source_manifest=record['original_source_manifest'],original_acquisition_owner=record['original_acquisition_owner'],stage1_selected_step=record['selected_step'],stage1_completed_updates=record['completed_updates'],original_selected_state_fingerprint=digest,original_full_restore_verified=True,original_restore_and_verification_seconds=restore_seconds,H_fingerprint=self.cache_fingerprints[-1],H_bytes=hidden.numel()*hidden.element_size(),H_cache_seconds=time.monotonic()-tick))
        del state,actual,optimizer,native;self.guard();return model,hidden
    def construct(self,arm,block,ancestors,caches):
        core=self.core.build_arm(self.t,self.F,arm,block,ancestors,caches,gru_backend=self.backend)
        for model in core['models']:model.to(self.device);model.tour_block=block
        audit=self.core.inspect_parameter_ownership(core);optimizers=self.core.make_optimizers(core,self.author.optimizer_class(self.t))
        for model,opt in zip(core['models'],optimizers):
            owned=[p for g in opt.param_groups for p in g['params']]
            require(len({id(p) for p in owned})==len(owned) and {id(p) for p in owned}=={id(p) for p in model.parameters() if p.requires_grad},'Complete live-only optimizer ownership')
        return core,optimizers,audit
    def construct_stage1(self,block,body):
        import stage1_native
        outside=self.fp(self.rng());core=stage1_native.build_P1(self.t,self.F,block,body)
        model=core['models'][0];model.to(self.device)
        require(self.fp(self.rng())==outside,'Isolated stage1 init altered global RNG')
        audit=stage1_native.inspect_parameter_ownership(core)
        optimizer=self.author.optimizer_class(self.t)(model.parameters())
        require(len(optimizer.state)==1 and optimizer.state['tf1_global']['updates']==0,'Fresh independent ordinary P1 optimizer')
        audit.update(init_seed=block+459000101+1000003*body,dropout_seed=block+459000111+1000003*body,post_initialization_cpu_fingerprint=self.fp(model.bodies[0].post_constructor_cpu_rng),outside_global_rng_unchanged=True,dropout_cpu_advanced_for_initialization=False)
        return core,optimizer,audit
    def native_forward(self,model,streams=None):
        return model(self.features,self.adjacency,self.degrees,None,None,None if streams is None else lambda i:self.scope(streams,i))
    def cache_selected_P1(self,model,block,body):
        model.eval();tick=time.monotonic()
        with self.t.no_grad():hidden=model.bodies[0].first(self.features,self.adjacency,self.degrees,0).detach()
        require(hidden.shape==(34493,32) and not hidden.requires_grad and self.t.isfinite(hidden).all().item(),'Ordinary selected eval H without dropout')
        self.caches.append(hidden);self.cache_fingerprints.append(self.fp(hidden));self.fixed+=(hidden,);self.versions+=(hidden._version,)
        self.guard();return hidden,dict(block=block,body=body,H_fingerprint=self.cache_fingerprints[-1],H_bytes=hidden.numel()*hidden.element_size(),H_cache_seconds=time.monotonic()-tick)
    def forward(self,model):return model(self.features,self.adjacency,self.degrees,self.tours[model.tour_block])
    def view(self,model,view,streams):return model.view_forward(self.features,self.adjacency,self.degrees,self.tours[model.tour_block],view,lambda i:self.scope(streams,i))
    def modes(self,model):return {name:module.training for name,module in model.named_modules()}
    def state(self,model,optimizer,streams,meta):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),module_modes=self.modes(model),global_rng=self.rng(),route_rng=streams,meta=meta)
    def save(self,path,model,optimizer,streams,meta):
        state=self.state(model,optimizer,streams,meta);digest=self.fp(state);state['fingerprint']=digest;self.t.save(state,path);return digest
    def restore(self,path,model,optimizer,streams,meta):
        state=self.t.load(path,map_location='cpu',weights_only=True);digest=state.pop('fingerprint')
        require(state['meta']==meta and self.fp(state)==digest,'Selected full-state serialized custody')
        saved=state['model'];actual=model.state_dict();require(set(saved)==set(actual),'Full saved model key authority')
        if 'fixed_H' in saved:
            require(any(model.fixed_H is h for h in self.caches),'Registered H is external immutable cache')
            index=next(i for i,h in enumerate(self.caches) if model.fixed_H is h)
            require(self.fp(saved['fixed_H'])==self.fp(model.fixed_H)==self.cache_fingerprints[index],'Saved H differs from external cache')
            learned={k:v for k,v in saved.items() if k!='fixed_H'};missing=model.load_state_dict(learned,strict=False)
            require(missing.missing_keys==['fixed_H'] and not missing.unexpected_keys,'Only immutable H excluded from copy restore')
        else:model.load_state_dict(saved,strict=True)
        optimizer.load_state_dict(state['optimizer']);modules=dict(model.named_modules())
        require(set(modules)==set(state['module_modes']) and all(type(v) is bool for v in state['module_modes'].values()),'All module mode authority')
        for name,module in modules.items():module.training=state['module_modes'][name]
        streams[:]=state['route_rng'];self.restore_rng(state['global_rng']);optimizer.zero_grad(set_to_none=True)
        require(self.fp(self.state(model,optimizer,streams,meta))==digest,'Exact model/Adam/all modes/global/route RNG restoration')
        self.guard();return digest
    def costs(self):
        return dict(seconds=time.monotonic()-self.started,peak_CUDA_allocated_bytes=self.t.cuda.max_memory_allocated(self.device),peak_CUDA_reserved_bytes=self.t.cuda.max_memory_reserved(self.device),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
