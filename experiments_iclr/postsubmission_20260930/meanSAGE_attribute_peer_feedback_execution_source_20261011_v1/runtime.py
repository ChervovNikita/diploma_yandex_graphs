"""Fixed meanSAGE runtime custody/RNG/checkpoint helpers; stdlib-only import."""
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
ARMS=('P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4')
BINDING_SHA='704783e8f71c2031da7e98b10849610a4811488debc6154550f51f844c4ac4c5'

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

def load(name):
    spec=importlib.util.spec_from_file_location('peer_fixed_'+name,HERE/'deps'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def admit(config,stage):
    require(config.get('blocks')==list(BLOCKS) and config.get('arms')==list(ARMS),'Frozen block/arm binding required')
    require(stage in ('custody','qualify','acquire','calibrate','score') and config.get('stage')==stage and config.get('execution_enabled') is True,'Exact root enabled stage required')
    require(os.environ.get('PILOT_OWNER_PID')==str(os.getppid()) and config.get('finite_external_owner') is True,'Direct finite owner required')
    checked(config['root_decision']);limits=config['limits']
    require(set(limits)=={'wall_seconds','cuda_bytes','rss_bytes'} and all(type(v) is int and v>0 for v in limits.values()),'Positive finite resource caps')
    manifest=json.loads(checked(config['source_manifest']).read_text())
    for rel,fixed in manifest['files'].items():
        got=receipt(HERE/rel);require(got['bytes']==fixed['bytes'] and got['sha256']==fixed['sha256'],'Fixed source differs: '+rel)
    require(receipt(HERE/'BINDING.json')['sha256']==BINDING_SHA,'Prospective probability-pool binding differs')
    return dict(stage=stage,root_decision=config['root_decision'],source_manifest=config['source_manifest'],binding=receipt(HERE/'BINDING.json'))

class Runtime:
    def __init__(self,config):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np,self.t,self.F=np,torch,F;self.config=config;self.started=time.monotonic()
        require(torch.get_default_dtype()==torch.float32,'Source float32 runtime required')
        require(config.get('expected_runtime_versions')=={'numpy':np.__version__,'torch':torch.__version__},'Root bound runtime versions required')
        self.device=torch.device('cuda:0');torch.cuda.set_device(self.device);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        self.core,self.author,self.helper=load('core'),load('author_model'),load('state_helpers')
        torch.cuda.reset_peak_memory_stats(self.device)
        self.custody=json.loads(checked(config['custody']).read_text())
        require(self.custody['complete'] and self.custody['blocks']==list(BLOCKS) and self.custody['role_rng']=='RandomState(block+5000011+223000019)','Whole fresh role binding required')
        with np.load(checked(self.custody['files']['graph.npz']),allow_pickle=False) as a:
            require(set(a.files)=={'x','edge_index'},'Graph must contain no targets');x,e=a['x'].copy(),a['edge_index'].copy()
        require(x.shape==(34493,8415) and x.dtype==np.float32 and np.isfinite(x).all() and np.logical_or(x==0,x==1).all(),'Full raw binary X')
        require(e.shape==(2,530417) and e.dtype==np.int64 and e.min()>=0 and e.max()<34493 and int((e[0]==e[1]).sum())==34493,'Full prepared native support')
        self.features=torch.from_numpy(x).to_sparse_coo().coalesce().to(self.device)
        self.adjacency=torch.sparse_coo_tensor(torch.from_numpy(e).flip(0),torch.ones(530417),(34493,34493)).coalesce().to(self.device)
        require(self.adjacency._nnz()==530417,'Prepared support changed')
        self.degrees=torch.bincount(self.adjacency.indices()[0],minlength=34493).to(torch.float32).unsqueeze(1)
        require(self.degrees.min().item()>=1,'Positive native degrees')
        self.C,self.D=self.core.incidence_from_raw_binary_features(torch,self.features)
        self.versions=tuple(a._version for a in (self.features,self.adjacency,self.degrees,self.C,self.D))
        self.graph=dict(nodes=34493,features=8415,classes=5,edges=530417,self_edges=34493,incidence_nnz=self.D._nnz(),feature_sha256=self.custody['feature_sha256'],graph_sha256=self.custody['graph_sha256'],C_sha256=self.fp(dict(indices=self.C.indices(),values=self.C.values(),shape=list(self.C.shape))),D_sha256=self.fp(dict(indices=self.D.indices(),values=self.D.values(),shape=list(self.D.shape))))
        del x,e;self.guard()
    def tensor(self,a):return self.t.from_numpy(a).to(self.device)
    def fp(self,v):return self.helper.fingerprint(self.t,v)
    def rng(self):return self.helper.rng(self.t,self.np,self.device)
    def restore_rng(self,v):self.helper.restore_rng(self.t,self.np,self.device,v)
    def guard(self):
        self.t.cuda.synchronize(self.device);l=self.config['limits']
        require(time.monotonic()-self.started<l['wall_seconds'] and self.t.cuda.max_memory_allocated(self.device)<=l['cuda_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=l['rss_bytes'],'Finite wall/RSS/CUDA cap exceeded')
        require(self.versions==tuple(a._version for a in (self.features,self.adjacency,self.degrees,self.C,self.D)),'Fixed graph/incidence input mutated')
    def roles(self,block,select):
        def labels(name,count,perclass):
            with self.np.load(checked(self.custody['files'][f'block_{block}/{name}.npz']),allow_pickle=False) as a:
                require(set(a.files)=={'ids','y'},'Role-only labels');ids,y=a['ids'].copy(),a['y'].copy()
            require(ids.shape==y.shape==(count,) and ids.dtype==y.dtype==self.np.int64 and len(self.np.unique(ids))==count and ids.min()>=0 and ids.max()<34493 and y.min()>=0 and y.max()<5 and (self.np.bincount(y,minlength=5)==perclass).all(),'Exact stratified role')
            return ids,y
        tid,ty=labels('train',100,20);v=dict(train_ids=self.tensor(tid),train_y=self.tensor(ty))
        sid=self.np.load(checked(self.custody['files'][f'block_{block}/select_ids.npy']),allow_pickle=False)
        require(sid.dtype==self.np.int64 and sid.shape==(150,) and len(self.np.unique(sid))==150 and sid.min()>=0 and sid.max()<34493 and not self.np.intersect1d(tid,sid).size,'Label-free SELECT IDs')
        v['select_ids']=self.tensor(sid)
        if select:
            sid2,sy=labels('select',150,30);require(self.np.array_equal(sid,sid2),'SELECT ID authority');v['select_y']=self.tensor(sy)
        if not select:return v
        for role,n in (('cal',150),('dev',34093)):
            ids=self.np.load(checked(self.custody['files'][f'block_{block}/{role}_ids.npy']),allow_pickle=False)
            require(ids.dtype==self.np.int64 and ids.shape==(n,),'Label-free export roles');v[role+'_ids_numpy']=ids
        all_ids=self.np.concatenate([tid,sid,v['cal_ids_numpy'],v['dev_ids_numpy']])
        require(self.np.array_equal(self.np.sort(all_ids),self.np.arange(34493,dtype=self.np.int64)),'Whole disjoint role partition')
        return v
    def streams(self,block,members,start_index=0):
        result=[]
        for m in range(start_index,start_index+members):
            seed=block+1000003*m;g=self.t.Generator(device='cpu').manual_seed(seed)
            # Match CPU source-constructor draw progression without another model.
            for shape in ((8415,32),(8415,32),(32,5),(32,5)):
                self.t.nn.init.xavier_uniform_(self.t.empty(shape),generator=g)
            ng=self.np.random.RandomState(seed).get_state()
            result.append(dict(python=random.Random(seed).getstate(),numpy=(ng[0],ng[1].tolist(),ng[2],ng[3],ng[4]),cpu=g.get_state(),cuda=self.t.Generator(device=self.device).manual_seed(seed).get_state()))
        return result
    @contextlib.contextmanager
    def scope(self,streams,index):
        outside=self.rng();self.restore_rng(streams[index])
        try:yield
        finally:streams[index]=self.rng();self.restore_rng(outside)
    def construct(self,arm,block):
        core=self.core.build_arm(self.t,self.F,arm,block)
        for model in core['models']:model.to(self.device)
        audit=self.core.inspect_parameter_ownership(core)
        optimizers=tuple(self.author.optimizer_class(self.t)(m.parameters()) for m in core['models'])
        for m,o in zip(core['models'],optimizers):
            params=[p for g in o.param_groups for p in g['params']]
            require(len({id(p) for p in params})==len(params) and {id(p) for p in params}=={id(p) for p in m.parameters() if p.requires_grad},'Complete disjoint optimizer ownership')
        return core,optimizers,audit
    def forward(self,model,streams=None):
        return model(self.features,self.adjacency,self.degrees,self.C,self.D,None if streams is None else lambda m:self.scope(streams,m))
    def save(self,path,model,optimizer,streams,meta):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_training=model.training,global_rng=self.rng(),route_rng=streams,meta=meta)
        state['fingerprint']=self.fp(state);self.t.save(state,path);return state['fingerprint']
    def restore(self,path,model,optimizer,streams,meta):
        state=self.t.load(path,map_location='cpu',weights_only=True);digest=state.pop('fingerprint')
        require(state['meta']==meta and self.fp(state)==digest,'Serialized selected-state custody')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);model.train(state['model_training'])
        streams[:]=state['route_rng'];self.restore_rng(state['global_rng'])
        actual=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_training=model.training,global_rng=self.rng(),route_rng=streams,meta=meta)
        require(self.fp(actual)==digest,'Exact model/Adam/mode/global/route RNG restore');return state
    def costs(self):
        return dict(seconds=time.monotonic()-self.started,peak_CUDA_allocated_bytes=self.t.cuda.max_memory_allocated(self.device),peak_CUDA_reserved_bytes=self.t.cuda.max_memory_reserved(self.device),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
