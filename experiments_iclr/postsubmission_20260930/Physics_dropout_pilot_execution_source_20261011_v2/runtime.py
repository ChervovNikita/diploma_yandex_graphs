"""Thin runtime reuse; importing this file uses stdlib only."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCKS = (8101, 8663, 9623)

def require(ok, message):
    if not ok: raise ValueError(message)

def route():
    require(socket.gethostname() == 'anogena-2-0', 'Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID], 'Excluded wrong GPU allocation')
    require(Path.cwd() == REPO, 'Run from authorized allocation repository')

def receipt(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): digest.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())

def checked(item):
    require(receipt(item['path']) == item, 'File receipt differs: ' + item['path'])
    return Path(item['path'])

def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)

def load(name):
    path = HERE / 'deps' / (name + '.py')
    spec = importlib.util.spec_from_file_location('pilot_' + name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module

def admit(config, stage):
    require(config.get('execution_enabled') is True and config.get('stage') == stage, 'Root must enable this exact stage')
    require(os.environ.get('PILOT_OWNER_PID') == str(os.getppid()), 'Launch through the direct finite owner')
    require(config.get('root_decision') and config.get('finite_external_owner') is True, 'Root decision and direct finite owner required')
    checked(config['root_decision'])
    limits = config['limits']
    require(set(limits) == {'wall_seconds','cuda_bytes','rss_bytes'} and all(type(x) is int and x > 0 for x in limits.values()), 'Positive finite wall/CUDA/RSS caps')
    manifest = json.loads(checked(config['source_manifest']).read_text())
    for rel, fixed in manifest['files'].items():
        actual = receipt(HERE / rel)
        require(actual['bytes'] == fixed['bytes'] and actual['sha256'] == fixed['sha256'], 'Execution source differs: ' + rel)
    for name, fixed in json.loads((HERE/'SOURCE_BINDINGS.json').read_text()).items():
        actual = receipt(HERE/'deps'/name)
        require(actual['bytes'] == fixed['bytes'] and actual['sha256'] == fixed['sha256'], 'Native dependency differs')
    return dict(stage=stage, root_decision=config['root_decision'], source_manifest=config['source_manifest'], seed_binding=receipt(HERE/'RUNTIME_SEEDS.json'))

class Runtime:
    def __init__(self, config, select=False):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np,self.t,self.F = np,torch,F
        require(torch.get_default_dtype() == torch.float32, 'Native float32 default')
        self.device = torch.device('cuda:0'); torch.cuda.set_device(self.device); torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        self.helper,self.models,self.common = load('comparator'),load('native_models'),load('native_common')
        self.joint,self.samplers,self.independent = load('joint_forward'),load('samplers'),load('independent_controller')
        self.original = load('native_tabm_training'); self.original.F,self.original.TABM_INITS = F,4
        self.config,self.started = config,time.monotonic()
        self.custody = json.loads(checked(config['custody']).read_text())
        require(self.custody['complete'] and self.custody['blocks'] == list(BLOCKS), 'Fresh complete three-block custodian')
        with np.load(checked(self.custody['files']['graph.npz']), allow_pickle=False) as a:
            require(set(a.files) == {'x','edge_index'}, 'Graph has no targets')
            x,edges = a['x'].copy(),a['edge_index'].copy()
        require(x.shape == (34493,8415) and x.dtype == np.float32 and np.isfinite(x).all() and np.logical_or(x == 0,x == 1).all(), 'Full raw binary features')
        require(edges.shape == (2,530417) and edges.dtype == np.int64 and edges.min() >= 0 and edges.max() < 34493 and int((edges[0] == edges[1]).sum()) == 34493, 'Full common support')
        self.data = SimpleNamespace(x=self.tensor(x),edge_index=self.tensor(edges),y=torch.zeros(34493,dtype=torch.long,device=self.device),train_mask=torch.zeros(34493,dtype=torch.bool,device=self.device))
        self.versions = self.data.x._version,self.data.edge_index._version
        del x,edges
        torch.cuda.reset_peak_memory_stats(self.device); self.guard()
    def tensor(self, x): return self.t.from_numpy(x).to(self.device)
    def guard(self):
        t=self.t; limits=self.config['limits']; t.cuda.synchronize(self.device)
        require(time.monotonic()-self.started < limits['wall_seconds'] and t.cuda.max_memory_allocated(self.device) <= limits['cuda_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits['rss_bytes'], 'Finite wall/CUDA/RSS cap exceeded')
        require(self.versions == (self.data.x._version,self.data.edge_index._version), 'Graph mutated')
    def roles(self, block, select=False):
        np=self.np
        with np.load(checked(self.custody['files'][f'block_{block}/train.npz']),allow_pickle=False) as a:
            require(set(a.files)=={'ids','y'}, 'TRAIN-only schema'); ids,y=a['ids'].copy(),a['y'].copy()
        require(ids.shape==y.shape==(100,) and ids.dtype==y.dtype==np.int64 and len(np.unique(ids))==100 and (np.bincount(y,minlength=5)==20).all(), 'Fresh TRAIN role')
        self.data.y.zero_();self.data.train_mask.zero_();self.data.y[self.tensor(ids)]=self.tensor(y);self.data.train_mask[self.tensor(ids)]=True
        value=dict(train_ids=self.tensor(ids),train_y=self.tensor(y))
        if select:
            with np.load(checked(self.custody['files'][f'block_{block}/select.npz']),allow_pickle=False) as a:
                require(set(a.files)=={'ids','y'}, 'SELECT-only schema'); sid,sy=a['ids'].copy(),a['y'].copy()
            require(sid.shape==sy.shape==(150,) and sid.dtype==sy.dtype==np.int64 and (np.bincount(sy,minlength=5)==30).all() and not np.intersect1d(ids,sid).size,'Fresh SELECT role')
            value.update(select_ids=self.tensor(sid),select_y=self.tensor(sy))
        return value
    def construct(self, seed, members):
        self.common.set_seed(seed)
        model=self.models.TABMModel('SAGE',2,8415,128,5,1,8,'LayerNorm',.2,tabm_inits=members,device=self.device).to(self.device)
        optimizer=self.t.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
        require({id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad},'Own optimizer')
        return model,optimizer,self.rng()
    def generator(self, block):
        value=self.t.Generator(device=self.device);value.manual_seed(block+211000007);return value
    def rng(self): return self.helper.global_rng(self.t,self.np,self.device)
    def restore_rng(self,state): self.helper.restore_rng(self.t,self.np,self.device,state)
    def fp(self,value): return self.helper.fingerprint(self.t,value)
    @contextlib.contextmanager
    def body_scope(self, streams, index):
        outside=self.rng(); self.restore_rng(streams[index])
        try: yield
        finally: streams[index]=self.rng();self.restore_rng(outside)
    def save(self,path,models,optimizers,streams,generator,meta):
        state=dict(models=[m.state_dict() for m in models],optimizers=[o.state_dict() for o in optimizers],global_rng=self.rng(),body_rng=streams,sampler_rng=None if generator is None else generator.get_state(),meta=meta)
        state['fingerprint']=self.fp(state)
        self.t.save(state,path);return state['fingerprint']
    def restore(self,path,models,optimizers,streams,generator,meta):
        state=self.t.load(path,map_location='cpu',weights_only=True)
        require(state['meta']==meta,'Checkpoint custody differs')
        digest=state.pop('fingerprint');require(self.fp(state)==digest,'Serialized checkpoint fingerprint')
        for m,s in zip(models,state['models']):m.load_state_dict(s,strict=True)
        for o,s in zip(optimizers,state['optimizers']):o.load_state_dict(s)
        streams[:]=state['body_rng'];self.restore_rng(state['global_rng'])
        if generator is not None:generator.set_state(state['sampler_rng'].cpu())
        actual=dict(models=[m.state_dict() for m in models],optimizers=[o.state_dict() for o in optimizers],global_rng=self.rng(),body_rng=streams,sampler_rng=None if generator is None else generator.get_state(),meta=meta)
        require(self.fp(actual)==digest,'Strict model+optimizer+global/body/sampler RNG restore failed')
        return state
    def raw(self,model,placement,members):
        model.eval()
        with self.t.no_grad():
            value=self.t.stack([self.joint.native_route_forward(model,self.data,self.data.x,m,placement=placement,torch=self.t) for m in range(members)])
        require(value.shape==(members,34493,5) and self.t.isfinite(value).all().item(),'Finite actual served raw member logits')
        return value
    def costs(self):
        return dict(seconds=time.monotonic()-self.started,peak_CUDA_allocated_bytes=self.t.cuda.max_memory_allocated(self.device),peak_CUDA_reserved_bytes=self.t.cuda.max_memory_reserved(self.device),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
